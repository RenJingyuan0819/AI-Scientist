"""Build practical-resolution Mollweide plates from public surveys.

The all-sky ultraviolet plate is resampled from the CDS GALEX GR6/7 far-UV and
near-UV HiPS. Gaps are filled with a log-linear fit against the WISE 12 µm
WSSA map. Visible light is the Gaia DR3 red/green/blue flux map. The Earth
plates are NASA global images in the same oval.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.units import deg
from astropy_healpix import HEALPix
from PIL import Image

from uvmap.projection import galactic_to_equatorial, image_to_sky

ROOT = Path(__file__).resolve().parents[1]
WEB_MAPS = ROOT / "web" / "maps"
CACHE = Path("/tmp/uv-sky-map-cache")

FUV_CUT = (0.0, 0.007)
NUV_CUT = (0.001, 0.1)
TILE_WIDTH = 512

SURVEYS = {
    "galex-fuv": {
        "base": "https://alasky.cds.unistra.fr/GALEX/GALEXGR6_7_FUV",
        "ext": "png",
        "kind": "gray",
    },
    "galex-nuv": {
        "base": "https://alasky.cds.unistra.fr/GALEX/GALEXGR6_7_NUV",
        "ext": "png",
        "kind": "gray",
    },
    "gaia": {
        "base": "https://alasky.cds.unistra.fr/ancillary/GaiaDR3/color-Rp-G-Bp-flux-map",
        "ext": "jpg",
        "kind": "rgb",
    },
    "wssa": {
        "base": "https://alasky.cds.unistra.fr/WSSA",
        "ext": "jpg",
        "kind": "gray",
    },
}

PLACES = [
    {
        "id": "centre",
        "name": "Galactic centre",
        "longitude": 0.0,
        "latitude": 0.0,
        "fov": 28,
        "zoom": 5.2,
        "blurb": "The brightest part of the visible Milky Way, and one of the darkest in the ultraviolet.",
    },
    {
        "id": "scorpius",
        "name": "Scorpius and Ophiuchus",
        "longitude": 353.0,
        "latitude": 18.0,
        "fov": 32,
        "zoom": 4.4,
        "blurb": "Dust above the Galactic centre, lit by hot young stars.",
    },
    {
        "id": "lmc",
        "name": "Large Magellanic Cloud",
        "longitude": 280.47,
        "latitude": -32.89,
        "fov": 14,
        "zoom": 8.0,
        "blurb": "A nearby star-forming galaxy, lower right of the oval.",
    },
    {
        "id": "pleiades",
        "name": "Pleiades",
        "longitude": 166.6,
        "latitude": -23.5,
        "fov": 8,
        "zoom": 11.0,
        "blurb": "Young cluster at galactic longitude 166.6°, latitude −23.5°.",
    },
    {
        "id": "orion",
        "name": "Orion",
        "longitude": 209.0,
        "latitude": -19.4,
        "fov": 20,
        "zoom": 6.2,
        "blurb": "Massive stars and the clouds they illuminate.",
    },
    {
        "id": "high-latitude",
        "name": "High-latitude field",
        "longitude": 180.0,
        "latitude": 55.0,
        "fov": 26,
        "zoom": 4.8,
        "blurb": "GALEX pointings with unobserved gaps between them.",
    },
]

EARTH_PLACES = [
    {"id": "atlantic", "name": "Atlantic", "longitude": -30.0, "latitude": 15.0, "zoom": 2.4},
    {"id": "europe", "name": "Europe and Africa", "longitude": 18.0, "latitude": 18.0, "zoom": 3.4},
    {"id": "east-asia", "name": "East Asia", "longitude": 115.0, "latitude": 28.0, "zoom": 3.4},
    {"id": "americas", "name": "The Americas", "longitude": -80.0, "latitude": 10.0, "zoom": 2.6},
    {"id": "north-cap", "name": "Northern cap", "longitude": 0.0, "latitude": 75.0, "zoom": 3.8},
    {"id": "south-cap", "name": "Southern cap", "longitude": 0.0, "latitude": -75.0, "zoom": 3.8},
]


def tile_url(base, order, pix, ext):
    directory = (pix // 10000) * 10000
    return f"{base}/Norder{order}/Dir{directory}/Npix{pix}.{ext}"


def fetch_tile(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    request = urllib.request.Request(url, headers={"User-Agent": "uv-sky-map-rebuild/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            dest.write_bytes(response.read())
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def download_survey(name, order, workers):
    spec = SURVEYS[name]
    ntiles = 12 * 4**order
    folder = CACHE / name / f"Norder{order}"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pix in range(ntiles):
        dest = folder / f"Npix{pix}.{spec['ext']}"
        url = tile_url(spec["base"], order, pix, spec["ext"])
        jobs.append((url, dest))
    found = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_tile, url, dest): dest for url, dest in jobs}
        for future in as_completed(futures):
            if future.result():
                found += 1
    print(f"{name}: {found}/{ntiles} tiles at order {order}")
    if found == 0:
        raise RuntimeError(f"no tiles downloaded for {name}")
    return folder, spec["ext"]


def load_tiles(folder, ext, kind):
    tiles = {}
    for path in sorted(folder.glob(f"Npix*.{ext}")):
        pix = int(path.stem.replace("Npix", ""))
        image = Image.open(path)
        array = np.asarray(image)
        if kind == "rgb":
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            tiles[pix] = array[..., :3]
        else:
            if array.ndim == 3 and array.shape[2] == 4:
                tiles[pix] = (array[..., 0], array[..., 3] > 0)
            elif array.ndim == 3:
                tiles[pix] = (array[..., 0], np.ones(array.shape[:2], dtype=bool))
            else:
                tiles[pix] = (array, np.ones(array.shape, dtype=bool))
    return tiles


def nest_xy(local, depth):
    """Tile-local column and row for a HiPS nested index.

    The raster stored by CDS puts the even bits of the nested index on the
    image row (top is row 0) and the odd bits on the column. That orientation
    was checked against the WSSA galactic plane.
    """
    local = np.asarray(local, dtype=np.int64)
    columns = np.zeros(local.shape, dtype=np.int32)
    rows = np.zeros(local.shape, dtype=np.int32)
    for bit in range(depth):
        rows |= ((local >> (2 * bit)) & 1).astype(np.int32) << bit
        columns |= ((local >> (2 * bit + 1)) & 1).astype(np.int32) << bit
    return columns, rows


def healpix_indices(ra, dec, order, width=TILE_WIDTH):
    nside = (1 << order) * width
    healpix = HEALPix(nside=nside, order="nested")
    ipix = np.asarray(healpix.lonlat_to_healpix(np.asarray(ra) * deg, np.asarray(dec) * deg), dtype=np.int64)
    shift = int(np.log2(width)) * 2
    tile = ipix >> shift
    local = ipix & ((width * width) - 1)
    columns, rows = nest_xy(local, int(np.log2(width)))
    return tile, rows, columns


def sample_gray(ra, dec, tiles, order):
    tile, rows, columns = healpix_indices(ra, dec, order)
    values = np.zeros(tile.shape, dtype=np.float32)
    valid = np.zeros(tile.shape, dtype=bool)
    for index, (gray, mask) in tiles.items():
        selected = tile == index
        if not np.any(selected):
            continue
        rr = rows[selected]
        cc = columns[selected]
        pix_mask = mask[rr, cc]
        valid[selected] = pix_mask
        values[selected] = np.where(pix_mask, gray[rr, cc], 0)
    return values, valid


def sample_rgb(ra, dec, tiles, order):
    tile, rows, columns = healpix_indices(ra, dec, order)
    values = np.zeros(tile.shape + (3,), dtype=np.float32)
    valid = np.zeros(tile.shape, dtype=bool)
    for index, rgb in tiles.items():
        selected = tile == index
        if not np.any(selected):
            continue
        taken = rgb[rows[selected], columns[selected]]
        values[selected] = taken
        valid[selected] = True
    return values, valid


def sky_coordinates(width, height, pad, longitude_increases):
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    grid_c, grid_r = np.meshgrid(columns, rows)
    longitude, latitude, inside = image_to_sky(
        grid_c, grid_r, width, height, pad, longitude_increases
    )
    return longitude, latitude, inside


def linear_from_stretch(gray, valid, low, high):
    scaled = (gray / 255.0) ** 2
    return np.where(valid, low + scaled * (high - low), np.nan)


def _bin_medians(dust, values, valid, edges):
    medians = np.empty(len(edges) - 1, dtype=np.float64)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = valid & (dust >= low) & (dust < high) & np.isfinite(values)
        medians[index] = np.median(values[selected]) if np.count_nonzero(selected) >= 40 else np.nan
    known = np.isfinite(medians)
    positions = np.arange(len(medians))
    if known.sum() == 0:
        medians[:] = np.nanmedian(values[valid])
    elif known.sum() == 1:
        medians[:] = medians[known][0]
    else:
        medians[~known] = np.interp(positions[~known], positions[known], medians[known])
    return medians


def _lookup_bins(dust, edges, medians):
    index = np.digitize(dust, edges) - 1
    index = np.clip(index, 0, len(medians) - 1)
    return medians[index]


def fill_ultraviolet(fuv, nuv, dust, inside, coeffs=None):
    """Measured pixels stay. Missing far-UV is scaled from near-UV.

    Empty pixels take the median ultraviolet of GALEX pixels with similar
    12 µm brightness. The inner plane, where that map saturates and GALEX
    did not look, is kept faint: dust there absorbs ultraviolet light.
    """
    both = inside & np.isfinite(fuv) & np.isfinite(nuv)
    nuv_only = inside & np.isfinite(nuv) & ~np.isfinite(fuv)
    neither = inside & ~np.isfinite(fuv) & ~np.isfinite(nuv)
    dust_ok = np.isfinite(dust)
    plane_level = 150.0
    if coeffs is None:
        edges = np.linspace(0.0, plane_level, 16)
        train = both & dust_ok & (dust < plane_level)
        if np.count_nonzero(train) < 1000:
            raise RuntimeError("not enough overlap to fit the ultraviolet gap model")
        nuv_median = _bin_medians(dust, nuv, train, edges)
        ratio_median = _bin_medians(dust, fuv / np.maximum(nuv, 1e-8), train, edges)
        ratio = float(np.nanmedian((fuv[both] / nuv[both])))
        fit_pixels = int(np.count_nonzero(train))
        faint = float(np.nanpercentile(nuv[both], 8))
    else:
        edges = np.asarray(coeffs["dustEdges"], dtype=np.float64)
        nuv_median = np.asarray(coeffs["nuvMedian"], dtype=np.float64)
        ratio_median = np.asarray(coeffs["fuvOverNuv"], dtype=np.float64)
        ratio = float(coeffs["medianFuvOverNuv"])
        fit_pixels = int(coeffs.get("fitPixels", 0))
        faint = float(coeffs["faintNuv"])
        plane_level = float(coeffs["planeDust"])

    fuv_out = fuv.copy()
    nuv_out = nuv.copy()
    provenance = np.zeros(fuv.shape, dtype=np.uint8)
    provenance[both] = 1

    provenance[nuv_only] = 2
    colour = _lookup_bins(np.where(dust_ok, dust, 0.0), edges, ratio_median)
    fuv_out[nuv_only] = nuv[nuv_only] * np.where(np.isfinite(colour[nuv_only]), colour[nuv_only], ratio)

    missing = neither & dust_ok
    plane = missing & (dust >= plane_level)
    cloudy = missing & (dust < plane_level)
    if np.any(cloudy):
        nuv_hat = _lookup_bins(dust, edges, nuv_median)
        ratio_hat = _lookup_bins(dust, edges, ratio_median)
        nuv_out[cloudy] = nuv_hat[cloudy]
        fuv_out[cloudy] = nuv_hat[cloudy] * ratio_hat[cloudy]
        provenance[cloudy] = 3
    if np.any(plane):
        nuv_out[plane] = faint
        fuv_out[plane] = faint * ratio
        provenance[plane] = 3

    hi = np.nanpercentile(nuv[both], 99.8)
    nuv_out = np.clip(nuv_out, 1e-8, hi)
    fuv_out = np.clip(fuv_out, 1e-8, hi)
    return fuv_out, nuv_out, provenance, {
        "dustEdges": [float(v) for v in edges],
        "nuvMedian": [float(v) for v in nuv_median],
        "fuvOverNuv": [float(v) for v in ratio_median],
        "medianFuvOverNuv": ratio,
        "planeDust": plane_level,
        "faintNuv": faint,
        "fitPixels": fit_pixels,
    }


def stretch_limits(values, valid, low_pct, high_pct):
    sample = values[valid]
    return np.percentile(sample, [low_pct, high_pct])


def uv_rgb(fuv, nuv, valid, ratio_limits, level_limits):
    ratio = np.log10(np.maximum(fuv, 1e-8) / np.maximum(nuv, 1e-8))
    t0, t1 = ratio_limits
    tone = np.clip((ratio - t0) / (t1 - t0 + 1e-8), 0.0, 1.0)
    gold = np.array([228.0, 186.0, 96.0])
    blue = np.array([52.0, 122.0, 216.0])
    violet = np.array([142.0, 90.0, 216.0])
    low = tone <= 0.55
    u = np.where(low, tone / 0.55, (tone - 0.55) / 0.45)
    start = np.where(low[..., None], gold, blue)
    end = np.where(low[..., None], blue, violet)
    color = start + (end - start) * u[..., None]
    level = np.log10(np.maximum(np.maximum(nuv, fuv * 0.35), 1e-8))
    lo, hi = level_limits
    brightness = np.clip((level - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    brightness = np.sqrt(brightness)
    rgb = color * (0.045 + 0.955 * brightness[..., None])
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    rgb = np.clip(rgb, 0, 255)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    out = np.dstack([rgb.astype(np.uint8), alpha])
    return out


def colorize_infrared(gray, valid):
    sample = gray[valid]
    lo, hi = np.percentile(sample, [2.0, 99.6])
    tone = np.clip((gray - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    tone = np.power(tone, 0.72)
    shadow = np.array([18.0, 8.0, 12.0])
    copper = np.array([186.0, 78.0, 24.0])
    pale = np.array([255.0, 220.0, 166.0])
    low = tone <= 0.62
    u = np.where(low, tone / 0.62, (tone - 0.62) / 0.38)
    start = np.where(low[..., None], shadow, copper)
    end = np.where(low[..., None], copper, pale)
    rgb = np.clip(start + (end - start) * u[..., None], 0, 255).astype(np.uint8)
    plate = np.zeros(gray.shape + (4,), dtype=np.uint8)
    plate[..., :3] = rgb
    plate[..., 3] = np.where(valid, 255, 0)
    return plate


def colorize_visible(rgb, valid):
    plate = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    luminance = rgb.mean(axis=2)
    sample = luminance[valid]
    lo, hi = np.percentile(sample, [1.0, 99.5])
    scale = np.clip((luminance - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    scale = np.power(scale, 0.85)
    gain = scale / np.maximum(luminance / 255.0, 1e-3)
    painted = np.clip(rgb * gain[..., None], 0, 255)
    plate[..., :3] = painted.astype(np.uint8)
    plate[..., 3] = np.where(valid, 255, 0)
    return plate


def provenance_overlay(provenance):
    overlay = np.zeros(provenance.shape + (4,), dtype=np.uint8)
    nuv_only = provenance == 2
    model = provenance == 3
    overlay[nuv_only] = (36, 176, 168, 95)
    overlay[model] = (214, 78, 132, 120)
    return overlay


def gnomonic_coordinates(size, longitude, latitude, fov_deg):
    half = np.tan(np.radians(fov_deg) / 2.0)
    xs = np.linspace(half, -half, size)
    ys = np.linspace(half, -half, size)
    x, y = np.meshgrid(xs, ys)
    lon0 = np.radians(longitude)
    lat0 = np.radians(latitude)
    rho = np.sqrt(x * x + y * y)
    angle = np.arctan(rho)
    sin_c = np.sin(angle)
    cos_c = np.cos(angle)
    safe = rho > 1e-8
    lat = np.arcsin(
        np.clip(
            cos_c * np.sin(lat0) + np.divide(y * sin_c * np.cos(lat0), rho, out=np.zeros_like(rho), where=safe),
            -1.0,
            1.0,
        )
    )
    lat = np.where(safe, lat, lat0)
    dlon = np.arctan2(
        x * sin_c,
        rho * np.cos(lat0) * cos_c - y * np.sin(lat0) * sin_c,
    )
    lon = lon0 + np.where(safe, dlon, 0.0)
    return np.degrees(lon) % 360.0, np.degrees(lat)


def save_image(path, array, quality=76):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array)
    if path.suffix == ".webp":
        image.save(path, "WEBP", quality=quality, method=4)
    elif path.suffix == ".png":
        image.save(path, "PNG", optimize=True)
    else:
        image.save(path)


def download_equirect(url, dest, width, height):
    if dest.exists() and dest.stat().st_size > 1000:
        image = Image.open(dest).convert("RGB")
        if image.size == (width, height):
            return np.asarray(image)
    request = urllib.request.Request(url, headers={"User-Agent": "uv-sky-map-rebuild/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    image = Image.open(dest).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        image.save(dest, quality=90)
    return np.asarray(image)


def worldview_url(layer, when, width, height):
    return (
        "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
        f"&FORMAT=image/jpeg&WIDTH={width}&HEIGHT={height}&CRS=EPSG:4326"
        f"&BBOX=-90,-180,90,180&TIME={when}&LAYERS={layer}"
    )


def composite_snapshots(layer, dates, width, height, cache_name):
    stack = []
    for when in dates:
        url = worldview_url(layer, when, width, height)
        dest = CACHE / "earth" / f"{cache_name}-{when}.jpg"
        print(f"earth {cache_name} {when}")
        stack.append(download_equirect(url, dest, width, height).astype(np.float32))
    cube = np.stack(stack, axis=0)
    valid = cube.max(axis=-1) > 14
    count = valid.sum(axis=0)
    painted = np.zeros_like(cube[0])
    for channel in range(3):
        channel_sum = (cube[..., channel] * valid).sum(axis=0)
        painted[..., channel] = np.divide(
            channel_sum, count, out=np.zeros(count.shape, dtype=np.float32), where=count > 0
        )
    return painted, count > 0


def fill_along_rows(image, valid):
    height, width, _ = image.shape
    for row in range(height):
        good = np.flatnonzero(valid[row])
        if good.size == 0:
            continue
        bad = np.flatnonzero(~valid[row])
        if bad.size == 0:
            continue
        position = np.searchsorted(good, bad)
        left_i = np.clip(position - 1, 0, good.size - 1)
        right_i = np.clip(position, 0, good.size - 1)
        left = good[left_i]
        right = good[right_i]
        dist_left = np.minimum(np.abs(bad - left), width - np.abs(bad - left))
        dist_right = np.minimum(np.abs(bad - right), width - np.abs(bad - right))
        choose = np.where(dist_left <= dist_right, left, right)
        image[row, bad] = image[row, choose]
        valid[row, bad] = True


def extrapolate_north_cap(image, valid, limit=78.0):
    """Fill the northern polar-night gap from the first bright latitude.

    Daily OMI maps in January stay empty well south of 78° N. The whole dark
    cap is replaced by that first bright row of measurements.
    """
    height = image.shape[0]
    latitudes = 90.0 - (np.arange(height) + 0.5) * 180.0 / height
    source = None
    for row in range(height):
        measured = valid[row]
        if measured.mean() > 0.45 and image[row][measured].mean() > 40:
            source = row
            break
    if source is None:
        return
    must = latitudes >= limit
    for row in range(source):
        if must[row] or valid[row].mean() < 0.45:
            image[row] = image[source]
            valid[row] = True
    return float(latitudes[source])


def grow_valid(image, valid, passes):
    height, width, _ = image.shape
    for _ in range(passes):
        up = np.zeros_like(valid)
        down = np.zeros_like(valid)
        left = np.zeros_like(valid)
        right = np.zeros_like(valid)
        up[:-1] = valid[1:]
        down[1:] = valid[:-1]
        left[:, :-1] = valid[:, 1:]
        right[:, 1:] = valid[:, :-1]
        grow = ~valid & (up | down | left | right)
        if not np.any(grow):
            break
        acc = np.zeros_like(image)
        weight = np.zeros((height, width), dtype=np.float32)
        for shift, mask in (((-1, 0), up), ((1, 0), down), ((0, -1), left), ((0, 1), right)):
            source = np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)
            acc[mask] += source[mask]
            weight[mask] += 1.0
        image[grow] = acc[grow] / weight[grow, None]
        valid[grow] = True


def paint_geographic(src, valid, width, height, pad):
    longitude, latitude, inside = sky_coordinates(width, height, pad, "right")
    source_h, source_w = src.shape[:2]
    x = (longitude + 180.0) / 360.0 * source_w - 0.5
    y = (90.0 - latitude) / 180.0 * source_h - 0.5
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    x0 = np.mod(x0, source_w)
    x1 = np.mod(x0 + 1, source_w)
    y0 = np.clip(y0, 0, source_h - 1)
    y1 = np.clip(y0 + 1, 0, source_h - 1)
    sample = src.astype(np.float32)
    v00 = sample[y0, x0]
    v01 = sample[y0, x1]
    v10 = sample[y1, x0]
    v11 = sample[y1, x1]
    wx = wx[..., None]
    wy = wy[..., None]
    painted = v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy
    plate = np.zeros((height, width, 4), dtype=np.uint8)
    rgb = np.clip(painted, 0, 255).astype(np.uint8)
    plate[..., :3] = rgb
    plate[..., 3] = np.where(inside, 255, 0)
    # Keep ocean / nodata dark rather than inventing a colour, but still inside the oval.
    if valid is not None:
        nearest_col = np.mod(np.rint(x).astype(np.int32), source_w)
        nearest_row = np.clip(np.rint(y).astype(np.int32), 0, source_h - 1)
        source_valid = valid[nearest_row, nearest_col]
        hole = inside & ~source_valid
        plate[hole, :3] = (8, 10, 16)
        plate[hole, 3] = 255
    return plate


def build_sky(order, width, pad, closeup):
    for name in ("galex-fuv", "galex-nuv", "gaia", "wssa"):
        download_survey(name, order, workers=24)
    folders = {name: CACHE / name / f"Norder{order}" for name in SURVEYS}
    print("loading tiles")
    fuv_tiles = load_tiles(folders["galex-fuv"], "png", "gray")
    nuv_tiles = load_tiles(folders["galex-nuv"], "png", "gray")
    gaia_tiles = load_tiles(folders["gaia"], "jpg", "rgb")
    wssa_tiles = load_tiles(folders["wssa"], "jpg", "gray")

    height = width // 2
    print(f"sampling sky {width}x{height}")
    longitude, latitude, inside = sky_coordinates(width, height, pad, "left")
    ra, dec = galactic_to_equatorial(np.mod(longitude, 360.0), latitude)
    fuv_gray, fuv_valid = sample_gray(ra, dec, fuv_tiles, order)
    nuv_gray, nuv_valid = sample_gray(ra, dec, nuv_tiles, order)
    gaia, gaia_valid = sample_rgb(ra, dec, gaia_tiles, order)
    dust_gray, dust_valid = sample_gray(ra, dec, wssa_tiles, order)

    fuv = linear_from_stretch(fuv_gray, fuv_valid & inside, *FUV_CUT)
    nuv = linear_from_stretch(nuv_gray, nuv_valid & inside, *NUV_CUT)
    dust = np.where(dust_valid & inside, dust_gray, np.nan)
    fuv_filled, nuv_filled, provenance, fit = fill_ultraviolet(fuv, nuv, dust, inside)

    both = provenance == 1
    ratio = np.log10(np.maximum(fuv[both], 1e-8) / np.maximum(nuv[both], 1e-8))
    ratio_limits = np.percentile(ratio, [12, 88])
    level = np.log10(np.maximum(np.maximum(nuv[both], fuv[both] * 0.35), 1e-8))
    level_limits = np.percentile(level, [4, 99.3])

    measured_valid = (provenance == 1) | (provenance == 2)
    measured = uv_rgb(np.where(both, fuv, nuv), np.where(np.isfinite(nuv), nuv, fuv), measured_valid, ratio_limits, level_limits)
    # Near-UV-only measured pixels are drawn as near-UV light, without the estimated far-UV.
    nuv_only = provenance == 2
    gold = uv_rgb(nuv * 0.25, nuv, nuv_only, ratio_limits, level_limits)
    measured[nuv_only] = gold[nuv_only]

    complete_valid = provenance > 0
    complete = uv_rgb(fuv_filled, nuv_filled, complete_valid, ratio_limits, level_limits)
    empty = inside & (provenance == 0)
    for plate in (complete, measured):
        plate[empty, :3] = (7, 9, 14)
        plate[empty, 3] = 255
        plate[~inside, 3] = 0

    visible = colorize_visible(gaia, gaia_valid & inside)
    visible[~inside, 3] = 0
    infrared = colorize_infrared(dust_gray, dust_valid & inside)
    infrared[~inside, 3] = 0
    overlay = provenance_overlay(provenance)
    overlay[~inside, 3] = 0

    sky_fraction = float(inside.mean())
    counts = {
        "bothBands": float((provenance == 1).sum() / inside.sum()),
        "nearUvOnly": float((provenance == 2).sum() / inside.sum()),
        "dustModel": float((provenance == 3).sum() / inside.sum()),
        "unfilled": float((inside & (provenance == 0)).sum() / inside.sum()),
    }
    print("coverage", counts, "ellipse", round(sky_fraction, 3))
    print("fit", fit)

    maps = WEB_MAPS
    save_image(maps / "sky-ultraviolet.webp", complete)
    save_image(maps / "sky-ultraviolet-measured.webp", measured)
    save_image(maps / "sky-visible.webp", visible)
    save_image(maps / "sky-infrared.webp", infrared)
    save_image(maps / "sky-provenance.webp", overlay, quality=70)
    prov_l = np.zeros(provenance.shape, dtype=np.uint8)
    prov_l[provenance == 1] = 80
    prov_l[provenance == 2] = 160
    prov_l[provenance == 3] = 240
    save_image(maps / "sky-provenance-code.png", prov_l)

    if closeup:
        print("close-ups")
        for place in PLACES:
            lon, lat = gnomonic_coordinates(closeup, place["longitude"], place["latitude"], place["fov"])
            cra, cdec = galactic_to_equatorial(lon, lat)
            cfuv_g, cfuv_v = sample_gray(cra, cdec, fuv_tiles, order)
            cnuv_g, cnuv_v = sample_gray(cra, cdec, nuv_tiles, order)
            cgaia, cgaia_v = sample_rgb(cra, cdec, gaia_tiles, order)
            cdust_g, cdust_v = sample_gray(cra, cdec, wssa_tiles, order)
            whole = np.ones(lon.shape, dtype=bool)
            cfuv = linear_from_stretch(cfuv_g, cfuv_v, *FUV_CUT)
            cnuv = linear_from_stretch(cnuv_g, cnuv_v, *NUV_CUT)
            cdust = np.where(cdust_v, cdust_g, np.nan)
            ff, nf, prov, _ = fill_ultraviolet(cfuv, cnuv, cdust, whole, coeffs=fit)
            # Keep the all-sky colour meaning, but stretch brightness on this field.
            local_level = np.log10(np.maximum(np.maximum(nf[prov > 0], ff[prov > 0] * 0.35), 1e-8))
            local_limits = np.percentile(local_level, [3, 99.4]) if np.any(prov > 0) else level_limits
            plate = uv_rgb(ff, nf, prov > 0, ratio_limits, local_limits)
            save_image(maps / f"closeup-{place['id']}-ultraviolet.webp", plate, quality=78)
            save_image(maps / f"closeup-{place['id']}-visible.webp", colorize_visible(cgaia, cgaia_v), quality=78)
            save_image(
                maps / f"closeup-{place['id']}-infrared.webp",
                colorize_infrared(cdust_g, cdust_v),
                quality=78,
            )
    return {
        "order": order,
        "coverage": counts,
        "fit": fit,
        "ratioLimits": [float(v) for v in ratio_limits],
        "equatorArcminPerPixel": 360.0 * 60.0 / width,
    }


def build_earth(width, pad):
    height = width // 2
    source_w, source_h = 2048, 1024
    ozone, ozone_valid = composite_snapshots(
        "OMI_Ozone_TOMS_Total_Column",
        [f"2016-01-{day:02d}" for day in (2, 6, 10, 14, 18, 22, 26)],
        source_w,
        source_h,
        "ozone",
    )
    fill_along_rows(ozone, ozone_valid)
    extrapolate_north_cap(ozone, ozone_valid, 78.0)

    temperature, temperature_valid = composite_snapshots(
        "MODIS_Terra_Land_Surface_Temp_Day",
        [f"2016-07-{day:02d}" for day in (2, 6, 10, 14, 18, 22, 26, 30)],
        source_w,
        source_h,
        "lst",
    )
    grow_valid(temperature, temperature_valid, passes=8)

    night = download_equirect(
        worldview_url("VIIRS_Black_Marble", "2016-06-01", source_w, source_h),
        CACHE / "earth" / "black-marble.jpg",
        source_w,
        source_h,
    )
    night_valid = night.max(axis=2) > 0

    maps = WEB_MAPS
    save_image(maps / "earth-ultraviolet.webp", paint_geographic(ozone, ozone_valid, width, height, pad))
    save_image(maps / "earth-visible.webp", paint_geographic(night, night_valid, width, height, pad))
    save_image(
        maps / "earth-infrared.webp",
        paint_geographic(temperature, temperature_valid, width, height, pad),
    )


def write_meta(width, sky_info, pad):
    meta = {
        "width": width,
        "height": width // 2,
        "pad": pad,
        "projection": "mollweide",
        "sky": {
            "frame": "galactic",
            "longitudeIncreases": "left",
            "centre": "galactic centre",
        },
        "earth": {
            "frame": "geographic",
            "longitudeIncreases": "right",
            "centre": "prime meridian",
        },
        "equatorArcminPerPixel": sky_info["equatorArcminPerPixel"],
        "hipsOrder": sky_info["order"],
        "coverage": sky_info["coverage"],
        "gapModel": sky_info["fit"],
        "assets": {
            "sky": {
                "ultraviolet": "maps/sky-ultraviolet.webp",
                "ultravioletMeasured": "maps/sky-ultraviolet-measured.webp",
                "visible": "maps/sky-visible.webp",
                "infrared": "maps/sky-infrared.webp",
                "provenance": "maps/sky-provenance.webp",
                "provenanceCode": "maps/sky-provenance-code.png",
            },
            "earth": {
                "ultraviolet": "maps/earth-ultraviolet.webp",
                "visible": "maps/earth-visible.webp",
                "infrared": "maps/earth-infrared.webp",
            },
        },
        "places": PLACES,
        "earthPlaces": EARTH_PLACES,
        "provenanceLegend": {
            "80": "GALEX far-UV and near-UV",
            "160": "GALEX near-UV only; far-UV estimated",
            "240": "No GALEX coverage; estimated from 12 µm dust",
        },
    }
    for place in PLACES:
        meta["assets"]["sky"].setdefault("closeups", {})[place["id"]] = {
            "ultraviolet": f"maps/closeup-{place['id']}-ultraviolet.webp",
            "visible": f"maps/closeup-{place['id']}-visible.webp",
            "infrared": f"maps/closeup-{place['id']}-infrared.webp",
        }
    path = WEB_MAPS / "meta.json"
    path.write_text(json.dumps(meta, indent=2))
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description="Build the ultraviolet sky plates")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--pad", type=float, default=0.012)
    parser.add_argument("--closeup", type=int, default=720)
    parser.add_argument("--skip-earth", action="store_true")
    parser.add_argument("--skip-closeups", action="store_true")
    parser.add_argument("--skip-sky", action="store_true")
    args = parser.parse_args()
    if args.width % 2:
        raise SystemExit("width must be even")
    sky_info = {
        "order": args.order,
        "coverage": {},
        "fit": {},
        "equatorArcminPerPixel": 360.0 * 60.0 / args.width,
    }
    if not args.skip_sky:
        sky_info = build_sky(args.order, args.width, args.pad, 0 if args.skip_closeups else args.closeup)
    if not args.skip_earth:
        build_earth(args.width, args.pad)
    write_meta(args.width, sky_info, args.pad)


if __name__ == "__main__":
    main()
