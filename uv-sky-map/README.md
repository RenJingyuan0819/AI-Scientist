# Ultraviolet sky

A Mollweide viewer for a practical all-sky ultraviolet plate, plus the same oval in visible light, 12 µm dust, and three views of Earth.

The plate is rebuilt from public surveys. The nside 2048 far-UV and near-UV products described at [menard.pha.jhu.edu/uvmap](https://menard.pha.jhu.edu/uvmap/) were not available as FITS or HiPS downloads when this viewer was made.

## Open the viewer

```bash
python3 -m http.server 8765 --directory web
```

Then open <http://127.0.0.1:8765/>. Drag to pan, scroll or use the zoom buttons to zoom, and switch Sky / Earth and Ultraviolet / Visible / Infrared. On the sky, Estimates hides the filled pixels and Provenance tints near-UV-only pixels (teal) and dust estimates (rose).

## Rebuild the plates

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 -m uvmap.build
```

Tiles are cached under `/tmp/uv-sky-map-cache`. The default plate is HiPS order 2 (about 1.7 arcmin in the source tiles) drawn at 4096 pixels across, about 5 arcmin per pixel on the equator of the oval. Close-ups of six fields are sampled from the same tiles.

```bash
PYTHONPATH=. python3 -m unittest tests.test_projection
```

## What the ultraviolet plate contains

- Far-UV and near-UV from the CDS HiPS of GALEX GR6/7 (`CDS/P/GALEXGR6_7/FUV` and `NUV`). Display stretches are inverted with the published pixel cuts, then clipped.
- Colour from the far-UV / near-UV ratio: violet where far-UV is relatively strong, gold where near-UV is relatively strong.
- Gaps: near-UV-only pixels keep the measured near-UV and a far-UV scaled from dust-matched overlap. Pixels with neither band take the median ultraviolet of GALEX pixels at similar WSSA 12 µm brightness. The saturated inner plane is left faint.
- Visible: Gaia DR3 Rp/G/Bp flux map. Infrared: WISE 12 µm WSSA.
- Earth: January 2016 Aura/OMI total ozone (orbital gaps filled across a week; the polar-night cap, empty down to about 66° N in this composite, copied from the first bright latitude), VIIRS Black Marble, and a July 2016 MODIS land-surface temperature composite.

## Still required for the full-resolution published map

The project page specifies HEALPix nside 2048 (about 1.7 arcmin) Galactic maps in photons cm⁻² s⁻¹ sr⁻¹ Å⁻¹, with per-pixel weights for GALEX, near-UV-informed far-UV, Swift-UVOT, FIMS/SPEAR, and the statistical model, plus a calibrated uncertainty. Building that product needs those files, or a new reduction of the GALEX tiles together with Swift-UVOT, FIMS/SPEAR, TD-1, and Gaia stellar fluxes. This repository does not reproduce that reduction.
