"""Mollweide and galactic coordinate helpers for the sky map.

The plate matches the usual astronomical Mollweide: equal area, galactic
coordinates, the Galactic Centre in the middle, longitude increasing to the
left, north at the top.
"""

from __future__ import annotations

import numpy as np

# IAU equatorial-to-galactic rotation (Hipparcos / J2000).
# galactic_xyz = EQUATORIAL_TO_GALACTIC @ equatorial_xyz
EQUATORIAL_TO_GALACTIC = np.array(
    [
        [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
        [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
        [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ],
    dtype=np.float64,
)

# Mollweide axes for a unit sphere. The ellipse is 2:1.
MOLLWEIDE_SPAN_X = 4.0 * np.sqrt(2.0)
MOLLWEIDE_SPAN_Y = 2.0 * np.sqrt(2.0)


def equatorial_to_galactic(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    xyz = np.stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)),
        axis=-1,
    )
    gal = xyz @ EQUATORIAL_TO_GALACTIC.T
    lon = np.degrees(np.arctan2(gal[..., 1], gal[..., 0])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(gal[..., 2], -1.0, 1.0)))
    return lon, lat


def galactic_to_equatorial(lon_deg, lat_deg):
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    xyz = np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)),
        axis=-1,
    )
    eq = xyz @ EQUATORIAL_TO_GALACTIC
    ra = np.degrees(np.arctan2(eq[..., 1], eq[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(eq[..., 2], -1.0, 1.0)))
    return ra, dec


def _mollweide_theta(latitude_rad, iterations=8):
    """Auxiliary angle θ in the Mollweide projection."""
    target = np.pi * np.sin(latitude_rad)
    theta = np.array(latitude_rad, dtype=np.float64, copy=True)
    polar = np.abs(np.sin(latitude_rad)) > 0.999999
    theta = np.where(polar, np.sign(np.sin(latitude_rad)) * (np.pi / 2.0), theta)
    for _ in range(iterations):
        deriv = 2.0 + 2.0 * np.cos(2.0 * theta)
        delta = (2.0 * theta + np.sin(2.0 * theta) - target) / np.where(
            np.abs(deriv) < 1e-8, 1.0, deriv
        )
        theta = np.where(polar, theta, theta - delta)
    return theta


def mollweide_forward(longitude_deg, latitude_deg, longitude_increases="left"):
    """Project spherical coordinates to Mollweide x, y (unit sphere)."""
    lon = np.array(longitude_deg, dtype=np.float64, copy=True)
    lat = np.array(latitude_deg, dtype=np.float64, copy=True)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    if longitude_increases == "left":
        lam = np.radians(-lon)
    elif longitude_increases == "right":
        lam = np.radians(lon)
    else:
        raise ValueError(longitude_increases)
    phi = np.radians(lat)
    theta = _mollweide_theta(phi)
    x = (2.0 * np.sqrt(2.0) / np.pi) * lam * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    return x, y


def mollweide_inverse(x, y):
    """Inverse Mollweide. Returns longitude and latitude in radians.

    Longitude is the raw plot longitude, positive to the right, in [-π, π].
    Callers flip the sign when astronomical longitude increases to the left.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.arcsin(np.clip(y / np.sqrt(2.0), -1.0, 1.0))
    latitude = np.arcsin(np.clip((2.0 * theta + np.sin(2.0 * theta)) / np.pi, -1.0, 1.0))
    cos_theta = np.cos(theta)
    longitude = np.empty_like(theta)
    safe = np.abs(cos_theta) > 1e-8
    longitude[safe] = np.pi * x[safe] / (2.0 * np.sqrt(2.0) * cos_theta[safe])
    longitude[~safe] = 0.0
    return longitude, latitude


def inside_mollweide(x, y):
    return (x / (2.0 * np.sqrt(2.0))) ** 2 + (y / np.sqrt(2.0)) ** 2 <= 1.0


def image_to_mollweide(col, row, width, height, pad):
    """Map pixel centers to Mollweide x, y. Row 0 is the top of the image."""
    x_norm = (col + 0.5) / width
    y_norm = (row + 0.5) / height
    usable = 1.0 - 2.0 * pad
    x = (x_norm - 0.5) * MOLLWEIDE_SPAN_X / usable
    y = (0.5 - y_norm) * MOLLWEIDE_SPAN_Y / usable
    return x, y


def mollweide_to_image(x, y, width, height, pad):
    usable = 1.0 - 2.0 * pad
    x_norm = 0.5 + x * usable / MOLLWEIDE_SPAN_X
    y_norm = 0.5 - y * usable / MOLLWEIDE_SPAN_Y
    col = x_norm * width - 0.5
    row = y_norm * height - 0.5
    return col, row


def image_to_sky(col, row, width, height, pad, longitude_increases="left"):
    """Pixel centers to longitude and latitude in degrees."""
    x, y = image_to_mollweide(col, row, width, height, pad)
    lam, phi = mollweide_inverse(x, y)
    latitude = np.degrees(phi)
    plot_lon = np.degrees(lam)
    if longitude_increases == "left":
        longitude = -plot_lon
    else:
        longitude = plot_lon
    longitude = np.mod(longitude + 180.0, 360.0) - 180.0
    return longitude, latitude, inside_mollweide(x, y)
