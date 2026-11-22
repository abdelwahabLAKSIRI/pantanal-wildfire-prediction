"""GOES-R ABI fixed-grid geolocation, and the Pantanal index window.

The ABI fire product is stored on the satellite's fixed grid: two scan-angle
axes (x east/west, y north/south, both radians) rather than lat/lon. Turning
those into geographic coordinates is the standard GOES-R transform documented in
the Product User Guide (Vol. 3, section 5.1.2.8).

Nothing here is Pantanal-specific except :func:`pantanal_window`, which finds the
smallest (row, col) rectangle of the 5424x5424 full disk covering a lat/lon box,
so every downstream file read touches ~0.1% of the array instead of all 29M
pixels.
"""

from __future__ import annotations

import numpy as np

# GRS80 / GOES-R constants, matching the goes_imager_projection attributes.
REQ = 6378137.0            # semi-major axis (m)
RPOL = 6356752.31414       # semi-minor axis (m)
H = 42164160.0             # satellite distance from Earth centre = 35786023 + REQ
LON_ORIGIN = -75.0         # sub-satellite longitude for GOES-16 (degrees)

_E2 = 1.0 - (RPOL ** 2) / (REQ ** 2)   # eccentricity squared
_RATIO = (REQ ** 2) / (RPOL ** 2)


def scan_to_lonlat(x, y, lon_origin=LON_ORIGIN):
    """Scan angles (radians) -> (lon, lat) in degrees.

    Points that miss the Earth's disk come back as NaN, which is what the
    quadratic's negative discriminant means physically.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")

    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)

    a = sin_x ** 2 + cos_x ** 2 * (cos_y ** 2 + _RATIO * sin_y ** 2)
    b = -2.0 * H * cos_x * cos_y
    c = H ** 2 - REQ ** 2

    disc = b ** 2 - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2.0 * a)
    rs = np.where(disc < 0, np.nan, rs)

    sx = rs * cos_x * cos_y
    sy = -rs * sin_x
    sz = rs * cos_x * sin_y

    lat = np.degrees(np.arctan(_RATIO * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
    lon = lon_origin - np.degrees(np.arctan(sy / (H - sx)))
    return lon, lat


def lonlat_to_scan(lon, lat, lon_origin=LON_ORIGIN):
    """(lon, lat) in degrees -> scan angles (x, y) in radians.

    The inverse of :func:`scan_to_lonlat`. Points hidden behind the limb come
    back as NaN.
    """
    lon = np.radians(np.asarray(lon, dtype="float64"))
    lat = np.radians(np.asarray(lat, dtype="float64"))
    lon0 = np.radians(lon_origin)

    # geodetic -> geocentric latitude
    lat_c = np.arctan((RPOL ** 2 / REQ ** 2) * np.tan(lat))
    rc = RPOL / np.sqrt(1.0 - _E2 * np.cos(lat_c) ** 2)

    sx = H - rc * np.cos(lat_c) * np.cos(lon - lon0)
    sy = -rc * np.cos(lat_c) * np.sin(lon - lon0)
    sz = rc * np.sin(lat_c)

    # Visibility test from the PUG: the point must be on the near side.
    visible = H * (H - sx) >= (sy ** 2 + _RATIO * sz ** 2)

    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return np.where(visible, x, np.nan), np.where(visible, y, np.nan)


def pantanal_window(x_axis, y_axis, lat_min, lat_max, lon_min, lon_max, pad=4):
    """Smallest (row_slice, col_slice) of the full disk covering a lat/lon box.

    Uses the inverse transform on a dense sample of the box's perimeter rather
    than only its four corners: the fixed grid is curvilinear, so the extreme
    scan angles generally sit on an edge, not at a corner.
    """
    n = 200
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)

    edge_lon = np.concatenate([lons, lons, np.full(n, lon_min), np.full(n, lon_max)])
    edge_lat = np.concatenate([np.full(n, lat_min), np.full(n, lat_max), lats, lats])

    xs, ys = lonlat_to_scan(edge_lon, edge_lat)
    if np.isnan(xs).any():
        raise ValueError("part of the requested box is not visible from GOES-16")

    # y_axis descends (north to south); x_axis ascends (west to east).
    col0 = int(np.searchsorted(x_axis, xs.min()) - pad)
    col1 = int(np.searchsorted(x_axis, xs.max()) + pad)
    row0 = int(np.searchsorted(-y_axis, -ys.max()) - pad)
    row1 = int(np.searchsorted(-y_axis, -ys.min()) + pad)

    col0, row0 = max(col0, 0), max(row0, 0)
    col1 = min(col1, len(x_axis))
    row1 = min(row1, len(y_axis))
    return slice(row0, row1), slice(col0, col1)


# --- Fire mask semantics -------------------------------------------------
# Codes 10-15 are fire; 30-35 are the same categories after temporal filtering.
FIRE_CODES_HIGH = {10, 11, 30, 31}                    # good + saturated
FIRE_CODES_ALL = {10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35}

# 100 = processed_no_fire_pixel: the pixel WAS examined and nothing was burning.
# This is the code that makes a 0 label an observation rather than an absence of
# one -- the thing the INPE overpass product could never provide.
NO_FIRE_CODE = 100

# Everything else is unobservable rather than fire-free: off-earth (40), local
# zenith angle block-out (50), sun-glint block-out (60), and the 120+ family of
# missing/saturated/invalid input and cloud codes. These must be excluded, not
# labelled 0.


def classify(mask: np.ndarray, high_confidence_only: bool = False):
    """Split a Mask array into (fire, observed_no_fire, unobservable)."""
    codes = FIRE_CODES_HIGH if high_confidence_only else FIRE_CODES_ALL
    fire = np.isin(mask, list(codes))
    no_fire = mask == NO_FIRE_CODE
    return fire, no_fire, ~(fire | no_fire)
