"""
load_s1_grd_pair.py

Load a pre- and post-event Sentinel-1 GRD SAR image pair using STAC.

What this module does
---------------------
This script is the "data ingestion boundary" of the OpenSAR project. It:

1) Searches a STAC catalog (Planetary Computer) for Sentinel-1 GRD scenes that
   intersect a given Area of Interest (AOI) and fall within two date windows
   (pre-event and post-event).

2) Filters candidate scenes to those that include BOTH VV and VH polarizations
   (required for the VV/VH/(VV-VH) RGB composite used in the Esri tutorial).

3) Selects one "best" pre scene and one "best" post scene, preferring that the
   orbit direction matches between them (ascending vs descending) because radar
   viewing geometry can affect backscatter.

4) Loads the selected scenes into xarray Datasets using odc-stac.
   IMPORTANT: This is the exact moment pixel values enter memory.
   Everything before this is metadata-only.

Outputs (written to disk)
-------------------------
- outputs/manifest_pre_all.csv  : all candidate pre scenes (transparent record)
- outputs/manifest_post_all.csv : all candidate post scenes
- data/interim/sar_pre.nc       : loaded pre-event pixels (NetCDF)
- data/interim/sar_post.nc      : loaded post-event pixels (NetCDF)

How to run
----------
From repository root:

    python src/stac/load_s1_grd_pair.py

Dependencies (typical)
----------------------
- pystac-client
- planetary-computer
- odc-stac
- xarray
- pandas
- pyyaml

Notes
-----
- This script uses a conservative, reproducible approach:
  * tight AOI bbox
  * small time windows
  * one scene each (pre, post)
- Later phases can extend this to time series, mosaics, and change detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml
from pystac_client import Client
from shapely.geometry import box
import planetary_computer as pc
import odc.stac
import xarray as xr


# -----------------------------------------------------------------------------
# Configuration: STAC endpoint + collection
# -----------------------------------------------------------------------------

# Planetary Computer STAC API endpoint
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Planetary Computer Sentinel-1 GRD collection name
COLLECTION = "sentinel-1-grd"


# -----------------------------------------------------------------------------
# Configuration: target dates and search windows
# -----------------------------------------------------------------------------
# These targets align with the Esri tutorial (Hurricane Harvey example).
# We allow a small window around each date to ensure we can find a usable scene.

TARGET_PRE_DATE = "2017-08-05"   # date string (YYYY-MM-DD)
TARGET_POST_DATE = "2017-08-29"  # date string (YYYY-MM-DD)

# Search windows (UTC). These are intentionally wider than the exact dates.
PRE_WINDOW = "2017-08-03T00:00:00Z/2017-08-07T23:59:59Z"
POST_WINDOW = "2017-08-27T00:00:00Z/2017-08-31T23:59:59Z"


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Data model: AOI
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AOI:
    """
    Area of Interest (AOI) configuration.

    Using a dataclass:
    - reduces boilerplate (Python auto-generates __init__, __repr__, etc.)
    - makes intent very clear: this is a small container of config values

    frozen=True makes AOI immutable (cannot be modified after creation),
    which is good for reproducible pipelines.

    Attributes:
        bbox: Bounding box in EPSG:4326 (lon/lat), formatted as:
              (minLon, minLat, maxLon, maxLat)
        crs: Coordinate reference system string (e.g., "EPSG:4326")
    """
    bbox: tuple[float, float, float, float]  # (minLon, minLat, maxLon, maxLat)
    crs: str


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def read_aoi(path: str | Path) -> AOI:
    """
    Read an Area of Interest (AOI) definition from a YAML config file.

    Expected YAML structure:
        bbox:
          min_lon: <float>
          min_lat: <float>
          max_lon: <float>
          max_lat: <float>
        crs: "EPSG:4326"   # optional, defaults to EPSG:4326

    Args:
        path: Path to the YAML config.

    Returns:
        AOI: An immutable AOI object (bbox + CRS).

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If required bbox keys are missing.
        ValueError: If bbox values cannot be parsed as floats.
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    b = cfg["bbox"]
    bbox = (
        float(b["min_lon"]),
        float(b["min_lat"]),
        float(b["max_lon"]),
        float(b["max_lat"]),
    )
    crs = str(cfg.get("crs", "EPSG:4326"))
    return AOI(bbox=bbox, crs=crs)


def _parse_item_datetime(item) -> datetime:
    """
    Parse a timezone-aware datetime from a STAC Item.

    STAC items typically provide `item.datetime` as a datetime object.
    If not present, fall back to `item.properties["datetime"]`.

    Args:
        item: A STAC Item.

    Returns:
        datetime: A timezone-aware datetime (UTC).
    """
    dt = item.datetime
    if dt is None:
        dt = datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def has_vv_vh(item) -> bool:
    """
    Determine whether a STAC Item contains BOTH VV and VH polarizations.

    Sentinel-1 GRD products may sometimes provide only one polarization.
    For the Esri RGB composite (VV, VH, VV-VH), we require both.

    Args:
        item: A STAC Item describing a Sentinel-1 acquisition.

    Returns:
        bool: True if both VV and VH are present; False otherwise.
    """
    # STAC SAR extension commonly uses sar:polarizations (list of strings).
    pols = item.properties.get("sar:polarizations") or item.properties.get("sar:polarization")
    if not pols:
        return False

    # Normalize to a list of uppercase strings.
    pols_list = pols if isinstance(pols, list) else [pols]
    pols_list = [p.upper() for p in pols_list]
    return ("VV" in pols_list) and ("VH" in pols_list)


def stac_search(aoi: AOI, datetime_range: str) -> list:
    """
    Search the STAC catalog for Sentinel-1 GRD scenes meeting criteria.

    Criteria:
    - Collection: sentinel-1-grd
    - Intersects AOI bbox
    - Falls within the provided datetime range
    - Has both VV and VH polarizations

    Args:
        aoi: Area of Interest (bbox + CRS).
        datetime_range: STAC datetime range string, e.g.:
                        "2017-08-03T00:00:00Z/2017-08-07T23:59:59Z"

    Returns:
        list: A list of STAC Items that match criteria.
    """
    client = Client.open(STAC_API)
    search = client.search(
        collections=[COLLECTION],
        bbox=aoi.bbox,
        datetime=datetime_range,
        max_items=500,
    )

    items = list(search.items())
    items = [it for it in items if has_vv_vh(it)]
    return items


def to_rows(items: Iterable) -> pd.DataFrame:
    """
    Convert STAC Items into a lightweight tabular manifest.

    This is helpful for:
    - transparency (what scenes were considered?)
    - debugging (why was a scene picked?)
    - documentation in your write-up

    Args:
        items: Iterable of STAC Items.

    Returns:
        pd.DataFrame: A DataFrame with key properties per Item.
    """
    rows = []
    for it in items:
        dt = _parse_item_datetime(it)
        rows.append(
            {
                "id": it.id,
                "datetime": dt.isoformat(),
                "platform": it.properties.get("platform"),
                "orbit_state": it.properties.get("sat:orbit_state"),
                "relative_orbit": it.properties.get("sat:relative_orbit"),
                "instrument_mode": it.properties.get("sar:instrument_mode"),
                "polarizations": it.properties.get("sar:polarizations"),
            }
        )
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    return df


def pick_closest(items: list, target_date: str, prefer_orbit_state: Optional[str] = None):
    """
    Select the STAC Item closest in time to a target date.

    Optionally prefers items with a specific orbit direction
    (ascending or descending). Matching orbit direction between pre/post
    scenes improves comparison because radar viewing geometry affects backscatter.

    Args:
        items: List of candidate STAC Items.
        target_date: Date string "YYYY-MM-DD" to match.
        prefer_orbit_state: Optional orbit direction to enforce
                            ("ascending" or "descending").

    Returns:
        The selected STAC Item (the closest match in time).

    Raises:
        ValueError: If items is empty.
    """
    if not items:
        raise ValueError("Cannot pick closest item from an empty list.")

    # If orbit preference is requested and matching items exist, restrict to those.
    if prefer_orbit_state:
        filtered = [it for it in items if it.properties.get("sat:orbit_state") == prefer_orbit_state]
        if filtered:
            items = filtered

    target = datetime.fromisoformat(target_date).replace(tzinfo=timezone.utc)

    def _distance_seconds(it) -> float:
        dt = _parse_item_datetime(it)
        return abs((dt - target).total_seconds())

    return sorted(items, key=_distance_seconds)[0]


def load_item_to_xarray(item, aoi: AOI) -> xr.Dataset:
    """
    Load a Sentinel-1 STAC Item into an xarray Dataset (pixels enter here).

    This function is the **pixel entry point** of the pipeline:
    - It signs asset URLs (required for many Planetary Computer assets).
    - It opens the underlying raster assets (VV and VH).
    - It reads pixel values into memory and returns them as an xarray.Dataset.

    Args:
        item: A STAC Item representing a Sentinel-1 GRD acquisition.
        aoi: Area of Interest used to spatially subset the raster.

    Returns:
        xr.Dataset: Dataset containing VV and VH variables aligned on a common grid,
                    clipped to the AOI bbox, with labeled coordinates.
    """
    # Sign the item so all its asset URLs are usable.
    signed_item = pc.sign(item)
    # Build an AOI polygon in lon/lat (EPSG:4326)
    # (Assuming your aoi object has .bbox as (west, south, east, north))
    west, south, east, north = aoi.bbox
    geom4326 = box(west, south, east, north)
    ds = odc.stac.load(
        [signed_item],
        measurements=["vv", "vh"],
        geopolygon=geom4326,   # AOI in lon/lat
        crs="EPSG:3857",       # output grid in meters (Web Mercator)
        resolution=10,         # 10 m pixels (reasonable default for S1 GRD style analysis)
        chunks={},             # optional: helps keep it lazy/dask-friendly
    )
   
    return ds


def ensure_dirs() -> Path:
    """
    Ensure the data/interim directory exists (ignored by git).

    Returns:
        Path: The created/existing interim data directory.
    """
    data_dir = Path("data/interim")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Run the end-to-end process:
    - Read AOI config
    - Search pre and post windows
    - Choose best pre/post items (matching orbit direction)
    - Save candidate manifests
    - Load pixel data into xarray
    - Save NetCDF caches
    """
    aoi = read_aoi("configs/aoi_bbox.yaml")

    # 1) Search STAC for candidate scenes
    pre_items = stac_search(aoi, PRE_WINDOW)
    post_items = stac_search(aoi, POST_WINDOW)

    if not pre_items or not post_items:
        raise RuntimeError(
            f"Search returned empty results. pre={len(pre_items)}, post={len(post_items)}.\n"
            "Possible fixes:\n"
            "- Widen the date windows\n"
            "- Confirm AOI bbox is correct\n"
            "- Inspect item properties for polarization fields\n"
        )

    # 2) Pick best PRE scene closest to target date
    pre_pick = pick_closest(pre_items, TARGET_PRE_DATE)

    # 3) Pick best POST scene closest to target date, preferring matching orbit direction
    orbit = pre_pick.properties.get("sat:orbit_state")
    post_pick = pick_closest(post_items, TARGET_POST_DATE, prefer_orbit_state=orbit)

    # 4) Save manifests for transparency and debugging
    to_rows(pre_items).to_csv(OUT_DIR / "manifest_pre_all.csv", index=False)
    to_rows(post_items).to_csv(OUT_DIR / "manifest_post_all.csv", index=False)

    # 5) Load pixel data (pixels enter pipeline inside load_item_to_xarray)
    ds_pre = load_item_to_xarray(pre_pick, aoi)
    ds_post = load_item_to_xarray(post_pick, aoi)

    # 6) Save lightweight caches (not committed to git)
    data_dir = ensure_dirs()
    ds_pre.to_netcdf(data_dir / "sar_pre.nc")
    ds_post.to_netcdf(data_dir / "sar_post.nc")

    # 7) Print selections (useful for your notes and write-up)
    print("Selected PRE :")
    print("  id        :", pre_pick.id)
    print("  datetime   :", _parse_item_datetime(pre_pick).isoformat())
    print("  orbit_state:", orbit)

    print("Selected POST:")
    print("  id        :", post_pick.id)
    print("  datetime   :", _parse_item_datetime(post_pick).isoformat())
    print("  orbit_state:", post_pick.properties.get("sat:orbit_state"))

    print("\nWrote:")
    print(f"  - {OUT_DIR / 'manifest_pre_all.csv'}")
    print(f"  - {OUT_DIR / 'manifest_post_all.csv'}")
    print(f"  - {data_dir / 'sar_pre.nc'}")
    print(f"  - {data_dir / 'sar_post.nc'}")


if __name__ == "__main__":
    main()
