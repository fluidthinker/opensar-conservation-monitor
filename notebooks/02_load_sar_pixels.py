# %%
"""
Notebook-style runner for loading Sentinel-1 GRD pre/post scenes (STAC -> xarray).

This file is meant to be run cell-by-cell in VS Code.
It imports the real ingestion code from src/stac/load_s1_grd_pair.py (your module),
so we learn interactively without turning the pipeline into a notebook-only artifact.
"""

# If VS Code complains about imports, make sure your workspace root is the repo root.
# Run this file from the repo root in VS Code, or set PYTHONPATH accordingly.


# %%
import sys
from pathlib import Path

repo_root = Path.cwd().resolve().parent  # notebooks/ -> repo root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print("Repo root:", repo_root)


# %%
from pathlib import Path
import pandas as pd

# Import your ingestion module (adjust this import based on where you saved it)
# If your file is at src/stac/load_s1_grd_pair.py, this import should work:
from src.stac.load_s1_grd_pair import (
    read_aoi,
    stac_search,
    to_rows,
    pick_closest,
    load_item_to_xarray,
    ensure_dirs,
    PRE_WINDOW,
    POST_WINDOW,
    TARGET_PRE_DATE,
    TARGET_POST_DATE,
    OUT_DIR,
)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)




# %%
# 1) Confirm repo structure + config exists

config_path = repo_root / "configs" / "aoi_bbox.yaml"
print("Checking:", config_path)

assert config_path.exists(), f"Missing {config_path}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Repo root OK. Config found.")

# %%
# 2) Load AOI (bbox) from YAML
aoi = read_aoi(config_path)
print("AOI:", aoi)

# %%
# 3) STAC search: PRE window
pre_items = stac_search(aoi, PRE_WINDOW)
print("Pre candidates:", len(pre_items))

pre_df = to_rows(pre_items)
pre_df.head(10)


# %%
# 4) STAC search: POST window
post_items = stac_search(aoi, POST_WINDOW)
print("Post candidates:", len(post_items))

post_df = to_rows(post_items)
post_df.head(10)

# %%
# 5) Save manifests (transparent record of what you considered)
OUT_DIR.mkdir(parents=True, exist_ok=True)
pre_df.to_csv(OUT_DIR / "manifest_pre_all.csv", index=False)
post_df.to_csv(OUT_DIR / "manifest_post_all.csv", index=False)
print("Wrote manifests to outputs/")

# %%
# 6) Select one pre + one post scene
# Pick closest pre
pre_pick = pick_closest(pre_items, TARGET_PRE_DATE)
orbit = pre_pick.properties.get("sat:orbit_state")

# Pick closest post, preferring same orbit direction as pre
post_pick = pick_closest(post_items, TARGET_POST_DATE, prefer_orbit_state=orbit)

print("Selected PRE :", pre_pick.id, pre_pick.datetime, orbit)
print("Selected POST:", post_pick.id, post_pick.datetime, post_pick.properties.get("sat:orbit_state"))



# %%
# 7) Load pixels into xarray (this is where pixels enter memory)
ds_pre = load_item_to_xarray(pre_pick, aoi)
ds_post = load_item_to_xarray(post_pick, aoi)

ds_pre

# %%
# 8) Inspect what's inside (variables, dims, coords)
print(ds_pre)
print("Data vars:", list(ds_pre.data_vars))
print("Dims:", ds_pre.dims)

# %%
# 9) Basic sanity checks: VV/VH present and shapes match
assert "vv" in ds_pre and "vh" in ds_pre, "Missing vv or vh in ds_pre"
assert "vv" in ds_post and "vh" in ds_post, "Missing vv or vh in ds_post"

assert ds_pre["vv"].shape == ds_pre["vh"].shape, "VV/VH mismatch in ds_pre"
assert ds_post["vv"].shape == ds_post["vh"].shape, "VV/VH mismatch in ds_post"

print("✅ VV/VH present and aligned (within each dataset).")

# %%
# 10) Save lightweight caches (NetCDF) for fast reuse
data_dir = ensure_dirs()
pre_path = data_dir / "sar_pre.nc"
post_path = data_dir / "sar_post.nc"

ds_pre.to_netcdf(pre_path)
ds_post.to_netcdf(post_path)

print("Wrote:", pre_path)
print("Wrote:", post_path)

# %%
# 11) OPTIONAL: quick pixel statistics (useful for stretch planning later)
def summarize(da, name: str):
    vals = da.data
    # vals could be dask-backed; convert safely
    import numpy as np
    arr = da.values
    arr = arr[np.isfinite(arr)]
    print(f"{name}: min={arr.min():.3f}, p2={np.percentile(arr,2):.3f}, median={np.percentile(arr,50):.3f}, p98={np.percentile(arr,98):.3f}, max={arr.max():.3f}")

summarize(ds_pre["vv"], "PRE VV")
summarize(ds_pre["vh"], "PRE VH")
summarize(ds_post["vv"], "POST VV")
summarize(ds_post["vh"], "POST VH")

# %%
