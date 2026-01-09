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
# 12) Reload from local NetCDF caches to avoid re-reading remote TIFF tiles during heavy computes
#     (Step 10 already wrote these)
import xarray as xr

pre_path = data_dir / "sar_pre.nc"
post_path = data_dir / "sar_post.nc"

print("Reloading local caches:")
print(" -", pre_path)
print(" -", post_path)

ds_pre = xr.open_dataset(pre_path, chunks={})
ds_post = xr.open_dataset(post_path, chunks={})

print("Loaded ds_pre:", ds_pre)
print("Loaded ds_post:", ds_post)



# %%
# 13) Sanity check: confirm PRE and POST share the same grid (critical before differencing)
same_x = ds_pre["x"].equals(ds_post["x"])
same_y = ds_pre["y"].equals(ds_post["y"])

print("Same x grid?", same_x)
print("Same y grid?", same_y)
print("PRE vv shape:", ds_pre["vv"].shape, "POST vv shape:", ds_post["vv"].shape)

if not (same_x and same_y):
    print("⚠️ Grids differ. You can still proceed for visualization, but DO NOT do pixel-wise change "
          "until resampling one dataset onto the other's grid.")


# %%
# 14) Prepare 2D arrays (drop time dimension) and derive VV−VH
vv_pre  = ds_pre["vv"].isel(time=0)
vh_pre  = ds_pre["vh"].isel(time=0)
vv_post = ds_post["vv"].isel(time=0)
vh_post = ds_post["vh"].isel(time=0)

diff_pre  = vv_pre - vh_pre
diff_post = vv_post - vh_post

print("2D shapes:", vv_pre.shape, vv_post.shape)


# %%
# 15) Compute SHARED stretch bounds (one set of p2/p98 per channel across BOTH dates)
def shared_percentiles(a: xr.DataArray, b: xr.DataArray, p=(2, 98)):
    stacked = xr.concat([a, b], dim="stack")
    q = stacked.quantile([p[0] / 100, p[1] / 100], dim=("stack", "y", "x"), skipna=True)
    lo = float(q.sel(quantile=p[0] / 100).values)
    hi = float(q.sel(quantile=p[1] / 100).values)
    return lo, hi

vv_lo, vv_hi   = shared_percentiles(vv_pre,   vv_post,  p=(2, 98))
vh_lo, vh_hi   = shared_percentiles(vh_pre,   vh_post,  p=(2, 98))
dif_lo, dif_hi = shared_percentiles(diff_pre, diff_post, p=(2, 98))

print("Shared VV  p2/p98:", vv_lo, vv_hi)
print("Shared VH  p2/p98:", vh_lo, vh_hi)
print("Shared DIF p2/p98:", dif_lo, dif_hi)


# %%
# 16) Build RGB composites using shared stretch:
#     R = VV, G = VH, B = VV − VH
def scale01(da: xr.DataArray, lo: float, hi: float) -> xr.DataArray:
    clipped = da.clip(min=lo, max=hi)
    return (clipped - lo) / (hi - lo)

rgb_pre = xr.concat(
    [
        scale01(vv_pre,   vv_lo,  vv_hi),   # R
        scale01(vh_pre,   vh_lo,  vh_hi),   # G
        scale01(diff_pre, dif_lo, dif_hi),  # B
    ],
    dim="band",
).assign_coords(band=["R(VV)", "G(VH)", "B(VV-VH)"])

rgb_post = xr.concat(
    [
        scale01(vv_post,   vv_lo,  vv_hi),
        scale01(vh_post,   vh_lo,  vh_hi),
        scale01(diff_post, dif_lo, dif_hi),
    ],
    dim="band",
).assign_coords(band=["R(VV)", "G(VH)", "B(VV-VH)"])

print("RGB PRE:", rgb_pre)
print("RGB POST:", rgb_post)


# %%
# 17) Quick visualization (shared stretch)
import matplotlib.pyplot as plt

def show_rgb(rgb: xr.DataArray, title: str):
    img = rgb.transpose("y", "x", "band").values
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

show_rgb(rgb_pre,  "PRE composite (shared stretch)")
show_rgb(rgb_post, "POST composite (shared stretch)")


# %%
# 18) Downsample for display-only (optional)
rgb_pre_small  = rgb_pre.isel(y=slice(None, None, 4), x=slice(None, None, 4))
rgb_post_small = rgb_post.isel(y=slice(None, None, 4), x=slice(None, None, 4))

show_rgb(rgb_pre_small,  "PRE composite (downsampled for display)")
show_rgb(rgb_post_small, "POST composite (downsampled for display)")


# %%
# 19) Simple change layers
dvv  = vv_post - vv_pre
dvh  = vh_post - vh_pre
ddif = diff_post - diff_pre

def qstats(da: xr.DataArray, name: str):
    q = da.quantile([0.02, 0.5, 0.98], dim=("y", "x"), skipna=True)
    print(f"{name}: p2={float(q.sel(quantile=0.02).values):.3f}, "
          f"median={float(q.sel(quantile=0.5).values):.3f}, "
          f"p98={float(q.sel(quantile=0.98).values):.3f}")

qstats(dvv,  "ΔVV (POST-PRE)")
qstats(dvh,  "ΔVH (POST-PRE)")
qstats(ddif, "Δ(VV−VH) (POST-PRE)")

# 20) Intentially deleted



# %%
# 21) Diagnose why the change looks flat: print robust stats for the linear deltas
def qstats(da, name):
    q = da.quantile([0.02, 0.5, 0.98], dim=("y", "x"), skipna=True)
    print(
        f"{name}: p2={float(q.sel(quantile=0.02).values):.3f}, "
        f"median={float(q.sel(quantile=0.5).values):.3f}, "
        f"p98={float(q.sel(quantile=0.98).values):.3f}"
    )

qstats(dvv,  "ΔVV linear (POST-PRE)")
qstats(dvh,  "ΔVH linear (POST-PRE)")
qstats(ddif, "Δ(VV−VH) linear (POST-PRE)")


# %%
# 22) Plot deltas with a robust, symmetric stretch around 0 + colorbar
#     This is the quickest way to reveal subtle change patterns.
import matplotlib.pyplot as plt
import numpy as np

def plot_delta(da, title, p=98):
    # robust symmetric limits around 0 using percentile of absolute values
    abs_p = float(np.nanpercentile(np.abs(da.values), p))
    plt.figure()
    plt.imshow(da.values, vmin=-abs_p, vmax=abs_p, cmap="RdBu_r")
    plt.title(f"{title} (±p{p} abs)")
    plt.axis("off")
    plt.colorbar(shrink=0.8)
    plt.show()

plot_delta(dvv,  "ΔVV linear (POST-PRE)", p=98)
plot_delta(dvh,  "ΔVH linear (POST-PRE)", p=98)
plot_delta(ddif, "Δ(VV−VH) linear (POST-PRE)", p=98)


# %%
# 23) Convert VV/VH to dB, then compute deltas in dB (usually much more interpretable)
#     NOTE: We add a tiny epsilon to avoid log(0).
eps = 1e-6

vv_pre_db  = 10.0 * np.log10(vv_pre + eps)
vh_pre_db  = 10.0 * np.log10(vh_pre + eps)
vv_post_db = 10.0 * np.log10(vv_post + eps)
vh_post_db = 10.0 * np.log10(vh_post + eps)

dvv_db = vv_post_db - vv_pre_db
dvh_db = vh_post_db - vh_pre_db

qstats(dvv_db, "ΔVV dB (POST-PRE)")
qstats(dvh_db, "ΔVH dB (POST-PRE)")


# %%
# 24) Plot dB deltas with robust symmetric stretch around 0
plot_delta(dvv_db, "ΔVV dB (POST-PRE)", p=98)
plot_delta(dvh_db, "ΔVH dB (POST-PRE)", p=98)









# %%
# XX) (Optional) Save portfolio-ready PNGs of PRE/POST composites (shared stretch)
#     Uses downsampled versions for speed/size; switch to rgb_pre/rgb_post for full-res.
out_dir = OUT_DIR  # from your module import
out_dir.mkdir(parents=True, exist_ok=True)

def save_rgb_png(rgb: xr.DataArray, path, title: str):
    img = rgb.transpose("y", "x", "band").values
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

save_rgb_png(rgb_pre_small,  out_dir / "pre_rgb_shared.png",  "PRE composite (shared stretch)")
save_rgb_png(rgb_post_small, out_dir / "post_rgb_shared.png", "POST composite (shared stretch)")

print("Wrote:")
print(" -", out_dir / "pre_rgb_shared.png")
print(" -", out_dir / "post_rgb_shared.png")


# %%
# XX) (Optional) Save change PNGs (downsampled for speed/size)
dvv_small  = dvv.isel(y=slice(None, None, 4), x=slice(None, None, 4))
dvh_small  = dvh.isel(y=slice(None, None, 4), x=slice(None, None, 4))
ddif_small = ddif.isel(y=slice(None, None, 4), x=slice(None, None, 4))

def save_layer_png(layer: xr.DataArray, path, title: str):
    plt.figure()
    plt.imshow(layer.values)
    plt.title(title)
    plt.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

save_layer_png(dvv_small,  out_dir / "change_dvv.png",  "ΔVV (POST - PRE)")
save_layer_png(dvh_small,  out_dir / "change_dvh.png",  "ΔVH (POST - PRE)")
save_layer_png(ddif_small, out_dir / "change_ddif.png", "Δ(VV−VH) (POST - PRE)")

print("Wrote:")
print(" -", out_dir / "change_dvv.png")
print(" -", out_dir / "change_dvh.png")
print(" -", out_dir / "change_ddif.png")
