# Open-Source SAR Pre/Post Comparison (Sentinel-1)

## Overview
This project was inspired by the ESRI ArcGIS Learn lesson:

- **Explore SAR satellite imagery — Identify flooded areas**  
  https://learn.arcgis.com/en/projects/explore-sar-satellite-imagery/#identify-flooded-areas

The lesson frames the scenario as follows:

> “In the last scenario, as an image analyst for the U.S. Federal Emergency Management Agency (FEMA), you will identify areas flooded by Hurricane Harvey in the area of Freeport, Texas. Working with analysis-ready SAR images captured before and after the hurricane, you'll learn about polarization bands and derive two color composites. Then you'll use the swipe tool to visualize the flooded areas.”

The goal of this project was to reproduce the **core pre/post SAR comparison workflow** from the ESRI lesson using **open-source Python tools**. The lesson’s guided scenario and reference interpretation provided important context for learning.

The focus is on **reproducible data access**, **SAR-aware scene selection**, and **honest visual comparison**, rather than automated classification or operational change detection.

This work demonstrates how Sentinel-1 SAR data can be accessed via STAC, loaded into xarray, and visualized in a consistent way using shared display settings.



---

## Motivation
Synthetic Aperture Radar (SAR) is widely used for flood monitoring and surface change analysis, but many workflows are tied to proprietary tools or abstract away key decisions.

This project was motivated by curiosity about:
- how SAR workflows can be implemented using open-source Python,
- how to ensure fair pre/post comparison, and
- how to make data access and selection decisions explicit and reproducible.


---

## Data
- **Sensor:** Sentinel-1 GRD  
- **Polarizations:** VV and VH  
- **Access:** Planetary Computer STAC API  
- **Context:** Pre- and post-event scenes around Hurricane Harvey (2017)

---
## Imagery context
- **AOI:** Freeport, Texas area (configured via `configs/aoi_bbox.yaml`)
- **Sensor/product:** Sentinel-1 GRD (VV/VH)
- **Pre-event target date:** 2017-08-05 (UTC)
- **Post-event target date:** 2017-08-29 (UTC)
- **Scene selection goal:** choose geometrically comparable acquisitions (same orbit direction) to support fair pre/post visual comparison.


---

## Method (High-Level)
1. Define an Area of Interest (AOI) using a bounding box.
2. Search the Planetary Computer STAC catalog for Sentinel-1 GRD scenes in pre- and post-event time windows.
3. Filter candidate scenes to those containing both VV and VH polarizations.
4. Select one pre-event and one post-event scene, preferring matching orbit direction to reduce geometric effects.
5. Load pixel data into xarray using `odc-stac`, clipping to the AOI and projecting to a shared grid.
6. Cache loaded datasets locally as NetCDF files for reproducibility.
7. Apply a **shared display stretch** to both dates.
8. Generate RGB SAR composites using:
   - Red = VV  
   - Green = VH  
   - Blue = VV − VH  

---

## Results

### Pre-event SAR composite
*(VV / VH / VV−VH, shared stretch)*
![Pre SAR composite](outputs/plots/pre_rgb_shared.png)


### Post-event SAR composite
*(VV / VH / VV−VH, shared stretch)*
![Post SAR composite](outputs/plots/post_rgb_shared.png)


These images are visualized using the **same display stretch** to avoid misleading contrast differences between dates.

---

## Note on units
**The SAR composites shown here are visualized using the native Sentinel-1 GRD backscatter values (linear scale), not decibels (dB).**  
A shared display stretch is applied to both pre- and post-event images to ensure honest visual comparison. Quantitative interpretation or threshold-based change detection is intentionally out of scope for this example.

---



## Transparency & Reproducibility
- All candidate pre- and post-event scenes are saved as CSV manifests for inspection.
- Loaded pixel data are cached locally as NetCDF files to separate remote data access from analysis.
- Scene selection criteria (time window, orbit direction, polarizations) are explicit in code.

---

## Limitations
- This project stops at visual comparison and does not attempt automated flood classification.
- SAR backscatter interpretation depends strongly on land cover and context.
- Further steps such as calibration, speckle filtering, dB conversion, and validation would be required for operational analysis.

---

## Future Work
- Incorporate land-cover context to aid interpretation.
- Convert backscatter to dB for quantitative analysis.
- Apply the same workflow to simpler change-detection examples (e.g., lake area change or vegetation indices).

---

## Tools & Libraries
- pystac-client  
- planetary-computer  
- odc-stac  
- xarray / rioxarray  
- pandas  
- matplotlib  
