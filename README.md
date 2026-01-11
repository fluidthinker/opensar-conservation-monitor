# Open-Source SAR Pre/Post Comparison (Sentinel-1)

## Overview
This project explores how to reproduce the core of an ESRI SAR pre/post comparison workflow using **open-source Python tools**. The focus is on **reproducible data access**, **SAR-aware scene selection**, and **honest visual comparison**, rather than automated classification or operational change detection.

The work demonstrates how Sentinel-1 SAR data can be accessed via STAC, loaded into xarray, and visualized in a consistent way using shared display settings.

---

## Motivation
Synthetic Aperture Radar (SAR) is widely used for flood monitoring and surface change analysis, but many workflows are tied to proprietary tools or abstract away key decisions.

This project was motivated by curiosity about:
- how SAR workflows can be implemented using open-source Python,
- how to ensure fair pre/post comparison, and
- how to make data access and selection decisions explicit and reproducible.

An ESRI Sentinel-1 tutorial provided the conceptual inspiration, but all implementation here uses open-source libraries.

---

## Data
- **Sensor:** Sentinel-1 GRD  
- **Polarizations:** VV and VH  
- **Access:** Planetary Computer STAC API  
- **Context:** Pre- and post-event scenes around Hurricane Harvey (2017)

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

![Pre SAR composite](outputs/pre_rgb_shared.png)

### Post-event SAR composite
*(VV / VH / VV−VH, shared stretch)*

![Post SAR composite](outputs/post_rgb_shared.png)

These images are visualized using the **same display stretch** to avoid misleading contrast differences between dates.

---

## Note on units
**The SAR composites shown here are visualized using the native Sentinel-1 GRD backscatter values (linear scale), not decibels (dB).**  
A shared display stretch is applied to both pre- and post-event images to ensure honest visual comparison. Quantitative interpretation or threshold-based change detection is intentionally out of scope for this example.

---

## Transparency & Reproducibility
- All candidate pre- and post-event scenes are s

