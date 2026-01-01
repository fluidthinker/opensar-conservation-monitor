# OpenSAR Conservation Monitor

Phase 1 goal:
Replicate the Esri tutorial workflow with open-source Python to create pre/post SAR RGB composites (VV, VH, VV–VH),
apply consistent display ranges, and enable honest visual comparison.

## Repo layout
- configs/: AOI and display settings
- src/: pipeline code (STAC search/load, SAR composites, change detection, optical verification)
- outputs/: committed figures/maps/reports (large rasters are ignored)
- data/: ignored cache and intermediate files
