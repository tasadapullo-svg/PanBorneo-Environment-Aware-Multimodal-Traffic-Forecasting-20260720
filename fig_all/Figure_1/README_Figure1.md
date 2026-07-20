# Figure 1 delivery

This folder contains one integrated GIS-style figure; it is not a multi-panel figure.

## Frozen-data facts

- FCD nodes: 51
- Frozen corridor coverage: 0.000–95.690 km
- Unique Open-Meteo weather grids: 15
- Unique CAMS-derived atmospheric-composition grids: 9
- Node-order mismatches: 0
- Coordinate mismatches against node_master_final.csv: 0

## Formats

- PDF and SVG: editable vector versions.
- PNG and TIFF: 600 dpi publication versions.
- Preview PNG: 200 dpi review version.

## Map conventions

- Filled circles: FCD nodes.
- Open circles: Open-Meteo weather grids.
- Open squares: CAMS-derived atmospheric-composition grids.
- No claim is made that CAMS grids are roadside or ground-station observations.
- No 131 km label is used; the figure reports the frozen 95.690 km coverage.

## Reproduction

Run `generate_figure1.py` with Python plus pandas, numpy, Pillow, and matplotlib. Cached OpenStreetMap tiles and Natural Earth GeoJSON are retained under `assets` for exact rerendering.
