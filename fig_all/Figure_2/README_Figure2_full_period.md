# Figure 2 delivery

This is one integrated figure with two narrow environmental context strips above one corridor-distance × time speed heatmap. It contains no (a), (b), or (c) panel labels.

## Automatically located frozen sources

- Model-ready panel: `01_final_data/final_model_features_frozen.parquet`
- Frozen scenario labels: `01_final_data/scenario_labels_final.parquet`
- Verified node order: `01_final_data/01_fcd_reference/node_order_verified.csv`

## Formal variables

- Timestamp: `timestamp_local`
- Traffic speed used by E1–E5: `current_speed_input`
- Original-input availability: `missing_mask`
- Rainfall predictor: `precipitation`
- AOD predictor: `aerosol_optical_depth`

The plotted rainfall and AOD strips use corridor medians of the actual frozen predictor columns. No scenario threshold was recalculated from the plotted data.

## Frozen study grid

- Period: 2025-12-08 13:00:00 to 2026-02-28 23:00:00
- Hours: 1979
- Nodes: 51
- Node-hours: 100929
- FCD-unavailable node-hours displayed in grey: 3835
- Traffic zero values: 0
- Environmental missing values: precipitation=0; AOD=0

`Figure2_plot_audit_full_period.csv` contains every plotted node-hour, source value, plotted value, environmental value, and frozen scenario assignment.
