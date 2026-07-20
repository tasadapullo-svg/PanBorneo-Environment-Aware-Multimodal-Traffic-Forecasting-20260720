# Phase 1 Final Data Freeze and Integrity Audit

Generated: 2026-07-18T21:53:04+08:00

## Outcome

- Frozen feature table: 100,929 node-hours, 51 nodes.
- Time range: 2025-12-08 13:00:00 to 2026-02-28 23:00:00 at 1-hour resolution.
- Targets remain point speed: current_speed(t+1h), current_speed(t+3h), current_speed(t+6h).
- Raw data were read-only. No original file was modified.
- One origin without a causal past FCD observation remains documented and is excluded from every model sample set; validation and test are unchanged.
- Total excluded noncausal origins: 1.
- Independent lag/rolling check: 18 features x 1,000 sampled rows, total mismatch count 0.
- Weather provenance: Open-Meteo Historical Weather API.
- Atmospheric provenance: CAMS-derived gridded atmospheric composition data accessed through Open-Meteo; these are not roadside or ground-station observations.

## Required audit files

- `final_node_order_audit.csv`
- `final_temporal_integrity.csv`
- `final_split_integrity.csv`
- `final_target_integrity.csv`
- `fcd_imputation_row_level_check.csv`
- `fcd_gap_distribution.csv`
- `fcd_imputation_summary.csv`
- `final_lag_rolling_audit.csv`
- `environment_training_only_imputation_audit.csv`

## Freeze rule

All later experiments read `01_final_data/final_model_features_frozen.parquet` and never write to it.
