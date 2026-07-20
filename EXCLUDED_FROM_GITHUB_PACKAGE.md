# Files intentionally excluded from the GitHub package

The LOCAL FULL BACKUP remains the authoritative complete frozen experiment package.

The GitHub package excludes only categories that are large, duplicative, or not necessary for a compact public reproducibility repository:

- `archive/`: historical frozen snapshots and nested ZIPs.
- duplicate root `audit/`: duplicates the canonical `02_audit/` evidence.
- `*/best_checkpoint/` and `.pt` model checkpoints.
- per-sample prediction Parquet files from E0, E1–E5, E6, scenario and reliability analyses.
- `09_E7_UQ/quantile_predictions.parquet` and UQ run-level Parquet predictions/checkpoints.
- `01_final_data/scenario_labels.csv` and `scenario_labels_final.csv` because compact Parquet equivalents are retained.
- `02_audit/fcd_imputation_row_level_check.csv`, a large row-level diagnostic export; summary/integrity audit files are retained.

Use the LOCAL FULL BACKUP when exact archived predictions/checkpoints or complete forensic audit material are required.
