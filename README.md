# PanBorneo-Multimodal-Traffic-Forecasting-Reproducibility Package

This repository-ready package is derived from the validated frozen experiment package generated on 2026-07-18 and 2026-07-19.

## Scope

Included:

- frozen study, model, and scenario configuration
- processed model-ready data and manifests
- integrity, leakage, imputation, and metric-reproduction audit summaries
- compact E0 baseline, E1-E5 multimodal, and E6 ablation results
- scenario, reliability, sensitivity, outage, uncertainty, statistical, and interpretability summaries
- Figure 1-7 and Table 1-4 source data
- manuscript figure assets under `fig_all/`
- experiment scripts and final evidence reports

Excluded from this GitHub-oriented package:

- historical archive snapshots and nested ZIP packages
- model checkpoint files
- large per-sample prediction exports
- large UQ checkpoint and run-prediction files
- duplicate large CSV scenario-label files when compact Parquet equivalents are retained
- duplicate full audit folders where canonical summaries are retained

These excluded files remain in the local full backup. See `EXCLUDED_FROM_GITHUB_PACKAGE.md`.

## Recommended Starting Points

1. `DATA_COMPLETENESS_AUDIT_20260719.md`
2. `14_final_reports/T2_TRANSPORTMETRICA_B_EVIDENCE_SUMMARY.md`
3. `FINAL_FINDING_AUDIT.md`
4. `00_final_freeze/`
5. `12_figure_source_data/`, `13_table_source_data/`, and `fig_all/`
6. `scripts/`

## Main Evidence Boundary

The frozen evidence supports a negative and conditional conclusion: traffic-only forecasting is difficult to beat overall, while environmental value is horizon- and regime-dependent. Negative results are intentionally retained.

## Public Release Caution

Before making the repository public, verify redistribution and attribution requirements for every upstream data source. This package organization does not itself grant redistribution rights for third-party data.

## Integrity

See `REPRODUCIBILITY_PACKAGE_MANIFEST.csv` for SHA256 checksums of included files. See `GITHUB_UPLOAD_INSTRUCTIONS.md` for upload commands and repository boundary notes.

