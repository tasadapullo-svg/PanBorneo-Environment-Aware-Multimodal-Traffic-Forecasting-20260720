# T2 GitHub Package Data Completeness Audit

Audit date: 2026-07-19

This audit checks whether `T2_TransportmetricaB_GitHub` is complete as a GitHub-ready reproducibility package, using the local full backup as the reference boundary:

`D:\OneDrive\桌面\20260718多数据源数据补全\论文数据\2026071901_200MB_ZIP_PARTS\T2_TransportmetricaB_LOCAL_FULL_20260719\T2_FINAL_EXPERIMENT_PACKAGE`

## Result

Status: PASS for GitHub upload.

The package contains the complete compact evidence chain needed for public reproducibility: frozen configuration, processed model-ready data, manifests, leakage and integrity audits, model summaries, scenario/reliability/sensitivity/UQ/statistical/interpretability outputs, figure/table source data, final reports, scripts, and manuscript figure assets.

## File-Level Scope Check

- Local full package reference files: 897
- GitHub target package files including `REPRODUCIBILITY_PACKAGE_MANIFEST.csv`: 546
- GitHub target package size: 62.94 MB
- Regenerated manifest rows, excluding the manifest itself: 545
- `fig_all` files copied from parent figure sources: 142
- `fig_all` size: 47.08 MB
- Files missing from the local full package comparison: 499
- Unexpected missing files: 0

All missing-from-full items are explained by the GitHub exclusion policy:

- `archive/` historical snapshots and nested release packages
- duplicate root `audit/` folder, because canonical evidence is retained in `02_audit/`
- model checkpoints under `best_checkpoint/`
- large per-sample prediction Parquet exports
- UQ run-level prediction Parquet files and `09_E7_UQ/quantile_predictions.parquet`
- duplicate large CSV versions of scenario labels where compact Parquet equivalents are retained
- large row-level FCD imputation diagnostic export where summary audit files are retained
- root full-package checksum file superseded by the regenerated GitHub manifest

## Required Evidence Check

- `T2_MISSING_REQUIRED_ITEMS.csv` has 0 required missing rows.
- `T2_REMAINING_OPTIONAL_ITEMS.csv` has 3 optional non-blocking rows: GRU benchmark, SHAP, and heavy-rain inferential analysis.
- `02_audit/FINAL_AUDIT_PASS3.csv` references no missing files.
- Figure evidence status: Figure 1 through Figure 7 are PASS.
- Table evidence status: Table 1 through Table 4 are PASS.
- Claim evidence status: all 7 final claims are PASS in the paper-evidence audit.

## Closed-Loop Data Structure

The closed-loop structure is present:

- `00_config/`: frozen study, scenario, and model configuration
- `00_final_freeze/`: frozen dataset, feature, sample, model, and checksum manifests
- `01_final_data/`: final model-ready data, sample labels, scenario labels, node order, FCD reference files, and experiment-ready dictionaries/manifests
- `02_audit/`: temporal, target, split, node-order, lag/rolling, imputation, and metric reproduction audits
- `03_E0_baselines/`: formal baseline metrics, run manifests, curves, and convergence/sample consistency audits
- `04_E1_E5/`: frozen traffic-only and multimodal model summaries with preprocessing records
- `05_E6_ablation/`: modality, rainfall, meteorology, and atmospheric ablation summaries
- `06_scenario_analysis/`: scenario sample counts and scenario metrics
- `07_reliability_analysis/`: reliability and environmental interaction metrics
- `08_sensitivity_analysis/`: outage and sensitivity checks
- `09_E7_UQ/`: uncertainty quantification metrics, calibration audit, preprocessing records, and summaries
- `10_statistical_tests/`: Holm-corrected results and statistical reports
- `11_interpretability/`: group permutation importance and interpretability summaries
- `12_figure_source_data/`: source data and captions for Figures 1-7
- `13_table_source_data/`: source data for Tables 1-4
- `14_final_reports/`: final evidence, experiment, statistical, and data-readiness reports
- `fig_all/`: manuscript figure assets copied from `D:\OneDrive\桌面\20260718多数据源数据补全\fig`
- `scripts/`: reproducibility and analysis scripts

## Data Integrity Notes

- The package preserves the negative and conditional finding. No test samples were removed to improve conclusions.
- The GitHub package is intentionally compact. The local full backup remains the authority for checkpoints, large prediction exports, archived snapshots, and forensic row-level diagnostics.
- Before making the repository public, verify redistribution terms for upstream third-party data sources.
