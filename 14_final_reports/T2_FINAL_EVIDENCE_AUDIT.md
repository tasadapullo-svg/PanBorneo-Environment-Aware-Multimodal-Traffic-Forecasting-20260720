# T2 Final Evidence Audit

Generated: 2026-07-18T22:12:01+08:00

Every reported main value traces to prediction-level parquet files. Figure and table source data were generated programmatically, not typed from manuscript values.

## Claim status

| claim_id | status | claim | evidence_files |
| --- | --- | --- | --- |
| 1 | PARTIALLY_SUPPORTED | Traffic-only forecasting provides the strongest overall performance. | 03_E0_baselines/metrics_summary.csv;04_E1_E5/metrics_summary.csv |
| 2 | SUPPORTED | Naive multimodal environmental augmentation does not universally improve traffic-speed forecasting. | 04_E1_E5/metrics_summary.csv;10_statistical_tests/holm_corrected_results.csv |
| 3 | SUPPORTED | Environmental predictive value varies by forecast horizon. | 05_E6_ablation/meteorology_ablation/metrics_summary.csv;06_scenario_analysis/scenario_metrics_summary.csv |
| 4 | PARTIALLY_SUPPORTED | Atmospheric context may improve 3-6h forecasting under elevated-AOD or compound environmental conditions. | 06_scenario_analysis/scenario_metrics_summary.csv;06_scenario_analysis/scenario_sample_counts.csv |
| 5 | SUPPORTED | The failure of full fusion is not solely caused by the February 17 FCD outage. | 08_sensitivity_analysis/outage_conclusion_stability.csv;08_sensitivity_analysis/sensitivity_comparison.csv |
| 6 | PARTIALLY_SUPPORTED | Environmental conditions affect forecasting uncertainty. | 09_E7_UQ/uq_metrics_by_scenario.csv |
| 7 | PARTIALLY_SUPPORTED | Environmental information is more useful under specific reliability regimes. | 07_reliability_analysis/interaction_metrics_summary.csv |

## Provenance rules

- E1-E5: 15/15 horizon×seed sample comparisons pass.
- TCN: 15/15 sample comparisons and convergence checks pass.
- E6: all 450 variant×horizon×seed sample comparisons pass.
- UQ: 45/45 calibration audits pass; test is never calibration.
- Interpretation: 15/15 frozen E5 checkpoints reproduce prediction files exactly.
- CAMS is described only as CAMS-derived gridded atmospheric composition data.
