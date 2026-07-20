# Final Negative / Conditional Finding Audit

| claim_id | claim | status | evidence | evidence_files |
| --- | --- | --- | --- | --- |
| 1 | Traffic-only forecasting provides the strongest overall performance. | PARTIALLY_SUPPORTED | E1 is best among E1-E5 at all horizons and best E0/E1 model at 1h/3h, but Seasonal HA at 6h (1.1874) is lower than E1 (1.2166). | 03_E0_baselines/metrics_summary.csv;04_E1_E5/metrics_summary.csv |
| 2 | Naive multimodal environmental augmentation does not universally improve traffic-speed forecasting. | SUPPORTED | Each frozen E2-E5 overall MAE exceeds E1 at 1h, 3h, and 6h; negative results are retained. | 04_E1_E5/metrics_summary.csv;10_statistical_tests/holm_corrected_results.csv |
| 3 | Environmental predictive value varies by forecast horizon. | SUPPORTED | Validation-selected M4 pressure has small test gains at 1h/3h and a loss at 6h; E4 conditional gains appear mainly at 3h/6h. | 05_E6_ablation/meteorology_ablation/metrics_summary.csv;06_scenario_analysis/scenario_metrics_summary.csv |
| 4 | Atmospheric context may improve 3-6h forecasting under elevated-AOD or compound environmental conditions. | PARTIALLY_SUPPORTED | E4 beats E1 in 4 of 6 elevated-AOD/compound horizon cells, specifically the 3h/6h cells, but event counts are limited and no strong scenario-level inferential claim is warranted. | 06_scenario_analysis/scenario_metrics_summary.csv;06_scenario_analysis/scenario_sample_counts.csv |
| 5 | The failure of full fusion is not solely caused by the February 17 FCD outage. | SUPPORTED | The E5-minus-E1 sign remains unfavorable after excluding 2026-02-17 at all 3 horizons; sign changes=0. | 08_sensitivity_analysis/outage_conclusion_stability.csv;08_sensitivity_analysis/sensitivity_comparison.csv |
| 6 | Environmental conditions affect forecasting uncertainty. | PARTIALLY_SUPPORTED | Prediction intervals vary descriptively by regime (mean compound width 17.090 versus dry 14.390), but a causal or formal interaction claim was not tested. | 09_E7_UQ/uq_metrics_by_scenario.csv |
| 7 | Environmental information is more useful under specific reliability regimes. | PARTIALLY_SUPPORTED | E4 beats E1 in 9 reliability-regime-horizon cells, concentrated in elevated-AOD/compound 3h-6h strata; the pattern is not universal. | 07_reliability_analysis/interaction_metrics_summary.csv |

NOT_SUPPORTED claims must not be retained as core conclusions. This audit contains no unsupported core claim.
