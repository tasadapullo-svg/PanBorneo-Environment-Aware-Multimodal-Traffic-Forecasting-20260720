# T2 Final Experiment Report

Generated: 2026-07-18T22:12:01+08:00

## Frozen E1-E5 overall test results

| experiment | horizon | MAE_mean | MAE_std | RMSE_mean | sMAPE_mean | R2_mean |
| --- | --- | --- | --- | --- | --- | --- |
| E1_traffic_only | 1 | 0.9994 | 0.0026 | 2.4467 | 1.3176 | 0.9768 |
| E1_traffic_only | 3 | 1.1750 | 0.0025 | 2.7111 | 1.5935 | 0.9715 |
| E1_traffic_only | 6 | 1.2166 | 0.0048 | 2.7529 | 1.6519 | 0.9706 |
| E2_rainfall | 1 | 1.1462 | 0.0130 | 2.5323 | 1.4841 | 0.9752 |
| E2_rainfall | 3 | 1.3193 | 0.0102 | 2.7760 | 1.7522 | 0.9701 |
| E2_rainfall | 6 | 1.4187 | 0.0176 | 2.9003 | 1.8705 | 0.9673 |
| E3_meteorology | 1 | 1.0316 | 0.0032 | 2.4790 | 1.3562 | 0.9762 |
| E3_meteorology | 3 | 1.2344 | 0.0018 | 2.7320 | 1.6553 | 0.9711 |
| E3_meteorology | 6 | 1.3326 | 0.0076 | 2.8289 | 1.7739 | 0.9689 |
| E4_atmospheric | 1 | 1.1864 | 0.0103 | 2.5096 | 1.5181 | 0.9756 |
| E4_atmospheric | 3 | 1.3873 | 0.0085 | 2.7998 | 1.8045 | 0.9696 |
| E4_atmospheric | 6 | 1.4487 | 0.0092 | 2.8755 | 1.8932 | 0.9679 |
| E5_full_fusion | 1 | 1.1682 | 0.0151 | 2.5154 | 1.5051 | 0.9755 |
| E5_full_fusion | 3 | 1.4082 | 0.0147 | 2.8274 | 1.8377 | 0.9690 |
| E5_full_fusion | 6 | 1.5000 | 0.0189 | 2.8885 | 1.9466 | 0.9676 |

Traffic-only E1 is the strongest frozen E1-E5 configuration at every horizon. Full Fusion does not improve overall test MAE and the negative result is retained without split, target, or sample changes.

## Formal E0 benchmark

| model | horizon | MAE_mean | MAE_std | RMSE_mean | sMAPE_mean | R2_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence | 1 | 1.0958 | 0.0000 | 3.3019 | 1.5333 | 0.9578 |
| Historical_Average | 1 | 1.9635 | 0.0000 | 3.8948 | 2.8812 | 0.9413 |
| Seasonal_Historical_Average | 1 | 1.1897 | 0.0000 | 2.8138 | 1.6297 | 0.9694 |
| Ridge | 1 | 1.2248 | 0.0000 | 2.7539 | 1.6542 | 0.9707 |
| XGBoost_traffic_baseline | 1 | 0.9994 | 0.0026 | 2.4467 | 1.3176 | 0.9768 |
| Persistence | 3 | 1.7023 | 0.0000 | 4.4425 | 2.5210 | 0.9235 |
| Historical_Average | 3 | 1.9629 | 0.0000 | 3.8950 | 2.8758 | 0.9412 |
| Seasonal_Historical_Average | 3 | 1.1934 | 0.0000 | 2.8219 | 1.6332 | 0.9691 |
| Ridge | 3 | 1.9627 | 0.0000 | 3.5668 | 2.7098 | 0.9507 |
| XGBoost_traffic_baseline | 3 | 1.1750 | 0.0025 | 2.7111 | 1.5935 | 0.9715 |
| Persistence | 6 | 2.2606 | 0.0000 | 5.4735 | 3.4596 | 0.8836 |
| Historical_Average | 6 | 1.9716 | 0.0000 | 3.9071 | 2.8880 | 0.9407 |
| Seasonal_Historical_Average | 6 | 1.1874 | 0.0000 | 2.8242 | 1.6272 | 0.9690 |
| Ridge | 6 | 2.2574 | 0.0000 | 3.7822 | 3.0975 | 0.9444 |
| XGBoost_traffic_baseline | 6 | 1.2166 | 0.0048 | 2.7529 | 1.6519 | 0.9706 |
| Formal_TCN_24h_causal | 1 | 1.4480 | 0.0394 | 3.3502 | 1.9299 | 0.9564 |
| Formal_TCN_24h_causal | 3 | 1.6889 | 0.1417 | 3.9198 | 2.2862 | 0.9381 |
| Formal_TCN_24h_causal | 6 | 1.7098 | 0.0975 | 3.5004 | 2.2857 | 0.9522 |

The former 8-epoch TCN is archived. The formal TCN uses a 100-epoch limit, patience 15, scheduler, clipping, training-only scaling, and validation-MAE checkpoint selection. Fourteen runs early-stopped; one reached a validation plateau at epoch 100.

## Best configuration within each E6 family and horizon

| ablation_group | variant | horizon | MAE_mean | MAE_std |
| --- | --- | --- | --- | --- |
| modality_ablation | A4_without_environmental_lag_rolling | 1 | 1.0732 | 0.0048 |
| modality_ablation | A4_without_environmental_lag_rolling | 3 | 1.3065 | 0.0127 |
| modality_ablation | A3_without_atmospheric_context | 6 | 1.4233 | 0.0067 |
| meteorology_ablation | M4_pressure | 1 | 0.9918 | 0.0040 |
| meteorology_ablation | M4_pressure | 3 | 1.1696 | 0.0030 |
| meteorology_ablation | M0_traffic_only | 6 | 1.2166 | 0.0048 |
| rainfall_ablation | R1_current_plus_rain_flag | 1 | 1.0688 | 0.0069 |
| rainfall_ablation | R0_current_rainfall_only | 3 | 1.2225 | 0.0054 |
| rainfall_ablation | R0_current_rainfall_only | 6 | 1.2788 | 0.0064 |
| atmospheric_ablation | P0_traffic_only | 1 | 0.9994 | 0.0026 |
| atmospheric_ablation | P0_traffic_only | 3 | 1.1750 | 0.0025 |
| atmospheric_ablation | P0_traffic_only | 6 | 1.2166 | 0.0048 |

M4 (Traffic-only + surface pressure) was selected solely by validation MAE as the best selective environmental model. Its small overall test differences versus E1 are not statistically significant after correction.

## Conditional interpretation

Atmospheric E4 is worse overall but shows descriptive 3 h/6 h gains in elevated-AOD and rain-plus-elevated-pollution scenarios. These are conditional findings, not universal multimodal superiority.
