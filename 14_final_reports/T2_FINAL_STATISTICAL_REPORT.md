# T2 Final Statistical Report

Generated: 2026-07-18T22:12:01+08:00

## Protocol

Errors are first averaged across the 51 nodes for each forecast origin and seed, then averaged across seeds. Confidence intervals use 5,000 paired moving-block bootstrap repetitions with contiguous 24-hour blocks. The secondary test is paired Wilcoxon on daily MAE, with Holm correction across the 15 main comparisons.

Delta convention: comparator minus E1; positive values mean the comparator is worse.

## Results

| comparison | horizon | delta_MAE | relative_change_percent | ci_lower | ci_upper | p_value | p_adjusted | effect_size | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_vs_E2 | 1 | 0.1464 | 14.5795 | 0.0803 | 0.2459 | 0.0803 | 0.8835 | 0.5604 | False |
| E1_vs_E2 | 3 | 0.1442 | 12.2391 | 0.0787 | 0.2572 | 0.0215 | 0.3223 | 0.7143 | False |
| E1_vs_E2 | 6 | 0.2017 | 16.5307 | 0.0901 | 0.3481 | 0.0215 | 0.3223 | 0.7143 | False |
| E1_vs_E3 | 1 | 0.0319 | 3.1810 | 0.0097 | 0.0952 | 0.4973 | 1.0000 | 0.2308 | False |
| E1_vs_E3 | 3 | 0.0594 | 5.0373 | 0.0188 | 0.1120 | 0.1677 | 1.0000 | 0.4505 | False |
| E1_vs_E3 | 6 | 0.1161 | 9.5133 | 0.0707 | 0.1505 | 0.0215 | 0.3223 | 0.7143 | False |
| E1_vs_E4 | 1 | 0.1866 | 18.5870 | 0.0991 | 0.2821 | 0.0803 | 0.8835 | 0.5604 | False |
| E1_vs_E4 | 3 | 0.2128 | 18.0589 | 0.0775 | 0.4334 | 0.2734 | 1.0000 | 0.3626 | False |
| E1_vs_E4 | 6 | 0.2326 | 19.0597 | 0.1163 | 0.4444 | 0.2163 | 1.0000 | 0.4066 | False |
| E1_vs_E5 | 1 | 0.1686 | 16.7913 | 0.1028 | 0.2869 | 0.1677 | 1.0000 | 0.4505 | False |
| E1_vs_E5 | 3 | 0.2336 | 19.8265 | 0.0922 | 0.4587 | 0.0681 | 0.8174 | 0.5824 | False |
| E1_vs_E5 | 6 | 0.2837 | 23.2507 | 0.1557 | 0.5272 | 0.1272 | 1.0000 | 0.4945 | False |
| E1_vs_best_selective_environmental | 1 | -0.0073 | -0.7316 | -0.0154 | 0.0028 | 0.4143 | 1.0000 | -0.2747 | False |
| E1_vs_best_selective_environmental | 3 | -0.0049 | -0.4170 | -0.0251 | 0.0351 | 1.0000 | 1.0000 | 0.0110 | False |
| E1_vs_best_selective_environmental | 6 | 0.0081 | 0.6654 | -0.0117 | 0.0536 | 0.7354 | 1.0000 | -0.1209 | False |

No daily Wilcoxon comparison remains significant after Holm correction. Bootstrap intervals nevertheless show positive mean degradation for the frozen E2-E5 overall comparisons, while the validation-selected M4 pressure model remains close to E1.
