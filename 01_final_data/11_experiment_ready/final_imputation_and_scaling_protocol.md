# Frozen Imputation, Scaling, and Sample Protocol

1. Source files are read-only and are never overwritten.
2. Traffic input uses the current observed value or node-wise causal last observation carried forward.
3. The sole origin with neither a current nor any prior node observation is marked `causal_input_available=False` and excluded from every E0-E5 model sample set; it is not filled from a later observation.
4. Continuous environmental and road predictors, when missing, use medians fitted only on the horizon-specific training rows and then frozen.
5. Categorical predictors use training categories with an explicit unknown level.
6. Scaling and all learned preprocessing use training rows only.
7. Targets are point speeds at t+1h, t+3h, and t+6h. Targets are never imputed.
8. Cross-split targets and targets outside the frozen timeline are excluded by the existing horizon-specific availability flags.
9. E1-E5 use identical horizon-specific sample IDs in each split and identical test observations for every seed.
10. Scenario labels are evaluation-only and are not included in any predictor matrix.
