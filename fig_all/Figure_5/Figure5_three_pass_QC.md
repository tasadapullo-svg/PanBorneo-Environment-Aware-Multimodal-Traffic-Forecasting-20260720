# Figure 5 three-pass quality check

## Pass 1 — source and statistical-direction audit: PASS

- Panel (a): 15 rows, five comparator configurations and three horizons.
- ΔMAE is comparator minus E1; negative values are favourable to the comparator.
- Point estimates fall inside the supplied 95% moving-block bootstrap intervals.
- Holm-adjusted significant comparisons: 0; no significance star was drawn.
- Validation-selected ΔMAE values: 1 h = -0.0073, 3 h = -0.0049, 6 h = +0.0081 km h⁻¹.

## Pass 2 — importance and scale audit: PASS

- Panel (b) uses only `scenario == Overall_test` (21 rows).
- Seven requested feature groups are present at 1, 3 and 6 h.
- Selected x-axis scale: log; max/min mean-importance ratio = 2039.6.
- Atmospheric mean importance: 1 h = 0.0906, 3 h = 0.2275, 6 h = 0.3128 ΔMAE.
- Caption explicitly states that permutation importance is not causal.

## Pass 3 — rendered-output review: PASS

- Exactly two panels, labelled (a) and (b).
- Main PNG: 4505 × 2461 px at 600.0 × 600.0 dpi.
- Preview PNG: 1352 × 742 px.
- Panel labels, category labels, zero line, CIs, markers, legend and log-scale ticks are visible without clipping or overlap.
