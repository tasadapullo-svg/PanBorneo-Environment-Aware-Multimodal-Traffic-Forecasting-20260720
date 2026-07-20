# Figure 7 three-pass quality check

## Pass 1 — frozen UQ source audit: PASS

- Source rows: 36 (four regimes × three models × three horizons).
- Every metric is aggregated across five seeds.
- The three models share identical samples within every scenario-horizon combination.
- Display names are Traffic-only, Pressure-augmented and Full fusion; “Best environmental model” is absent.

## Pass 2 — metric and scale audit: PASS

- Nominal coverage: 0.90.
- PICP range: 0.912830–0.962842; cells at/above nominal: 36/36.
- MPIW range: 13.331679–18.028203 km h⁻¹.
- Panel (a) uses its own PICP colourbar; panel (b) uses a separate MPIW colourbar.
- PINAW and Pinball Loss are excluded from the main figure and retained in `Figure7_supplementary_PINAW_Pinball.csv`.

## Pass 3 — rendered-output review: PASS

- Exactly two aligned heatmap panels.
- Main PNG: 4453 × 3289 px at 600.0 × 600.0 dpi.
- Preview PNG: 1336 × 984 px.
- All 72 displayed values, group headers, model labels, scenario labels, thin horizon separators and both colourbars are visible without clipping or overlap.
