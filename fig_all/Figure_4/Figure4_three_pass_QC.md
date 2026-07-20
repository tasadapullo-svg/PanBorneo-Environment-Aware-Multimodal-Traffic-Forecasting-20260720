# Figure 4 three-pass quality check

## Pass 1 — frozen-data audit: PASS

- Source rows: 15 (five configurations × three horizons).
- All rows use the test split and `seed_count = 5`.
- Test sample counts are identical across E1–E5 within each horizon.
- E5-versus-E1 annotations were calculated from source values: 1 h = +16.9%, 3 h = +19.8%, 6 h = +23.3%.

## Pass 2 — statistical encoding audit: PASS

- Only MAE is plotted.
- Every point is `MAE_mean`; every error bar is `± MAE_std` across five seeds.
- Horizons are encoded redundantly by colour, marker and line style.
- No bar chart and no significance star is present.

## Pass 3 — rendered-output review: PASS

- One integrated figure with no panel labels.
- Main PNG: 4022 × 2291 px at 600.0 × 600.0 dpi.
- Preview PNG: 1210 × 689 px.
- Axis labels, legend, markers, error bars and all three E5 annotations are visible without clipping or overlap.
