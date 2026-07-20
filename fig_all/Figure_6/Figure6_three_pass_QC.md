# Figure 6 three-pass quality check

## Pass 1 — frozen-cell construction audit: PASS

- Source rows: 144; calculated heatmap cells: 108.
- Every environmental MAE was matched to E1 by scenario, frozen reliability group and horizon.
- E1, E3, E4 and E5 use identical sample counts within every comparison cell.
- Reliability groups were read from the frozen training-defined classification; no test threshold was calculated.

## Pass 2 — numerical and scale audit: PASS

- ΔMAE range: -0.137661 to +0.408584 km h⁻¹.
- Shared normalization for all panels: vmin = -0.450, vcenter = 0, vmax = 0.450.
- Negative cells: 22; positive cells: 86.
- Eight lowest ΔMAE cells are retained in `Figure6_validation.json`/`Figure6_heatmap_cell_audit.csv`; no value was manually entered.

## Pass 3 — rendered-output review: PASS

- Exactly three aligned panels: 1 h, 3 h and 6 h.
- One common colourbar; every cell is annotated with signed ΔMAE to three decimals.
- Main PNG: 4527 × 3350 px at 600.0 × 600.0 dpi.
- Preview PNG: 1360 × 1006 px.
- Panel titles, row labels, column labels, annotations, scenario separators and colourbar are visible without clipping or overlap.
- No significance star appears.
