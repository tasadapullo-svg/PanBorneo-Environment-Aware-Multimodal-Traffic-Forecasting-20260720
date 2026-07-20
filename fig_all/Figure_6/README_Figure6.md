# Figure 6 deliverables

- Submission PNG: `Figure6_conditional_environmental_value.png` (600 dpi)
- Preview PNG: `Figure6_conditional_environmental_value_preview.png`
- Vector PDF: `Figure6_conditional_environmental_value.pdf`
- Editable SVG: `Figure6_conditional_environmental_value.svg`
- Exact 108-cell heatmap audit: `Figure6_heatmap_cell_audit.csv`
- Validation summary: `Figure6_validation.json`
- Manuscript caption: `Figure6_caption.md`
- Three-pass QC: `Figure6_three_pass_QC.md`
- Reproducible generator: `generate_figure6.py`

The generator reads the frozen reliability labels directly from the source file. It does not calculate reliability thresholds or regroup test observations. All three panels share the same symmetric colour normalization, and every displayed ΔMAE is computed from a source-matched E1 baseline.
