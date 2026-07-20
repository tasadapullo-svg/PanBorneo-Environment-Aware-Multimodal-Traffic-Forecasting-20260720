# Figure 5 deliverables

- Submission PNG: `Figure5_selective_environmental_value_and_importance.png` (600 dpi)
- Preview PNG: `Figure5_selective_environmental_value_and_importance_preview.png`
- Vector PDF: `Figure5_selective_environmental_value_and_importance.pdf`
- Editable SVG: `Figure5_selective_environmental_value_and_importance.svg`
- Panel (a) plotted-data audit: `Figure5a_effect_size_plotted_data.csv`
- Panel (b) plotted-data audit: `Figure5b_importance_plotted_data.csv`
- Validation summary: `Figure5_validation.json`
- Manuscript caption: `Figure5_caption.md`
- Three-pass QC: `Figure5_three_pass_QC.md`
- Reproducible generator: `generate_figure5.py`

Panel (a) uses the frozen comparator-minus-E1 direction and the supplied moving-block bootstrap intervals. Panel (b) is restricted to `Overall_test`; its axis scale is selected by the generator from the actual importance distribution. Permutation importance is treated as predictive dependence, not a causal effect.
