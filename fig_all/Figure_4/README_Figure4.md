# Figure 4 deliverables

- Submission PNG: `Figure4_overall_frozen_multimodal_performance.png` (600 dpi)
- Preview PNG: `Figure4_overall_frozen_multimodal_performance_preview.png`
- Vector PDF: `Figure4_overall_frozen_multimodal_performance.pdf`
- Editable SVG: `Figure4_overall_frozen_multimodal_performance.svg`
- Exact plotted values and computed annotations: `Figure4_plotted_data_audit.csv`
- Validation summary: `Figure4_validation.json`
- Manuscript caption: `Figure4_caption.md`
- Three-pass QC: `Figure4_three_pass_QC.md`
- Reproducible generator: `generate_figure4.py`

Only MAE is plotted. RMSE, sMAPE and R² remain outside the figure. Error bars use the frozen `MAE_std` column and all E5-versus-E1 percentages are computed from the source data at run time; no result is manually entered into the plotting code.
