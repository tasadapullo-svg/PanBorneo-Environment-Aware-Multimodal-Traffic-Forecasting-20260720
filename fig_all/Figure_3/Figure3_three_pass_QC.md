# Figure 3 three-pass quality check

## Pass 1 — frozen evidence and terminology: PASS

- All 11 protocol checks passed against `source_data.csv`, `study_config_final.yaml` and `model_manifest.yaml`.
- Weather is named Open-Meteo historical weather.
- Atmospheric inputs are named CAMS-derived gridded atmospheric-composition data; no roadside/ground-station claim is used.
- Causal LOCF, training-only environmental imputation, target-boundary purge and test-independent calibration are explicit.

## Pass 2 — output integrity: PASS

- Main PNG dimensions: 4458 × 3258 px.
- Embedded PNG resolution: 600.0 × 600.0 dpi.
- Preview dimensions: 1337 × 977 px.
- Vector PDF and editable SVG were generated from the same plotting object.
- SHA-256 hashes are recorded in `Figure3_output_manifest.csv`.

## Pass 3 — visual and journal-style review: PASS

- One integrated figure; no (a)/(b)/(c) labels.
- Four-stage left-to-right reading order is unambiguous.
- No clipping, overlap, cartoon icon, network-internals diagram or decorative formula was retained.
- White background, greyscale containers and one restrained accent colour are used.
- The two safeguards, **NO FUTURE LEAKAGE** and **FROZEN TEST SET**, remain prominent at final size.
