# T2 Final Data Readiness Report

Generated: 2026-07-18T22:12:01+08:00

## Readiness outcome

**PASS — the frozen dataset is ready for submission-grade experiments.**

- Frozen node-hours: 100,929; nodes: 51; verified physical node order: PASS.
- Resolution: 1 hour; horizons: 1 h, 3 h, 6 h; point-speed target definitions are unchanged.
- Split/target integrity rows passing: 9/9.
- Independent lag/rolling checks: 18 features × 1,000 rows; total mismatches: 0.
- Traffic missingness: causal LOCF with missing mask, gap length, coverage, and reliability; one origin without a past observation is documented and excluded from every model.
- Environmental missingness: training-only medians/categories; no validation/test statistics are used.
- Weather source: Open-Meteo Historical Weather API.
- Atmospheric source: CAMS-derived gridded atmospheric composition data accessed through Open-Meteo; not roadside or ground-station observations.
- Evaluation-only pollution labels are absent from predictors.

## Evidence

See `01_final_data/00_final_freeze/` and `02_audit/` for manifests, row-level FCD checks, node/time/target audits, and lag/rolling recomputation.
