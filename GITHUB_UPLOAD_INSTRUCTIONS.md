# GitHub Upload Instructions

This folder is ready to upload as the compact GitHub reproducibility package:

`D:\OneDrive\桌面\20260718多数据源数据补全\T2_TransportmetricaB_GitHub`

## Recommended Repository Description

Transportmetrica B reproducibility package for T2 multimodal environmental data completion and traffic-speed forecasting experiments.

## Suggested GitHub README Summary

This repository contains the compact reproducibility package for the T2 Transportmetrica B experiment. It includes frozen configuration, processed model-ready data, audit reports, baseline and multimodal experiment summaries, ablation/scenario/reliability/sensitivity/UQ/statistical/interpretability outputs, figure and table source data, manuscript figure assets, and scripts.

Large regenerated artifacts such as model checkpoints, per-sample prediction exports, archived snapshots, and forensic row-level diagnostics are intentionally excluded from the GitHub package and remain in the local full backup.

## Upload Steps

Run these commands inside:

`D:\OneDrive\桌面\20260718多数据源数据补全\T2_TransportmetricaB_GitHub`

```powershell
git init
git add .
git commit -m "Add T2 Transportmetrica B reproducibility package"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

If GitHub rejects large files, check:

```powershell
git ls-files | ForEach-Object { Get-Item $_ } | Sort-Object Length -Descending | Select-Object -First 20 FullName,Length
```

The intended package size is about 63 MB before Git metadata, mostly from manuscript figure assets in `fig_all/`.

## Upload Boundary

Do upload:

- compact processed data and manifests
- audit and final reports
- source data for figures and tables
- scripts
- manuscript-ready figure assets in `fig_all/`

Do not upload from the local full backup unless a reviewer specifically asks:

- checkpoints
- per-sample prediction Parquet exports
- nested archives and ZIP files
- duplicate full audit snapshots
- large row-level diagnostic exports

The exclusion rationale is documented in `EXCLUDED_FROM_GITHUB_PACKAGE.md`.

## Recommended First Files for Reviewers

1. `README.md`
2. `DATA_COMPLETENESS_AUDIT_20260719.md`
3. `14_final_reports/T2_TRANSPORTMETRICA_B_EVIDENCE_SUMMARY.md`
4. `FINAL_FINDING_AUDIT.md`
5. `REPRODUCIBILITY_PACKAGE_MANIFEST.csv`

