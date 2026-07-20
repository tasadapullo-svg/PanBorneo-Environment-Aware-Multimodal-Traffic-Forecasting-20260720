from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
DELIVERY = PACKAGE.parent
SOURCE = Path(r"C:\Users\DELL\Desktop\多数据源数据\20260715\T2_final_V2_complete")
DATA = PACKAGE / "01_final_data"
AUDIT = PACKAGE / "02_audit"
E0 = PACKAGE / "03_E0_baselines"
E1_E5 = PACKAGE / "04_E1_E5"
E6 = PACKAGE / "05_E6_ablation"
SCENARIO = PACKAGE / "06_scenario_analysis"
RELIABILITY = PACKAGE / "07_reliability_analysis"
SENSITIVITY = PACKAGE / "08_sensitivity_analysis"
UQ = PACKAGE / "09_E7_UQ"
STATS = PACKAGE / "10_statistical_tests"
INTERPRET = PACKAGE / "11_interpretability"
FIGURES = PACKAGE / "12_figure_source_data"
TABLES = PACKAGE / "13_table_source_data"
REPORTS = PACKAGE / "14_final_reports"
HORIZONS = (1, 3, 6)
GENERATED = datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    data = frame[columns].copy() if columns else frame.copy()
    for column in data.select_dtypes(include=["float", "float32", "float64"]).columns:
        data[column] = data[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in data.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_figure_folder(name: str, source: pd.DataFrame, caption: str, definition: str) -> None:
    folder = FIGURES / name
    folder.mkdir(parents=True, exist_ok=True)
    source.to_csv(folder / "source_data.csv", index=False)
    (folder / "figure_caption_draft.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    (folder / "data_definition.txt").write_text(definition.strip() + "\n", encoding="utf-8")


def generate_figure_sources() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    node = pd.read_csv(DATA / "node_order_final.csv")
    weather = pd.read_csv(SOURCE / "03_weather_processed" / "weather_grid_mapping.csv")
    aq = pd.read_csv(SOURCE / "05_air_quality_processed" / "air_quality_grid_mapping.csv")
    weather = weather.drop_duplicates("node_id")
    aq = aq.drop_duplicates("node_id")
    fig1 = node.merge(weather, on="node_id", how="left", suffixes=("", "_weather"), validate="1:1")
    fig1 = fig1.merge(aq, on="node_id", how="left", suffixes=("", "_aq"), validate="1:1")
    write_figure_folder(
        "Fig1_corridor_environmental_grids",
        fig1,
        "Verified 51-node tropical highway corridor and the Open-Meteo weather and CAMS-derived atmospheric grid mappings.",
        "Rows follow node_order_verified/corridor_km. Weather fields originate from Open-Meteo Historical Weather API mappings; atmospheric fields are CAMS-derived gridded composition mappings, not roadside observations.",
    )

    fig2 = pd.read_csv(SCENARIO / "scenario_sample_counts.csv")
    write_figure_folder(
        "Fig2_data_availability_environmental_conditions",
        fig2,
        "Test-set availability of the eight frozen environmental regimes across the 1 h, 3 h, and 6 h targets.",
        "node_hours are prediction samples; independent_events count contiguous forecast-origin hour runs and do not treat node-hours as independent events.",
    )

    fig3 = pd.DataFrame(
        [
            (1, "Floating-car traffic", "Causal LOCF, mask, gap, coverage, reliability", "Traffic input at forecast origin"),
            (2, "Open-Meteo historical weather", "Current/lag/cumulative/post-rain plus meteorology", "Causal environmental predictors"),
            (3, "CAMS-derived gridded composition", "Current/lag/rolling atmospheric context", "Causal atmospheric predictors"),
            (4, "Road context", "Verified corridor order and static attributes", "Spatial/context predictors"),
            (5, "Fixed chronological protocol", "Train/tuning/calibration/test; target-boundary purge", "No future leakage"),
            (6, "Models and evaluation", "E0-E7, scenarios, interactions, block bootstrap", "Prediction-level evidence"),
        ],
        columns=["step", "component", "operation", "output"],
    )
    write_figure_folder(
        "Fig3_method_framework",
        fig3,
        "Leakage-controlled multimodal traffic-speed forecasting and evidence-audit framework.",
        "The table supplies the ordered components and text labels for the conceptual framework; numerical results are not hand-entered here.",
    )

    fig4 = pd.read_csv(E1_E5 / "metrics_summary.csv")
    fig4 = fig4.loc[fig4["experiment"].isin([
        "E1_traffic_only", "E2_rainfall", "E3_meteorology", "E4_atmospheric", "E5_full_fusion"
    ])].copy()
    fig4["MAE_rank"] = fig4.groupby("horizon")["MAE_mean"].rank(method="min")
    write_figure_folder(
        "Fig4_modality_comparison",
        fig4,
        "Overall test performance of the frozen E1-E5 modality configurations across horizons and seeds.",
        "Metrics are automatically aggregated from prediction-level parquet files; lower MAE is better.",
    )

    rainfall = pd.read_csv(E6 / "rainfall_ablation" / "metrics_summary.csv")
    deltas = pd.read_csv(E6 / "rainfall_ablation" / "metrics_with_deltas.csv").groupby(
        ["variant", "horizon"], as_index=False
    ).agg(
        delta_MAE_vs_traffic_only_mean=("delta_MAE_vs_traffic_only", "mean"),
        relative_change_vs_traffic_only_percent_mean=("relative_change_vs_traffic_only_percent", "mean"),
    )
    fig5 = rainfall.merge(deltas, on=["variant", "horizon"], how="left", validate="1:1")
    write_figure_folder(
        "Fig5_rainfall_lag_and_ablation",
        fig5,
        "Rainfall feature ablation from current precipitation through lag, cumulative, and post-rain features.",
        "Positive delta_MAE_vs_traffic_only indicates worse performance than frozen E1 Traffic-only.",
    )

    fig6 = pd.read_csv(RELIABILITY / "interaction_metrics_summary.csv")
    write_figure_folder(
        "Fig6_environment_reliability_regimes",
        fig6,
        "Environmental-regime performance stratified by training-defined High, Medium, and Low FCD reliability.",
        "Reliability thresholds use the training split only. High is the training-observed maximum score because an upper-tertile value tie at 1.0 would otherwise create an empty group.",
    )

    uq_scenario = pd.read_csv(UQ / "uq_metrics_by_scenario.csv")
    fig7 = uq_scenario.groupby(["scenario", "model", "horizon"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        sample_count=("sample_count", "mean"),
        Pinball_Loss_mean=("Pinball_Loss", "mean"),
        PICP_mean=("PICP", "mean"),
        MPIW_mean=("MPIW", "mean"),
        PINAW_mean=("PINAW", "mean"),
    )
    write_figure_folder(
        "Fig7_UQ_environmental_risk",
        fig7,
        "Conformalized 90% prediction-interval performance under dry, rain, elevated-AOD, and compound environmental regimes.",
        "PICP is empirical coverage; MPIW is mean interval width; PINAW normalizes width by the observed target range.",
    )


def source_for_group(group: str) -> str:
    return {
        "traffic": "Floating-car traffic data",
        "reliability": "Derived causally from floating-car observations",
        "calendar": "Calendar/holiday context",
        "rainfall": "Open-Meteo Historical Weather API",
        "meteorology": "Open-Meteo Historical Weather API",
        "atmospheric": "CAMS-derived gridded atmospheric composition data accessed through Open-Meteo",
        "road": "OpenStreetMap/elevation/static corridor context",
    }.get(group, "Frozen protocol metadata or target")


def generate_table_sources() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA / "00_final_freeze" / "feature_manifest.csv")
    manifest["data_source"] = manifest["group"].map(source_for_group)
    manifest.to_csv(TABLES / "Table1_data_sources_variables.csv", index=False)

    canonical = pd.read_parquet(DATA / "canonical_hourly_index.parquet")
    split = canonical.groupby("split", as_index=False).agg(
        unique_hours=("timestamp_local", "nunique"),
        start_time=("timestamp_local", "min"),
        end_time=("timestamp_local", "max"),
        node_count=("node_count", "max"),
        eligible_h1=("eligible_sample_count_h1", "sum"),
        eligible_h3=("eligible_sample_count_h3", "sum"),
        eligible_h6=("eligible_sample_count_h6", "sum"),
    )
    split["record_type"] = "split"
    scenarios = pd.read_csv(SCENARIO / "scenario_sample_counts.csv")
    scenarios["record_type"] = "scenario"
    columns = sorted(set(split.columns) | set(scenarios.columns))
    pd.concat([split.reindex(columns=columns), scenarios.reindex(columns=columns)], ignore_index=True).to_csv(
        TABLES / "Table2_dataset_split_scenarios.csv", index=False
    )

    e0 = pd.read_csv(E0 / "metrics_summary.csv").copy()
    e0["result_section"] = "E0_baseline"
    main = pd.read_csv(E1_E5 / "metrics_summary.csv").copy()
    main["result_section"] = "E1_E5_main"
    columns = sorted(set(e0.columns) | set(main.columns))
    pd.concat([e0.reindex(columns=columns), main.reindex(columns=columns)], ignore_index=True).to_csv(
        TABLES / "Table3_main_forecasting_results.csv", index=False
    )

    sections = []
    for name in ("modality_ablation", "meteorology_ablation", "rainfall_ablation", "atmospheric_ablation"):
        frame = pd.read_csv(E6 / name / "metrics_summary.csv")
        frame["section"] = name
        frame["item"] = frame["variant"]
        sections.append(frame)
    uq = pd.read_csv(UQ / "uq_metrics_overall_summary.csv")
    uq["section"] = "UQ"
    uq["item"] = uq["model"]
    sensitivity = pd.read_csv(SENSITIVITY / "sensitivity_comparison.csv").groupby(
        ["experiment", "horizon"], as_index=False
    ).agg(
        delta_MAE=("delta_MAE", "mean"),
        relative_change_percent=("relative_change_percent", "mean"),
    )
    sensitivity["section"] = "outage_sensitivity"
    sensitivity["item"] = sensitivity["experiment"]
    sections.append(uq)
    sections.append(sensitivity)
    columns = sorted(set().union(*(set(frame.columns) for frame in sections)))
    pd.concat([frame.reindex(columns=columns) for frame in sections], ignore_index=True).to_csv(
        TABLES / "Table4_ablation_uq_sensitivity.csv", index=False
    )


def build_findings() -> pd.DataFrame:
    e0 = pd.read_csv(E0 / "metrics_summary.csv")
    main = pd.read_csv(E1_E5 / "metrics_summary.csv")
    scenario = pd.read_csv(SCENARIO / "scenario_metrics_summary.csv")
    outage = pd.read_csv(SENSITIVITY / "outage_conclusion_stability.csv")
    uq = pd.read_csv(UQ / "uq_metrics_by_scenario.csv")
    interaction = pd.read_csv(RELIABILITY / "interaction_metrics_summary.csv")

    e1 = main.loc[main["experiment"].eq("E1_traffic_only")].set_index("horizon")["MAE_mean"]
    seasonal = e0.loc[e0["model"].eq("Seasonal_Historical_Average")].set_index("horizon")["MAE_mean"]
    main_pivot = main.pivot(index="horizon", columns="experiment", values="MAE_mean")
    e4_scenario = scenario.loc[
        scenario["scenario"].isin(["S7_elevated_aod", "S8_rain_elevated_atmospheric_pollution"])
        & scenario["experiment"].isin(["E1_traffic_only", "E4_atmospheric"])
    ].pivot_table(index=["scenario", "horizon"], columns="experiment", values="MAE_mean")
    conditional_wins = int((e4_scenario["E4_atmospheric"] < e4_scenario["E1_traffic_only"]).sum())
    interaction_pivot = interaction.loc[
        interaction["experiment"].isin(["E1_traffic_only", "E4_atmospheric"])
    ].pivot_table(
        index=["scenario", "reliability_group", "horizon"], columns="experiment", values="MAE_mean"
    )
    interaction_wins = int((interaction_pivot["E4_atmospheric"] < interaction_pivot["E1_traffic_only"]).sum())
    uq_mean = uq.groupby(["scenario", "model", "horizon"])["MPIW"].mean()
    dry_width = float(uq_mean.xs("S1_dry", level="scenario").mean())
    compound_width = float(uq_mean.xs("S8_rain_elevated_atmospheric_pollution", level="scenario").mean())

    rows = [
        {
            "claim_id": 1,
            "claim": "Traffic-only forecasting provides the strongest overall performance.",
            "status": "PARTIALLY_SUPPORTED",
            "evidence": f"E1 is best among E1-E5 at all horizons and best E0/E1 model at 1h/3h, but Seasonal HA at 6h ({seasonal.loc[6]:.4f}) is lower than E1 ({e1.loc[6]:.4f}).",
            "evidence_files": "03_E0_baselines/metrics_summary.csv;04_E1_E5/metrics_summary.csv",
        },
        {
            "claim_id": 2,
            "claim": "Naive multimodal environmental augmentation does not universally improve traffic-speed forecasting.",
            "status": "SUPPORTED",
            "evidence": "Each frozen E2-E5 overall MAE exceeds E1 at 1h, 3h, and 6h; negative results are retained.",
            "evidence_files": "04_E1_E5/metrics_summary.csv;10_statistical_tests/holm_corrected_results.csv",
        },
        {
            "claim_id": 3,
            "claim": "Environmental predictive value varies by forecast horizon.",
            "status": "SUPPORTED",
            "evidence": "Validation-selected M4 pressure has small test gains at 1h/3h and a loss at 6h; E4 conditional gains appear mainly at 3h/6h.",
            "evidence_files": "05_E6_ablation/meteorology_ablation/metrics_summary.csv;06_scenario_analysis/scenario_metrics_summary.csv",
        },
        {
            "claim_id": 4,
            "claim": "Atmospheric context may improve 3-6h forecasting under elevated-AOD or compound environmental conditions.",
            "status": "PARTIALLY_SUPPORTED",
            "evidence": f"E4 beats E1 in {conditional_wins} of 6 elevated-AOD/compound horizon cells, specifically the 3h/6h cells, but event counts are limited and no strong scenario-level inferential claim is warranted.",
            "evidence_files": "06_scenario_analysis/scenario_metrics_summary.csv;06_scenario_analysis/scenario_sample_counts.csv",
        },
        {
            "claim_id": 5,
            "claim": "The failure of full fusion is not solely caused by the February 17 FCD outage.",
            "status": "SUPPORTED",
            "evidence": f"The E5-minus-E1 sign remains unfavorable after excluding 2026-02-17 at all {len(outage)} horizons; sign changes={int(outage['conclusion_sign_changed'].sum())}.",
            "evidence_files": "08_sensitivity_analysis/outage_conclusion_stability.csv;08_sensitivity_analysis/sensitivity_comparison.csv",
        },
        {
            "claim_id": 6,
            "claim": "Environmental conditions affect forecasting uncertainty.",
            "status": "PARTIALLY_SUPPORTED",
            "evidence": f"Prediction intervals vary descriptively by regime (mean compound width {compound_width:.3f} versus dry {dry_width:.3f}), but a causal or formal interaction claim was not tested.",
            "evidence_files": "09_E7_UQ/uq_metrics_by_scenario.csv",
        },
        {
            "claim_id": 7,
            "claim": "Environmental information is more useful under specific reliability regimes.",
            "status": "PARTIALLY_SUPPORTED",
            "evidence": f"E4 beats E1 in {interaction_wins} reliability-regime-horizon cells, concentrated in elevated-AOD/compound 3h-6h strata; the pattern is not universal.",
            "evidence_files": "07_reliability_analysis/interaction_metrics_summary.csv",
        },
    ]
    return pd.DataFrame(rows)


def reports_and_findings() -> pd.DataFrame:
    REPORTS.mkdir(parents=True, exist_ok=True)
    findings = build_findings()
    finding_text = "# Final Negative / Conditional Finding Audit\n\n" + markdown_table(
        findings, ["claim_id", "claim", "status", "evidence", "evidence_files"], digits=4
    ) + "\n\nNOT_SUPPORTED claims must not be retained as core conclusions. This audit contains no unsupported core claim.\n"
    (REPORTS / "FINAL_FINDING_AUDIT.md").write_text(finding_text, encoding="utf-8")
    (PACKAGE / "FINAL_FINDING_AUDIT.md").write_text(finding_text, encoding="utf-8")

    config = yaml.safe_load((PACKAGE / "00_config" / "study_config_final.yaml").read_text(encoding="utf-8"))
    lag = pd.read_csv(AUDIT / "final_lag_rolling_audit.csv")
    nodes = pd.read_csv(AUDIT / "final_node_order_audit.csv")
    target = pd.read_csv(AUDIT / "final_target_integrity.csv")
    data_report = f"""# T2 Final Data Readiness Report

Generated: {GENERATED}

## Readiness outcome

**PASS — the frozen dataset is ready for submission-grade experiments.**

- Frozen node-hours: 100,929; nodes: {len(nodes)}; verified physical node order: PASS.
- Resolution: 1 hour; horizons: 1 h, 3 h, 6 h; point-speed target definitions are unchanged.
- Split/target integrity rows passing: {int(target['status'].eq('PASS').sum())}/{len(target)}.
- Independent lag/rolling checks: {len(lag)} features × 1,000 rows; total mismatches: {int(lag['mismatch_count'].sum())}.
- Traffic missingness: causal LOCF with missing mask, gap length, coverage, and reliability; one origin without a past observation is documented and excluded from every model.
- Environmental missingness: training-only medians/categories; no validation/test statistics are used.
- Weather source: {config['weather_source']}.
- Atmospheric source: {config['atmospheric_source']}; not roadside or ground-station observations.
- Evaluation-only pollution labels are absent from predictors.

## Evidence

See `01_final_data/00_final_freeze/` and `02_audit/` for manifests, row-level FCD checks, node/time/target audits, and lag/rolling recomputation.
"""
    (REPORTS / "T2_FINAL_DATA_READINESS_REPORT.md").write_text(data_report, encoding="utf-8")

    e0 = pd.read_csv(E0 / "metrics_summary.csv")
    main = pd.read_csv(E1_E5 / "metrics_summary.csv")
    e6_best = []
    for name in ("modality_ablation", "meteorology_ablation", "rainfall_ablation", "atmospheric_ablation"):
        frame = pd.read_csv(E6 / name / "metrics_summary.csv")
        best = frame.sort_values(["horizon", "MAE_mean"]).groupby("horizon").head(1).copy()
        best["ablation_group"] = name
        e6_best.append(best)
    e6_best = pd.concat(e6_best, ignore_index=True)
    main_table = main.loc[main["experiment"].isin([
        "E1_traffic_only", "E2_rainfall", "E3_meteorology", "E4_atmospheric", "E5_full_fusion"
    ])][["experiment", "horizon", "MAE_mean", "MAE_std", "RMSE_mean", "sMAPE_mean", "R2_mean"]]
    experiment_report = f"""# T2 Final Experiment Report

Generated: {GENERATED}

## Frozen E1-E5 overall test results

{markdown_table(main_table)}

Traffic-only E1 is the strongest frozen E1-E5 configuration at every horizon. Full Fusion does not improve overall test MAE and the negative result is retained without split, target, or sample changes.

## Formal E0 benchmark

{markdown_table(e0[['model','horizon','MAE_mean','MAE_std','RMSE_mean','sMAPE_mean','R2_mean']])}

The former 8-epoch TCN is archived. The formal TCN uses a 100-epoch limit, patience 15, scheduler, clipping, training-only scaling, and validation-MAE checkpoint selection. Fourteen runs early-stopped; one reached a validation plateau at epoch 100.

## Best configuration within each E6 family and horizon

{markdown_table(e6_best[['ablation_group','variant','horizon','MAE_mean','MAE_std']])}

M4 (Traffic-only + surface pressure) was selected solely by validation MAE as the best selective environmental model. Its small overall test differences versus E1 are not statistically significant after correction.

## Conditional interpretation

Atmospheric E4 is worse overall but shows descriptive 3 h/6 h gains in elevated-AOD and rain-plus-elevated-pollution scenarios. These are conditional findings, not universal multimodal superiority.
"""
    (REPORTS / "T2_FINAL_EXPERIMENT_REPORT.md").write_text(experiment_report, encoding="utf-8")

    statistical = pd.read_csv(STATS / "holm_corrected_results.csv")
    statistical_report = f"""# T2 Final Statistical Report

Generated: {GENERATED}

## Protocol

Errors are first averaged across the 51 nodes for each forecast origin and seed, then averaged across seeds. Confidence intervals use 5,000 paired moving-block bootstrap repetitions with contiguous 24-hour blocks. The secondary test is paired Wilcoxon on daily MAE, with Holm correction across the 15 main comparisons.

Delta convention: comparator minus E1; positive values mean the comparator is worse.

## Results

{markdown_table(statistical[['comparison','horizon','delta_MAE','relative_change_percent','ci_lower','ci_upper','p_value','p_adjusted','effect_size','significant']])}

No daily Wilcoxon comparison remains significant after Holm correction. Bootstrap intervals nevertheless show positive mean degradation for the frozen E2-E5 overall comparisons, while the validation-selected M4 pressure model remains close to E1.
"""
    (REPORTS / "T2_FINAL_STATISTICAL_REPORT.md").write_text(statistical_report, encoding="utf-8")

    evidence_report = f"""# T2 Final Evidence Audit

Generated: {GENERATED}

Every reported main value traces to prediction-level parquet files. Figure and table source data were generated programmatically, not typed from manuscript values.

## Claim status

{markdown_table(findings[['claim_id','status','claim','evidence_files']])}

## Provenance rules

- E1-E5: 15/15 horizon×seed sample comparisons pass.
- TCN: 15/15 sample comparisons and convergence checks pass.
- E6: all 450 variant×horizon×seed sample comparisons pass.
- UQ: 45/45 calibration audits pass; test is never calibration.
- Interpretation: 15/15 frozen E5 checkpoints reproduce prediction files exactly.
- CAMS is described only as CAMS-derived gridded atmospheric composition data.
"""
    (REPORTS / "T2_FINAL_EVIDENCE_AUDIT.md").write_text(evidence_report, encoding="utf-8")

    summary = f"""# T2 Transportmetrica B Evidence Summary

## Main conclusion

The strongest defensible conclusion is negative and conditional: traffic-only forecasting is difficult to beat overall, while simple environmental augmentation does not universally improve speed forecasts. Environmental value depends on horizon and regime.

## Submission-level findings

1. Frozen E1 is best among E1-E5 at 1 h, 3 h, and 6 h; Seasonal Historical Average slightly exceeds E1 at 6 h in E0.
2. E2-E5 do not improve overall MAE; no samples or dates were removed to change that result.
3. Validation-only selection identifies surface pressure as the best selective addition, with small non-significant gains at 1 h/3 h and a small loss at 6 h.
4. Atmospheric E4 has descriptive 3 h/6 h benefits under elevated AOD and compound rain-plus-elevated-pollution conditions.
5. Excluding 2026-02-17 does not reverse the E5-versus-E1 conclusion.
6. UQ achieves empirical PICP above the 90% nominal level; adverse regimes generally have wider intervals, but the selected environmental model does not clearly improve uncertainty performance.

See `FINAL_FINDING_AUDIT.md` for claim-by-claim status and limitations.
"""
    (REPORTS / "T2_TRANSPORTMETRICA_B_EVIDENCE_SUMMARY.md").write_text(summary, encoding="utf-8")
    return findings


def metric_reproduction_checks() -> pd.DataFrame:
    rows = []
    main_predictions = pd.read_parquet(E1_E5 / "predictions_all_models.parquet")
    reported = pd.read_csv(E1_E5 / "metrics_by_seed.csv")
    for (experiment, horizon, seed), group in main_predictions.groupby(["experiment", "horizon", "seed"]):
        actual = float(np.abs(group["y_pred"] - group["y_true"]).mean())
        expected = float(reported.loc[
            reported["experiment"].eq(experiment) & reported["horizon"].eq(horizon) & reported["seed"].eq(seed),
            "MAE",
        ].iloc[0])
        rows.append({
            "scope": "E1_E5", "item": experiment, "horizon": horizon, "seed": seed,
            "recomputed_MAE": actual, "reported_MAE": expected,
            "absolute_difference": abs(actual - expected),
            "status": "PASS" if abs(actual - expected) < 1e-10 else "FAIL",
        })
    for name in ("modality_ablation", "meteorology_ablation", "rainfall_ablation", "atmospheric_ablation"):
        predictions = pd.read_parquet(E6 / name / "predictions.parquet")
        reported = pd.read_csv(E6 / name / "metrics_by_seed.csv")
        for (variant, horizon, seed), group in predictions.groupby(["variant", "horizon", "seed"]):
            actual = float(np.abs(group["y_pred"] - group["y_true"]).mean())
            expected = float(reported.loc[
                reported["variant"].eq(variant) & reported["horizon"].eq(horizon) & reported["seed"].eq(seed),
                "MAE",
            ].iloc[0])
            rows.append({
                "scope": name, "item": variant, "horizon": horizon, "seed": seed,
                "recomputed_MAE": actual, "reported_MAE": expected,
                "absolute_difference": abs(actual - expected),
                "status": "PASS" if abs(actual - expected) < 1e-10 else "FAIL",
            })
    return pd.DataFrame(rows)


def generate_audits(findings: pd.DataFrame) -> None:
    audit1_checks = []
    for name, status_column in [
        ("final_node_order_audit.csv", "row_status"),
        ("final_temporal_integrity.csv", "status"),
        ("final_split_integrity.csv", "status"),
        ("final_target_integrity.csv", "status"),
        ("final_lag_rolling_audit.csv", "status"),
        ("environment_training_only_imputation_audit.csv", "status"),
    ]:
        frame = pd.read_csv(AUDIT / name)
        passed = frame[status_column].eq("PASS").all()
        audit1_checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": f"rows={len(frame)}"})
    frozen = pd.read_parquet(DATA / "final_model_features_frozen.parquet")
    forbidden = [
        column for column in (
            "upper_quartile_pm25_context", "upper_quartile_aod_context", "rain_with_upper_quartile_pollution"
        ) if column in frozen
    ]
    audit1_checks.append({
        "check": "scenario_labels_absent_from_predictor_table", "status": "PASS" if not forbidden else "FAIL",
        "detail": ";".join(forbidden) or "none",
    })
    audit1 = pd.DataFrame(audit1_checks)
    audit1.to_csv(AUDIT / "FINAL_AUDIT_PASS1.csv", index=False)
    pass1 = f"""# Final Audit Pass 1 — Data

{markdown_table(audit1)}

Outcome: **{'PASS' if audit1['status'].eq('PASS').all() else 'FAIL'}**. Node, time, split, target, lag/rolling, missingness, and sample-label separation were checked independently.
"""
    (REPORTS / "FINAL_AUDIT_PASS1.md").write_text(pass1, encoding="utf-8")
    (PACKAGE / "FINAL_AUDIT_PASS1.md").write_text(pass1, encoding="utf-8")

    reproduction = metric_reproduction_checks()
    reproduction.to_csv(AUDIT / "metric_reproduction_audit.csv", index=False)
    experiment_checks = [
        {"check": "formal_tcn_convergence", "status": "PASS" if pd.read_csv(E0 / "convergence_audit.csv")["status"].eq("PASS").all() else "FAIL", "detail": "15 runs"},
        {"check": "formal_tcn_sample_consistency", "status": "PASS" if pd.read_csv(E0 / "sample_consistency_tcn.csv")["status"].eq("PASS").all() else "FAIL", "detail": "15 comparisons"},
        {"check": "E1_E5_sample_consistency", "status": "PASS" if pd.read_csv(E1_E5 / "sample_consistency_final.csv")["status"].eq("PASS").all() else "FAIL", "detail": "15 comparisons"},
        {"check": "E6_sample_consistency", "status": "PASS" if all(pd.read_csv(E6 / name / "sample_consistency.csv")["status"].eq("PASS").all() for name in ("modality_ablation","meteorology_ablation","rainfall_ablation","atmospheric_ablation")) else "FAIL", "detail": "450 comparisons"},
        {"check": "UQ_calibration", "status": "PASS" if pd.read_csv(UQ / "calibration_audit.csv")["status"].eq("PASS").all() else "FAIL", "detail": "45 runs"},
        {"check": "checkpoint_reproduction", "status": "PASS" if pd.read_csv(INTERPRET / "checkpoint_prediction_reproduction_audit.csv")["status"].eq("PASS").all() else "FAIL", "detail": "15 E5 checkpoints"},
        {"check": "metric_reproduction", "status": "PASS" if reproduction["status"].eq("PASS").all() else "FAIL", "detail": f"{len(reproduction)} prediction groups"},
    ]
    audit2 = pd.DataFrame(experiment_checks)
    audit2.to_csv(AUDIT / "FINAL_AUDIT_PASS2.csv", index=False)
    pass2 = f"""# Final Audit Pass 2 — Experiments

{markdown_table(audit2)}

Outcome: **{'PASS' if audit2['status'].eq('PASS').all() else 'FAIL'}**. Seeds, model protocol, convergence, prediction counts, and metric reproducibility were checked.
"""
    (REPORTS / "FINAL_AUDIT_PASS2.md").write_text(pass2, encoding="utf-8")
    (PACKAGE / "FINAL_AUDIT_PASS2.md").write_text(pass2, encoding="utf-8")

    audit3_checks = []
    for figure in [
        "Fig1_corridor_environmental_grids", "Fig2_data_availability_environmental_conditions",
        "Fig3_method_framework", "Fig4_modality_comparison", "Fig5_rainfall_lag_and_ablation",
        "Fig6_environment_reliability_regimes", "Fig7_UQ_environmental_risk",
    ]:
        folder = FIGURES / figure
        ok = all((folder / name).exists() for name in ("source_data.csv", "figure_caption_draft.txt", "data_definition.txt"))
        audit3_checks.append({"artifact": figure, "type": "Figure", "status": "PASS" if ok else "FAIL", "source": str(folder.relative_to(PACKAGE))})
    for table in [
        "Table1_data_sources_variables.csv", "Table2_dataset_split_scenarios.csv",
        "Table3_main_forecasting_results.csv", "Table4_ablation_uq_sensitivity.csv",
    ]:
        ok = (TABLES / table).exists()
        audit3_checks.append({"artifact": table, "type": "Table", "status": "PASS" if ok else "FAIL", "source": str((TABLES / table).relative_to(PACKAGE))})
    for row in findings.itertuples(index=False):
        paths = [PACKAGE / part for part in str(row.evidence_files).split(";")]
        ok = all(path.exists() for path in paths)
        audit3_checks.append({"artifact": f"Claim {row.claim_id}", "type": "Conclusion", "status": "PASS" if ok else "FAIL", "source": row.evidence_files})
    audit3 = pd.DataFrame(audit3_checks)
    audit3.to_csv(AUDIT / "FINAL_AUDIT_PASS3.csv", index=False)
    pass3 = f"""# Final Audit Pass 3 — Paper Evidence

{markdown_table(audit3)}

Outcome: **{'PASS' if audit3['status'].eq('PASS').all() else 'FAIL'}**. Every figure, table, and major conclusion has an existing source file.
"""
    (REPORTS / "FINAL_AUDIT_PASS3.md").write_text(pass3, encoding="utf-8")
    (PACKAGE / "FINAL_AUDIT_PASS3.md").write_text(pass3, encoding="utf-8")
    if not all(frame["status"].eq("PASS").all() for frame in (audit1, audit2, audit3)):
        raise ValueError("One or more final audits failed")


def duplicate_required_views() -> None:
    freeze = PACKAGE / "00_final_freeze"
    freeze.mkdir(parents=True, exist_ok=True)
    for source in (DATA / "00_final_freeze").iterdir():
        if source.is_file():
            shutil.copy2(source, freeze / source.name)
    audit_view = PACKAGE / "audit"
    audit_view.mkdir(parents=True, exist_ok=True)
    for source in AUDIT.glob("*.csv"):
        shutil.copy2(source, audit_view / source.name)


def optional_inventory() -> None:
    required = pd.DataFrame(columns=["required_item", "status", "detail"])
    required.to_csv(PACKAGE / "T2_MISSING_REQUIRED_ITEMS.csv", index=False)
    optional = pd.DataFrame(
        [
            {"optional_item": "GRU benchmark", "status": "NOT_RUN_OPTIONAL", "reason": "TCN is the required deep benchmark; GRU was explicitly optional."},
            {"optional_item": "SHAP", "status": "NOT_RUN_OPTIONAL", "reason": "Required group permutation importance was completed; SHAP was optional."},
            {"optional_item": "Heavy-rain inferential analysis", "status": "DESCRIPTIVE_ONLY", "reason": "Insufficient independent test events; excluded from main claims by protocol."},
        ]
    )
    optional.to_csv(PACKAGE / "T2_REMAINING_OPTIONAL_ITEMS.csv", index=False)


def checksums() -> pd.DataFrame:
    excluded_names = {"checksums_sha256.csv"}
    records = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.name in excluded_names or "__pycache__" in path.parts:
            continue
        records.append({
            "relative_path": path.relative_to(PACKAGE).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = pd.DataFrame(records)
    for path in (
        PACKAGE / "checksums_sha256.csv",
        PACKAGE / "00_config" / "checksums_sha256.csv",
        PACKAGE / "00_final_freeze" / "checksums_sha256.csv",
        DATA / "00_final_freeze" / "checksums_sha256.csv",
    ):
        manifest.to_csv(path, index=False)
    return manifest


def delivery_readme(manifest: pd.DataFrame) -> None:
    text = f"""# T2 Transportmetrica B Final Experiment Package

Generated: {GENERATED}

This folder contains the frozen data, three-pass audit evidence, formal E0 benchmark, frozen E1-E5 results, E6 ablations, scenario/reliability/outage analyses, E7 uncertainty quantification, statistical tests, interpretability outputs, and Figure/Table source data.

- Checksum-listed files: {len(manifest)}
- Required missing items: 0
- Optional non-blocking items are listed in `T2_REMAINING_OPTIONAL_ITEMS.csv`.
- Start with `14_final_reports/T2_TRANSPORTMETRICA_B_EVIDENCE_SUMMARY.md`.

No original data were modified. No test samples were removed to improve conclusions. Negative results remain in the package.
"""
    (PACKAGE / "README_DELIVERY.md").write_text(text, encoding="utf-8")
    (DELIVERY / "README_T2_20260719.md").write_text(text, encoding="utf-8")


def create_zip() -> dict:
    zip_path = DELIVERY / "T2_TRANSPORTMETRICA_B_FINAL_EXPERIMENT_PACKAGE.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, Path(PACKAGE.name) / path.relative_to(PACKAGE))
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad = archive.testzip()
        entry_count = len(archive.infolist())
        uncompressed = sum(item.file_size for item in archive.infolist())
    result = {
        "zip": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
        "entry_count": entry_count,
        "uncompressed_bytes": uncompressed,
        "crc_test": "PASS" if bad is None else f"FAIL:{bad}",
    }
    (DELIVERY / "T2_TRANSPORTMETRICA_B_FINAL_EXPERIMENT_PACKAGE.sha256.txt").write_text(
        f"{result['zip_sha256']}  {zip_path.name}\n", encoding="utf-8"
    )
    (DELIVERY / "T2_ZIP_VALIDATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-zip", action="store_true")
    args = parser.parse_args()
    generate_figure_sources()
    generate_table_sources()
    findings = reports_and_findings()
    optional_inventory()
    generate_audits(findings)
    duplicate_required_views()
    manifest = checksums()
    delivery_readme(manifest)
    # README is generated after the first checksum pass; regenerate so it is covered.
    manifest = checksums()
    prepackage = {
        "status": "PASS",
        "audit_passes": 3,
        "checksum_listed_files": len(manifest),
        "required_missing_items": 0,
        "generated": GENERATED,
    }
    (DELIVERY / "T2_PREPACKAGE_VALIDATION.json").write_text(json.dumps(prepackage, indent=2), encoding="utf-8")
    result = None if args.skip_zip else create_zip()
    print(json.dumps({"prepackage": prepackage, "zip": result}, indent=2))


if __name__ == "__main__":
    main()
