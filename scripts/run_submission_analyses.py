from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "01_final_data"
E1_E5 = PACKAGE / "04_E1_E5"
E6 = PACKAGE / "05_E6_ablation"
SCENARIO_OUT = PACKAGE / "06_scenario_analysis"
RELIABILITY_OUT = PACKAGE / "07_reliability_analysis"
SENSITIVITY_OUT = PACKAGE / "08_sensitivity_analysis"
STATS_OUT = PACKAGE / "10_statistical_tests"
SOURCE = Path(r"C:\Users\DELL\Desktop\多数据源数据\20260715\T2_final_V2_complete")

HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)
MAIN_EXPERIMENTS = {
    "E1_traffic_only": "E1_traffic_only",
    "E2_rainfall": "E2_rainfall",
    "E3_meteorology": "E3_meteorology",
    "E4_atmospheric": "E4_atmospheric",
    "E5_full_fusion": "E5_full_fusion",
}
COMPARISON_EXPERIMENTS = ["E1_traffic_only", "E2_rainfall", "E3_meteorology", "E4_atmospheric", "E5_full_fusion"]
SCENARIOS = [
    "S1_dry",
    "S2_rain",
    "S3_widespread_rain",
    "S4_post_rain_1_3h",
    "S5_post_rain_4_6h",
    "S6_elevated_pm2_5",
    "S7_elevated_aod",
    "S8_rain_elevated_atmospheric_pollution",
]
INTERACTION_SCENARIOS = [
    "S1_dry",
    "S2_rain",
    "S7_elevated_aod",
    "S8_rain_elevated_atmospheric_pollution",
]
INTERACTION_MODELS = ["E1_traffic_only", "E3_meteorology", "E4_atmospheric", "E5_full_fusion"]


def ensure_dirs() -> None:
    for path in (SCENARIO_OUT, RELIABILITY_OUT, SENSITIVITY_OUT, STATS_OUT):
        path.mkdir(parents=True, exist_ok=True)
    (SCENARIO_OUT / "horizon_analysis").mkdir(parents=True, exist_ok=True)
    (SCENARIO_OUT / "rain_event_analysis").mkdir(parents=True, exist_ok=True)


def sample_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(values.astype(str))).encode()).hexdigest()


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    error = y_pred - y_true
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    total = np.square(y_true - y_true.mean()).sum()
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "sMAPE": float(np.where(denominator > 1e-12, np.abs(error) / denominator, 0).mean() * 100),
        "R2": float(1 - np.square(error).sum() / total) if total > 0 else float("nan"),
    }


def metric_summary(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["seed_count"] = group["seed"].nunique() if "seed" in group else np.nan
        row["sample_count_mean"] = float(group["sample_count"].mean()) if "sample_count" in group else np.nan
        for metric in ("MAE", "RMSE", "sMAPE", "R2"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def load_main_predictions() -> pd.DataFrame:
    tables = []
    for experiment, folder in MAIN_EXPERIMENTS.items():
        frame = pd.read_parquet(E1_E5 / folder / "predictions.parquet").copy()
        frame["experiment"] = experiment
        tables.append(frame)
    predictions = pd.concat(tables, ignore_index=True)
    predictions["forecast_origin"] = pd.to_datetime(predictions["forecast_origin"])
    predictions["target_timestamp"] = pd.to_datetime(predictions["target_timestamp"])
    predictions.to_parquet(E1_E5 / "predictions_all_models.parquet", index=False)

    metrics = pd.concat(
        [pd.read_csv(E1_E5 / folder / "metrics_by_seed.csv") for folder in MAIN_EXPERIMENTS.values()],
        ignore_index=True,
    )
    metrics.to_csv(E1_E5 / "metrics_by_seed.csv", index=False)
    summaries = pd.concat(
        [pd.read_csv(E1_E5 / folder / "metrics_summary.csv") for folder in MAIN_EXPERIMENTS.values()],
        ignore_index=True,
    )
    summaries.to_csv(E1_E5 / "metrics_summary.csv", index=False)

    consistency_rows = []
    for horizon in HORIZONS:
        for seed in SEEDS:
            group = predictions.loc[predictions["horizon"].eq(horizon) & predictions["seed"].eq(seed)]
            hashes = group.groupby("experiment")["sample_id"].agg(sample_hash)
            counts = group.groupby("experiment")["sample_id"].size()
            truth_hashes = group.assign(pair=group["sample_id"].astype(str) + "|" + group["y_true"].astype(str)).groupby(
                "experiment"
            )["pair"].agg(sample_hash)
            consistency_rows.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "experiment_count": group["experiment"].nunique(),
                    "sample_count": int(counts.iloc[0]),
                    "all_counts_identical": counts.nunique() == 1,
                    "all_sample_id_hashes_identical": hashes.nunique() == 1,
                    "all_y_true_hashes_identical": truth_hashes.nunique() == 1,
                    "duplicate_sample_ids": int(
                        group.groupby("experiment")["sample_id"].apply(lambda x: x.duplicated().sum()).sum()
                    ),
                    "status": "PASS"
                    if group["experiment"].nunique() == 5
                    and counts.nunique() == 1
                    and hashes.nunique() == 1
                    and truth_hashes.nunique() == 1
                    else "FAIL",
                }
            )
    consistency = pd.DataFrame(consistency_rows)
    consistency.to_csv(E1_E5 / "sample_consistency_final.csv", index=False)
    if not consistency["status"].eq("PASS").all():
        raise ValueError("E1-E5 sample consistency failed")
    return predictions


def long_labels() -> pd.DataFrame:
    labels = pd.read_parquet(DATA / "scenario_labels_final.parquet")
    labels["timestamp_local"] = pd.to_datetime(labels["timestamp_local"])
    tables = []
    common = [
        "node_id", "timestamp_local", "split", "uq_partition", "precipitation",
        "reliability_score", "reliability_group_final", *SCENARIOS,
    ]
    for horizon in HORIZONS:
        table = labels[common + [f"sample_id_h{horizon}"]].rename(
            columns={f"sample_id_h{horizon}": "sample_id"}
        )
        table = table.loc[table["sample_id"].notna()].copy()
        table["horizon"] = horizon
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)

    raw = pd.read_parquet(
        SOURCE / "08_multimodal_panel" / "multimodal_hourly_panel.parquet",
        columns=[
            "node_id", "timestamp_local", "corridor_rain_event_id", "wet_grid_fraction",
            "corridor_max_rainfall", "corridor_cumulative_rainfall", "current_speed_input",
        ],
    )
    raw["timestamp_local"] = pd.to_datetime(raw["timestamp_local"])
    result = result.merge(raw, on=["node_id", "timestamp_local"], how="left", validate="m:1")
    result.to_parquet(DATA / "sample_labels_long.parquet", index=False)
    return result


def contiguous_events(timestamps: pd.Series) -> int:
    unique = pd.Series(pd.to_datetime(timestamps).dropna().unique()).sort_values()
    if len(unique) == 0:
        return 0
    return int(unique.diff().gt(pd.Timedelta(hours=1)).sum() + 1)


def scenario_analysis(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[predictions["experiment"].isin(INTERACTION_MODELS)].copy()
    selected = selected.merge(
        labels, on=["sample_id", "node_id", "horizon"], how="left", validate="m:1"
    )
    count_rows = []
    metric_rows = []
    prediction_tables = []
    for horizon in HORIZONS:
        horizon_labels = labels.loc[labels["horizon"].eq(horizon) & labels["split"].eq("test")]
        for scenario in SCENARIOS:
            label_subset = horizon_labels.loc[horizon_labels[scenario].fillna(False)]
            count_rows.append(
                {
                    "scenario": scenario,
                    "split": "test",
                    "horizon": horizon,
                    "node_hours": len(label_subset),
                    "unique_hours": label_subset["timestamp_local"].nunique(),
                    "unique_dates": label_subset["timestamp_local"].dt.date.nunique(),
                    "independent_events": contiguous_events(label_subset["timestamp_local"]),
                    "number_of_nodes": label_subset["node_id"].nunique(),
                }
            )
            subset = selected.loc[selected["horizon"].eq(horizon) & selected[scenario].fillna(False)].copy()
            subset["scenario"] = scenario
            prediction_tables.append(subset)
            for (experiment, model, seed), group in subset.groupby(["experiment", "model", "seed"], sort=False):
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "experiment": experiment,
                        "model": model,
                        "horizon": horizon,
                        "seed": int(seed),
                        "sample_count": len(group),
                        "unique_hours": group["forecast_origin"].nunique(),
                        "unique_dates": group["forecast_origin"].dt.date.nunique(),
                        "independent_events": contiguous_events(group["forecast_origin"]),
                        **metric_values(group),
                    }
                )
    counts = pd.DataFrame(count_rows)
    metrics = pd.DataFrame(metric_rows)
    scenario_predictions = pd.concat(prediction_tables, ignore_index=True)
    counts.to_csv(SCENARIO_OUT / "scenario_sample_counts.csv", index=False)
    metrics.to_csv(SCENARIO_OUT / "scenario_metrics_by_seed.csv", index=False)
    metric_summary(metrics, ["scenario", "experiment", "model", "horizon"]).to_csv(
        SCENARIO_OUT / "scenario_metrics_summary.csv", index=False
    )
    scenario_predictions.to_parquet(SCENARIO_OUT / "scenario_predictions.parquet", index=False)
    return selected


def reliability_interaction(selected: pd.DataFrame) -> None:
    metric_rows = []
    count_rows = []
    prediction_tables = []
    for scenario in INTERACTION_SCENARIOS:
        for reliability in ("High", "Medium", "Low"):
            subset = selected.loc[
                selected[scenario].fillna(False)
                & selected["reliability_group_final"].eq(reliability)
            ].copy()
            subset["scenario"] = scenario
            subset["reliability_group"] = reliability
            prediction_tables.append(subset)
            for horizon in HORIZONS:
                label_count = subset.loc[
                    subset["horizon"].eq(horizon)
                    & subset["experiment"].eq("E1_traffic_only")
                    & subset["seed"].eq(SEEDS[0])
                ]
                count_rows.append(
                    {
                        "scenario": scenario,
                        "reliability_group": reliability,
                        "horizon": horizon,
                        "sample_count": len(label_count),
                        "unique_hours": label_count["forecast_origin"].nunique(),
                        "unique_dates": label_count["forecast_origin"].dt.date.nunique(),
                        "independent_events": contiguous_events(label_count["forecast_origin"]),
                    }
                )
            for (experiment, model, horizon, seed), group in subset.groupby(
                ["experiment", "model", "horizon", "seed"], sort=False
            ):
                metric_rows.append(
                    {
                        "scenario": scenario,
                        "reliability_group": reliability,
                        "experiment": experiment,
                        "model": model,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "sample_count": len(group),
                        **metric_values(group),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RELIABILITY_OUT / "interaction_metrics_by_seed.csv", index=False)
    metric_summary(
        metrics, ["scenario", "reliability_group", "experiment", "model", "horizon"]
    ).to_csv(RELIABILITY_OUT / "interaction_metrics_summary.csv", index=False)
    pd.DataFrame(count_rows).to_csv(RELIABILITY_OUT / "interaction_sample_counts.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_parquet(
        RELIABILITY_OUT / "interaction_predictions.parquet", index=False
    )


def outage_sensitivity(selected: pd.DataFrame) -> None:
    overall_rows = []
    detailed_rows = []
    sets = {
        "Test-A_all": selected,
        "Test-B_excluding_20260217": selected.loc[
            selected["forecast_origin"].dt.date.ne(pd.Timestamp("2026-02-17").date())
        ],
    }
    for test_set, frame in sets.items():
        for (experiment, model, horizon, seed), group in frame.groupby(
            ["experiment", "model", "horizon", "seed"], sort=False
        ):
            overall_rows.append(
                {
                    "test_set": test_set,
                    "experiment": experiment,
                    "model": model,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "sample_count": len(group),
                    **metric_values(group),
                }
            )
        for scenario in SCENARIOS:
            for (experiment, model, horizon, seed), group in frame.loc[frame[scenario].fillna(False)].groupby(
                ["experiment", "model", "horizon", "seed"], sort=False
            ):
                detailed_rows.append(
                    {
                        "test_set": test_set,
                        "stratum_type": "scenario",
                        "stratum": scenario,
                        "experiment": experiment,
                        "model": model,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "sample_count": len(group),
                        **metric_values(group),
                    }
                )
        for reliability in ("High", "Medium", "Low"):
            for (experiment, model, horizon, seed), group in frame.loc[
                frame["reliability_group_final"].eq(reliability)
            ].groupby(["experiment", "model", "horizon", "seed"], sort=False):
                detailed_rows.append(
                    {
                        "test_set": test_set,
                        "stratum_type": "reliability",
                        "stratum": reliability,
                        "experiment": experiment,
                        "model": model,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "sample_count": len(group),
                        **metric_values(group),
                    }
                )
    overall = pd.DataFrame(overall_rows)
    overall.loc[overall["test_set"].eq("Test-A_all")].to_csv(
        SENSITIVITY_OUT / "all_test_metrics.csv", index=False
    )
    overall.loc[overall["test_set"].eq("Test-B_excluding_20260217")].to_csv(
        SENSITIVITY_OUT / "excluding_20260217_metrics.csv", index=False
    )
    pd.DataFrame(detailed_rows).to_csv(SENSITIVITY_OUT / "stratified_sensitivity_metrics.csv", index=False)
    keys = ["experiment", "model", "horizon", "seed"]
    all_set = overall.loc[overall["test_set"].eq("Test-A_all")].drop(columns="test_set")
    excluded = overall.loc[overall["test_set"].eq("Test-B_excluding_20260217")].drop(columns="test_set")
    comparison = all_set.merge(excluded, on=keys, suffixes=("_all", "_excluding"), validate="1:1")
    comparison["delta_MAE"] = comparison["MAE_excluding"] - comparison["MAE_all"]
    comparison["relative_change_percent"] = comparison["delta_MAE"] / comparison["MAE_all"] * 100
    comparison.to_csv(SENSITIVITY_OUT / "sensitivity_comparison.csv", index=False)

    mean_metrics = overall.groupby(["test_set", "experiment", "horizon"])["MAE"].mean().reset_index()
    pivot = mean_metrics.pivot(index=["test_set", "horizon"], columns="experiment", values="MAE").reset_index()
    pivot["E5_minus_E1"] = pivot["E5_full_fusion"] - pivot["E1_traffic_only"]
    signs = pivot.pivot(index="horizon", columns="test_set", values="E5_minus_E1")
    conclusion = pd.DataFrame(
        {
            "horizon": signs.index,
            "E5_minus_E1_all": signs["Test-A_all"],
            "E5_minus_E1_excluding": signs["Test-B_excluding_20260217"],
            "conclusion_sign_changed": np.sign(signs["Test-A_all"]) != np.sign(signs["Test-B_excluding_20260217"]),
        }
    ).reset_index(drop=True)
    conclusion.to_csv(SENSITIVITY_OUT / "outage_conclusion_stability.csv", index=False)


def best_environmental_selection() -> tuple[str, str, pd.DataFrame]:
    logs = []
    for path in E6.rglob("*_training_log.csv"):
        frame = pd.read_csv(path)
        if "best_validation_MAE" not in frame:
            continue
        frame = frame.loc[frame["best_validation_MAE"].notna()].copy()
        if len(frame):
            frame["ablation_group"] = path.parents[1].name
            logs.append(frame)
    candidates = pd.concat(logs, ignore_index=True)
    summary = candidates.groupby(["ablation_group", "variant"], as_index=False).agg(
        validation_MAE_mean=("best_validation_MAE", "mean"),
        validation_MAE_std=("best_validation_MAE", "std"),
        validation_runs=("best_validation_MAE", "size"),
    ).sort_values("validation_MAE_mean")
    summary["selected_for_UQ_and_primary_selective_comparison"] = False
    summary.loc[summary.index[0], "selected_for_UQ_and_primary_selective_comparison"] = True
    summary["selection_uses_test"] = False
    summary.to_csv(STATS_OUT / "environmental_best_model_selection.csv", index=False)
    row = summary.iloc[0]
    return str(row["ablation_group"]), str(row["variant"]), summary


def horizon_analysis(predictions: pd.DataFrame, best_group: str, best_variant: str) -> None:
    rows = []
    for (experiment, horizon, seed), group in predictions.groupby(["experiment", "horizon", "seed"]):
        rows.append(
            {"experiment": experiment, "horizon": horizon, "seed": seed, **metric_values(group)}
        )
    metrics = pd.DataFrame(rows)
    ranking = metrics.groupby(["experiment", "horizon"], as_index=False).agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")
    )
    ranking["rank"] = ranking.groupby("horizon")["MAE_mean"].rank(method="min")
    ranking.sort_values(["horizon", "rank"]).to_csv(
        SCENARIO_OUT / "horizon_analysis" / "model_ranking_by_horizon.csv", index=False
    )
    e1 = ranking.loc[ranking["experiment"].eq("E1_traffic_only"), ["horizon", "MAE_mean"]].rename(
        columns={"MAE_mean": "E1_MAE"}
    )
    gain = ranking.merge(e1, on="horizon", how="left")
    gain["delta_MAE_model_minus_E1"] = gain["MAE_mean"] - gain["E1_MAE"]
    gain["relative_change_percent"] = gain["delta_MAE_model_minus_E1"] / gain["E1_MAE"] * 100
    gain.to_csv(
        SCENARIO_OUT / "horizon_analysis" / "environmental_gain_by_horizon.csv", index=False
    )
    (SCENARIO_OUT / "horizon_analysis" / "selection_note.json").write_text(
        json.dumps({"best_validation_selected_group": best_group, "variant": best_variant}, indent=2),
        encoding="utf-8",
    )


def rain_event_analysis(predictions: pd.DataFrame, labels: pd.DataFrame) -> None:
    events = pd.read_csv(SOURCE / "10_feature_data" / "corridor_rain_events_final.csv")
    events["corridor_rain_event_id"] = pd.to_numeric(events["corridor_rain_event_id"], errors="coerce")
    joined = predictions.merge(
        labels[
            [
                "sample_id", "node_id", "horizon", "corridor_rain_event_id", "wet_grid_fraction",
                "corridor_max_rainfall", "corridor_cumulative_rainfall", "current_speed_input",
            ]
        ],
        on=["sample_id", "node_id", "horizon"], how="left", validate="m:1",
    )
    joined = joined.loc[joined["corridor_rain_event_id"].notna()].copy()
    metric_rows = []
    for (event_id, horizon), event_group in joined.groupby(["corridor_rain_event_id", "horizon"]):
        meta = events.loc[events["corridor_rain_event_id"].eq(event_id)]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        unique_truth = event_group.loc[
            event_group["experiment"].eq("E1_traffic_only") & event_group["seed"].eq(SEEDS[0])
        ]
        row = {
            "rain_event_id": int(event_id),
            "horizon": int(horizon),
            "event_start": meta["start_time"],
            "event_end": meta["end_time"],
            "duration_hours": meta["wet_duration_hours"],
            "maximum_rainfall": meta["corridor_max_rainfall"],
            "cumulative_rainfall": meta["corridor_cumulative_rainfall"],
            "affected_grid_fraction": meta["maximum_wet_grid_fraction"],
            "traffic_speed_change": float((unique_truth["y_true"] - unique_truth["current_speed_input"]).mean()),
            "sample_count": len(unique_truth),
        }
        for experiment, model_group in event_group.groupby("experiment"):
            row[f"{experiment}_MAE"] = float(np.abs(model_group["y_pred"] - model_group["y_true"]).mean())
        metric_rows.append(row)
    result = pd.DataFrame(metric_rows).sort_values(["rain_event_id", "horizon"])
    event_count = result["rain_event_id"].nunique()
    result["analysis_strength"] = "DESCRIPTIVE_ONLY" if event_count < 10 else "INFERENTIAL_SUPPORTED"
    result.to_csv(SCENARIO_OUT / "rain_event_analysis" / "rain_event_performance.csv", index=False)
    correlations = []
    for horizon, group in result.groupby("horizon"):
        for driver in ("maximum_rainfall", "cumulative_rainfall", "duration_hours", "affected_grid_fraction"):
            for experiment in COMPARISON_EXPERIMENTS:
                value_col = f"{experiment}_MAE"
                if value_col not in group or len(group) < 3:
                    continue
                x_rank = stats.rankdata(group[driver].to_numpy(float))
                y_rank = stats.rankdata(group[value_col].to_numpy(float))
                x_centered = x_rank - x_rank.mean()
                y_centered = y_rank - y_rank.mean()
                denominator = math.sqrt(float(np.square(x_centered).sum() * np.square(y_centered).sum()))
                coefficient = float((x_centered * y_centered).sum() / denominator) if denominator else 0.0
                if len(group) > 2 and abs(coefficient) < 1:
                    t_statistic = coefficient * math.sqrt((len(group) - 2) / max(1 - coefficient**2, 1e-12))
                    p_value = float(2 * stats.t.sf(abs(t_statistic), df=len(group) - 2))
                else:
                    p_value = float("nan")
                correlations.append(
                    {
                        "horizon": horizon,
                        "driver": driver,
                        "experiment": experiment,
                        "event_count": len(group),
                        "spearman_rho": coefficient,
                        "p_value": p_value,
                        "interpretation": "descriptive_only" if event_count < 10 else "inferential",
                    }
                )
    pd.DataFrame(correlations).to_csv(
        SCENARIO_OUT / "rain_event_analysis" / "event_metric_correlations.csv", index=False
    )


def best_variant_predictions(best_group: str, best_variant: str) -> pd.DataFrame:
    predictions = pd.read_parquet(E6 / best_group / "predictions.parquet")
    result = predictions.loc[predictions["variant"].eq(best_variant)].copy()
    result["experiment"] = "Best_selective_environmental"
    return result


def moving_block_bootstrap(diff: np.ndarray, repetitions: int, block: int, seed: int) -> np.ndarray:
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n / block)
    max_start = max(n - block + 1, 1)
    starts = rng.integers(0, max_start, size=(repetitions, blocks_needed))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(repetitions, -1)[:, :n]
    indices = np.minimum(indices, n - 1)
    return diff[indices].mean(axis=1)


def rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = (m - rank) * values[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def statistical_tests(predictions: pd.DataFrame, best_predictions: pd.DataFrame) -> None:
    combined = pd.concat([predictions, best_predictions], ignore_index=True)
    comparisons = [
        ("E1_vs_E2", "E2_rainfall"),
        ("E1_vs_E3", "E3_meteorology"),
        ("E1_vs_E4", "E4_atmospheric"),
        ("E1_vs_E5", "E5_full_fusion"),
        ("E1_vs_best_selective_environmental", "Best_selective_environmental"),
    ]
    bootstrap_rows = []
    wilcoxon_rows = []
    for comparison_number, (comparison, comparator) in enumerate(comparisons):
        for horizon in HORIZONS:
            subset = combined.loc[
                combined["horizon"].eq(horizon)
                & combined["experiment"].isin(["E1_traffic_only", comparator])
            ].copy()
            subset["absolute_error"] = np.abs(subset["y_pred"] - subset["y_true"])
            origin = subset.groupby(["experiment", "seed", "forecast_origin"])["absolute_error"].mean().reset_index()
            origin = origin.groupby(["experiment", "forecast_origin"])["absolute_error"].mean().unstack("experiment")
            origin = origin.dropna(subset=["E1_traffic_only", comparator]).sort_index()
            difference = origin[comparator] - origin["E1_traffic_only"]
            delta = float(difference.mean())
            reference = float(origin["E1_traffic_only"].mean())
            bootstrap = moving_block_bootstrap(
                difference.to_numpy(), repetitions=5000, block=24, seed=20260719 + comparison_number * 10 + horizon
            )
            p_bootstrap = min(1.0, 2 * min(float((bootstrap <= 0).mean()), float((bootstrap >= 0).mean())))
            bootstrap_rows.append(
                {
                    "comparison": comparison,
                    "comparator": comparator,
                    "horizon": horizon,
                    "delta_MAE": delta,
                    "delta_convention": "comparator_minus_E1; positive means comparator is worse",
                    "relative_change_percent": delta / reference * 100,
                    "ci_lower": float(np.quantile(bootstrap, 0.025)),
                    "ci_upper": float(np.quantile(bootstrap, 0.975)),
                    "bootstrap_p_value": p_bootstrap,
                    "origin_hours": len(origin),
                    "block_hours": 24,
                    "bootstrap_repetitions": 5000,
                }
            )

            daily = origin.resample("1D").mean().dropna(subset=["E1_traffic_only", comparator])
            daily_difference = (daily[comparator] - daily["E1_traffic_only"]).to_numpy()
            try:
                statistic, p_value = stats.wilcoxon(daily_difference, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            wilcoxon_rows.append(
                {
                    "comparison": comparison,
                    "comparator": comparator,
                    "horizon": horizon,
                    "delta_MAE": delta,
                    "relative_change_percent": delta / reference * 100,
                    "daily_pairs": len(daily),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "effect_size": rank_biserial(daily_difference),
                    "effect_size_definition": "matched-pairs rank-biserial; positive means comparator worse",
                }
            )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    wilcoxon_frame = pd.DataFrame(wilcoxon_rows)
    wilcoxon_frame["p_adjusted"] = holm_adjust(wilcoxon_frame["p_value"])
    wilcoxon_frame["significant"] = wilcoxon_frame["p_adjusted"].lt(0.05)
    bootstrap_frame.to_csv(STATS_OUT / "paired_block_bootstrap.csv", index=False)
    wilcoxon_frame.to_csv(STATS_OUT / "wilcoxon_daily.csv", index=False)
    final = bootstrap_frame.merge(
        wilcoxon_frame[
            ["comparison", "comparator", "horizon", "p_value", "p_adjusted", "effect_size", "significant"]
        ],
        on=["comparison", "comparator", "horizon"], validate="1:1",
    )
    final.to_csv(STATS_OUT / "holm_corrected_results.csv", index=False)


def main() -> None:
    ensure_dirs()
    predictions = load_main_predictions()
    labels = long_labels()
    selected = scenario_analysis(predictions, labels)
    reliability_interaction(selected)
    outage_sensitivity(selected)
    best_group, best_variant, _ = best_environmental_selection()
    horizon_analysis(predictions, best_group, best_variant)
    rain_event_analysis(predictions, labels)
    best_predictions = best_variant_predictions(best_group, best_variant)
    statistical_tests(predictions, best_predictions)
    print(
        json.dumps(
            {
                "status": "PASS",
                "best_environmental_group_validation_only": best_group,
                "best_environmental_variant_validation_only": best_variant,
                "E1_E5_consistency_checks": 15,
                "bootstrap_repetitions": 5000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
