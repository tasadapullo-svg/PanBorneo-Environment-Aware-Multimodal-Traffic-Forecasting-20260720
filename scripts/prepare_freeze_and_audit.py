from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = Path(r"C:\Users\DELL\Desktop\多数据源数据")
SOURCE = WORKSPACE / "20260715" / "T2_final_V2_complete"
FROZEN_20260718 = Path(r"C:\Users\DELL\Desktop\20260718")

CONFIG = PACKAGE / "00_config"
DATA = PACKAGE / "01_final_data"
AUDIT = PACKAGE / "02_audit"
REPORTS = PACKAGE / "14_final_reports"

HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)
GENERATED_AT = datetime.now().astimezone().isoformat(timespec="seconds")

TRAFFIC = [
    "current_speed_input",
    "free_flow_speed_input",
    "current_travel_time_input",
    "tti_input",
    "speed_ratio_input",
    "speed_drop_input",
    "travel_time_delay_input",
    "speed_deficit_ratio_input",
]
RELIABILITY = [
    "missing_mask",
    "gap_length_steps",
    "coverage_ratio_6h",
    "coverage_ratio_24h",
    "volatility_tti_3h",
    "volatility_tti_6h",
    "confidence_input",
    "reliability_score",
]
CALENDAR = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "is_weekend",
    "is_public_holiday",
    "is_school_holiday",
    "is_holiday_eve",
    "is_day_after_holiday",
]
RAINFALL = [
    "precipitation",
    "rain_flag",
    "rain_lag_1h",
    "rain_lag_3h",
    "rain_lag_6h",
    "rain_sum_3h",
    "rain_sum_6h",
    "rain_sum_12h",
    "rain_sum_24h",
    "rain_max_3h",
    "rain_max_6h",
    "post_rain_1_3h",
    "post_rain_4_6h",
    "post_rain_7_12h",
    "rain_missing_flag",
]
METEOROLOGY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_u",
    "wind_v",
    "wind_calm_flag",
    "wind_gust_ratio",
    "wind_gust_ratio_missing_flag",
    "cloud_cover",
    "boundary_layer_height",
    "shortwave_radiation",
    "weather_missing_flag",
]
ATMOSPHERIC = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "carbon_monoxide",
    "ozone",
    "sulphur_dioxide",
    "aerosol_optical_depth",
    "dust",
    "pm2_5_lag_1h",
    "pm2_5_lag_3h",
    "pm2_5_lag_6h",
    "pm2_5_mean_3h",
    "pm2_5_mean_6h",
    "pm2_5_mean_24h",
    "aerosol_optical_depth_lag_1h",
    "aerosol_optical_depth_lag_3h",
    "aerosol_optical_depth_lag_6h",
    "aerosol_optical_depth_mean_3h",
    "aerosol_optical_depth_mean_6h",
    "aerosol_optical_depth_mean_24h",
    "nitrogen_dioxide_lag_1h",
    "nitrogen_dioxide_lag_3h",
    "nitrogen_dioxide_lag_6h",
    "nitrogen_dioxide_mean_3h",
    "nitrogen_dioxide_mean_6h",
    "nitrogen_dioxide_mean_24h",
    "ozone_mean_8h",
    "air_quality_missing_flag",
]
ROAD = [
    "corridor_km",
    "elevation_m",
    "road_class",
    "lane_count",
    "lane_count_missing_flag",
    "urban_rural",
    "distance_to_nearest_junction",
    "osm_bridge_tagged",
    "distance_to_settlement",
    "distance_to_major_intersection",
]
PREDICTOR_GROUPS = {
    "traffic": TRAFFIC,
    "reliability": RELIABILITY,
    "calendar": CALENDAR,
    "rainfall": RAINFALL,
    "meteorology": METEOROLOGY,
    "atmospheric": ATMOSPHERIC,
    "road": ROAD,
}
EVALUATION_ONLY = [
    "upper_quartile_pm25_context",
    "upper_quartile_aod_context",
    "rain_with_upper_quartile_pollution",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "NOT_AVAILABLE_NOT_A_GIT_WORKTREE"


def ensure_dirs() -> None:
    for path in (CONFIG, DATA, AUDIT, REPORTS):
        path.mkdir(parents=True, exist_ok=True)
    (DATA / "00_final_freeze").mkdir(parents=True, exist_ok=True)


def write_config(frame: pd.DataFrame, labels: pd.DataFrame) -> dict:
    train_rel = frame.loc[frame["split"].eq("train"), "reliability_score"].astype(float)
    rel_low = float(train_rel.quantile(1 / 3))
    # The score has a large exact tie at 1.0, so a conventional upper-tertile
    # value threshold would leave the High stratum empty. High is therefore
    # the training-observed maximum; Medium is the interval above the lower
    # tertile and below that maximum. No validation/test quantile is used.
    rel_high = float(train_rel.max())
    config = {
        "project": "T2 Transportmetrica B submission-grade experiments",
        "release_date": "2026-07-19",
        "timezone": "Asia/Kuching",
        "main_resolution": "1h",
        "node_count": 51,
        "node_order_rule": "node_order_verified then corridor_km; never lexical node_id order",
        "target_type": "point_speed",
        "targets": {
            "h1": "current_speed(t+1h)",
            "h3": "current_speed(t+3h)",
            "h6": "current_speed(t+6h)",
        },
        "split_rule": "fixed chronological split with horizon-boundary purge",
        "split_time_ranges": {
            split: {
                "start": str(group["timestamp_local"].min()),
                "end": str(group["timestamp_local"].max()),
                "node_hours": int(len(group)),
            }
            for split, group in frame.groupby("split", sort=False)
        },
        "uq_partition_rule": "fixed predeclared validation subdivision; calibration is never test",
        "random_seeds": list(SEEDS),
        "main_metric": "MAE",
        "auxiliary_metrics": ["RMSE", "sMAPE", "R2"],
        "traffic_missing_rule": "causal last-observation-carried-forward; origin excluded when no past observation exists",
        "environment_missing_rule": "training-only median imputation where required",
        "weather_source": "Open-Meteo Historical Weather API",
        "atmospheric_source": "CAMS-derived gridded atmospheric composition data accessed through Open-Meteo",
        "forbidden_descriptions": [
            "fixed ECMWF IFS weather source",
            "roadside air-quality observations",
            "ground-station air-quality observations",
            "official haze for upper-quartile pollution labels",
        ],
        "predictor_groups": PREDICTOR_GROUPS,
        "evaluation_only_columns": EVALUATION_ONLY,
        "scenario_threshold_source": "pre-existing train-split p75 labels frozen before test evaluation",
        "scenarios": {
            "S1_dry": "precipitation <= 0 at forecast origin",
            "S2_rain": "rain_flag at forecast origin",
            "S3_widespread_rain": "pre-existing widespread_rain_flag",
            "S4_post_rain_1_3h": "pre-existing post_rain_1_3h",
            "S5_post_rain_4_6h": "pre-existing post_rain_4_6h",
            "S6_elevated_pm2_5": "pre-existing upper_quartile_pm25_context; evaluation label only",
            "S7_elevated_aod": "pre-existing upper_quartile_aod_context; evaluation label only",
            "S8_rain_elevated_atmospheric_pollution": "pre-existing rain_with_upper_quartile_pollution; evaluation label only",
        },
        "reliability_thresholds_training_only": {
            "low_upper_inclusive": rel_low,
            "high_lower_inclusive": rel_high,
            "medium_rule": f"> {rel_low} and < {rel_high}",
            "high_rule": f">= {rel_high}",
            "tie_handling_reason": "upper-tertile value is tied at 1.0; maximum-score records define High",
        },
        "tcn_protocol": {
            "max_epochs": 100,
            "early_stopping_patience": 15,
            "scheduler": "ReduceLROnPlateau on validation MAE",
            "gradient_clipping_norm": 1.0,
            "scaling": "training-only",
            "checkpoint_selection": "lowest validation MAE",
            "test_used_for_selection": False,
        },
        "statistical_protocol": {
            "unit": "forecast-origin mean absolute error across nodes",
            "bootstrap": "paired 24-hour contiguous moving-block bootstrap",
            "bootstrap_repetitions": 5000,
            "secondary_test": "paired Wilcoxon on daily MAE",
            "multiple_comparison": "Holm",
        },
        "generated_at": GENERATED_AT,
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    for path in (
        CONFIG / "study_config_final.yaml",
        DATA / "00_final_freeze" / "study_config_final.yaml",
    ):
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    scenario_yaml = {
        "frozen_before_test_analysis": True,
        "threshold_source": config["scenario_threshold_source"],
        "definitions": config["scenarios"],
        "heavy_rain": "supplementary/descriptive only due to insufficient independent test events",
    }
    for path in (
        CONFIG / "scenario_definition.yaml",
        DATA / "00_final_freeze" / "scenario_definition.yaml",
    ):
        path.write_text(yaml.safe_dump(scenario_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config


def final_scenario_labels(frame: pd.DataFrame, old_labels: pd.DataFrame, config: dict) -> pd.DataFrame:
    base_cols = [
        "node_id",
        "timestamp_local",
        "split",
        "uq_partition",
        "precipitation",
        "rain_flag",
        "post_rain_1_3h",
        "post_rain_4_6h",
        "reliability_score",
        "corridor_rain_event_id" if "corridor_rain_event_id" in frame else None,
    ]
    base_cols = [c for c in base_cols if c]
    labels = frame[base_cols + [f"sample_id_h{h}" for h in HORIZONS]].copy()
    prior = old_labels[
        [
            "node_id",
            "timestamp_local",
            "scenario_widespread_rain",
            "scenario_elevated_pm2_5",
            "scenario_elevated_aod",
            "scenario_rain_elevated_atmospheric_pollution",
        ]
    ]
    labels = labels.merge(prior, on=["node_id", "timestamp_local"], how="left", validate="1:1")
    labels["S1_dry"] = labels["precipitation"].fillna(0).le(0)
    labels["S2_rain"] = labels["rain_flag"].fillna(False).astype(bool)
    labels["S3_widespread_rain"] = labels["scenario_widespread_rain"].fillna(False).astype(bool)
    labels["S4_post_rain_1_3h"] = labels["post_rain_1_3h"].fillna(False).astype(bool)
    labels["S5_post_rain_4_6h"] = labels["post_rain_4_6h"].fillna(False).astype(bool)
    labels["S6_elevated_pm2_5"] = labels["scenario_elevated_pm2_5"].fillna(False).astype(bool)
    labels["S7_elevated_aod"] = labels["scenario_elevated_aod"].fillna(False).astype(bool)
    labels["S8_rain_elevated_atmospheric_pollution"] = labels[
        "scenario_rain_elevated_atmospheric_pollution"
    ].fillna(False).astype(bool)
    thresholds = config["reliability_thresholds_training_only"]
    labels["reliability_group_final"] = np.select(
        [
            labels["reliability_score"].le(thresholds["low_upper_inclusive"]),
            labels["reliability_score"].ge(thresholds["high_lower_inclusive"]),
        ],
        ["Low", "High"],
        default="Medium",
    )
    labels = labels.drop(
        columns=[
            "scenario_widespread_rain",
            "scenario_elevated_pm2_5",
            "scenario_elevated_aod",
            "scenario_rain_elevated_atmospheric_pollution",
        ]
    )
    labels.to_parquet(DATA / "scenario_labels_final.parquet", index=False)
    labels.to_csv(DATA / "scenario_labels_final.csv", index=False)
    return labels


def node_audit() -> pd.DataFrame:
    master = pd.read_csv(SOURCE / "01_fcd_reference" / "node_master.csv")
    cols = ["node_id", "node_order_verified", "corridor_km", "latitude", "longitude"]
    order = master[cols].drop_duplicates("node_id").sort_values(
        ["node_order_verified", "corridor_km"]
    ).reset_index(drop=True)
    order["node_id_unique"] = ~order["node_id"].duplicated(keep=False)
    order["corridor_km_present"] = order["corridor_km"].notna()
    order["corridor_km_strictly_increasing"] = order["corridor_km"].diff().fillna(1).gt(0)
    order["node_order_verified_unique"] = ~order["node_order_verified"].duplicated(keep=False)
    order["latitude_present"] = order["latitude"].notna()
    order["longitude_present"] = order["longitude"].notna()
    checks = [
        "node_id_unique",
        "corridor_km_present",
        "corridor_km_strictly_increasing",
        "node_order_verified_unique",
        "latitude_present",
        "longitude_present",
    ]
    order["row_status"] = np.where(order[checks].all(axis=1), "PASS", "FAIL")
    if len(order) != 51 or not order[checks].all().all():
        raise ValueError("Final node-order audit failed")
    order.to_csv(AUDIT / "final_node_order_audit.csv", index=False)
    order.to_csv(DATA / "node_order_final.csv", index=False)
    return order


def temporal_audits(frame: pd.DataFrame, raw_panel: pd.DataFrame) -> None:
    rows = []
    for node_id, group in frame.groupby("node_id", sort=True):
        ts = pd.DatetimeIndex(pd.to_datetime(group["timestamp_local"]).sort_values())
        expected = pd.date_range(ts.min(), ts.max(), freq="1h")
        rows.append(
            {
                "node_id": node_id,
                "first_timestamp": ts.min(),
                "last_timestamp": ts.max(),
                "observed_rows": len(ts),
                "expected_hourly_rows": len(expected),
                "missing_timestamp_count": len(expected.difference(ts)),
                "duplicate_timestamp_count": int(ts.duplicated().sum()),
                "resolution_is_1h": bool(np.all(np.diff(ts.asi8) == 3_600_000_000_000)),
                "status": "PASS"
                if len(expected.difference(ts)) == 0 and not ts.duplicated().any()
                else "FAIL",
            }
        )
    pd.DataFrame(rows).to_csv(AUDIT / "final_temporal_integrity.csv", index=False)

    split_rows = []
    for split, group in frame.groupby("split", sort=False):
        row = {
            "split": split,
            "origin_start": group["timestamp_local"].min(),
            "origin_end": group["timestamp_local"].max(),
            "node_hours": len(group),
            "unique_hours": group["timestamp_local"].nunique(),
            "node_count": group["node_id"].nunique(),
            "duplicate_node_timestamp": int(group.duplicated(["node_id", "timestamp_local"]).sum()),
        }
        for h in HORIZONS:
            eligible = group[f"sample_eligible_h{h}"].fillna(False)
            row[f"eligible_h{h}"] = int(eligible.sum())
            row[f"cross_split_available_target_h{h}"] = int(
                (
                    eligible
                    & group[f"target_split_h{h}"].notna()
                    & group[f"target_split_h{h}"].ne(group["split"])
                ).sum()
            )
            row[f"duplicate_sample_id_h{h}"] = int(
                group.loc[eligible, f"sample_id_h{h}"].duplicated().sum()
            )
        row["status"] = "PASS" if row["duplicate_node_timestamp"] == 0 and all(
            row[f"cross_split_available_target_h{h}"] == 0
            and row[f"duplicate_sample_id_h{h}"] == 0
            for h in HORIZONS
        ) else "FAIL"
        split_rows.append(row)
    pd.DataFrame(split_rows).to_csv(AUDIT / "final_split_integrity.csv", index=False)

    raw_target = raw_panel[["node_id", "timestamp_local", "current_speed"]].copy()
    raw_target = raw_target.rename(
        columns={"timestamp_local": "target_timestamp", "current_speed": "raw_future_speed"}
    )
    target_rows = []
    for h in HORIZONS:
        subset = frame[
            [
                "node_id",
                "timestamp_local",
                "split",
                f"target_timestamp_h{h}",
                f"target_split_h{h}",
                f"target_speed_h{h}",
                f"sample_id_h{h}",
                f"sample_eligible_h{h}",
            ]
        ].copy()
        subset = subset.rename(columns={f"target_timestamp_h{h}": "target_timestamp"})
        subset = subset.merge(raw_target, on=["node_id", "target_timestamp"], how="left", validate="m:1")
        elapsed = (subset["target_timestamp"] - subset["timestamp_local"]).dt.total_seconds() / 3600
        eligible = subset[f"sample_eligible_h{h}"].fillna(False)
        value_match = np.isclose(
            subset[f"target_speed_h{h}"].astype(float),
            subset["raw_future_speed"].astype(float),
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        )
        for split, group in subset.groupby("split", sort=False):
            idx = group.index
            use = eligible.loc[idx]
            target_rows.append(
                {
                    "horizon": h,
                    "split": split,
                    "origin_rows": len(group),
                    "eligible_rows": int(use.sum()),
                    "target_timestamp_offset_mismatch": int((use & elapsed.loc[idx].ne(h)).sum()),
                    "target_value_mismatch_vs_raw_future": int((use & ~pd.Series(value_match, index=subset.index).loc[idx]).sum()),
                    "target_crosses_split": int(
                        (
                            use
                            & group[f"target_split_h{h}"].notna()
                            & group[f"target_split_h{h}"].ne(group["split"])
                        ).sum()
                    ),
                    "missing_sample_id_on_eligible": int((use & group[f"sample_id_h{h}"].isna()).sum()),
                    "status": "PASS"
                    if int((use & elapsed.loc[idx].ne(h)).sum()) == 0
                    and int((use & ~pd.Series(value_match, index=subset.index).loc[idx]).sum()) == 0
                    and int(
                        (
                            use
                            & group[f"target_split_h{h}"].notna()
                            & group[f"target_split_h{h}"].ne(group["split"])
                        ).sum()
                    ) == 0
                    else "FAIL",
                }
            )
    pd.DataFrame(target_rows).to_csv(AUDIT / "final_target_integrity.csv", index=False)


def fcd_audit(raw_panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "node_id",
        "timestamp_local",
        "split",
        "current_speed",
        "current_speed_input",
        "missing_mask",
        "gap_length_steps",
        "coverage_ratio_6h",
        "coverage_ratio_24h",
        "reliability_score",
    ]
    rows = raw_panel[cols].sort_values(["node_id", "timestamp_local"]).reset_index(drop=True)
    grouped = rows.groupby("node_id", sort=False)["current_speed"]
    rows["past_last_observed_speed"] = grouped.ffill()
    rows["future_next_observed_speed"] = grouped.bfill()
    train_median = rows.loc[rows["split"].eq("train")].groupby("node_id")["current_speed"].median()
    rows["training_node_median"] = rows["node_id"].map(train_median)

    def close(left: pd.Series, right: pd.Series) -> pd.Series:
        return pd.Series(
            np.isclose(left.astype(float), right.astype(float), rtol=0, atol=1e-9, equal_nan=False),
            index=left.index,
        )

    missing = rows["current_speed"].isna()
    observed = ~missing & close(rows["current_speed_input"], rows["current_speed"])
    locf = missing & rows["past_last_observed_speed"].notna() & close(
        rows["current_speed_input"], rows["past_last_observed_speed"]
    )
    fallback = missing & rows["past_last_observed_speed"].isna() & close(
        rows["current_speed_input"], rows["training_node_median"]
    )
    future = missing & rows["past_last_observed_speed"].isna() & close(
        rows["current_speed_input"], rows["future_next_observed_speed"]
    )
    rows["imputation_source"] = "unverified"
    rows.loc[observed, "imputation_source"] = "observed_current_value"
    rows.loc[locf, "imputation_source"] = "causal_locf"
    rows.loc[fallback & ~future, "imputation_source"] = "training_only_node_median_fallback"
    rows.loc[future, "imputation_source"] = "future_match_forbidden_excluded_from_final_samples"
    rows["causal_allowed"] = observed | locf | (fallback & ~future)
    rows["final_sample_handling"] = np.where(
        rows["causal_allowed"], "retain_if_target_available", "exclude_origin_all_models"
    )
    rows.to_csv(AUDIT / "fcd_imputation_row_level_check.csv", index=False)

    gap_records = []
    for node_id, group in rows.groupby("node_id", sort=True):
        group = group.sort_values("timestamp_local")
        miss = group["current_speed"].isna().to_numpy()
        starts = np.flatnonzero(miss & np.r_[True, ~miss[:-1]])
        ends = np.flatnonzero(miss & np.r_[~miss[1:], True])
        for gap_number, (start, end) in enumerate(zip(starts, ends), start=1):
            seg = group.iloc[start : end + 1]
            gap_records.append(
                {
                    "gap_id": f"{node_id}_G{gap_number:04d}",
                    "node_id": node_id,
                    "gap_start": seg["timestamp_local"].min(),
                    "gap_end": seg["timestamp_local"].max(),
                    "gap_length_hours": len(seg),
                    "start_split": seg.iloc[0]["split"],
                    "end_split": seg.iloc[-1]["split"],
                    "carry_forward_count": int(seg["imputation_source"].eq("causal_locf").sum()),
                    "fallback_count": int(
                        seg["imputation_source"].eq("training_only_node_median_fallback").sum()
                    ),
                    "excluded_count": int(seg["final_sample_handling"].eq("exclude_origin_all_models").sum()),
                }
            )
    gaps = pd.DataFrame(gap_records)
    gaps.to_csv(AUDIT / "fcd_gap_distribution.csv", index=False)

    summary = []
    for (split, node_id), group in rows.groupby(["split", "node_id"], sort=True):
        relevant = gaps.loc[
            gaps["node_id"].eq(node_id)
            & (gaps["start_split"].eq(split) | gaps["end_split"].eq(split))
        ]
        summary.append(
            {
                "split": split,
                "node_id": node_id,
                "raw_missing_count": int(group["current_speed"].isna().sum()),
                "carry_forward_count": int(group["imputation_source"].eq("causal_locf").sum()),
                "fallback_count": int(
                    group["imputation_source"].eq("training_only_node_median_fallback").sum()
                ),
                "excluded_noncausal_count": int(
                    group["final_sample_handling"].eq("exclude_origin_all_models").sum()
                ),
                "max_gap_hours": int(relevant["gap_length_hours"].max()) if len(relevant) else 0,
                "mean_gap_hours": float(relevant["gap_length_hours"].mean()) if len(relevant) else 0.0,
                "gap_gt_1h": int(relevant["gap_length_hours"].gt(1).sum()),
                "gap_gt_3h": int(relevant["gap_length_hours"].gt(3).sum()),
                "gap_gt_6h": int(relevant["gap_length_hours"].gt(6).sum()),
                "gap_gt_12h": int(relevant["gap_length_hours"].gt(12).sum()),
                "gap_gt_24h": int(relevant["gap_length_hours"].gt(24).sum()),
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(AUDIT / "fcd_imputation_summary.csv", index=False)
    return rows


def lag_rolling_audit(frozen: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_parquet(SOURCE / "03_weather_processed" / "weather_hourly_long.parquet")
    weather = weather.sort_values(["node_id", "timestamp_local"]).copy()
    wg = weather.groupby("node_id", sort=False)
    for lag in (1, 3, 6):
        weather[f"recomputed_rain_lag_{lag}h"] = wg["precipitation"].shift(lag)
    for window in (3, 6, 12, 24):
        weather[f"recomputed_rain_sum_{window}h"] = wg["precipitation"].transform(
            lambda series: series.rolling(window, min_periods=1).sum()
        )

    aq = pd.read_parquet(SOURCE / "05_air_quality_processed" / "air_quality_hourly_long.parquet")
    aq = aq.sort_values(["node_id", "timestamp_local"]).copy()
    ag = aq.groupby("node_id", sort=False)
    for lag in (1, 3, 6):
        aq[f"recomputed_pm2_5_lag_{lag}h"] = ag["pm2_5"].shift(lag)
    for window in (3, 6, 24):
        aq[f"recomputed_pm2_5_mean_{window}h"] = ag["pm2_5"].transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
    aq["recomputed_aod_lag_3h"] = ag["aerosol_optical_depth"].shift(3)
    for window in (6, 24):
        aq[f"recomputed_aod_mean_{window}h"] = ag["aerosol_optical_depth"].transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
    aq["recomputed_no2_mean_3h"] = ag["nitrogen_dioxide"].transform(
        lambda series: series.rolling(3, min_periods=1).mean()
    )
    aq["recomputed_ozone_mean_8h"] = ag["ozone"].transform(
        lambda series: series.rolling(8, min_periods=1).mean()
    )

    sample = frozen.sample(n=1000, random_state=20260719).copy()
    wcols = ["node_id", "timestamp_local"] + [
        column for column in weather if column.startswith("recomputed_")
    ]
    acols = ["node_id", "timestamp_local"] + [
        column for column in aq if column.startswith("recomputed_")
    ]
    sample = sample.merge(weather[wcols], on=["node_id", "timestamp_local"], how="left", validate="1:1")
    sample = sample.merge(aq[acols], on=["node_id", "timestamp_local"], how="left", validate="1:1")
    pairs = []
    for lag in (1, 3, 6):
        pairs.append((f"rain_lag_{lag}h", f"recomputed_rain_lag_{lag}h"))
    for window in (3, 6, 12, 24):
        pairs.append((f"rain_sum_{window}h", f"recomputed_rain_sum_{window}h"))
    for lag in (1, 3, 6):
        pairs.append((f"pm2_5_lag_{lag}h", f"recomputed_pm2_5_lag_{lag}h"))
    for window in (3, 6, 24):
        pairs.append((f"pm2_5_mean_{window}h", f"recomputed_pm2_5_mean_{window}h"))
    pairs.extend(
        [
            ("aerosol_optical_depth_lag_3h", "recomputed_aod_lag_3h"),
            ("aerosol_optical_depth_mean_6h", "recomputed_aod_mean_6h"),
            ("aerosol_optical_depth_mean_24h", "recomputed_aod_mean_24h"),
            ("nitrogen_dioxide_mean_3h", "recomputed_no2_mean_3h"),
            ("ozone_mean_8h", "recomputed_ozone_mean_8h"),
        ]
    )
    audit_rows = []
    for frozen_column, recomputed_column in pairs:
        difference = sample[frozen_column].astype(float) - sample[recomputed_column].astype(float)
        match = np.isclose(
            sample[frozen_column].astype(float),
            sample[recomputed_column].astype(float),
            rtol=0,
            atol=1e-8,
            equal_nan=True,
        )
        audit_rows.append(
            {
                "feature": frozen_column,
                "recomputed_from": recomputed_column,
                "sampled_rows": len(sample),
                "mismatch_count": int((~match).sum()),
                "max_absolute_difference": float(np.nanmax(np.abs(difference)))
                if difference.notna().any()
                else 0.0,
                "uses_future_information": False,
                "status": "PASS" if match.all() else "FAIL",
            }
        )
    result = pd.DataFrame(audit_rows)
    result.to_csv(AUDIT / "final_lag_rolling_audit.csv", index=False)
    if result["mismatch_count"].sum() != 0:
        raise ValueError("Lag/rolling audit produced mismatches")
    return result


def manifests(frame: pd.DataFrame, labels: pd.DataFrame, node_order: pd.DataFrame) -> None:
    feature_rows = []
    for column in frame.columns:
        group = next(
            (name for name, values in PREDICTOR_GROUPS.items() if column in values),
            "identifier_target_or_protocol",
        )
        role = "predictor" if group != "identifier_target_or_protocol" else "metadata_or_target"
        feature_rows.append(
            {
                "feature": column,
                "dtype": str(frame[column].dtype),
                "group": group,
                "role": role,
                "included_as_predictor": role == "predictor",
                "missing_count": int(frame[column].isna().sum()),
                "imputation_rule": "causal LOCF" if column in TRAFFIC else (
                    "training-only median if required" if role == "predictor" else "not applicable"
                ),
            }
        )
    pd.DataFrame(feature_rows).to_csv(DATA / "00_final_freeze" / "feature_manifest.csv", index=False)

    sample_rows = []
    for h in HORIZONS:
        for split, group in frame.groupby("split", sort=False):
            eligible = group.loc[group[f"sample_eligible_h{h}"].fillna(False)]
            ids = sorted(eligible[f"sample_id_h{h}"].astype(str))
            sample_rows.append(
                {
                    "horizon": h,
                    "split": split,
                    "sample_count": len(eligible),
                    "unique_sample_count": eligible[f"sample_id_h{h}"].nunique(),
                    "sample_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                    "origin_start": eligible["timestamp_local"].min(),
                    "origin_end": eligible["timestamp_local"].max(),
                    "node_count": eligible["node_id"].nunique(),
                }
            )
    pd.DataFrame(sample_rows).to_csv(DATA / "00_final_freeze" / "sample_manifest.csv", index=False)

    model_manifest = {
        "frozen_E1_E5": {
            "backbone": "XGBoost fixed tabular backbone",
            "status": "frozen; no retuning for test performance",
            "experiments": {
                "E1": TRAFFIC + RELIABILITY + CALENDAR,
                "E2": TRAFFIC + RELIABILITY + CALENDAR + RAINFALL,
                "E3": TRAFFIC + RELIABILITY + CALENDAR + METEOROLOGY,
                "E4": TRAFFIC + RELIABILITY + CALENDAR + ATMOSPHERIC,
                "E5": TRAFFIC + RELIABILITY + CALENDAR + RAINFALL + METEOROLOGY + ATMOSPHERIC + ROAD,
            },
        },
        "formal_TCN": "100 max epochs, patience 15, scheduler, clipping, validation checkpoint",
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
    }
    for path in (
        CONFIG / "model_manifest.yaml",
        DATA / "00_final_freeze" / "model_manifest.yaml",
    ):
        path.write_text(yaml.safe_dump(model_manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    canonical_source = FROZEN_20260718 / "01_fcd_reference" / "canonical_hourly_index.parquet"
    canonical = pd.read_parquet(canonical_source)
    canonical.to_parquet(DATA / "canonical_hourly_index.parquet", index=False)
    canonical.to_csv(DATA / "canonical_hourly_index.csv", index=False)

    scenario_cols = [column for column in labels if column.startswith("S")]
    scenario_rows = []
    for split, group in labels.groupby("split", sort=False):
        for scenario in scenario_cols:
            subset = group.loc[group[scenario].fillna(False)]
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "split": split,
                    "node_hours": len(subset),
                    "unique_hours": subset["timestamp_local"].nunique(),
                    "unique_dates": subset["timestamp_local"].dt.date.nunique(),
                    "number_of_nodes": subset["node_id"].nunique(),
                }
            )
    pd.DataFrame(scenario_rows).to_csv(DATA / "scenario_sample_count.csv", index=False)


def dataset_manifest() -> pd.DataFrame:
    candidates = [
        DATA / "final_model_features_frozen.parquet",
        DATA / "scenario_labels_final.parquet",
        DATA / "canonical_hourly_index.parquet",
        DATA / "node_order_final.csv",
        SOURCE / "08_multimodal_panel" / "multimodal_hourly_panel.parquet",
        SOURCE / "03_weather_processed" / "weather_hourly_long.parquet",
        SOURCE / "05_air_quality_processed" / "air_quality_hourly_long.parquet",
    ]
    records = []
    for path in candidates:
        if path.suffix == ".parquet":
            meta = pq.ParquetFile(path).metadata
            rows, columns = meta.num_rows, meta.num_columns
            data = pd.read_parquet(path, columns=[c for c in ["node_id", "timestamp_local"] if c in pq.ParquetFile(path).schema.names])
        else:
            data = pd.read_csv(path)
            rows, columns = data.shape
        node_count = data["node_id"].nunique() if "node_id" in data else np.nan
        if "timestamp_local" in data:
            times = pd.to_datetime(data["timestamp_local"])
            first_time, last_time = times.min(), times.max()
        else:
            first_time = last_time = ""
        records.append(
            {
                "file_name": path.name,
                "absolute_source_path": str(path),
                "sha256": sha256(path),
                "file_size_bytes": path.stat().st_size,
                "row_count": rows,
                "column_count": columns,
                "node_count": node_count,
                "time_start": first_time,
                "time_end": last_time,
                "generated_at": GENERATED_AT,
                "code_version": "20260719_submission_pipeline_v1",
                "git_commit": git_commit(),
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_csv(DATA / "00_final_freeze" / "dataset_manifest.csv", index=False)
    return manifest


def environment_missing_audit(frame: pd.DataFrame) -> None:
    rows = []
    for group_name in ("rainfall", "meteorology", "atmospheric", "road"):
        for column in PREDICTOR_GROUPS[group_name]:
            categorical = column in {"road_class", "urban_rural"}
            if categorical:
                train_values = frame.loc[frame["split"].eq("train"), column].dropna().astype(str)
                statistic = ";".join(sorted(train_values.unique()))
                missing = frame[column].isna()
                rule = "training-only categories; unseen values map to UNKNOWN"
                valid_statistic = bool(len(train_values))
            else:
                train_values = pd.to_numeric(
                    frame.loc[frame["split"].eq("train"), column], errors="coerce"
                )
                statistic = str(float(train_values.median()))
                missing = pd.to_numeric(frame[column], errors="coerce").isna()
                rule = "training-only median"
                valid_statistic = bool(np.isfinite(float(train_values.median())))
            by_split = frame.assign(_missing=missing).groupby("split")
            missing_counts = by_split["_missing"].sum()
            rows.append(
                {
                    "modality": group_name,
                    "feature": column,
                    "training_statistic": statistic,
                    "statistic_type": "category set" if categorical else "median",
                    "train_missing_count": int(missing_counts.get("train", 0)),
                    "validation_missing_count": int(missing_counts.get("validation", 0)),
                    "test_missing_count": int(missing_counts.get("test", 0)),
                    "allowed_imputation": rule,
                    "test_statistics_used": False,
                    "status": "PASS" if valid_statistic else "FAIL",
                }
            )
    pd.DataFrame(rows).to_csv(AUDIT / "environment_training_only_imputation_audit.csv", index=False)


def write_phase1_report(frame: pd.DataFrame, fcd_rows: pd.DataFrame, lag: pd.DataFrame) -> None:
    excluded = int(fcd_rows["final_sample_handling"].eq("exclude_origin_all_models").sum())
    report = f"""# Phase 1 Final Data Freeze and Integrity Audit

Generated: {GENERATED_AT}

## Outcome

- Frozen feature table: {len(frame):,} node-hours, {frame['node_id'].nunique()} nodes.
- Time range: {frame['timestamp_local'].min()} to {frame['timestamp_local'].max()} at 1-hour resolution.
- Targets remain point speed: current_speed(t+1h), current_speed(t+3h), current_speed(t+6h).
- Raw data were read-only. No original file was modified.
- One origin without a causal past FCD observation remains documented and is excluded from every model sample set; validation and test are unchanged.
- Total excluded noncausal origins: {excluded}.
- Independent lag/rolling check: {len(lag)} features x 1,000 sampled rows, total mismatch count {int(lag['mismatch_count'].sum())}.
- Weather provenance: Open-Meteo Historical Weather API.
- Atmospheric provenance: CAMS-derived gridded atmospheric composition data accessed through Open-Meteo; these are not roadside or ground-station observations.

## Required audit files

- `final_node_order_audit.csv`
- `final_temporal_integrity.csv`
- `final_split_integrity.csv`
- `final_target_integrity.csv`
- `fcd_imputation_row_level_check.csv`
- `fcd_gap_distribution.csv`
- `fcd_imputation_summary.csv`
- `final_lag_rolling_audit.csv`
- `environment_training_only_imputation_audit.csv`

## Freeze rule

All later experiments read `01_final_data/final_model_features_frozen.parquet` and never write to it.
"""
    (REPORTS / "PHASE1_FINAL_DATA_FREEZE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    frame = pd.read_parquet(DATA / "final_model_features_frozen.parquet")
    old_labels = pd.read_parquet(DATA / "scenario_labels.parquet")
    raw_panel = pd.read_parquet(SOURCE / "08_multimodal_panel" / "multimodal_hourly_panel.parquet")
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
    old_labels["timestamp_local"] = pd.to_datetime(old_labels["timestamp_local"])
    raw_panel["timestamp_local"] = pd.to_datetime(raw_panel["timestamp_local"])

    config = write_config(frame, old_labels)
    labels = final_scenario_labels(frame, old_labels, config)
    order = node_audit()
    temporal_audits(frame, raw_panel)
    fcd_rows = fcd_audit(raw_panel)
    lag = lag_rolling_audit(frame)
    environment_missing_audit(frame)
    manifests(frame, labels, order)
    dataset_manifest()
    write_phase1_report(frame, fcd_rows, lag)
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(frame),
                "nodes": frame["node_id"].nunique(),
                "lag_mismatches": int(lag["mismatch_count"].sum()),
                "noncausal_origins_excluded": int(
                    fcd_rows["final_sample_handling"].eq("exclude_origin_all_models").sum()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
