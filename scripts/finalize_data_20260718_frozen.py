from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


OUTPUT = Path(__file__).resolve().parents[1]
WORKSPACE = OUTPUT.parent
SOURCE = WORKSPACE / "20260715" / "T2_final_V2_complete"
SOURCE_ZIP = WORKSPACE / "20260715.zip"

AUDIT_DIR = OUTPUT / "09_audit_reports"
REFERENCE_DIR = OUTPUT / "01_fcd_reference"
FEATURE_DIR = OUTPUT / "10_feature_data"
READY_DIR = OUTPUT / "11_experiment_ready"
CONFIG_DIR = OUTPUT / "00_config"

HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)
SCENARIO_COLUMNS = {
    "dry": "scenario_dry",
    "rain": "scenario_rain",
    "widespread_rain": "scenario_widespread_rain",
    "post_rain_1_3h": "scenario_post_rain_1_3h",
    "elevated_pm2_5": "scenario_elevated_pm2_5",
    "elevated_aod": "scenario_elevated_aod",
    "rain_elevated_atmospheric_pollution": "scenario_rain_elevated_atmospheric_pollution",
}

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
    "atmospheric_context": ATMOSPHERIC,
    "road_context": ROAD,
}

EXCLUDED_SCENARIO_PREDICTORS = [
    "upper_quartile_pm25_context",
    "upper_quartile_aod_context",
    "rain_with_upper_quartile_pollution",
    "any_grid_rain_flag",
    "localized_rain_flag",
    "widespread_rain_flag",
    "localized_high_intensity_flag",
    "corridor_average_rain_flag",
]


def ensure_directories() -> None:
    for path in (AUDIT_DIR, REFERENCE_DIR, FEATURE_DIR, READY_DIR, CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(
        np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, equal_nan=False),
        index=a.index,
    )


def identify_gap_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for node_id, group in frame.groupby("node_id", sort=False):
        group = group.sort_values("timestamp_local")
        missing = group["current_speed"].isna().to_numpy()
        positions = group.index.to_numpy()
        i = 0
        gap_number = 0
        while i < len(group):
            if not missing[i]:
                i += 1
                continue
            gap_number += 1
            j = i
            while j + 1 < len(group) and missing[j + 1]:
                j += 1
            idx = positions[i : j + 1]
            segment = frame.loc[idx]
            methods = sorted(segment["imputation_class"].unique())
            rows.append(
                {
                    "gap_id": f"{node_id}_G{gap_number:04d}",
                    "node_id": node_id,
                    "gap_start": segment["timestamp_local"].min(),
                    "gap_end": segment["timestamp_local"].max(),
                    "gap_length_hours": int(len(segment)),
                    "start_split": str(segment.iloc[0]["split"]),
                    "end_split": str(segment.iloc[-1]["split"]),
                    "cross_split_gap": bool(segment["split"].nunique() > 1),
                    "imputation_classes": ";".join(methods),
                    "causal_row_count": int(segment["imputation_is_causal"].sum()),
                    "noncausal_or_unavailable_row_count": int((~segment["imputation_is_causal"]).sum()),
                    "input_matches_next_valid_count": int(segment["input_matches_future_next_valid"].sum()),
                }
            )
            i = j + 1
    return pd.DataFrame(rows)


def build_fcd_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    audit_rows = panel[cols].sort_values(["node_id", "timestamp_local"]).reset_index(drop=True)
    by_node = audit_rows.groupby("node_id", sort=False)["current_speed"]
    audit_rows["past_last_observed_speed"] = by_node.ffill()
    audit_rows["future_next_observed_speed"] = by_node.bfill()

    observed_match = audit_rows["current_speed"].notna() & close(
        audit_rows["current_speed_input"], audit_rows["current_speed"]
    )
    missing = audit_rows["current_speed"].isna()
    locf_match = missing & audit_rows["past_last_observed_speed"].notna() & close(
        audit_rows["current_speed_input"], audit_rows["past_last_observed_speed"]
    )
    future_match = (
        missing
        & audit_rows["past_last_observed_speed"].isna()
        & audit_rows["future_next_observed_speed"].notna()
        & close(audit_rows["current_speed_input"], audit_rows["future_next_observed_speed"])
    )

    train_medians = (
        audit_rows.loc[audit_rows["split"].eq("train")]
        .groupby("node_id")["current_speed"]
        .median()
    )
    audit_rows["training_node_median"] = audit_rows["node_id"].map(train_medians)
    median_match = (
        missing
        & ~future_match
        & audit_rows["past_last_observed_speed"].isna()
        & close(audit_rows["current_speed_input"], audit_rows["training_node_median"])
    )

    audit_rows["imputation_class"] = "unverified_or_mismatch"
    audit_rows.loc[observed_match, "imputation_class"] = "observed_current_value"
    audit_rows.loc[locf_match, "imputation_class"] = "causal_locf"
    audit_rows.loc[median_match, "imputation_class"] = "training_node_median_fallback"
    audit_rows.loc[future_match, "imputation_class"] = "initial_no_past_matches_future_next_value"
    audit_rows["imputation_is_causal"] = observed_match | locf_match
    audit_rows["input_matches_future_next_valid"] = future_match
    audit_rows["causal_input_available"] = audit_rows["imputation_is_causal"]

    gap_runs = identify_gap_runs(audit_rows)
    gap_runs.to_csv(AUDIT_DIR / "fcd_gap_distribution.csv", index=False)

    gap_agg = []
    for (node_id, split), group in audit_rows.groupby(["node_id", "split"], sort=True):
        node_gaps = gap_runs.loc[
            gap_runs["node_id"].eq(node_id)
            & (gap_runs["start_split"].eq(split) | gap_runs["end_split"].eq(split))
        ]
        gap_agg.append(
            {
                "split": split,
                "node_id": node_id,
                "total_rows": int(len(group)),
                "raw_missing_count": int(group["current_speed"].isna().sum()),
                "observed_current_count": int(group["imputation_class"].eq("observed_current_value").sum()),
                "carry_forward_count": int(group["imputation_class"].eq("causal_locf").sum()),
                "median_imputation_count": int(
                    group["imputation_class"].eq("training_node_median_fallback").sum()
                ),
                "future_value_match_count": int(
                    group["imputation_class"].eq("initial_no_past_matches_future_next_value").sum()
                ),
                "unverified_noncausal_count": int(
                    group["imputation_class"].eq("unverified_or_mismatch").sum()
                ),
                "max_gap_hours": int(node_gaps["gap_length_hours"].max()) if len(node_gaps) else 0,
                "mean_gap_hours": float(node_gaps["gap_length_hours"].mean()) if len(node_gaps) else 0.0,
                "gap_gt_1h": int(node_gaps["gap_length_hours"].gt(1).sum()),
                "gap_gt_3h": int(node_gaps["gap_length_hours"].gt(3).sum()),
                "gap_gt_6h": int(node_gaps["gap_length_hours"].gt(6).sum()),
                "gap_gt_12h": int(node_gaps["gap_length_hours"].gt(12).sum()),
                "causal_audit_pass_original": bool(group["imputation_is_causal"].all()),
                "final_handling": (
                    "retain"
                    if group["imputation_is_causal"].all()
                    else "exclude_noncausal_origin_from_all_model_sample_sets"
                ),
            }
        )
    node_split_audit = pd.DataFrame(gap_agg)
    node_split_audit.to_csv(AUDIT_DIR / "fcd_input_imputation_audit.csv", index=False)

    summary_rows: list[dict] = []
    for label, group in [("ALL", audit_rows)]:
        summary_rows.append(_imputation_summary_row(label, group, gap_runs))
    for split, group in audit_rows.groupby("split", sort=False):
        split_gaps = gap_runs.loc[
            gap_runs["start_split"].eq(split) | gap_runs["end_split"].eq(split)
        ]
        summary_rows.append(_imputation_summary_row(str(split), group, split_gaps))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(AUDIT_DIR / "fcd_imputation_summary.csv", index=False)
    audit_rows.to_parquet(AUDIT_DIR / "fcd_input_imputation_row_audit.parquet", index=False)
    return audit_rows, gap_runs, summary


def _imputation_summary_row(label: str, rows: pd.DataFrame, gaps: pd.DataFrame) -> dict:
    class_counts = rows["imputation_class"].value_counts()
    return {
        "scope": label,
        "total_rows": int(len(rows)),
        "raw_missing_count": int(rows["current_speed"].isna().sum()),
        "observed_current_count": int(class_counts.get("observed_current_value", 0)),
        "carry_forward_count": int(class_counts.get("causal_locf", 0)),
        "median_imputation_count": int(class_counts.get("training_node_median_fallback", 0)),
        "future_value_match_count": int(
            class_counts.get("initial_no_past_matches_future_next_value", 0)
        ),
        "unverified_noncausal_count": int(class_counts.get("unverified_or_mismatch", 0)),
        "causal_rows": int(rows["imputation_is_causal"].sum()),
        "noncausal_or_unavailable_rows": int((~rows["imputation_is_causal"]).sum()),
        "gap_event_count": int(len(gaps)),
        "max_gap_hours": int(gaps["gap_length_hours"].max()) if len(gaps) else 0,
        "mean_gap_hours": float(gaps["gap_length_hours"].mean()) if len(gaps) else 0.0,
        "original_audit_status": "PASS" if rows["imputation_is_causal"].all() else "FAIL",
        "frozen_release_status": "PASS_AFTER_NONCAUSAL_ORIGIN_EXCLUSION",
    }


def build_node_order_and_adjacency() -> pd.DataFrame:
    master = pd.read_csv(SOURCE / "01_fcd_reference" / "node_master.csv")
    required = ["node_id", "node_order_verified", "corridor_km", "latitude", "longitude"]
    missing = [column for column in required if column not in master.columns]
    if missing:
        raise ValueError(f"Node master is missing {missing}")
    node_order = (
        master[required]
        .drop_duplicates("node_id")
        .sort_values(["node_order_verified", "corridor_km"])
        .reset_index(drop=True)
    )
    node_order["node_order_final"] = np.arange(1, len(node_order) + 1)
    node_order["order_rule"] = "node_order_verified_then_corridor_km"
    node_order["strictly_increasing_corridor_km"] = bool(
        node_order["corridor_km"].diff().dropna().gt(0).all()
    )
    if len(node_order) != 51 or not node_order["strictly_increasing_corridor_km"].all():
        raise ValueError("Verified node order failed the 51-node monotonicity requirement")

    node_order.to_csv(REFERENCE_DIR / "node_order_final.csv", index=False)
    node_order.to_csv(REFERENCE_DIR / "node_order_verified.csv", index=False)
    node_order.to_csv(REFERENCE_DIR / "node_master_final.csv", index=False)
    node_order.to_csv(READY_DIR / "node_order_final.csv", index=False)

    corridor = node_order["corridor_km"].to_numpy(float)
    distance = np.abs(corridor[:, None] - corridor[None, :])
    binary = np.zeros_like(distance, dtype=np.int8)
    for index in range(len(node_order) - 1):
        binary[index, index + 1] = 1
        binary[index + 1, index] = 1
    np.save(READY_DIR / "adjacency_distance.npy", distance)
    np.save(READY_DIR / "adjacency_binary.npy", binary)
    (READY_DIR / "node_order.json").write_text(
        json.dumps(node_order["node_id"].tolist(), indent=2), encoding="utf-8"
    )
    distance_csv = pd.DataFrame(distance, index=node_order["node_id"], columns=node_order["node_id"])
    distance_csv.index.name = "node_id"
    distance_csv.to_csv(READY_DIR / "adjacency_distance.csv")
    binary_csv = pd.DataFrame(binary, index=node_order["node_id"], columns=node_order["node_id"])
    binary_csv.index.name = "node_id"
    binary_csv.to_csv(READY_DIR / "adjacency_binary.csv")

    order_audit = pd.DataFrame(
        [
            {"check": "node_count_is_51", "status": "PASS", "detail": len(node_order)},
            {
                "check": "node_id_unique",
                "status": "PASS" if node_order["node_id"].is_unique else "FAIL",
                "detail": int(node_order["node_id"].nunique()),
            },
            {
                "check": "verified_order_is_1_to_51",
                "status": "PASS"
                if node_order["node_order_verified"].tolist() == list(range(1, 52))
                else "FAIL",
                "detail": "node_order_verified",
            },
            {
                "check": "corridor_km_strictly_increasing",
                "status": "PASS"
                if node_order["corridor_km"].diff().dropna().gt(0).all()
                else "FAIL",
                "detail": "sorted by verified order",
            },
            {
                "check": "binary_adjacency_symmetric",
                "status": "PASS" if np.array_equal(binary, binary.T) else "FAIL",
                "detail": int(binary.sum()),
            },
            {
                "check": "distance_adjacency_symmetric",
                "status": "PASS" if np.allclose(distance, distance.T) else "FAIL",
                "detail": float(distance.max()),
            },
        ]
    )
    order_audit.to_csv(AUDIT_DIR / "node_order_audit.csv", index=False)
    return node_order


def make_scenario_labels(features: pd.DataFrame) -> pd.DataFrame:
    labels = features[
        [
            "node_id",
            "timestamp_local",
            "split",
            "uq_partition",
            "precipitation",
            "rain_flag",
            "widespread_rain_flag",
            "post_rain_1_3h",
            "upper_quartile_pm25_context",
            "upper_quartile_aod_context",
            "rain_with_upper_quartile_pollution",
        ]
        + [f"sample_id_h{h}" for h in HORIZONS]
        + [f"target_available_h{h}" for h in HORIZONS]
    ].copy()
    labels["scenario_dry"] = features["precipitation"].fillna(0).le(0)
    labels["scenario_rain"] = features["rain_flag"].fillna(False).astype(bool)
    labels["scenario_widespread_rain"] = features["widespread_rain_flag"].fillna(False).astype(bool)
    labels["scenario_post_rain_1_3h"] = features["post_rain_1_3h"].fillna(False).astype(bool)
    labels["scenario_elevated_pm2_5"] = (
        features["upper_quartile_pm25_context"].fillna(False).astype(bool)
    )
    labels["scenario_elevated_aod"] = (
        features["upper_quartile_aod_context"].fillna(False).astype(bool)
    )
    labels["scenario_rain_elevated_atmospheric_pollution"] = (
        features["rain_with_upper_quartile_pollution"].fillna(False).astype(bool)
    )
    return labels


def build_frozen_features(
    features: pd.DataFrame, panel_audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility = panel_audit[
        ["node_id", "timestamp_local", "causal_input_available", "imputation_class"]
    ]
    frozen = features.merge(eligibility, on=["node_id", "timestamp_local"], how="left", validate="1:1")
    if frozen["causal_input_available"].isna().any():
        raise ValueError("Causal eligibility did not merge one-to-one")
    invalid = ~frozen["causal_input_available"]
    frozen.loc[invalid, TRAFFIC] = np.nan
    for horizon in HORIZONS:
        frozen[f"sample_eligible_h{horizon}"] = (
            frozen[f"target_available_h{horizon}"].fillna(False).astype(bool)
            & frozen["causal_input_available"].astype(bool)
        )

    labels = make_scenario_labels(features)
    scenario_drop = [column for column in EXCLUDED_SCENARIO_PREDICTORS if column in frozen.columns]
    frozen = frozen.drop(columns=scenario_drop)
    frozen.to_parquet(FEATURE_DIR / "final_model_features_frozen.parquet", index=False)
    labels.to_parquet(FEATURE_DIR / "scenario_labels.parquet", index=False)
    labels.to_csv(FEATURE_DIR / "scenario_labels.csv", index=False)

    manifest_rows: list[dict] = []
    for modality, columns in PREDICTOR_GROUPS.items():
        for column in columns:
            if column not in frozen.columns:
                raise ValueError(f"Required frozen predictor {column} is missing")
            manifest_rows.append(
                {
                    "variable": column,
                    "modality": modality,
                    "included_as_predictor": True,
                    "role": "predictor",
                    "missing_rule": (
                        "causal_locf_and_origin_exclusion_if_no_past"
                        if column in TRAFFIC
                        else "training_only_median_if_required"
                    ),
                    "notes": "",
                }
            )
    for source_column, scenario_name in [
        ("upper_quartile_pm25_context", "elevated_pm2_5"),
        ("upper_quartile_aod_context", "elevated_aod"),
        ("rain_with_upper_quartile_pollution", "rain_elevated_atmospheric_pollution"),
        ("any_grid_rain_flag", "rain_scenario_support"),
        ("localized_rain_flag", "rain_scenario_support"),
        ("widespread_rain_flag", "widespread_rain"),
        ("localized_high_intensity_flag", "supplementary_only"),
        ("corridor_average_rain_flag", "rain_scenario_support"),
    ]:
        manifest_rows.append(
            {
                "variable": source_column,
                "modality": "scenario_label",
                "included_as_predictor": False,
                "role": "scenario_or_evaluation_label",
                "missing_rule": "not_applicable",
                "notes": scenario_name,
            }
        )
    feature_manifest = pd.DataFrame(manifest_rows)
    feature_manifest.to_csv(READY_DIR / "final_model_feature_list.csv", index=False)
    return frozen, labels


def contiguous_event_count(timestamps: pd.Series) -> int:
    values = pd.Series(pd.to_datetime(timestamps).drop_duplicates().sort_values())
    if values.empty:
        return 0
    return int(values.diff().fillna(pd.Timedelta(hours=2)).gt(pd.Timedelta(hours=1)).sum())


def build_canonical_and_counts(frozen: pd.DataFrame, labels: pd.DataFrame) -> None:
    canonical = (
        frozen.groupby("timestamp_local", sort=True)
        .agg(
            split=("split", "first"),
            uq_partition=("uq_partition", "first"),
            node_count=("node_id", "nunique"),
            causal_input_node_count=("causal_input_available", "sum"),
        )
        .reset_index()
    )
    canonical["timestamp_utc"] = (
        canonical["timestamp_local"]
        .dt.tz_localize("Asia/Kuching")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    canonical["date"] = canonical["timestamp_local"].dt.date.astype(str)
    canonical["hour"] = canonical["timestamp_local"].dt.hour
    for horizon in HORIZONS:
        counts = frozen.groupby("timestamp_local")[f"sample_eligible_h{horizon}"].sum()
        canonical[f"eligible_sample_count_h{horizon}"] = canonical["timestamp_local"].map(counts).astype(int)
    canonical.to_parquet(REFERENCE_DIR / "canonical_hourly_index.parquet", index=False)
    canonical.to_csv(REFERENCE_DIR / "canonical_hourly_index.csv", index=False)

    split_rows = []
    for horizon in HORIZONS:
        eligible_column = f"sample_eligible_h{horizon}"
        for split, group in frozen.groupby("split", sort=False):
            eligible = group.loc[group[eligible_column]]
            split_rows.append(
                {
                    "split": split,
                    "horizon_hours": horizon,
                    "origin_start": group["timestamp_local"].min(),
                    "origin_end": group["timestamp_local"].max(),
                    "all_origin_rows": int(len(group)),
                    "eligible_samples": int(len(eligible)),
                    "unique_origin_hours": int(eligible["timestamp_local"].nunique()),
                    "unique_nodes": int(eligible["node_id"].nunique()),
                    "target_start": eligible[f"target_timestamp_h{horizon}"].min(),
                    "target_end": eligible[f"target_timestamp_h{horizon}"].max(),
                }
            )
    pd.DataFrame(split_rows).to_csv(REFERENCE_DIR / "split_summary.csv", index=False)

    counts_rows: list[dict] = []
    label_keys = ["node_id", "timestamp_local"] + list(SCENARIO_COLUMNS.values())
    label_flags = labels[label_keys]
    scenario_frame = frozen.merge(label_flags, on=["node_id", "timestamp_local"], how="left", validate="1:1")
    for horizon in HORIZONS:
        eligible_column = f"sample_eligible_h{horizon}"
        for split, group in scenario_frame.loc[scenario_frame[eligible_column]].groupby("split", sort=False):
            for scenario, flag in SCENARIO_COLUMNS.items():
                subset = group.loc[group[flag].fillna(False)]
                counts_rows.append(
                    {
                        "split": split,
                        "horizon_hours": horizon,
                        "scenario": scenario,
                        "sample_count": int(len(subset)),
                        "unique_hours": int(subset["timestamp_local"].nunique()),
                        "unique_dates": int(subset["timestamp_local"].dt.date.nunique()),
                        "unique_nodes": int(subset["node_id"].nunique()),
                        "independent_time_events": contiguous_event_count(subset["timestamp_local"]),
                    }
                )
    pd.DataFrame(counts_rows).to_csv(READY_DIR / "scenario_sample_count.csv", index=False)


def infer_modality(column: str) -> str:
    for modality, columns in PREDICTOR_GROUPS.items():
        if column in columns:
            return modality
    if column.startswith("target_"):
        return "target"
    if column.startswith("sample_") or column in {
        "node_id",
        "timestamp_local",
        "split",
        "uq_partition",
        "causal_input_available",
        "imputation_class",
    }:
        return "identifier_or_protocol"
    return "derived_or_supporting"


def infer_role(column: str) -> str:
    if any(column in columns for columns in PREDICTOR_GROUPS.values()):
        return "predictor"
    if column.startswith("target_speed"):
        return "target"
    if column.startswith("target_timestamp"):
        return "target_timestamp"
    if column.startswith("sample_id"):
        return "sample_identifier"
    if column.startswith("sample_eligible") or column.startswith("target_available"):
        return "eligibility_flag"
    return "metadata_or_supporting"


def infer_unit(column: str) -> str:
    exact = {
        "current_speed_input": "km/h",
        "free_flow_speed_input": "km/h",
        "temperature_2m": "degC",
        "dew_point_2m": "degC",
        "surface_pressure": "hPa",
        "precipitation": "mm",
        "pm2_5": "ug/m3",
        "pm10": "ug/m3",
        "nitrogen_dioxide": "ug/m3",
        "ozone": "ug/m3",
        "sulphur_dioxide": "ug/m3",
        "carbon_monoxide": "ug/m3",
        "aerosol_optical_depth": "dimensionless",
        "corridor_km": "km",
        "elevation_m": "m",
        "boundary_layer_height": "m",
        "shortwave_radiation": "W/m2",
    }
    if column in exact:
        return exact[column]
    if "timestamp" in column:
        return "Asia/Kuching local time unless named UTC"
    if column.endswith("_flag") or column.startswith("is_") or column.startswith("post_rain"):
        return "boolean"
    if "ratio" in column or "score" in column:
        return "dimensionless"
    if "hours" in column or "_h" in column or "gap_length" in column:
        return "hours or hourly steps"
    return "see source variable definition"


def source_for_modality(modality: str) -> str:
    return {
        "traffic": "Floating-car traffic records",
        "reliability": "Derived causally from floating-car traffic availability",
        "calendar": "Deterministic local calendar features",
        "rainfall": "Open-Meteo Historical Weather API",
        "meteorology": "Open-Meteo Historical Weather API",
        "atmospheric_context": "CAMS-derived gridded atmospheric composition data accessed through Open-Meteo",
        "road_context": "Verified corridor chainage and valid OSM-derived static context",
        "target": "Observed floating-car point speed at t+h",
        "identifier_or_protocol": "Frozen experiment protocol",
    }.get(modality, "Derived from frozen source package")


def build_data_dictionary(frozen: pd.DataFrame, labels: pd.DataFrame) -> None:
    rows = []
    for column in frozen.columns:
        modality = infer_modality(column)
        role = infer_role(column)
        rows.append(
            {
                "variable": column,
                "dtype": str(frozen[column].dtype),
                "modality": modality,
                "role": role,
                "unit": infer_unit(column),
                "missing_count_frozen": int(frozen[column].isna().sum()),
                "imputation_or_availability_rule": (
                    "causal LOCF; exclude origin if no past observation"
                    if column in TRAFFIC
                    else "training-only median if continuous predictor remains missing"
                    if role == "predictor"
                    else "not imputed"
                ),
                "source_description": source_for_modality(modality),
                "description": column.replace("_", " "),
            }
        )
    for scenario, column in SCENARIO_COLUMNS.items():
        rows.append(
            {
                "variable": column,
                "dtype": str(labels[column].dtype),
                "modality": "scenario_label",
                "role": "evaluation_only_not_predictor",
                "unit": "boolean",
                "missing_count_frozen": int(labels[column].isna().sum()),
                "imputation_or_availability_rule": "not imputed",
                "source_description": "Predeclared scenario definition; elevated thresholds derived from training split only",
                "description": scenario.replace("_", " "),
            }
        )
    pd.DataFrame(rows).to_csv(READY_DIR / "data_dictionary.csv", index=False)


def write_config() -> None:
    config = {
        "project_name": "T2_environment_aware_multimodal_forecasting",
        "project_title": "Environment-Aware Multimodal Traffic Speed Forecasting for a Tropical Highway Corridor Using Floating-Car, Weather, and Gridded Atmospheric Data",
        "release_date": "2026-07-18",
        "timezone": "Asia/Kuching",
        "main_resolution": "1h",
        "node_count": 51,
        "target_type": "point_speed",
        "target_definitions": {
            "h1": "current_speed(t+1h)",
            "h3": "current_speed(t+3h)",
            "h6": "current_speed(t+6h)",
        },
        "forecast_horizons_hours": list(HORIZONS),
        "history_windows_hours": [24, 72, 168],
        "split_rule": "fixed_chronological_split_with_horizon_boundary_purge",
        "scaler_rule": "training_only",
        "node_order_rule": "node_order_verified_then_corridor_km",
        "traffic_missing_rule": "causal_last_observation_carried_forward; origin excluded if no past observation",
        "environment_missing_rule": "training_only_median_if_required",
        "random_seeds": list(SEEDS),
        "main_metric": "MAE",
        "auxiliary_metrics": ["RMSE", "sMAPE", "R2"],
        "weather_source": "Open-Meteo Historical Weather API",
        "atmospheric_source": "CAMS-derived gridded atmospheric composition data accessed through Open-Meteo",
        "source_zip_sha256": sha256(SOURCE_ZIP),
        "predictor_groups": PREDICTOR_GROUPS,
        "evaluation_only_columns": EXCLUDED_SCENARIO_PREDICTORS,
        "formal_stage_scope": ["data_finalization", "E0", "E1", "E2", "E3", "E4", "E5"],
    }
    (CONFIG_DIR / "study_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_protocol() -> None:
    text = """# Frozen Imputation, Scaling, and Sample Protocol

1. Source files are read-only and are never overwritten.
2. Traffic input uses the current observed value or node-wise causal last observation carried forward.
3. The sole origin with neither a current nor any prior node observation is marked `causal_input_available=False` and excluded from every E0-E5 model sample set; it is not filled from a later observation.
4. Continuous environmental and road predictors, when missing, use medians fitted only on the horizon-specific training rows and then frozen.
5. Categorical predictors use training categories with an explicit unknown level.
6. Scaling and all learned preprocessing use training rows only.
7. Targets are point speeds at t+1h, t+3h, and t+6h. Targets are never imputed.
8. Cross-split targets and targets outside the frozen timeline are excluded by the existing horizon-specific availability flags.
9. E1-E5 use identical horizon-specific sample IDs in each split and identical test observations for every seed.
10. Scenario labels are evaluation-only and are not included in any predictor matrix.
"""
    (READY_DIR / "final_imputation_and_scaling_protocol.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    panel = pd.read_parquet(SOURCE / "08_multimodal_panel" / "multimodal_hourly_panel.parquet")
    features = pd.read_parquet(SOURCE / "10_feature_data" / "final_model_features_hourly.parquet")
    if len(panel) != len(features):
        raise ValueError("Panel and feature row counts differ")
    if panel[["node_id", "timestamp_local"]].duplicated().any():
        raise ValueError("Panel keys are not unique")
    if features[["node_id", "timestamp_local"]].duplicated().any():
        raise ValueError("Feature keys are not unique")

    panel_audit, _, _ = build_fcd_audit(panel)
    build_node_order_and_adjacency()
    frozen, labels = build_frozen_features(features, panel_audit)
    build_canonical_and_counts(frozen, labels)
    build_data_dictionary(frozen, labels)
    write_config()
    write_protocol()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "rows": len(frozen),
                "causal_unavailable_origins": int((~frozen["causal_input_available"]).sum()),
                "source_zip_sha256": sha256(SOURCE_ZIP),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
