from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
FEATURE_PATH = PACKAGE / "01_final_data" / "final_model_features_frozen.parquet"
E1_E5 = PACKAGE / "04_E1_E5"
ROOT = PACKAGE / "05_E6_ablation"
HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)

TRAFFIC = [
    "current_speed_input", "free_flow_speed_input", "current_travel_time_input", "tti_input",
    "speed_ratio_input", "speed_drop_input", "travel_time_delay_input", "speed_deficit_ratio_input",
]
RELIABILITY = [
    "missing_mask", "gap_length_steps", "coverage_ratio_6h", "coverage_ratio_24h",
    "volatility_tti_3h", "volatility_tti_6h", "confidence_input", "reliability_score",
]
CALENDAR = [
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos", "is_weekend",
    "is_public_holiday", "is_school_holiday", "is_holiday_eve", "is_day_after_holiday",
]
RAINFALL = [
    "precipitation", "rain_flag", "rain_lag_1h", "rain_lag_3h", "rain_lag_6h",
    "rain_sum_3h", "rain_sum_6h", "rain_sum_12h", "rain_sum_24h", "rain_max_3h",
    "rain_max_6h", "post_rain_1_3h", "post_rain_4_6h", "post_rain_7_12h",
    "rain_missing_flag",
]
METEOROLOGY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
    "wind_speed_10m", "wind_gusts_10m", "wind_u", "wind_v", "wind_calm_flag",
    "wind_gust_ratio", "wind_gust_ratio_missing_flag", "cloud_cover",
    "boundary_layer_height", "shortwave_radiation", "weather_missing_flag",
]
ATMOSPHERIC = [
    "pm2_5", "pm10", "nitrogen_dioxide", "carbon_monoxide", "ozone", "sulphur_dioxide",
    "aerosol_optical_depth", "dust", "pm2_5_lag_1h", "pm2_5_lag_3h", "pm2_5_lag_6h",
    "pm2_5_mean_3h", "pm2_5_mean_6h", "pm2_5_mean_24h",
    "aerosol_optical_depth_lag_1h", "aerosol_optical_depth_lag_3h",
    "aerosol_optical_depth_lag_6h", "aerosol_optical_depth_mean_3h",
    "aerosol_optical_depth_mean_6h", "aerosol_optical_depth_mean_24h",
    "nitrogen_dioxide_lag_1h", "nitrogen_dioxide_lag_3h", "nitrogen_dioxide_lag_6h",
    "nitrogen_dioxide_mean_3h", "nitrogen_dioxide_mean_6h", "nitrogen_dioxide_mean_24h",
    "ozone_mean_8h", "air_quality_missing_flag",
]
ROAD = [
    "corridor_km", "elevation_m", "road_class", "lane_count", "lane_count_missing_flag",
    "urban_rural", "distance_to_nearest_junction", "osm_bridge_tagged",
    "distance_to_settlement", "distance_to_major_intersection",
]
CATEGORICAL = {"road_class", "urban_rural"}
BASE = TRAFFIC + RELIABILITY + CALENDAR
FULL = BASE + RAINFALL + METEOROLOGY + ATMOSPHERIC + ROAD

ENV_LAG_ROLL = [
    column for column in RAINFALL + ATMOSPHERIC
    if any(token in column for token in ("_lag_", "_sum_", "_mean_", "_max_", "post_rain"))
]

GROUPS = {
    "modality_ablation": {
        "A0_full_fusion": FULL,
        "A1_without_rainfall": BASE + METEOROLOGY + ATMOSPHERIC + ROAD,
        "A2_without_meteorology": BASE + RAINFALL + ATMOSPHERIC + ROAD,
        "A3_without_atmospheric_context": BASE + RAINFALL + METEOROLOGY + ROAD,
        "A4_without_environmental_lag_rolling": [c for c in FULL if c not in ENV_LAG_ROLL],
        "A5_without_reliability": TRAFFIC + CALENDAR + RAINFALL + METEOROLOGY + ATMOSPHERIC + ROAD,
        "A6_without_road_context": BASE + RAINFALL + METEOROLOGY + ATMOSPHERIC,
    },
    "meteorology_ablation": {
        "M0_traffic_only": BASE,
        "M1_temperature": BASE + ["temperature_2m", "weather_missing_flag"],
        "M2_humidity_dew_point": BASE + ["relative_humidity_2m", "dew_point_2m", "weather_missing_flag"],
        "M3_wind": BASE + [
            "wind_speed_10m", "wind_gusts_10m", "wind_u", "wind_v", "wind_calm_flag",
            "wind_gust_ratio", "wind_gust_ratio_missing_flag", "weather_missing_flag",
        ],
        "M4_pressure": BASE + ["surface_pressure", "weather_missing_flag"],
        "M5_cloud_boundary_layer_radiation": BASE + [
            "cloud_cover", "boundary_layer_height", "shortwave_radiation", "weather_missing_flag",
        ],
        "M6_temperature_humidity": BASE + [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "weather_missing_flag",
        ],
        "M7_wind_boundary_layer": BASE + [
            "wind_speed_10m", "wind_gusts_10m", "wind_u", "wind_v", "wind_calm_flag",
            "wind_gust_ratio", "wind_gust_ratio_missing_flag", "boundary_layer_height",
            "weather_missing_flag",
        ],
        "M8_full_meteorology": BASE + METEOROLOGY,
    },
    "rainfall_ablation": {
        "R0_current_rainfall_only": BASE + ["precipitation"],
        "R1_current_plus_rain_flag": BASE + ["precipitation", "rain_flag"],
        "R2_current_flag_plus_lag": BASE + [
            "precipitation", "rain_flag", "rain_lag_1h", "rain_lag_3h", "rain_lag_6h",
        ],
        "R3_current_flag_lag_cumulative": BASE + [
            "precipitation", "rain_flag", "rain_lag_1h", "rain_lag_3h", "rain_lag_6h",
            "rain_sum_3h", "rain_sum_6h", "rain_sum_12h", "rain_sum_24h",
        ],
        "R4_current_flag_lag_cumulative_post_rain": BASE + [
            "precipitation", "rain_flag", "rain_lag_1h", "rain_lag_3h", "rain_lag_6h",
            "rain_sum_3h", "rain_sum_6h", "rain_sum_12h", "rain_sum_24h",
            "post_rain_1_3h", "post_rain_4_6h", "post_rain_7_12h",
        ],
        "R5_full_rainfall": BASE + RAINFALL,
    },
    "atmospheric_ablation": {
        "P0_traffic_only": BASE,
        "P1_pm2_5_only": BASE + ["pm2_5", "air_quality_missing_flag"],
        "P2_aod_only": BASE + ["aerosol_optical_depth", "air_quality_missing_flag"],
        "P3_pm2_5_plus_aod": BASE + ["pm2_5", "aerosol_optical_depth", "air_quality_missing_flag"],
        "P4_gaseous_pollutants": BASE + [
            "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide",
            "air_quality_missing_flag",
        ],
        "P5_particulate_context": BASE + [
            "pm2_5", "pm10", "aerosol_optical_depth", "air_quality_missing_flag",
        ],
        "P6_full_atmospheric_context": BASE + ATMOSPHERIC,
        "P7_full_atmospheric_without_lag_rolling": BASE + [
            c for c in ATMOSPHERIC if c not in ENV_LAG_ROLL
        ],
    },
}

FROZEN_ANCHORS = {
    ("modality_ablation", "A0_full_fusion"): "E5_full_fusion",
    ("meteorology_ablation", "M0_traffic_only"): "E1_traffic_only",
    ("meteorology_ablation", "M8_full_meteorology"): "E3_meteorology",
    ("rainfall_ablation", "R5_full_rainfall"): "E2_rainfall",
    ("atmospheric_ablation", "P0_traffic_only"): "E1_traffic_only",
    ("atmospheric_ablation", "P6_full_atmospheric_context"): "E4_atmospheric",
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def xgb_parameters(seed: int) -> dict:
    return {
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.05,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_state": int(seed),
        "n_jobs": 4,
        "early_stopping_rounds": 60,
    }


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    total = np.square(y_true - y_true.mean()).sum()
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "sMAPE": float(np.where(denominator > 1e-12, np.abs(error) / denominator, 0).mean() * 100),
        "R2": float(1 - np.square(error).sum() / total) if total > 0 else float("nan"),
    }


def sample_hash(values: pd.Series) -> str:
    return hashlib.sha256("\n".join(sorted(values.astype(str))).encode()).hexdigest()


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            output[column] = frame[column].astype(float)
        else:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    return output.replace([np.inf, -np.inf], np.nan)


def prepare(frame: pd.DataFrame, horizon: int, features: list[str]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    rows = frame.loc[frame[f"sample_eligible_h{horizon}"].fillna(False)].copy()
    rows = rows.sort_values(["timestamp_local", "node_id"]).reset_index(drop=True)
    train_mask = rows["split"].eq("train").to_numpy()
    numeric_columns = [c for c in features if c not in CATEGORICAL]
    categorical_columns = [c for c in features if c in CATEGORICAL]
    numeric = numeric_frame(rows, numeric_columns)
    medians = {}
    for column in numeric_columns:
        median = float(numeric.loc[train_mask, column].median())
        if not math.isfinite(median):
            raise ValueError(f"Training median missing for {column}")
        medians[column] = median
        numeric[column] = numeric[column].fillna(median)
    parts = [numeric]
    categories = {}
    for column in ["node_id"] + categorical_columns:
        train_categories = sorted(rows.loc[train_mask, column].fillna("UNKNOWN").astype(str).unique())
        values = rows[column].fillna("UNKNOWN").astype(str)
        values = values.where(values.isin(train_categories), "UNKNOWN")
        all_categories = train_categories + (["UNKNOWN"] if "UNKNOWN" not in train_categories else [])
        categories[column] = all_categories
        encoded = pd.get_dummies(
            pd.Categorical(values, categories=all_categories), prefix=column, dtype=np.float32
        )
        encoded.index = rows.index
        parts.append(encoded)
    design = pd.concat(parts, axis=1).astype(np.float32)
    return rows, design.to_numpy(np.float32), {
        "training_medians": medians,
        "training_categories": categories,
        "encoded_features": design.columns.astype(str).tolist(),
    }


def prediction_frame(rows: pd.DataFrame, horizon: int, y_pred: np.ndarray, variant: str, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": rows[f"sample_id_h{horizon}"].astype(str).to_numpy(),
            "node_id": rows["node_id"].astype(str).to_numpy(),
            "forecast_origin": pd.to_datetime(rows["timestamp_local"]).to_numpy(),
            "target_timestamp": pd.to_datetime(rows[f"target_timestamp_h{horizon}"]).to_numpy(),
            "horizon": horizon,
            "y_true": rows[f"target_speed_h{horizon}"].astype(float).to_numpy(),
            "y_pred": np.asarray(y_pred, dtype=float),
            "model": f"{variant}_xgboost",
            "variant": variant,
            "seed": seed,
            "split": "test",
        }
    )


def frozen_anchor(group_name: str, variant: str) -> pd.DataFrame:
    source = FROZEN_ANCHORS[(group_name, variant)]
    predictions = pd.read_parquet(E1_E5 / source / "predictions.parquet").copy()
    predictions["model"] = f"{variant}_xgboost_frozen_anchor"
    predictions["variant"] = variant
    return predictions


def run_variant(frame: pd.DataFrame, group_name: str, variant: str, features: list[str]) -> None:
    out = ROOT / group_name
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    prediction_path = runs / f"{variant}_predictions.parquet"
    log_path = runs / f"{variant}_training_log.csv"
    config_path = runs / f"{variant}_config.yaml"
    if prediction_path.exists() and log_path.exists() and config_path.exists():
        print(f"SKIP {group_name}/{variant}", flush=True)
        return
    if (group_name, variant) in FROZEN_ANCHORS:
        predictions = frozen_anchor(group_name, variant)
        logs = []
        for (horizon, seed), subset in predictions.groupby(["horizon", "seed"]):
            logs.append(
                {
                    "group": group_name,
                    "variant": variant,
                    "horizon": horizon,
                    "seed": seed,
                    "source": f"frozen_{FROZEN_ANCHORS[(group_name, variant)]}",
                    "best_iteration": "frozen",
                    "elapsed_seconds": 0,
                    "test_used_for_selection": False,
                }
            )
        predictions.to_parquet(prediction_path, index=False)
        pd.DataFrame(logs).to_csv(log_path, index=False)
    else:
        prediction_tables = []
        logs = []
        for horizon in HORIZONS:
            rows, design, preprocessing = prepare(frame, horizon, features)
            train_mask = rows["split"].eq("train").to_numpy()
            validation_mask = rows["split"].eq("validation").to_numpy()
            test_mask = rows["split"].eq("test").to_numpy()
            y = rows[f"target_speed_h{horizon}"].to_numpy(np.float32)
            for seed in SEEDS:
                set_seed(seed)
                start = time.time()
                model = xgb.XGBRegressor(**xgb_parameters(seed))
                model.fit(
                    design[train_mask], y[train_mask],
                    eval_set=[(design[validation_mask], y[validation_mask])], verbose=False,
                )
                prediction_tables.append(
                    prediction_frame(rows.loc[test_mask], horizon, model.predict(design[test_mask]), variant, seed)
                )
                logs.append(
                    {
                        "group": group_name,
                        "variant": variant,
                        "horizon": horizon,
                        "seed": seed,
                        "source": "new_fixed_protocol_fit",
                        "train_samples": int(train_mask.sum()),
                        "validation_samples": int(validation_mask.sum()),
                        "test_samples": int(test_mask.sum()),
                        "input_feature_count": len(features),
                        "encoded_feature_count": design.shape[1],
                        "best_iteration": int(model.best_iteration),
                        "best_validation_MAE": float(model.best_score),
                        "elapsed_seconds": time.time() - start,
                        "test_used_for_selection": False,
                    }
                )
                print(
                    f"DONE {group_name}/{variant} h{horizon} seed{seed} "
                    f"best_iter={model.best_iteration} sec={logs[-1]['elapsed_seconds']:.2f}",
                    flush=True,
                )
        predictions = pd.concat(prediction_tables, ignore_index=True)
        predictions.to_parquet(prediction_path, index=False)
        pd.DataFrame(logs).to_csv(log_path, index=False)
    config = {
        "group": group_name,
        "variant": variant,
        "features": features,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "backbone": "fixed XGBoost protocol identical to frozen E1-E5",
        "training_only_imputation": True,
        "validation_only_early_stopping": True,
        "test_used_for_selection": False,
        "frozen_anchor": FROZEN_ANCHORS.get((group_name, variant)),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, model, horizon), group in metrics.groupby(["variant", "model", "horizon"], sort=False):
        row = {
            "variant": variant,
            "model": model,
            "horizon": int(horizon),
            "split": "test",
            "seed_count": group["seed"].nunique(),
            "sample_count_per_seed": int(group["sample_count"].iloc[0]),
        }
        for metric in ("MAE", "RMSE", "sMAPE", "R2"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def finalize_group(group_name: str) -> None:
    out = ROOT / group_name
    variants = GROUPS[group_name]
    files = [out / "runs" / f"{variant}_predictions.parquet" for variant in variants]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise ValueError(f"Missing {len(missing)} variant outputs for {group_name}")
    predictions = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    metric_rows = []
    consistency_rows = []
    reference = pd.read_parquet(E1_E5 / "E1_traffic_only" / "predictions.parquet")
    for (variant, model, horizon, seed), subset in predictions.groupby(
        ["variant", "model", "horizon", "seed"], sort=False
    ):
        metric_rows.append(
            {
                "variant": variant,
                "model": model,
                "horizon": int(horizon),
                "seed": int(seed),
                "split": "test",
                "sample_count": len(subset),
                "unique_hours": subset["forecast_origin"].nunique(),
                **metric_values(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy()),
            }
        )
        reference_ids = reference.loc[
            reference["horizon"].eq(horizon) & reference["seed"].eq(seed), "sample_id"
        ]
        consistency_rows.append(
            {
                "variant": variant,
                "horizon": horizon,
                "seed": seed,
                "sample_count": len(subset),
                "reference_count": len(reference_ids),
                "sample_id_sha256": sample_hash(subset["sample_id"]),
                "reference_sha256": sample_hash(reference_ids),
                "status": "PASS"
                if len(subset) == len(reference_ids)
                and sample_hash(subset["sample_id"]) == sample_hash(reference_ids)
                else "FAIL",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    summary = summarize(metrics)
    consistency = pd.DataFrame(consistency_rows)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    consistency.to_csv(out / "sample_consistency.csv", index=False)
    if not consistency["status"].eq("PASS").all():
        raise ValueError(f"Sample consistency failed for {group_name}")

    e1 = pd.read_csv(E1_E5 / "E1_traffic_only" / "metrics_by_seed.csv")[
        ["horizon", "seed", "MAE"]
    ].rename(columns={"MAE": "traffic_only_MAE"})
    metrics = metrics.merge(e1, on=["horizon", "seed"], how="left", validate="m:1")
    metrics["delta_MAE_vs_traffic_only"] = metrics["MAE"] - metrics["traffic_only_MAE"]
    metrics["relative_change_vs_traffic_only_percent"] = (
        metrics["delta_MAE_vs_traffic_only"] / metrics["traffic_only_MAE"] * 100
    )
    if group_name == "modality_ablation":
        full = metrics.loc[metrics["variant"].eq("A0_full_fusion"), ["horizon", "seed", "MAE"]].rename(
            columns={"MAE": "full_fusion_MAE"}
        )
        metrics = metrics.merge(full, on=["horizon", "seed"], how="left", validate="m:1")
        metrics["delta_MAE_vs_full"] = metrics["MAE"] - metrics["full_fusion_MAE"]
    metrics.to_csv(out / "metrics_with_deltas.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS) + ["all"], default="all")
    parser.add_argument("--variant")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    frame = pd.read_parquet(FEATURE_PATH)
    groups = list(GROUPS) if args.group == "all" else [args.group]
    for group_name in groups:
        variants = GROUPS[group_name]
        selected = {args.variant: variants[args.variant]} if args.variant else variants
        for variant, features in selected.items():
            run_variant(frame, group_name, variant, features)
        if args.finalize or args.variant is None:
            finalize_group(group_name)
    print(json.dumps({"status": "PASS", "groups": groups}, indent=2), flush=True)


if __name__ == "__main__":
    main()
