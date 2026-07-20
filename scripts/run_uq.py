from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

import run_e6_ablations as e6


PACKAGE = Path(__file__).resolve().parents[1]
FEATURE_PATH = PACKAGE / "01_final_data" / "final_model_features_frozen.parquet"
LABEL_PATH = PACKAGE / "01_final_data" / "sample_labels_long.parquet"
SELECTION_PATH = PACKAGE / "10_statistical_tests" / "environmental_best_model_selection.csv"
OUT = PACKAGE / "09_E7_UQ"

HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)
QUANTILES = np.array([0.05, 0.50, 0.95], dtype=np.float32)
NOMINAL_COVERAGE = 0.90


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_specs() -> tuple[dict[str, list[str]], dict]:
    selection = pd.read_csv(SELECTION_PATH)
    chosen = selection.loc[selection["selected_for_UQ_and_primary_selective_comparison"]].iloc[0]
    group = str(chosen["ablation_group"])
    variant = str(chosen["variant"])
    features = e6.GROUPS[group][variant]
    specs = {
        "UQ_T1_traffic_only": e6.BASE,
        f"UQ_T2_best_environmental_{variant}": features,
        "UQ_aux_full_fusion": e6.FULL,
    }
    evidence = {
        "selected_group": group,
        "selected_variant": variant,
        "validation_MAE_mean": float(chosen["validation_MAE_mean"]),
        "selection_uses_test": False,
    }
    return specs, evidence


def uq_parameters(seed: int) -> dict:
    return {
        "n_estimators": 700,
        "max_depth": 6,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.05,
        "objective": "reg:quantileerror",
        "quantile_alpha": QUANTILES,
        "eval_metric": "quantile",
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_state": seed,
        "n_jobs": 4,
        "early_stopping_rounds": 60,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinball(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    residual = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.maximum(quantile * residual, (quantile - 1) * residual).mean())


def uq_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_true"].to_numpy(float)
    q05 = frame["q05"].to_numpy(float)
    q50 = frame["q50"].to_numpy(float)
    q95 = frame["q95"].to_numpy(float)
    lower = frame["q05_conformal"].to_numpy(float)
    upper = frame["q95_conformal"].to_numpy(float)
    width = upper - lower
    y_range = float(y.max() - y.min())
    return {
        "Pinball_q05": pinball(y, q05, 0.05),
        "Pinball_q50": pinball(y, q50, 0.50),
        "Pinball_q95": pinball(y, q95, 0.95),
        "Pinball_Loss": float(
            np.mean([pinball(y, q05, 0.05), pinball(y, q50, 0.50), pinball(y, q95, 0.95)])
        ),
        "PICP": float(((y >= lower) & (y <= upper)).mean()),
        "MPIW": float(width.mean()),
        "PINAW": float(width.mean() / y_range) if y_range > 0 else float("nan"),
        "median_MAE": float(np.abs(q50 - y).mean()),
    }


def finite_sample_level(calibration_count: int) -> float:
    return min(1.0, np.ceil((calibration_count + 1) * NOMINAL_COVERAGE) / calibration_count)


def run_one(
    frame: pd.DataFrame,
    model_name: str,
    features: list[str],
    horizon: int,
    seed: int,
) -> tuple[pd.DataFrame, dict, dict]:
    set_seed(seed)
    rows, design, preprocessing = e6.prepare(frame, horizon, features)
    partition = rows["uq_partition"].astype(str)
    train_mask = partition.eq("train").to_numpy()
    tuning_mask = partition.eq("tuning").to_numpy()
    calibration_mask = partition.eq("calibration").to_numpy()
    test_mask = partition.eq("test").to_numpy()
    y = rows[f"target_speed_h{horizon}"].to_numpy(np.float32)
    if not all(mask.sum() > 0 for mask in (train_mask, tuning_mask, calibration_mask, test_mask)):
        raise ValueError(f"Missing UQ partition for h{horizon}")

    start = time.time()
    model = xgb.XGBRegressor(**uq_parameters(seed))
    model.fit(
        design[train_mask],
        y[train_mask],
        eval_set=[(design[tuning_mask], y[tuning_mask])],
        verbose=False,
    )
    calibration_raw = np.asarray(model.predict(design[calibration_mask]), dtype=float)
    test_raw = np.asarray(model.predict(design[test_mask]), dtype=float)
    if calibration_raw.ndim != 2 or calibration_raw.shape[1] != 3:
        raise ValueError(f"Unexpected multi-quantile prediction shape: {calibration_raw.shape}")
    calibration_sorted = np.sort(calibration_raw, axis=1)
    test_sorted = np.sort(test_raw, axis=1)
    calibration_crossing_count = int(np.any(np.diff(calibration_raw, axis=1) < 0, axis=1).sum())
    test_crossing_count = int(np.any(np.diff(test_raw, axis=1) < 0, axis=1).sum())
    y_calibration = y[calibration_mask].astype(float)
    nonconformity = np.maximum.reduce(
        [
            calibration_sorted[:, 0] - y_calibration,
            y_calibration - calibration_sorted[:, 2],
            np.zeros(len(y_calibration)),
        ]
    )
    level = finite_sample_level(len(nonconformity))
    correction = float(np.quantile(nonconformity, level, method="higher"))
    lower = test_sorted[:, 0] - correction
    upper = test_sorted[:, 2] + correction
    test_rows = rows.loc[test_mask]
    predictions = pd.DataFrame(
        {
            "sample_id": test_rows[f"sample_id_h{horizon}"].astype(str).to_numpy(),
            "node_id": test_rows["node_id"].astype(str).to_numpy(),
            "forecast_origin": pd.to_datetime(test_rows["timestamp_local"]).to_numpy(),
            "target_timestamp": pd.to_datetime(test_rows[f"target_timestamp_h{horizon}"]).to_numpy(),
            "horizon": horizon,
            "y_true": y[test_mask].astype(float),
            "q05_raw": test_raw[:, 0],
            "q50_raw": test_raw[:, 1],
            "q95_raw": test_raw[:, 2],
            "q05": test_sorted[:, 0],
            "q50": test_sorted[:, 1],
            "q95": test_sorted[:, 2],
            "q05_conformal": lower,
            "q95_conformal": upper,
            "interval_width": upper - lower,
            "covered": (y[test_mask] >= lower) & (y[test_mask] <= upper),
            "model": model_name,
            "seed": seed,
            "split": "test",
        }
    )
    checkpoint_dir = OUT / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{model_name}_h{horizon}_seed{seed}.json"
    model.save_model(checkpoint)
    training = {
        "model": model_name,
        "horizon": horizon,
        "seed": seed,
        "feature_count": len(features),
        "encoded_feature_count": design.shape[1],
        "train_samples": int(train_mask.sum()),
        "tuning_samples": int(tuning_mask.sum()),
        "calibration_samples": int(calibration_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "best_iteration": int(model.best_iteration),
        "best_tuning_quantile_loss": float(model.best_score),
        "elapsed_seconds": time.time() - start,
        "test_used_for_model_selection": False,
        "checkpoint": str(checkpoint.relative_to(PACKAGE)),
        "checkpoint_sha256": sha256(checkpoint),
    }
    calibration = {
        "model": model_name,
        "horizon": horizon,
        "seed": seed,
        "nominal_coverage": NOMINAL_COVERAGE,
        "finite_sample_quantile_level": level,
        "conformal_correction": correction,
        "calibration_samples": int(calibration_mask.sum()),
        "calibration_start": rows.loc[calibration_mask, "timestamp_local"].min(),
        "calibration_end": rows.loc[calibration_mask, "timestamp_local"].max(),
        "calibration_partition": "calibration",
        "test_used_for_calibration": False,
        "calibration_quantile_crossing_rows_before_sort": calibration_crossing_count,
        "test_quantile_crossing_rows_before_sort": test_crossing_count,
        "quantile_crossing_rule": "row-wise monotone sorting before conformal calibration",
        "nonconformity_rule": "max(q05-y, y-q95, 0)",
    }
    preprocessing_path = OUT / "preprocessing" / f"{model_name}_h{horizon}.json"
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    if not preprocessing_path.exists():
        preprocessing_path.write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    return predictions, training, calibration


def aggregate_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> None:
    overall_rows = []
    for (model, horizon, seed), group in predictions.groupby(["model", "horizon", "seed"], sort=False):
        overall_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "seed": seed,
                "sample_count": len(group),
                "nominal_coverage": NOMINAL_COVERAGE,
                **uq_metrics(group),
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT / "uq_metrics_overall.csv", index=False)
    summary_rows = []
    for (model, horizon), group in overall.groupby(["model", "horizon"], sort=False):
        row = {"model": model, "horizon": horizon, "seed_count": group["seed"].nunique()}
        for metric in ("Pinball_Loss", "PICP", "MPIW", "PINAW", "median_MAE"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "uq_metrics_overall_summary.csv", index=False)

    joined = predictions.merge(
        labels,
        on=["sample_id", "node_id", "horizon"],
        how="left",
        validate="m:1",
    )
    scenario_columns = [
        "S1_dry",
        "S2_rain",
        "S7_elevated_aod",
        "S8_rain_elevated_atmospheric_pollution",
    ]
    scenario_rows = []
    for scenario in scenario_columns:
        for (model, horizon, seed), group in joined.loc[joined[scenario].fillna(False)].groupby(
            ["model", "horizon", "seed"], sort=False
        ):
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "horizon": horizon,
                    "seed": seed,
                    "sample_count": len(group),
                    **uq_metrics(group),
                }
            )
    pd.DataFrame(scenario_rows).to_csv(OUT / "uq_metrics_by_scenario.csv", index=False)

    reliability_rows = []
    for reliability in ("High", "Medium", "Low"):
        for (model, horizon, seed), group in joined.loc[
            joined["reliability_group_final"].eq(reliability)
        ].groupby(["model", "horizon", "seed"], sort=False):
            reliability_rows.append(
                {
                    "reliability_group": reliability,
                    "model": model,
                    "horizon": horizon,
                    "seed": seed,
                    "sample_count": len(group),
                    **uq_metrics(group),
                }
            )
    pd.DataFrame(reliability_rows).to_csv(OUT / "uq_metrics_by_reliability.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs, selection = select_specs()
    frame = pd.read_parquet(FEATURE_PATH)
    labels = pd.read_parquet(LABEL_PATH)
    predictions_all = []
    training_rows = []
    calibration_rows = []
    run_dir = OUT / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for model_name, features in specs.items():
        for horizon in HORIZONS:
            for seed in SEEDS:
                pred_path = run_dir / f"{model_name}_h{horizon}_seed{seed}.parquet"
                train_path = run_dir / f"{model_name}_h{horizon}_seed{seed}_training.json"
                calibration_path = run_dir / f"{model_name}_h{horizon}_seed{seed}_calibration.json"
                if pred_path.exists() and train_path.exists() and calibration_path.exists():
                    predictions_all.append(pd.read_parquet(pred_path))
                    training_rows.append(json.loads(train_path.read_text(encoding="utf-8")))
                    calibration_rows.append(json.loads(calibration_path.read_text(encoding="utf-8")))
                    continue
                predictions, training, calibration = run_one(frame, model_name, features, horizon, seed)
                predictions.to_parquet(pred_path, index=False)
                train_path.write_text(json.dumps(training, indent=2, default=str), encoding="utf-8")
                calibration_path.write_text(json.dumps(calibration, indent=2, default=str), encoding="utf-8")
                predictions_all.append(predictions)
                training_rows.append(training)
                calibration_rows.append(calibration)
                print(
                    f"DONE {model_name} h{horizon} seed{seed} "
                    f"best_iter={training['best_iteration']} qhat={calibration['conformal_correction']:.6f}",
                    flush=True,
                )
    predictions = pd.concat(predictions_all, ignore_index=True)
    predictions.to_parquet(OUT / "quantile_predictions.parquet", index=False)
    pd.DataFrame(training_rows).to_csv(OUT / "training_log.csv", index=False)
    calibration_audit = pd.DataFrame(calibration_rows)
    calibration_audit["status"] = np.where(~calibration_audit["test_used_for_calibration"], "PASS", "FAIL")
    calibration_audit.to_csv(OUT / "calibration_audit.csv", index=False)
    aggregate_metrics(predictions, labels)
    config = {
        "models": {name: features for name, features in specs.items()},
        "validation_only_environmental_selection": selection,
        "quantiles": QUANTILES.tolist(),
        "nominal_coverage": NOMINAL_COVERAGE,
        "calibration_partition": "fixed calibration subset of validation",
        "test_used_for_calibration": False,
        "conformal_rule": "finite-sample higher quantile of max(q05-y, y-q95, 0)",
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
    }
    (OUT / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "runs": len(training_rows), **selection}, indent=2))


if __name__ == "__main__":
    main()
