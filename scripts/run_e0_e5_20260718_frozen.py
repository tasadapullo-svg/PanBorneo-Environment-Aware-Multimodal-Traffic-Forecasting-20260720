from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import xgboost as xgb
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from finalize_data import (  # noqa: E402
    ATMOSPHERIC,
    CALENDAR,
    HORIZONS,
    METEOROLOGY,
    RAINFALL,
    RELIABILITY,
    ROAD,
    SEEDS,
    TRAFFIC,
)


FEATURE_PATH = OUTPUT / "10_feature_data" / "final_model_features_frozen.parquet"
EXPERIMENT_ROOT = OUTPUT / "12_experiments"
AUDIT_DIR = OUTPUT / "09_audit_reports"
READY_DIR = OUTPUT / "11_experiment_ready"
FIGURE_DIR = OUTPUT / "T2_FIGURE_SOURCE_DATA"
TABLE_DIR = OUTPUT / "T2_TABLE_SOURCE_DATA"

EXPERIMENT_FEATURES = {
    "E1_traffic_only": TRAFFIC + RELIABILITY + CALENDAR,
    "E2_rainfall": TRAFFIC + RELIABILITY + CALENDAR + RAINFALL,
    "E3_meteorology": TRAFFIC + RELIABILITY + CALENDAR + METEOROLOGY,
    "E4_atmospheric": TRAFFIC + RELIABILITY + CALENDAR + ATMOSPHERIC,
    "E5_full_fusion": TRAFFIC
    + RELIABILITY
    + CALENDAR
    + RAINFALL
    + METEOROLOGY
    + ATMOSPHERIC
    + ROAD,
}

MODEL_NAMES = {
    "E1_traffic_only": "E1_traffic_only_xgboost",
    "E2_rainfall": "E2_rainfall_aware_xgboost",
    "E3_meteorology": "E3_meteorology_aware_xgboost",
    "E4_atmospheric": "E4_atmospheric_context_xgboost",
    "E5_full_fusion": "E5_full_fusion_xgboost",
}

CATEGORICAL = {"road_class", "urban_rural"}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(np.where(denominator > 1e-12, np.abs(error) / denominator, 0.0)) * 100)
    total = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = float(1.0 - np.sum(np.square(error)) / total) if total > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "R2": r2}


def prediction_frame(
    rows: pd.DataFrame,
    horizon: int,
    y_pred: np.ndarray,
    model: str,
    seed: int,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "sample_id": rows[f"sample_id_h{horizon}"].astype(str).to_numpy(),
            "node_id": rows["node_id"].astype(str).to_numpy(),
            "forecast_origin": pd.to_datetime(rows["timestamp_local"]).to_numpy(),
            "target_timestamp": pd.to_datetime(rows[f"target_timestamp_h{horizon}"]).to_numpy(),
            "horizon": horizon,
            "y_true": rows[f"target_speed_h{horizon}"].astype(float).to_numpy(),
            "y_pred": np.asarray(y_pred, dtype=float),
            "model": model,
            "seed": int(seed),
            "split": rows["split"].astype(str).to_numpy(),
        }
    )
    return output


def metric_row(predictions: pd.DataFrame, experiment: str) -> dict:
    values = metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy())
    return {
        "experiment": experiment,
        "model": predictions["model"].iloc[0],
        "horizon": int(predictions["horizon"].iloc[0]),
        "seed": int(predictions["seed"].iloc[0]),
        "split": str(predictions["split"].iloc[0]),
        "sample_count": int(len(predictions)),
        "unique_hours": int(predictions["forecast_origin"].nunique()),
        **values,
    }


def summarize_metrics(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["MAE", "RMSE", "sMAPE", "R2"]
    grouped = metrics_by_seed.groupby(["experiment", "model", "horizon", "split"], sort=False)
    rows = []
    for keys, group in grouped:
        row = dict(zip(["experiment", "model", "horizon", "split"], keys))
        row["seed_count"] = int(group["seed"].nunique())
        row["sample_count_per_seed"] = int(group["sample_count"].iloc[0])
        for metric in metric_names:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def sample_hash(values: pd.Series) -> str:
    joined = "\n".join(sorted(values.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def experiment_dir(name: str) -> Path:
    path = EXPERIMENT_ROOT / name
    (path / "best_checkpoint").mkdir(parents=True, exist_ok=True)
    return path


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        if pd.api.types.is_bool_dtype(frame[column]):
            output[column] = frame[column].astype(float)
        else:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    return output.replace([np.inf, -np.inf], np.nan)


def prepare_tabular(
    frame: pd.DataFrame,
    horizon: int,
    features: list[str],
    experiment: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[dict], dict]:
    eligible = frame.loc[frame[f"sample_eligible_h{horizon}"]].copy()
    eligible = eligible.sort_values(["timestamp_local", "node_id"]).reset_index(drop=True)
    train_mask = eligible["split"].eq("train").to_numpy()
    numeric_columns = [column for column in features if column not in CATEGORICAL]
    categorical_columns = [column for column in features if column in CATEGORICAL]

    numeric = numeric_frame(eligible, numeric_columns)
    imputation_rows: list[dict] = []
    for column in numeric_columns:
        median = float(numeric.loc[train_mask, column].median())
        if not math.isfinite(median):
            raise ValueError(f"Training median unavailable for {experiment} h{horizon} {column}")
        missing_by_split = eligible.assign(_missing=numeric[column].isna()).groupby("split")["_missing"].sum()
        imputation_rows.append(
            {
                "experiment": experiment,
                "horizon": horizon,
                "feature": column,
                "training_median": median,
                "train_missing_count": int(missing_by_split.get("train", 0)),
                "validation_missing_count": int(missing_by_split.get("validation", 0)),
                "test_missing_count": int(missing_by_split.get("test", 0)),
                "fit_scope": "training_only",
            }
        )
        numeric[column] = numeric[column].fillna(median)

    encoded_parts = [numeric]
    category_manifest: dict[str, list[str]] = {}
    categorical_columns = ["node_id"] + categorical_columns
    for column in categorical_columns:
        training_categories = sorted(eligible.loc[train_mask, column].fillna("UNKNOWN").astype(str).unique())
        categories = training_categories + (["UNKNOWN"] if "UNKNOWN" not in training_categories else [])
        category_manifest[column] = categories
        values = eligible[column].fillna("UNKNOWN").astype(str)
        values = values.where(values.isin(training_categories), "UNKNOWN")
        categorical = pd.Categorical(values, categories=categories)
        encoded = pd.get_dummies(categorical, prefix=column, dtype=np.float32)
        encoded.index = eligible.index
        encoded_parts.append(encoded)

    design = pd.concat(encoded_parts, axis=1).astype(np.float32)
    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "training_categories": category_manifest,
        "encoded_feature_count": int(design.shape[1]),
    }
    return eligible, design.to_numpy(np.float32), design.columns.astype(str).tolist(), imputation_rows, metadata


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


def run_main_experiments(frame: pd.DataFrame) -> None:
    all_imputation_rows: list[dict] = []
    sample_rows: list[dict] = []
    for experiment, feature_columns in EXPERIMENT_FEATURES.items():
        print(f"START {experiment}", flush=True)
        out_dir = experiment_dir(experiment)
        all_predictions: list[pd.DataFrame] = []
        metric_rows: list[dict] = []
        training_rows: list[dict] = []
        experiment_config = {
            "experiment": experiment,
            "model": MODEL_NAMES[experiment],
            "backbone": "XGBoost fixed tabular backbone",
            "features": feature_columns,
            "node_identity": "training-category one-hot",
            "horizons": list(HORIZONS),
            "seeds": list(SEEDS),
            "model_selection_split": "validation",
            "test_used_for_selection": False,
            "xgboost_parameters_except_seed": xgb_parameters(SEEDS[0]) | {"random_state": "per_seed"},
        }
        (out_dir / "config.yaml").write_text(
            yaml.safe_dump(experiment_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        for horizon in HORIZONS:
            rows, X, encoded_names, imputation_rows, preprocessing = prepare_tabular(
                frame, horizon, feature_columns, experiment
            )
            all_imputation_rows.extend(imputation_rows)
            train_mask = rows["split"].eq("train").to_numpy()
            validation_mask = rows["split"].eq("validation").to_numpy()
            test_mask = rows["split"].eq("test").to_numpy()
            y = rows[f"target_speed_h{horizon}"].astype(np.float32).to_numpy()
            y_train = y[train_mask]
            y_validation = y[validation_mask]

            preprocessing_path = out_dir / f"preprocessing_h{horizon}.json"
            preprocessing_path.write_text(
                json.dumps(
                    {
                        **preprocessing,
                        "encoded_feature_names": encoded_names,
                        "training_only_imputation": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            test_ids = rows.loc[test_mask, f"sample_id_h{horizon}"]
            for seed in SEEDS:
                set_seed(seed)
                start = time.time()
                model = xgb.XGBRegressor(**xgb_parameters(seed))
                model.fit(
                    X[train_mask],
                    y_train,
                    eval_set=[(X[validation_mask], y_validation)],
                    verbose=False,
                )
                y_pred = model.predict(X[test_mask])
                elapsed = time.time() - start
                predictions = prediction_frame(
                    rows.loc[test_mask], horizon, y_pred, MODEL_NAMES[experiment], seed
                )
                all_predictions.append(predictions)
                metric_rows.append(metric_row(predictions, experiment))
                checkpoint = out_dir / "best_checkpoint" / f"{experiment}_h{horizon}_seed{seed}.json"
                model.save_model(checkpoint)
                training_rows.append(
                    {
                        "experiment": experiment,
                        "horizon": horizon,
                        "seed": seed,
                        "train_samples": int(train_mask.sum()),
                        "validation_samples": int(validation_mask.sum()),
                        "test_samples": int(test_mask.sum()),
                        "encoded_feature_count": int(X.shape[1]),
                        "best_iteration": int(getattr(model, "best_iteration", xgb_parameters(seed)["n_estimators"] - 1)),
                        "elapsed_seconds": elapsed,
                        "selection_split": "validation",
                        "test_used_for_selection": False,
                        "checkpoint": checkpoint.name,
                    }
                )
                sample_rows.append(
                    {
                        "experiment": experiment,
                        "horizon": horizon,
                        "seed": seed,
                        "split": "test",
                        "sample_count": int(len(test_ids)),
                        "sample_id_sha256": sample_hash(test_ids),
                        "duplicate_sample_ids": int(test_ids.duplicated().sum()),
                    }
                )
                print(
                    f"DONE {experiment} h{horizon} seed{seed} "
                    f"MAE={metric_rows[-1]['MAE']:.6f} sec={elapsed:.2f}",
                    flush=True,
                )

        prediction_table = pd.concat(all_predictions, ignore_index=True)
        metric_table = pd.DataFrame(metric_rows)
        prediction_table.to_parquet(out_dir / "predictions.parquet", index=False)
        metric_table.to_csv(out_dir / "metrics_by_seed.csv", index=False)
        summarize_metrics(metric_table).to_csv(out_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(training_rows).to_csv(out_dir / "training_log.csv", index=False)

    pd.DataFrame(all_imputation_rows).to_csv(
        READY_DIR / "training_only_imputation_statistics.csv", index=False
    )
    sample_table = pd.DataFrame(sample_rows)
    sample_table.to_csv(AUDIT_DIR / "e1_e5_sample_id_consistency.csv", index=False)
    fairness_rows = []
    for (horizon, seed), group in sample_table.groupby(["horizon", "seed"]):
        fairness_rows.append(
            {
                "horizon": horizon,
                "seed": seed,
                "experiment_count": int(group["experiment"].nunique()),
                "sample_count": int(group["sample_count"].iloc[0]),
                "all_sample_counts_identical": bool(group["sample_count"].nunique() == 1),
                "all_sample_hashes_identical": bool(group["sample_id_sha256"].nunique() == 1),
                "all_duplicate_counts_zero": bool(group["duplicate_sample_ids"].eq(0).all()),
                "status": "PASS"
                if group["experiment"].nunique() == 5
                and group["sample_count"].nunique() == 1
                and group["sample_id_sha256"].nunique() == 1
                and group["duplicate_sample_ids"].eq(0).all()
                else "FAIL",
            }
        )
    pd.DataFrame(fairness_rows).to_csv(AUDIT_DIR / "e1_e5_fairness_audit.csv", index=False)


def historical_predictions(
    rows: pd.DataFrame, horizon: int, model: str
) -> np.ndarray:
    train = rows.loc[rows["split"].eq("train")].copy()
    test = rows.loc[rows["split"].eq("test")].copy()
    target = f"target_speed_h{horizon}"
    if model == "Historical_Average":
        global_mean = float(train[target].mean())
        node_mean = train.groupby("node_id")[target].mean()
        return test["node_id"].map(node_mean).fillna(global_mean).to_numpy(float)
    if model == "Seasonal_Historical_Average":
        train["target_hour"] = train[f"target_timestamp_h{horizon}"].dt.hour
        train["target_dow"] = train[f"target_timestamp_h{horizon}"].dt.dayofweek
        test["target_hour"] = test[f"target_timestamp_h{horizon}"].dt.hour
        test["target_dow"] = test[f"target_timestamp_h{horizon}"].dt.dayofweek
        exact = train.groupby(["node_id", "target_dow", "target_hour"])[target].mean()
        node_hour = train.groupby(["node_id", "target_hour"])[target].mean()
        node = train.groupby("node_id")[target].mean()
        global_mean = float(train[target].mean())
        keys = pd.MultiIndex.from_frame(test[["node_id", "target_dow", "target_hour"]])
        pred = pd.Series(exact.reindex(keys).to_numpy(), index=test.index)
        missing = pred.isna()
        if missing.any():
            hour_keys = pd.MultiIndex.from_frame(test.loc[missing, ["node_id", "target_hour"]])
            pred.loc[missing] = node_hour.reindex(hour_keys).to_numpy()
        pred = pred.fillna(test["node_id"].map(node)).fillna(global_mean)
        return pred.to_numpy(float)
    raise ValueError(model)


def ridge_fit_predict(
    rows: pd.DataFrame,
    X: np.ndarray,
    horizon: int,
    checkpoint: Path,
) -> tuple[np.ndarray, dict]:
    train_mask = rows["split"].eq("train").to_numpy()
    validation_mask = rows["split"].eq("validation").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()
    y = rows[f"target_speed_h{horizon}"].to_numpy(float)
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std < 1e-8] = 1.0
    X_scaled = (X - mean) / std
    best_alpha = None
    best_mae = float("inf")
    best_weights = None
    validation_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.as_tensor(X_scaled[train_mask], dtype=torch.float32, device=device)
    X_validation_tensor = torch.as_tensor(X_scaled[validation_mask], dtype=torch.float32, device=device)
    X_test_tensor = torch.as_tensor(X_scaled[test_mask], dtype=torch.float32, device=device)
    y_train_tensor = torch.as_tensor(y[train_mask], dtype=torch.float32, device=device)
    ones_train = torch.ones((len(X_train_tensor), 1), dtype=torch.float32, device=device)
    ones_validation = torch.ones((len(X_validation_tensor), 1), dtype=torch.float32, device=device)
    ones_test = torch.ones((len(X_test_tensor), 1), dtype=torch.float32, device=device)
    design_train = torch.cat([X_train_tensor, ones_train], dim=1)
    design_validation = torch.cat([X_validation_tensor, ones_validation], dim=1)
    design_test = torch.cat([X_test_tensor, ones_test], dim=1)
    gram = design_train.T @ design_train
    rhs = design_train.T @ y_train_tensor
    for alpha in (0.1, 1.0, 10.0, 100.0):
        penalty = torch.eye(gram.shape[0], dtype=torch.float32, device=device) * float(alpha)
        penalty[-1, -1] = 0.0  # Do not regularize the intercept.
        weights = torch.linalg.solve(gram + penalty, rhs)
        pred = (design_validation @ weights).detach().cpu().numpy()
        mae = float(np.mean(np.abs(pred - y[validation_mask])))
        validation_results.append({"alpha": alpha, "validation_MAE": mae})
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_weights = weights.detach().cpu().numpy()
    if best_weights is None:
        raise RuntimeError("Ridge selection failed")
    joblib.dump(
        {
            "coefficient": best_weights[:-1],
            "intercept": float(best_weights[-1]),
            "training_mean": mean,
            "training_std": std,
            "best_alpha": best_alpha,
            "validation_results": validation_results,
            "solver": "closed_form_normal_equation_torch",
        },
        checkpoint,
    )
    test_prediction = (
        design_test @ torch.as_tensor(best_weights, dtype=torch.float32, device=device)
    ).detach().cpu().numpy()
    return test_prediction, {
        "best_alpha": best_alpha,
        "best_validation_MAE": best_mae,
        "validation_results": validation_results,
    }


class CausalTCN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2)
        self.residual = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = torch.nn.functional.pad(x, (2, 0))
        h = torch.relu(self.conv1(h))
        h = self.dropout(h)
        h = torch.nn.functional.pad(h, (4, 0))
        h = self.conv2(h)
        h = torch.relu(h + residual)
        return self.head(h[:, :, -1]).squeeze(-1)


def sequence_indices(sorted_frame: pd.DataFrame, mask: np.ndarray, history: int) -> np.ndarray:
    positions = np.flatnonzero(mask)
    in_node_position = sorted_frame["_node_position"].to_numpy()[positions]
    positions = positions[in_node_position >= history - 1]
    offsets = np.arange(-(history - 1), 1)
    indices = positions[:, None] + offsets[None, :]
    nodes = sorted_frame["node_id"].to_numpy()
    if not np.all(nodes[indices[:, 0]] == nodes[indices[:, -1]]):
        raise ValueError("TCN sequence crossed a node boundary")
    return indices


def run_tcn_horizon(
    frame: pd.DataFrame,
    horizon: int,
    seed: int,
    checkpoint: Path,
) -> tuple[pd.DataFrame, dict]:
    set_seed(seed)
    history = 24
    tcn_features = TRAFFIC + RELIABILITY + CALENDAR
    sorted_frame = frame.sort_values(["node_id", "timestamp_local"]).reset_index(drop=True).copy()
    sorted_frame["_node_position"] = sorted_frame.groupby("node_id").cumcount()
    numeric = numeric_frame(sorted_frame, tcn_features)
    fit_rows = sorted_frame["split"].eq("train") & sorted_frame["causal_input_available"]
    medians = numeric.loc[fit_rows].median()
    if medians.isna().any():
        raise ValueError("TCN training median missing")
    numeric = numeric.fillna(medians)
    means = numeric.loc[fit_rows].mean()
    stds = numeric.loc[fit_rows].std(ddof=0).replace(0, 1.0)
    flat = ((numeric - means) / stds).to_numpy(np.float32)

    eligible = sorted_frame[f"sample_eligible_h{horizon}"].to_numpy(bool)
    train_mask = eligible & sorted_frame["split"].eq("train").to_numpy()
    validation_mask = eligible & sorted_frame["split"].eq("validation").to_numpy()
    test_mask = eligible & sorted_frame["split"].eq("test").to_numpy()
    train_indices = sequence_indices(sorted_frame, train_mask, history)
    validation_indices = sequence_indices(sorted_frame, validation_mask, history)
    test_indices = sequence_indices(sorted_frame, test_mask, history)
    test_origin_positions = test_indices[:, -1]

    target = sorted_frame[f"target_speed_h{horizon}"].to_numpy(np.float32)
    y_train = target[train_indices[:, -1]]
    y_validation = target[validation_indices[:, -1]]
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-8:
        y_std = 1.0

    X_train = torch.from_numpy(flat[train_indices].transpose(0, 2, 1))
    X_validation = torch.from_numpy(flat[validation_indices].transpose(0, 2, 1))
    X_test = torch.from_numpy(flat[test_indices].transpose(0, 2, 1))
    y_train_tensor = torch.from_numpy((y_train - y_mean) / y_std)
    y_validation_tensor = torch.from_numpy((y_validation - y_mean) / y_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The complete 24 h sequence tensors fit comfortably on the project GPU.
    # Keeping them resident avoids thousands of small host-to-device transfers.
    X_train = X_train.to(device)
    X_validation = X_validation.to(device)
    X_test = X_test.to(device)
    y_train_tensor = y_train_tensor.to(device)
    y_validation_tensor = y_validation_tensor.to(device)
    model = CausalTCN(len(tcn_features), hidden_channels=16, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    best_state = None
    best_val_mae = float("inf")
    best_epoch = 0
    patience = 2
    stale = 0
    epoch_log = []
    start = time.time()
    batch_size = 16384 if device.type == "cuda" else 1024
    for epoch in range(1, 9):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        permutation = torch.randperm(len(X_train), device=device)
        for start_index in range(0, len(X_train), batch_size):
            selection = permutation[start_index : start_index + batch_size]
            xb = X_train[selection]
            yb = y_train_tensor[selection]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_function(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(xb)
            train_count += len(xb)
        model.eval()
        validation_predictions = []
        with torch.no_grad():
            for start_index in range(0, len(X_validation), batch_size):
                xb = X_validation[start_index : start_index + batch_size]
                validation_predictions.append(model(xb).cpu().numpy())
        validation_pred = np.concatenate(validation_predictions) * y_std + y_mean
        validation_mae = float(np.mean(np.abs(validation_pred - y_validation)))
        epoch_log.append(
            {
                "epoch": epoch,
                "training_smooth_l1": train_loss_sum / max(train_count, 1),
                "validation_MAE": validation_mae,
            }
        )
        if validation_mae < best_val_mae - 1e-5:
            best_val_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("TCN did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_predictions = []
    with torch.no_grad():
        for start_index in range(0, len(X_test), batch_size):
            xb = X_test[start_index : start_index + batch_size]
            test_predictions.append(model(xb).cpu().numpy())
    y_pred = np.concatenate(test_predictions) * y_std + y_mean
    elapsed = time.time() - start
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "CausalTCN",
            "input_features": tcn_features,
            "history_hours": history,
            "training_feature_medians": medians.to_dict(),
            "training_feature_means": means.to_dict(),
            "training_feature_stds": stds.to_dict(),
            "training_target_mean": y_mean,
            "training_target_std": y_std,
            "best_epoch": best_epoch,
            "best_validation_MAE": best_val_mae,
            "seed": seed,
        },
        checkpoint,
    )
    test_rows = sorted_frame.iloc[test_origin_positions]
    predictions = prediction_frame(test_rows, horizon, y_pred, "TCN_24h_causal_benchmark", seed)
    log = {
        "experiment": "E0_baselines",
        "model": "TCN_24h_causal_benchmark",
        "horizon": horizon,
        "seed": seed,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "best_epoch": best_epoch,
        "best_validation_MAE": best_val_mae,
        "elapsed_seconds": elapsed,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "epoch_log": json.dumps(epoch_log),
    }
    del X_train, X_validation, X_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, log


def run_e0(frame: pd.DataFrame, skip_tcn: bool = False) -> None:
    out_dir = experiment_dir("E0_baselines")
    config = {
        "experiment": "E0_baselines",
        "models": [
            "Persistence",
            "Historical Average",
            "Seasonal Historical Average",
            "Ridge",
            "XGBoost (same fitted traffic-only backbone as E1)",
            "TCN 24h causal benchmark",
        ],
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "deterministic_models_replicated_across_seeds": [
            "Persistence",
            "Historical Average",
            "Seasonal Historical Average",
            "Ridge",
        ],
        "selection_split": "validation",
        "test_used_for_selection": False,
    }
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    prediction_tables: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    training_logs: list[dict] = []
    e1_predictions = pd.read_parquet(EXPERIMENT_ROOT / "E1_traffic_only" / "predictions.parquet")

    for horizon in HORIZONS:
        rows, X, encoded_names, _, _ = prepare_tabular(
            frame, horizon, EXPERIMENT_FEATURES["E1_traffic_only"], "E0_baselines"
        )
        test_mask = rows["split"].eq("test").to_numpy()
        test_rows = rows.loc[test_mask]
        deterministic = {
            "Persistence": test_rows["current_speed_input"].to_numpy(float),
            "Historical_Average": historical_predictions(rows, horizon, "Historical_Average"),
            "Seasonal_Historical_Average": historical_predictions(
                rows, horizon, "Seasonal_Historical_Average"
            ),
        }
        ridge_checkpoint = out_dir / "best_checkpoint" / f"Ridge_h{horizon}.joblib"
        ridge_pred, ridge_info = ridge_fit_predict(rows, X, horizon, ridge_checkpoint)
        deterministic["Ridge"] = ridge_pred
        (out_dir / f"ridge_validation_h{horizon}.json").write_text(
            json.dumps(ridge_info, indent=2), encoding="utf-8"
        )
        training_logs.append(
            {
                "experiment": "E0_baselines",
                "model": "Ridge",
                "horizon": horizon,
                "seed": "deterministic_all_seeds",
                "train_samples": int(rows["split"].eq("train").sum()),
                "validation_samples": int(rows["split"].eq("validation").sum()),
                "test_samples": int(test_mask.sum()),
                "best_alpha": ridge_info["best_alpha"],
                "best_validation_MAE": ridge_info["best_validation_MAE"],
                "selection_split": "validation",
                "test_used_for_selection": False,
            }
        )

        for seed in SEEDS:
            for model_name, y_pred in deterministic.items():
                predictions = prediction_frame(test_rows, horizon, y_pred, model_name, seed)
                prediction_tables.append(predictions)
                metric_rows.append(metric_row(predictions, "E0_baselines"))

            xgb_source = e1_predictions.loc[
                e1_predictions["horizon"].eq(horizon) & e1_predictions["seed"].eq(seed)
            ].copy()
            xgb_source["model"] = "XGBoost_traffic_baseline"
            prediction_tables.append(xgb_source)
            metric_rows.append(metric_row(xgb_source, "E0_baselines"))

            if not skip_tcn:
                checkpoint = out_dir / "best_checkpoint" / f"TCN_h{horizon}_seed{seed}.pt"
                tcn_predictions, tcn_log = run_tcn_horizon(frame, horizon, seed, checkpoint)
                if sample_hash(tcn_predictions["sample_id"]) != sample_hash(test_rows[f"sample_id_h{horizon}"]):
                    raise ValueError("TCN test sample IDs differ from the frozen E0 test set")
                prediction_tables.append(tcn_predictions)
                metric_rows.append(metric_row(tcn_predictions, "E0_baselines"))
                training_logs.append(tcn_log)
                print(
                    f"DONE TCN h{horizon} seed{seed} MAE={metric_rows[-1]['MAE']:.6f} "
                    f"sec={tcn_log['elapsed_seconds']:.2f}",
                    flush=True,
                )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    metric_table = pd.DataFrame(metric_rows)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    metric_table.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    summarize_metrics(metric_table).to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(training_logs).to_csv(out_dir / "training_log.csv", index=False)


def aggregate_results() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_summaries = []
    for experiment in ["E0_baselines", *EXPERIMENT_FEATURES.keys()]:
        path = EXPERIMENT_ROOT / experiment
        metrics_path = path / "metrics_by_seed.csv"
        summary_path = path / "metrics_summary.csv"
        if metrics_path.exists():
            all_metrics.append(pd.read_csv(metrics_path))
        if summary_path.exists():
            all_summaries.append(pd.read_csv(summary_path))
    metrics_by_seed = pd.concat(all_metrics, ignore_index=True)
    metrics_summary = pd.concat(all_summaries, ignore_index=True)
    metrics_by_seed.to_csv(EXPERIMENT_ROOT / "E0_E5_metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(EXPERIMENT_ROOT / "E0_E5_metrics_summary.csv", index=False)
    metrics_by_seed.to_csv(TABLE_DIR / "Table03_E0_E5_metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(TABLE_DIR / "Table03_E0_E5_metrics_summary.csv", index=False)

    main = metrics_by_seed.loc[metrics_by_seed["experiment"].isin(EXPERIMENT_FEATURES)].copy()
    pivot = main.pivot_table(index=["horizon", "seed"], columns="experiment", values="MAE")
    required = list(EXPERIMENT_FEATURES)
    if not all(column in pivot.columns for column in required):
        raise ValueError("Main experiment aggregation is incomplete")
    stability = pivot.reset_index()
    stability["delta_MAE_E1_minus_E5"] = stability["E1_traffic_only"] - stability["E5_full_fusion"]
    stability["relative_improvement_percent"] = (
        stability["delta_MAE_E1_minus_E5"] / stability["E1_traffic_only"] * 100
    )
    stability["full_fusion_better"] = stability["delta_MAE_E1_minus_E5"].gt(0)
    stability.to_csv(EXPERIMENT_ROOT / "E5_vs_E1_gain_by_horizon_seed.csv", index=False)
    stability.to_csv(FIGURE_DIR / "Fig04_full_fusion_gain_by_horizon_seed.csv", index=False)

    stability_summary = (
        stability.groupby("horizon")
        .agg(
            seed_count=("seed", "nunique"),
            E1_MAE_mean=("E1_traffic_only", "mean"),
            E5_MAE_mean=("E5_full_fusion", "mean"),
            delta_MAE_mean=("delta_MAE_E1_minus_E5", "mean"),
            delta_MAE_std=("delta_MAE_E1_minus_E5", "std"),
            relative_improvement_percent_mean=("relative_improvement_percent", "mean"),
            full_fusion_win_count=("full_fusion_better", "sum"),
        )
        .reset_index()
    )
    stability_summary["stable_gain_all_seeds"] = stability_summary["full_fusion_win_count"].eq(
        stability_summary["seed_count"]
    )
    stability_summary.to_csv(EXPERIMENT_ROOT / "E5_vs_E1_gain_summary.csv", index=False)
    stability_summary.to_csv(TABLE_DIR / "Table03_E5_vs_E1_gain_summary.csv", index=False)

    modality_summary = metrics_summary.loc[metrics_summary["experiment"].isin(EXPERIMENT_FEATURES)].copy()
    modality_summary.to_csv(FIGURE_DIR / "Fig04_E1_E5_modality_comparison.csv", index=False)

    environment = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (EXPERIMENT_ROOT / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["main", "e0", "aggregate", "all"], default="all")
    parser.add_argument("--skip-tcn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(FEATURE_PATH)
    if args.stage in {"main", "all"}:
        run_main_experiments(frame)
    if args.stage in {"e0", "all"}:
        run_e0(frame, skip_tcn=args.skip_tcn)
    if args.stage in {"aggregate", "all"}:
        aggregate_results()


if __name__ == "__main__":
    main()
