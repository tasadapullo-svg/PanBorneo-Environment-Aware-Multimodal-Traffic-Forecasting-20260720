from __future__ import annotations

import argparse
import copy
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
import torch.nn as nn
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
FEATURE_PATH = PACKAGE / "01_final_data" / "final_model_features_frozen.parquet"
OUT = PACKAGE / "03_E0_baselines"
ARCHIVE = PACKAGE / "archive" / "undertrained_tcn"

HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)
MAX_EPOCHS = 100
PATIENCE = 15
HISTORY = 24
MIN_DELTA = 1e-4
GRADIENT_CLIP_NORM = 1.0

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
FEATURES = TRAFFIC + RELIABILITY + CALENDAR


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fixed seeds define the experimental replications. Optimized cuDNN kernels
    # are allowed because the protocol does not require bitwise GPU replay.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def sample_hash(values: pd.Series) -> str:
    joined = "\n".join(sorted(values.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


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


class FormalCausalTCN(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 16, dropout: float = 0.10):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size=3, dilation=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2)
        self.residual = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        hidden = torch.nn.functional.pad(x, (2, 0))
        hidden = self.dropout(torch.relu(self.conv1(hidden)))
        hidden = torch.nn.functional.pad(hidden, (4, 0))
        hidden = torch.relu(self.conv2(hidden) + residual)
        return self.head(hidden[:, :, -1]).squeeze(-1)


def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in FEATURES:
        if pd.api.types.is_bool_dtype(frame[column]):
            output[column] = frame[column].astype(float)
        else:
            output[column] = pd.to_numeric(frame[column], errors="coerce")
    return output.replace([np.inf, -np.inf], np.nan)


def sequence_indices(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    positions = np.flatnonzero(mask)
    positions = positions[frame["_node_position"].to_numpy()[positions] >= HISTORY - 1]
    indices = positions[:, None] + np.arange(-(HISTORY - 1), 1)[None, :]
    nodes = frame["node_id"].to_numpy()
    if not np.all(nodes[indices[:, 0]] == nodes[indices[:, -1]]):
        raise ValueError("TCN sequence crossed a node boundary")
    timestamps = frame["timestamp_local"].to_numpy(dtype="datetime64[h]")
    elapsed = timestamps[indices[:, -1]] - timestamps[indices[:, 0]]
    if not np.all(elapsed.astype("timedelta64[h]").astype(int) == HISTORY - 1):
        raise ValueError("TCN history is not a contiguous causal 24-hour window")
    return indices


def prepare(frame: pd.DataFrame, horizon: int) -> dict:
    sorted_frame = frame.sort_values(["node_id", "timestamp_local"]).reset_index(drop=True).copy()
    sorted_frame["_node_position"] = sorted_frame.groupby("node_id").cumcount()
    numeric = numeric_frame(sorted_frame)
    fit_mask = sorted_frame["split"].eq("train") & sorted_frame["causal_input_available"]
    medians = numeric.loc[fit_mask].median()
    numeric = numeric.fillna(medians)
    means = numeric.loc[fit_mask].mean()
    stds = numeric.loc[fit_mask].std(ddof=0).replace(0, 1.0)
    flat = ((numeric - means) / stds).to_numpy(np.float32)

    eligible = sorted_frame[f"sample_eligible_h{horizon}"].fillna(False).to_numpy(bool)
    train_mask = eligible & sorted_frame["split"].eq("train").to_numpy()
    validation_mask = eligible & sorted_frame["split"].eq("validation").to_numpy()
    test_mask = eligible & sorted_frame["split"].eq("test").to_numpy()
    train_indices = sequence_indices(sorted_frame, train_mask)
    validation_indices = sequence_indices(sorted_frame, validation_mask)
    test_indices = sequence_indices(sorted_frame, test_mask)
    target = sorted_frame[f"target_speed_h{horizon}"].to_numpy(np.float32)
    y_train = target[train_indices[:, -1]]
    y_validation = target[validation_indices[:, -1]]
    y_test = target[test_indices[:, -1]]
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-8)
    return {
        "frame": sorted_frame,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "X_train": torch.from_numpy(flat[train_indices].transpose(0, 2, 1)),
        "X_validation": torch.from_numpy(flat[validation_indices].transpose(0, 2, 1)),
        "X_test": torch.from_numpy(flat[test_indices].transpose(0, 2, 1)),
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "y_mean": y_mean,
        "y_std": y_std,
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
    }


def batched_predict(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            if batch.device != device:
                batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions.append(model(batch).float().cpu().numpy())
    return np.concatenate(predictions)


def train_one(frame: pd.DataFrame, horizon: int, seed: int) -> tuple[pd.DataFrame, list[dict], dict]:
    set_seed(seed)
    data = prepare(frame, horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    batch_size = 16384 if use_amp else 2048
    prediction_batch = 65536 if use_amp else 4096
    # The complete 24-hour tensors fit on the project GPU. Keeping them resident
    # removes repeated host-to-device transfers without changing any samples.
    X_train = data["X_train"].to(device)
    X_validation = data["X_validation"].to(device)
    X_test = data["X_test"].to(device)
    y_train_scaled = torch.from_numpy(
        (data["y_train"] - data["y_mean"]) / data["y_std"]
    ).to(device)

    model = FormalCausalTCN(len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=MIN_DELTA, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.SmoothL1Loss()
    best_state = None
    best_validation_mae = float("inf")
    best_epoch = 0
    stale = 0
    curves: list[dict] = []
    training_start = time.time()
    stop_reason = "max_epochs"

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        permutation = torch.randperm(len(X_train), device=device)
        loss_sum = 0.0
        count = 0
        max_gradient_norm = 0.0
        for start in range(0, len(X_train), batch_size):
            selection = permutation[start : start + batch_size]
            xb = X_train[selection]
            yb = y_train_scaled[selection]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(xb)
                loss = loss_function(prediction, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM).detach().cpu()
            )
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(xb)
            count += len(xb)

        validation_scaled = batched_predict(
            model, X_validation, device, prediction_batch, use_amp
        )
        validation_prediction = validation_scaled * data["y_std"] + data["y_mean"]
        validation_mae = float(np.abs(validation_prediction - data["y_validation"]).mean())
        scheduler.step(validation_mae)
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = validation_mae < best_validation_mae - MIN_DELTA
        if improved:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        curves.append(
            {
                "model": "Formal_TCN_24h_causal",
                "horizon": horizon,
                "seed": seed,
                "epoch": epoch,
                "training_smooth_l1": loss_sum / max(count, 1),
                "validation_MAE": validation_mae,
                "learning_rate": current_lr,
                "max_preclip_gradient_norm": max_gradient_norm,
                "improved_checkpoint": improved,
                "stale_epochs": stale,
                "epoch_seconds": time.time() - epoch_start,
            }
        )
        print(
            f"TCN h{horizon} seed{seed} epoch={epoch:03d} "
            f"val_MAE={validation_mae:.6f} best={best_validation_mae:.6f} "
            f"lr={current_lr:.6g} stale={stale} sec={curves[-1]['epoch_seconds']:.2f}",
            flush=True,
        )
        if stale >= PATIENCE:
            stop_reason = "early_stopping_patience_reached"
            break

    if best_state is None:
        raise RuntimeError("Formal TCN failed to produce a checkpoint")
    model.load_state_dict(best_state)
    test_scaled = batched_predict(model, X_test, device, prediction_batch, use_amp)
    y_pred = test_scaled * data["y_std"] + data["y_mean"]
    positions = data["test_indices"][:, -1]
    test_rows = data["frame"].iloc[positions]
    predictions = pd.DataFrame(
        {
            "sample_id": test_rows[f"sample_id_h{horizon}"].astype(str).to_numpy(),
            "node_id": test_rows["node_id"].astype(str).to_numpy(),
            "forecast_origin": pd.to_datetime(test_rows["timestamp_local"]).to_numpy(),
            "target_timestamp": pd.to_datetime(test_rows[f"target_timestamp_h{horizon}"]).to_numpy(),
            "horizon": horizon,
            "y_true": data["y_test"].astype(float),
            "y_pred": y_pred.astype(float),
            "model": "Formal_TCN_24h_causal",
            "seed": seed,
            "split": "test",
        }
    )

    checkpoint_dir = OUT / "best_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"Formal_TCN_h{horizon}_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_class": "FormalCausalTCN",
            "input_features": FEATURES,
            "history_hours": HISTORY,
            "training_feature_medians": data["medians"],
            "training_feature_means": data["means"],
            "training_feature_stds": data["stds"],
            "training_target_mean": data["y_mean"],
            "training_target_std": data["y_std"],
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "best_epoch": best_epoch,
            "best_validation_MAE": best_validation_mae,
            "stop_reason": stop_reason,
            "seed": seed,
            "horizon": horizon,
            "test_used_for_selection": False,
        },
        checkpoint,
    )
    manifest = {
        "model": "Formal_TCN_24h_causal",
        "horizon": horizon,
        "seed": seed,
        "train_samples": len(data["train_indices"]),
        "validation_samples": len(data["validation_indices"]),
        "test_samples": len(data["test_indices"]),
        "epochs_run": len(curves),
        "best_epoch": best_epoch,
        "best_validation_MAE": best_validation_mae,
        "final_validation_MAE": curves[-1]["validation_MAE"],
        "stop_reason": stop_reason,
        "scheduler_enabled": True,
        "gradient_clipping_norm": GRADIENT_CLIP_NORM,
        "train_only_scaling": True,
        "test_used_for_selection": False,
        "checkpoint": str(checkpoint.relative_to(PACKAGE)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "test_sample_id_sha256": sample_hash(predictions["sample_id"]),
        "elapsed_seconds": time.time() - training_start,
        "convergence_status": "PASS_EARLY_STOPPED"
        if stop_reason == "early_stopping_patience_reached"
        else "PASS_MAX_EPOCH_CHECKPOINT_SELECTED",
    }
    return predictions, curves, manifest


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), group in metrics.groupby(["model", "horizon"], sort=False):
        row = {
            "experiment": "E0_baselines",
            "model": model,
            "horizon": horizon,
            "split": "test",
            "seed_count": group["seed"].nunique(),
            "sample_count_per_seed": int(group["sample_count"].iloc[0]),
        }
        for metric in ("MAE", "RMSE", "sMAPE", "R2"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def finalize() -> None:
    run_files = sorted((OUT / "runs").glob("predictions_h*_seed*.parquet"))
    curve_files = sorted((OUT / "runs").glob("curves_h*_seed*.csv"))
    manifest_files = sorted((OUT / "runs").glob("manifest_h*_seed*.json"))
    if len(run_files) != 15 or len(curve_files) != 15 or len(manifest_files) != 15:
        raise ValueError(
            f"Formal TCN is incomplete: predictions={len(run_files)}, curves={len(curve_files)}, manifests={len(manifest_files)}"
        )
    tcn_predictions = pd.concat([pd.read_parquet(path) for path in run_files], ignore_index=True)
    curves = pd.concat([pd.read_csv(path) for path in curve_files], ignore_index=True)
    manifests = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in manifest_files])

    archived_predictions = pd.read_parquet(ARCHIVE / "predictions.parquet")
    main_predictions = archived_predictions.loc[
        ~archived_predictions["model"].eq("TCN_24h_causal_benchmark")
    ].copy()
    predictions = pd.concat([main_predictions, tcn_predictions], ignore_index=True)
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    tcn_predictions.to_parquet(OUT / "formal_tcn_predictions.parquet", index=False)
    curves.to_csv(OUT / "training_curves.csv", index=False)
    manifests.to_csv(OUT / "best_checkpoint_manifest.csv", index=False)

    metric_rows = []
    for (model, horizon, seed), group in predictions.groupby(["model", "horizon", "seed"], sort=False):
        metric_rows.append(
            {
                "experiment": "E0_baselines",
                "model": model,
                "horizon": int(horizon),
                "seed": int(seed),
                "split": "test",
                "sample_count": len(group),
                "unique_hours": group["forecast_origin"].nunique(),
                **metric_values(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
    summarize(metrics).to_csv(OUT / "metrics_summary.csv", index=False)

    convergence = manifests[
        [
            "model",
            "horizon",
            "seed",
            "epochs_run",
            "best_epoch",
            "best_validation_MAE",
            "final_validation_MAE",
            "stop_reason",
            "scheduler_enabled",
            "gradient_clipping_norm",
            "train_only_scaling",
            "test_used_for_selection",
            "convergence_status",
        ]
    ].copy()
    convergence["epoch_limit_at_least_100"] = MAX_EPOCHS >= 100
    convergence["early_stopping_patience_is_15"] = PATIENCE == 15
    terminal_slopes = []
    terminal_relative_ranges = []
    plateau_flags = []
    for row in convergence.itertuples(index=False):
        window = curves.loc[
            curves["horizon"].eq(row.horizon) & curves["seed"].eq(row.seed)
        ].tail(PATIENCE)
        x = window["epoch"].to_numpy(float)
        y = window["validation_MAE"].to_numpy(float)
        slope = float(((x - x.mean()) * (y - y.mean())).sum() / np.square(x - x.mean()).sum())
        relative_range = float(
            (window["validation_MAE"].max() - window["validation_MAE"].min())
            / max(window["validation_MAE"].mean(), 1e-12)
        )
        terminal_slopes.append(slope)
        terminal_relative_ranges.append(relative_range)
        plateau_flags.append(
            bool(abs(slope) < 0.002 and relative_range < 0.05 and window["learning_rate"].iloc[-1] < 0.003)
        )
    convergence["terminal_15_epoch_validation_slope"] = terminal_slopes
    convergence["terminal_15_epoch_relative_range"] = terminal_relative_ranges
    convergence["converged_plateau_at_max"] = plateau_flags
    convergence["convergence_evidence"] = np.where(
        convergence["stop_reason"].eq("early_stopping_patience_reached"),
        "EARLY_STOPPING",
        np.where(convergence["converged_plateau_at_max"], "VALIDATION_PLATEAU_AT_MAX_EPOCH", "INSUFFICIENT"),
    )
    convergence["status"] = np.where(
        convergence["convergence_evidence"].ne("INSUFFICIENT")
        & convergence["scheduler_enabled"]
        & convergence["train_only_scaling"]
        & ~convergence["test_used_for_selection"],
        "PASS",
        "FAIL",
    )
    convergence.to_csv(OUT / "convergence_audit.csv", index=False)

    reference = pd.read_parquet(
        PACKAGE / "04_E1_E5" / "E1_traffic_only" / "predictions.parquet"
    )
    sample_rows = []
    for horizon in HORIZONS:
        reference_ids = reference.loc[
            reference["horizon"].eq(horizon) & reference["seed"].eq(SEEDS[0]), "sample_id"
        ]
        for seed in SEEDS:
            candidate = tcn_predictions.loc[
                tcn_predictions["horizon"].eq(horizon) & tcn_predictions["seed"].eq(seed),
                "sample_id",
            ]
            sample_rows.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "reference_count": len(reference_ids),
                    "tcn_count": len(candidate),
                    "reference_sha256": sample_hash(reference_ids),
                    "tcn_sha256": sample_hash(candidate),
                    "status": "PASS"
                    if len(reference_ids) == len(candidate)
                    and sample_hash(reference_ids) == sample_hash(candidate)
                    else "FAIL",
                }
            )
    sample_audit = pd.DataFrame(sample_rows)
    sample_audit.to_csv(OUT / "sample_consistency_tcn.csv", index=False)
    if not convergence["status"].eq("PASS").all() or not sample_audit["status"].eq("PASS").all():
        raise ValueError("Formal TCN convergence or sample audit failed")

    config = {
        "experiment": "E0_baselines_submission_grade",
        "old_tcn": "preserved in archive/undertrained_tcn and excluded from main E0",
        "models": sorted(predictions["model"].unique().tolist()),
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "formal_tcn": {
            "history_hours": HISTORY,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "scheduler": "ReduceLROnPlateau",
            "gradient_clipping_norm": GRADIENT_CLIP_NORM,
            "checkpoint_selection": "validation MAE",
            "test_used_for_selection": False,
        },
    }
    (OUT / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "formal_tcn_runs": 15}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize()
        return
    frame = pd.read_parquet(FEATURE_PATH)
    jobs = (
        [(h, s) for h in HORIZONS for s in SEEDS]
        if args.all
        else [(args.horizon, args.seed)]
    )
    if any(h is None or s is None for h, s in jobs):
        parser.error("Specify --all or both --horizon and --seed")
    for horizon, seed in jobs:
        pred_path = OUT / "runs" / f"predictions_h{horizon}_seed{seed}.parquet"
        curve_path = OUT / "runs" / f"curves_h{horizon}_seed{seed}.csv"
        manifest_path = OUT / "runs" / f"manifest_h{horizon}_seed{seed}.json"
        if pred_path.exists() and curve_path.exists() and manifest_path.exists():
            print(f"SKIP completed TCN h{horizon} seed{seed}", flush=True)
            continue
        predictions, curves, manifest = train_one(frame, horizon, seed)
        predictions.to_parquet(pred_path, index=False)
        pd.DataFrame(curves).to_csv(curve_path, index=False)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"DONE formal TCN h{horizon} seed{seed} epochs={manifest['epochs_run']} "
            f"best_epoch={manifest['best_epoch']} val={manifest['best_validation_MAE']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
