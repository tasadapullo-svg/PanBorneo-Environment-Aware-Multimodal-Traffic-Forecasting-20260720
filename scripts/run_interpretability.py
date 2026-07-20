from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import run_e6_ablations as e6


PACKAGE = Path(__file__).resolve().parents[1]
FEATURE_PATH = PACKAGE / "01_final_data" / "final_model_features_frozen.parquet"
LABEL_PATH = PACKAGE / "01_final_data" / "sample_labels_long.parquet"
E5 = PACKAGE / "04_E1_E5" / "E5_full_fusion"
OUT = PACKAGE / "11_interpretability"
HORIZONS = (1, 3, 6)
SEEDS = (42, 2025, 20260623, 20260715, 20260718)
REPETITIONS = 10

FEATURE_GROUPS = {
    "Traffic": e6.TRAFFIC,
    "Reliability": e6.RELIABILITY,
    "Calendar": e6.CALENDAR,
    "Rainfall": e6.RAINFALL,
    "Meteorology": e6.METEOROLOGY,
    "Atmospheric": e6.ATMOSPHERIC,
    "Road": e6.ROAD,
}
SCENARIOS = {
    "Overall_test": None,
    "Rain": "S2_rain",
    "Elevated_AOD": "S7_elevated_aod",
    "Rain_elevated_pollution": "S8_rain_elevated_atmospheric_pollution",
}


def group_indices(encoded_names: list[str]) -> dict[str, list[int]]:
    result = {}
    for group, features in FEATURE_GROUPS.items():
        indices = []
        for index, encoded in enumerate(encoded_names):
            if encoded in features:
                indices.append(index)
            elif encoded.startswith("node_id_") and group == "Traffic":
                indices.append(index)
            elif any(encoded.startswith(f"{column}_") for column in features if column in e6.CATEGORICAL):
                indices.append(index)
        result[group] = sorted(set(indices))
        if not result[group]:
            raise ValueError(f"No encoded columns found for {group}")
    covered = sorted(set(index for indices in result.values() for index in indices))
    if covered != list(range(len(encoded_names))):
        missing = [encoded_names[index] for index in set(range(len(encoded_names))) - set(covered)]
        raise ValueError(f"Encoded features not assigned to a group: {missing}")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(FEATURE_PATH)
    labels = pd.read_parquet(LABEL_PATH)
    frozen_predictions = pd.read_parquet(E5 / "predictions.parquet")
    importance_rows = []
    audit_rows = []

    for horizon in HORIZONS:
        rows, design, preprocessing = e6.prepare(frame, horizon, e6.FULL)
        test_mask = rows["split"].eq("test").to_numpy()
        X_test = design[test_mask]
        test_rows = rows.loc[test_mask].reset_index(drop=True)
        test_ids = test_rows[f"sample_id_h{horizon}"].astype(str)
        horizon_labels = labels.loc[labels["horizon"].eq(horizon)].set_index("sample_id")
        encoded_names = preprocessing["encoded_features"]
        indices_by_group = group_indices(encoded_names)

        for seed in SEEDS:
            checkpoint = E5 / "best_checkpoint" / f"E5_full_fusion_h{horizon}_seed{seed}.json"
            model = xgb.XGBRegressor()
            model.load_model(checkpoint)
            model.set_params(device="cuda")
            baseline_prediction = model.predict(X_test)
            frozen = frozen_predictions.loc[
                frozen_predictions["horizon"].eq(horizon) & frozen_predictions["seed"].eq(seed)
            ].set_index("sample_id").loc[test_ids]
            difference = np.abs(baseline_prediction - frozen["y_pred"].to_numpy(float))
            audit_rows.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "sample_count": len(test_ids),
                    "max_absolute_prediction_difference": float(difference.max()),
                    "mean_absolute_prediction_difference": float(difference.mean()),
                    "status": "PASS" if np.allclose(baseline_prediction, frozen["y_pred"], rtol=0, atol=1e-5) else "FAIL",
                }
            )
            y_true = frozen["y_true"].to_numpy(float)
            for scenario_number, (scenario, label_column) in enumerate(SCENARIOS.items()):
                if label_column is None:
                    selection = np.ones(len(test_ids), dtype=bool)
                else:
                    selection = horizon_labels.loc[test_ids, label_column].fillna(False).to_numpy(bool)
                if not selection.any():
                    continue
                X_scenario = X_test[selection]
                y_scenario = y_true[selection]
                baseline_scenario = baseline_prediction[selection]
                baseline_mae = float(np.abs(baseline_scenario - y_scenario).mean())
                for group_number, (group, column_indices) in enumerate(indices_by_group.items()):
                    for repetition in range(REPETITIONS):
                        rng = np.random.default_rng(
                            seed + 10_000 * horizon + 100 * scenario_number + 10 * group_number + repetition
                        )
                        permutation = rng.permutation(len(X_scenario))
                        permuted = X_scenario.copy()
                        permuted[:, column_indices] = X_scenario[permutation][:, column_indices]
                        permuted_prediction = model.predict(permuted)
                        permuted_mae = float(np.abs(permuted_prediction - y_scenario).mean())
                        importance_rows.append(
                            {
                                "model": "E5_full_fusion_xgboost",
                                "horizon": horizon,
                                "seed": seed,
                                "scenario": scenario,
                                "feature_group": group,
                                "repetition": repetition + 1,
                                "sample_count": int(selection.sum()),
                                "encoded_columns_permuted": len(column_indices),
                                "baseline_MAE": baseline_mae,
                                "permuted_MAE": permuted_mae,
                                "delta_MAE": permuted_mae - baseline_mae,
                                "importance_convention": "positive delta means the group was useful",
                            }
                        )
            print(f"DONE importance h{horizon} seed{seed}", flush=True)

    importance = pd.DataFrame(importance_rows)
    audit = pd.DataFrame(audit_rows)
    importance.to_csv(OUT / "group_permutation_importance.csv", index=False)
    summary = importance.groupby(
        ["model", "horizon", "scenario", "feature_group"], as_index=False
    ).agg(
        seed_count=("seed", "nunique"),
        repetitions=("repetition", "size"),
        sample_count=("sample_count", "first"),
        delta_MAE_mean=("delta_MAE", "mean"),
        delta_MAE_std=("delta_MAE", "std"),
        delta_MAE_median=("delta_MAE", "median"),
    )
    summary["rank_within_horizon_scenario"] = summary.groupby(
        ["horizon", "scenario"]
    )["delta_MAE_mean"].rank(ascending=False, method="min")
    summary.to_csv(OUT / "feature_importance_by_scenario.csv", index=False)
    audit.to_csv(OUT / "checkpoint_prediction_reproduction_audit.csv", index=False)
    if not audit["status"].eq("PASS").all():
        raise ValueError("E5 checkpoint predictions did not reproduce frozen predictions")
    config = {
        "method": "group permutation importance",
        "model": "frozen E5 full-fusion XGBoost",
        "feature_groups": FEATURE_GROUPS,
        "scenarios": SCENARIOS,
        "repetitions_per_seed_horizon_scenario_group": REPETITIONS,
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "test_used_for_feature_selection": False,
        "SHAP": "not run; optional supplementary analysis",
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "SHAP_NOT_RUN.md").write_text(
        "# SHAP status\n\nSHAP was optional and was not used. The required group permutation importance was completed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "importance_rows": len(importance), "audit_runs": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
