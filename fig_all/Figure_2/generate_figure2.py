from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / "T2_FINAL_EXPERIMENT_PACKAGE"
FINAL_DATA = PACKAGE / "01_final_data"
OUT = PROJECT / "SCI_FIGURES" / "Figure_2"

FINAL_PNG = OUT / "Figure2_spatiotemporal_traffic_environmental_dynamics_full_period.png"
PREVIEW_PNG = OUT / "Figure2_spatiotemporal_traffic_environmental_dynamics_full_period_preview.png"
AUDIT_CSV = OUT / "Figure2_plot_audit_full_period.csv"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema.names)


def locate_frozen_panel() -> Path:
    required = {
        "node_id",
        "timestamp_local",
        "split",
        "corridor_km",
        "current_speed_input",
        "missing_mask",
        "precipitation",
        "aerosol_optical_depth",
    }
    candidates = []
    for path in FINAL_DATA.rglob("*.parquet"):
        if required.issubset(parquet_columns(path)):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No frozen model-ready parquet contains the required traffic and environmental fields")
    preferred = [path for path in candidates if path.name == "final_model_features_frozen.parquet"]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Ambiguous frozen model-ready panels: {candidates}")


def locate_scenario_labels() -> Path:
    required = {
        "node_id",
        "timestamp_local",
        "split",
        "S2_rain",
        "S7_elevated_aod",
        "S8_rain_elevated_atmospheric_pollution",
    }
    candidates = []
    for path in FINAL_DATA.rglob("*.parquet"):
        if required.issubset(parquet_columns(path)):
            candidates.append(path)
    preferred = [path for path in candidates if path.name == "scenario_labels_final.parquet"]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Unable to select one frozen scenario-label parquet: {candidates}")


def validate_feature_manifest() -> dict:
    manifest_path = FINAL_DATA / "00_final_freeze" / "feature_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    required = ["current_speed_input", "precipitation", "aerosol_optical_depth"]
    selected = manifest.loc[manifest["feature"].isin(required)].set_index("feature")
    missing = sorted(set(required) - set(selected.index))
    if missing:
        raise AssertionError(f"Feature manifest is missing {missing}")
    for feature in required:
        included = str(selected.loc[feature, "included_as_predictor"]).lower() == "true"
        if not included:
            raise AssertionError(f"{feature} is not recorded as an included predictor")
    return {
        "feature_manifest": str(manifest_path.relative_to(PACKAGE)).replace("\\", "/"),
        "feature_manifest_sha256": sha256(manifest_path),
        "speed_field": "current_speed_input",
        "rainfall_predictor_field": "precipitation",
        "aod_predictor_field": "aerosol_optical_depth",
    }


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    panel_path = locate_frozen_panel()
    scenario_path = locate_scenario_labels()
    feature_info = validate_feature_manifest()
    order_path = FINAL_DATA / "01_fcd_reference" / "node_order_verified.csv"

    columns = [
        "node_id",
        "timestamp_local",
        "split",
        "current_speed_input",
        "missing_mask",
        "corridor_km",
        "precipitation",
        "corridor_mean_rainfall",
        "aerosol_optical_depth",
    ]
    panel = pd.read_parquet(panel_path, columns=columns).copy()
    panel["timestamp_local"] = pd.to_datetime(panel["timestamp_local"], errors="raise")

    scenario_columns = [
        "node_id",
        "timestamp_local",
        "split",
        "S2_rain",
        "S7_elevated_aod",
        "S8_rain_elevated_atmospheric_pollution",
    ]
    scenario = pd.read_parquet(scenario_path, columns=scenario_columns).copy()
    scenario["timestamp_local"] = pd.to_datetime(scenario["timestamp_local"], errors="raise")

    order = pd.read_csv(order_path)[["node_id", "node_order_verified", "corridor_km"]].copy()
    order["node_order_verified"] = pd.to_numeric(order["node_order_verified"], errors="raise").astype(int)
    order["corridor_km"] = pd.to_numeric(order["corridor_km"], errors="raise")
    order = order.sort_values("node_order_verified").reset_index(drop=True)

    if panel.duplicated(["node_id", "timestamp_local"]).any():
        raise AssertionError("Duplicate node-hour rows in the frozen study-period panel")
    if scenario.duplicated(["node_id", "timestamp_local"]).any():
        raise AssertionError("Duplicate node-hour rows in the frozen study-period scenario labels")
    if panel["node_id"].nunique() != 51:
        raise AssertionError("The frozen study-period panel does not contain 51 nodes")
    node_hours = panel.groupby("timestamp_local")["node_id"].nunique()
    if not node_hours.eq(51).all():
        raise AssertionError("The frozen study-period node × time grid is incomplete")
    timestamps = pd.Index(sorted(panel["timestamp_local"].unique()))
    if len(timestamps) > 1:
        deltas = pd.Series(timestamps).diff().dropna()
        if not deltas.eq(pd.Timedelta(hours=1)).all():
            raise AssertionError("The frozen study-period timestamps are not a continuous hourly index")

    panel = panel.merge(order, on="node_id", suffixes=("", "_verified"), validate="many_to_one")
    if not np.allclose(panel["corridor_km"], panel["corridor_km_verified"], atol=1e-10, rtol=0):
        raise AssertionError("corridor_km disagrees with node_order_verified.csv")
    panel = panel.drop(columns="corridor_km_verified")
    order_from_panel = (
        panel[["node_id", "node_order_verified", "corridor_km"]]
        .drop_duplicates()
        .sort_values("corridor_km")
        .reset_index(drop=True)
    )
    if not np.array_equal(order_from_panel["node_id"], order["node_id"]):
        raise AssertionError("corridor_km order disagrees with verified node order")

    merged = panel.merge(
        scenario,
        on=["node_id", "timestamp_local", "split"],
        how="left",
        validate="one_to_one",
    )
    if merged[["S2_rain", "S7_elevated_aod", "S8_rain_elevated_atmospheric_pollution"]].isna().any().any():
        raise AssertionError("Frozen scenario labels failed to join to all study-period node-hours")

    for column in [
        "current_speed_input",
        "missing_mask",
        "corridor_km",
        "precipitation",
        "corridor_mean_rainfall",
        "aerosol_optical_depth",
        "S2_rain",
        "S7_elevated_aod",
        "S8_rain_elevated_atmospheric_pollution",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="raise")

    hourly = merged.groupby("timestamp_local", sort=True).agg(
        corridor_median_precipitation=("precipitation", "median"),
        frozen_corridor_mean_rainfall=("corridor_mean_rainfall", "median"),
        corridor_median_aod=("aerosol_optical_depth", "median"),
        rain_fraction=("S2_rain", "mean"),
        elevated_aod_fraction=("S7_elevated_aod", "mean"),
        compound_fraction=("S8_rain_elevated_atmospheric_pollution", "mean"),
    )
    hourly["scenario_context"] = "None"
    hourly.loc[hourly["elevated_aod_fraction"].ge(0.5), "scenario_context"] = "Elevated AOD"
    hourly.loc[hourly["rain_fraction"].ge(0.5), "scenario_context"] = "Rain"
    hourly.loc[hourly["compound_fraction"].ge(0.5), "scenario_context"] = "Compound"

    merged = merged.merge(hourly.reset_index(), on="timestamp_local", how="left", validate="many_to_one")
    merged["plotted_current_speed"] = merged["current_speed_input"].where(merged["missing_mask"].eq(0), np.nan)
    merged["plotted_speed_missing"] = merged["missing_mask"].ne(0)
    merged["plot_missing_reason"] = np.where(merged["plotted_speed_missing"], "original_FCD_missing", "")
    merged = merged.sort_values(["timestamp_local", "node_order_verified"]).reset_index(drop=True)

    expected_rows = 51 * len(hourly)
    if len(merged) != expected_rows:
        raise AssertionError(f"Unexpected study grid rows: {len(merged)} != {expected_rows}")
    source_speed_na = merged["current_speed_input"].isna()
    if source_speed_na.any():
        # The frozen package documents one causal-start origin without any past
        # observation. Preserve it as unavailable; never fill it for plotting.
        if not merged.loc[source_speed_na, "missing_mask"].eq(1).all():
            raise AssertionError("Unmasked current_speed_input NA detected in the frozen study-period panel")
        if not merged.loc[source_speed_na, "timestamp_local"].eq(hourly.index.min()).all():
            raise AssertionError("current_speed_input NA detected outside the documented causal-start hour")
    if (merged["current_speed_input"] == 0).any():
        raise AssertionError("Zero-valued traffic speed detected; verify before plotting")

    summary = {
        "status": "PASS",
        "model_ready_panel": str(panel_path.relative_to(PACKAGE)).replace("\\", "/"),
        "model_ready_panel_sha256": sha256(panel_path),
        "scenario_labels": str(scenario_path.relative_to(PACKAGE)).replace("\\", "/"),
        "scenario_labels_sha256": sha256(scenario_path),
        "node_order_file": str(order_path.relative_to(PACKAGE)).replace("\\", "/"),
        "node_order_file_sha256": sha256(order_path),
        **feature_info,
        "timestamp_field": "timestamp_local",
        "study_start": str(hourly.index.min()),
        "study_end": str(hourly.index.max()),
        "study_hours": int(len(hourly)),
        "node_count": int(merged["node_id"].nunique()),
        "node_hour_rows": int(len(merged)),
        "corridor_km_min": float(merged["corridor_km"].min()),
        "corridor_km_max": float(merged["corridor_km"].max()),
        "duplicate_node_hours": 0,
        "missing_mask_rows": int(merged["plotted_speed_missing"].sum()),
        "source_speed_na": int(source_speed_na.sum()),
        "source_speed_na_rule": "only masked causal-start origins without a past observation are permitted and remain unplotted",
        "source_speed_zero": int((merged["current_speed_input"] == 0).sum()),
        "plotted_speed_na": int(merged["plotted_current_speed"].isna().sum()),
        "precipitation_na": int(merged["precipitation"].isna().sum()),
        "aod_na": int(merged["aerosol_optical_depth"].isna().sum()),
        "speed_observed_min": float(merged["plotted_current_speed"].min()),
        "speed_observed_max": float(merged["plotted_current_speed"].max()),
        "rainfall_strip": "corridor median of frozen precipitation predictor",
        "aod_strip": "corridor median of frozen aerosol_optical_depth predictor",
        "scenario_shading_source": "frozen S2_rain, S7_elevated_aod, and S8_rain_elevated_atmospheric_pollution labels",
        "scenario_shading_aggregation": "hour is shaded when >=50% of 51 nodes carry the frozen label; precedence Compound > Rain > Elevated AOD",
        "scenario_context_hours": {str(k): int(v) for k, v in hourly["scenario_context"].value_counts().to_dict().items()},
        "interpretation_constraint": "temporal coincidence/association only; no causal claim",
    }
    return merged, hourly, order, summary


def center_edges(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two centers are required")
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = (values[:-1] + values[1:]) / 2.0
    edges[0] = values[0] - (values[1] - values[0]) / 2.0 if lower is None else lower
    edges[-1] = values[-1] + (values[-1] - values[-2]) / 2.0 if upper is None else upper
    return edges


def scenario_runs(classes: list[str]) -> list[tuple[int, int, str]]:
    runs = []
    start = 0
    current = classes[0]
    for index in range(1, len(classes)):
        if classes[index] != current:
            if current != "None":
                runs.append((start, index - 1, current))
            start = index
            current = classes[index]
    if current != "None":
        runs.append((start, len(classes) - 1, current))
    return runs


def make_figure(merged: pd.DataFrame, hourly: pd.DataFrame, order: pd.DataFrame) -> plt.Figure:
    timestamps = pd.DatetimeIndex(hourly.index)
    timestamp_num = mdates.date2num(timestamps.to_pydatetime())
    time_edges = center_edges(timestamp_num)

    ordered_nodes = order.sort_values("node_order_verified")
    node_ids = ordered_nodes["node_id"].tolist()
    corridor_km = ordered_nodes["corridor_km"].to_numpy(dtype=float)
    corridor_edges = center_edges(corridor_km, lower=float(corridor_km.min()), upper=float(corridor_km.max()))

    speed = (
        merged.pivot(index="node_id", columns="timestamp_local", values="plotted_current_speed")
        .reindex(index=node_ids, columns=timestamps)
        .to_numpy(dtype=float)
    )
    if speed.shape != (51, len(timestamps)):
        raise AssertionError(f"Unexpected heatmap matrix shape: {speed.shape}")

    rainfall = hourly["corridor_median_precipitation"].to_numpy(dtype=float)
    aod = hourly["corridor_median_aod"].to_numpy(dtype=float)
    context = hourly["scenario_context"].astype(str).tolist()

    fig = plt.figure(figsize=(7.08, 5.70), facecolor="white")
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 0.028],
        height_ratios=[0.38, 0.38, 4.35],
        left=0.135,
        right=0.925,
        bottom=0.185,
        top=0.975,
        hspace=0.075,
        wspace=0.075,
    )
    ax_rain = fig.add_subplot(grid[0, 0])
    ax_aod = fig.add_subplot(grid[1, 0], sharex=ax_rain)
    ax_speed = fig.add_subplot(grid[2, 0], sharex=ax_rain)
    cax_rain = fig.add_subplot(grid[0, 1])
    cax_aod = fig.add_subplot(grid[1, 1])
    cax_speed = fig.add_subplot(grid[2, 1])

    rain_cmap = mpl.colormaps["Blues"].copy()
    rain_cmap.set_bad("#BDBDBD")
    aod_cmap = mpl.colormaps["Purples"].copy()
    aod_cmap.set_bad("#BDBDBD")
    speed_cmap = mpl.colormaps["cividis"].copy()
    speed_cmap.set_bad("#BDBDBD")

    rain_max = max(1.0, float(np.nanmax(rainfall)))
    aod_max = max(0.1, float(np.nanmax(aod)))
    rain_mesh = ax_rain.pcolormesh(
        time_edges,
        np.array([0.0, 1.0]),
        rainfall[np.newaxis, :],
        cmap=rain_cmap,
        norm=Normalize(vmin=0.0, vmax=rain_max),
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    aod_mesh = ax_aod.pcolormesh(
        time_edges,
        np.array([0.0, 1.0]),
        aod[np.newaxis, :],
        cmap=aod_cmap,
        norm=Normalize(vmin=0.0, vmax=aod_max),
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    speed_mesh = ax_speed.pcolormesh(
        time_edges,
        corridor_edges,
        speed,
        cmap=speed_cmap,
        norm=Normalize(vmin=20.0, vmax=100.0),
        shading="flat",
        rasterized=True,
        zorder=1,
    )

    scenario_colors = {
        "Rain": "#5B9BD5",
        "Elevated AOD": "#7A6F8E",
        "Compound": "#9A6A45",
    }
    for start, end, label in scenario_runs(context):
        left = time_edges[start]
        right = time_edges[end + 1]
        for axis in (ax_rain, ax_aod, ax_speed):
            axis.axvspan(left, right, facecolor=scenario_colors[label], alpha=0.065, linewidth=0, zorder=3)

    day_boundaries = pd.date_range(timestamps.min().normalize() + pd.Timedelta(days=1), timestamps.max().normalize(), freq="1D")
    for date in day_boundaries:
        x = mdates.date2num(date.to_pydatetime())
        for axis in (ax_rain, ax_aod, ax_speed):
            axis.axvline(x, color="white", linewidth=0.35, alpha=0.55, zorder=4)

    for axis in (ax_rain, ax_aod):
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#6B7280")

    ax_rain.set_ylabel("Rainfall\n(mm h$^{-1}$)", rotation=0, ha="right", va="center", labelpad=8)
    ax_aod.set_ylabel("AOD\n(unitless)", rotation=0, ha="right", va="center", labelpad=8)

    rain_bar = fig.colorbar(rain_mesh, cax=cax_rain)
    rain_bar.locator = MaxNLocator(nbins=3)
    rain_bar.update_ticks()
    rain_bar.ax.tick_params(labelsize=6.2, width=0.45, length=2, pad=1)
    rain_bar.outline.set_linewidth(0.5)
    aod_bar = fig.colorbar(aod_mesh, cax=cax_aod)
    aod_bar.locator = MaxNLocator(nbins=3)
    aod_bar.update_ticks()
    aod_bar.ax.tick_params(labelsize=6.2, width=0.45, length=2, pad=1)
    aod_bar.outline.set_linewidth(0.5)

    ax_speed.set_ylim(float(corridor_km.min()), float(corridor_km.max()))
    ax_speed.set_ylabel("Corridor distance (km)")
    ax_speed.set_yticks([0, 20, 40, 60, 80, float(corridor_km.max())])
    ax_speed.set_yticklabels(["0", "20", "40", "60", "80", f"{corridor_km.max():.1f}"])
    ax_speed.set_xlabel("Study period (local time)", labelpad=7)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax_speed.xaxis.set_major_locator(locator)
    ax_speed.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax_speed.set_xlim(time_edges[0], time_edges[-1])
    for spine in ax_speed.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#374151")

    speed_bar = fig.colorbar(speed_mesh, cax=cax_speed)
    speed_bar.set_label("Traffic speed (km h$^{-1}$)", rotation=90, labelpad=8)
    speed_bar.set_ticks([20, 40, 60, 80, 100])
    speed_bar.ax.tick_params(labelsize=7, width=0.5, length=2.5, pad=2)
    speed_bar.outline.set_linewidth(0.55)

    legend_handles = [
        Patch(facecolor=scenario_colors["Rain"], alpha=0.28, edgecolor="none", label="Rain"),
        Patch(facecolor=scenario_colors["Elevated AOD"], alpha=0.28, edgecolor="none", label="Elevated AOD"),
        Patch(facecolor=scenario_colors["Compound"], alpha=0.28, edgecolor="none", label="Compound"),
        Patch(facecolor="#BDBDBD", edgecolor="#6B7280", linewidth=0.5, label="FCD unavailable"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.035),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.6,
        handleheight=0.8,
        title="Faint shading: frozen scenario membership in ≥50% of nodes",
        title_fontsize=6.8,
    )
    legend._legend_box.align = "center"

    fig.text(
        0.135,
        0.112,
        "Grey heatmap cells indicate original FCD unavailability; causal LOCF values are not shown in those cells.",
        ha="left",
        va="center",
        fontsize=6.3,
        color="#4B5563",
    )
    return fig


def write_audit(merged: pd.DataFrame, summary: dict) -> None:
    audit_columns = [
        "timestamp_local",
        "node_id",
        "node_order_verified",
        "corridor_km",
        "split",
        "current_speed_input",
        "missing_mask",
        "plotted_current_speed",
        "plotted_speed_missing",
        "plot_missing_reason",
        "precipitation",
        "corridor_median_precipitation",
        "frozen_corridor_mean_rainfall",
        "aerosol_optical_depth",
        "corridor_median_aod",
        "S2_rain",
        "S7_elevated_aod",
        "S8_rain_elevated_atmospheric_pollution",
        "rain_fraction",
        "elevated_aod_fraction",
        "compound_fraction",
        "scenario_context",
    ]
    audit = merged[audit_columns].copy()
    audit["timestamp_local"] = audit["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M:%S")
    audit.to_csv(AUDIT_CSV, index=False, encoding="utf-8-sig", na_rep="", float_format="%.10g")
    summary["plot_audit_csv"] = AUDIT_CSV.name
    summary["plot_audit_rows"] = int(len(audit))
    summary["plot_audit_sha256"] = sha256(AUDIT_CSV)


def save_figure(fig: plt.Figure) -> list[dict]:
    fig.savefig(FINAL_PNG, dpi=600, facecolor="white", format="png")
    fig.savefig(PREVIEW_PNG, dpi=200, facecolor="white", format="png")
    records = []
    for path in (FINAL_PNG, PREVIEW_PNG):
        with Image.open(path) as image:
            dpi = image.info.get("dpi")
            if isinstance(dpi, tuple):
                dpi = tuple(float(value) for value in dpi)
            records.append(
                {
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "pixel_width": image.width,
                    "pixel_height": image.height,
                    "dpi": dpi,
                }
            )
    return records


def write_documentation(summary: dict, output_records: list[dict]) -> None:
    (OUT / "Figure2_audit_summary_full_period.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "Figure2_output_manifest_full_period.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "size_bytes", "sha256", "pixel_width", "pixel_height", "dpi"],
        )
        writer.writeheader()
        writer.writerows(output_records)

    caption = f"""# Figure 2 caption

**Figure 2. Spatiotemporal traffic and environmental dynamics during the full frozen study period.** The heatmap shows traffic speed across the verified 0.000–{summary['corridor_km_max']:.3f} km corridor from {summary['study_start']} to {summary['study_end']} (local time). Traffic values are the frozen `current_speed_input` used by E1–E5; cells identified by `missing_mask=1` are displayed separately in grey rather than showing their causal last-observation-carried-forward values. The aligned strips show the corridor-median frozen `precipitation` predictor and corridor-median frozen `aerosol_optical_depth` predictor. Faint shading identifies hours when at least 50% of the 51 nodes carry the corresponding frozen scenario label, with Compound taking precedence over Rain and Elevated AOD for display. Shading reports temporal coincidence only and does not imply that environmental conditions caused the observed traffic pattern.
"""
    (OUT / "Figure2_caption_full_period.md").write_text(caption, encoding="utf-8")

    readme = f"""# Figure 2 delivery

This is one integrated figure with two narrow environmental context strips above one corridor-distance × time speed heatmap. It contains no (a), (b), or (c) panel labels.

## Automatically located frozen sources

- Model-ready panel: `{summary['model_ready_panel']}`
- Frozen scenario labels: `{summary['scenario_labels']}`
- Verified node order: `{summary['node_order_file']}`

## Formal variables

- Timestamp: `timestamp_local`
- Traffic speed used by E1–E5: `current_speed_input`
- Original-input availability: `missing_mask`
- Rainfall predictor: `precipitation`
- AOD predictor: `aerosol_optical_depth`

The plotted rainfall and AOD strips use corridor medians of the actual frozen predictor columns. No scenario threshold was recalculated from the plotted data.

## Frozen study grid

- Period: {summary['study_start']} to {summary['study_end']}
- Hours: {summary['study_hours']}
- Nodes: {summary['node_count']}
- Node-hours: {summary['node_hour_rows']}
- FCD-unavailable node-hours displayed in grey: {summary['missing_mask_rows']}
- Traffic zero values: {summary['source_speed_zero']}
- Environmental missing values: precipitation={summary['precipitation_na']}; AOD={summary['aod_na']}

`{summary['plot_audit_csv']}` contains every plotted node-hour, source value, plotted value, environmental value, and frozen scenario assignment.
"""
    (OUT / "README_Figure2_full_period.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    merged, hourly, order, summary = load_and_validate()
    write_audit(merged, summary)
    figure = make_figure(merged, hourly, order)
    output_records = save_figure(figure)
    plt.close(figure)
    write_documentation(summary, output_records)
    print(json.dumps({"status": "PASS", **summary, "outputs": output_records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
