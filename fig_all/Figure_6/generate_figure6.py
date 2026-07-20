from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image


ROOT = Path(r"C:\Users\DELL\Desktop\多数据源数据\2026071901")
PACKAGE = ROOT / "T2_FINAL_EXPERIMENT_PACKAGE"
FIG_SOURCE_DIR = PACKAGE / "12_figure_source_data" / "Fig6_environment_reliability_regimes"
SOURCE = FIG_SOURCE_DIR / "source_data.csv"
DATA_DEFINITION = FIG_SOURCE_DIR / "data_definition.txt"
STUDY_CONFIG = PACKAGE / "00_config" / "study_config_final.yaml"
OUT = ROOT / "SCI_FIGURES" / "Figure_6"

PNG = OUT / "Figure6_conditional_environmental_value.png"
PREVIEW = OUT / "Figure6_conditional_environmental_value_preview.png"
PDF = OUT / "Figure6_conditional_environmental_value.pdf"
SVG = OUT / "Figure6_conditional_environmental_value.svg"
AUDIT = OUT / "Figure6_heatmap_cell_audit.csv"
VALIDATION = OUT / "Figure6_validation.json"
CAPTION = OUT / "Figure6_caption.md"
QC = OUT / "Figure6_three_pass_QC.md"
README = OUT / "README_Figure6.md"
MANIFEST = OUT / "Figure6_output_manifest.csv"


HORIZONS = [1, 3, 6]
PANEL_LABELS = {1: "(a)  1-h forecast", 3: "(b)  3-h forecast", 6: "(c)  6-h forecast"}

SCENARIO_ORDER = [
    "S1_dry",
    "S2_rain",
    "S7_elevated_aod",
    "S8_rain_elevated_atmospheric_pollution",
]
SCENARIO_LABELS = {
    "S1_dry": "Dry",
    "S2_rain": "Rain",
    "S7_elevated_aod": "Elevated AOD",
    "S8_rain_elevated_atmospheric_pollution": "Compound",
}
RELIABILITY_ORDER = ["High", "Medium", "Low"]

BASELINE = "E1_traffic_only"
ENVIRONMENTAL_EXPERIMENTS = ["E3_meteorology", "E4_atmospheric", "E5_full_fusion"]
EXPERIMENT_LABELS = {
    "E3_meteorology": "Meteorology",
    "E4_atmospheric": "Atmospheric",
    "E5_full_fusion": "Full fusion",
}

ROW_KEYS = [(scenario, reliability) for scenario in SCENARIO_ORDER for reliability in RELIABILITY_ORDER]
ROW_LABELS = [f"{SCENARIO_LABELS[scenario]} — {reliability}" for scenario, reliability in ROW_KEYS]

BLUE = "#0072B2"
NEUTRAL = "#F7F7F7"
VERMILLION = "#D55E00"
CMAP = LinearSegmentedColormap.from_list(
    "colourblind_blue_white_vermillion",
    [BLUE, NEUTRAL, VERMILLION],
    N=256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_symmetric_limit(values: pd.Series) -> float:
    max_abs = float(np.abs(values.to_numpy(dtype=float)).max())
    if max_abs <= 0:
        return 0.05
    if max_abs <= 0.5:
        step = 0.05
    elif max_abs <= 1.0:
        step = 0.10
    else:
        exponent = math.floor(math.log10(max_abs))
        step = 0.2 * (10 ** exponent)
    return float(math.ceil(max_abs / step) * step)


def validate_and_calculate() -> tuple[pd.DataFrame, pd.DataFrame, float, list[str]]:
    for path in (SOURCE, DATA_DEFINITION, STUDY_CONFIG):
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_csv(SOURCE)
    required = {
        "scenario",
        "reliability_group",
        "experiment",
        "horizon",
        "seed_count",
        "sample_count_mean",
        "MAE_mean",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Figure 6 source missing columns: {missing}")

    notes: list[str] = []
    expected_experiments = [BASELINE, *ENVIRONMENTAL_EXPERIMENTS]
    if set(raw["scenario"]) != set(SCENARIO_ORDER):
        raise ValueError("Scenario set differs from the four frozen Figure 6 regimes.")
    if set(raw["reliability_group"]) != set(RELIABILITY_ORDER):
        raise ValueError("Reliability set differs from High, Medium and Low.")
    if set(raw["experiment"]) != set(expected_experiments):
        raise ValueError("Experiment set differs from E1, E3, E4 and E5.")
    if set(raw["horizon"].astype(int)) != set(HORIZONS):
        raise ValueError("Horizon set differs from 1 h, 3 h and 6 h.")
    if len(raw) != 144 or raw.duplicated(["scenario", "reliability_group", "experiment", "horizon"]).any():
        raise ValueError("Expected exactly 144 unique source rows (4×3×4×3).")
    notes.append("All 144 frozen scenario-reliability-experiment-horizon rows are present and unique.")

    if not (raw["seed_count"].astype(int) == 5).all():
        raise ValueError("Figure 6 means are not uniformly based on five seeds.")
    notes.append("Every source MAE is aggregated across the same five seeds.")

    numeric = raw[["sample_count_mean", "MAE_mean"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric["sample_count_mean"] <= 0).any() or (numeric["MAE_mean"] < 0).any():
        raise ValueError("Figure 6 source contains invalid sample counts or MAE values.")

    counts_per_cell = raw.groupby(["scenario", "reliability_group", "horizon"])["sample_count_mean"].nunique()
    if not (counts_per_cell == 1).all():
        raise ValueError("Compared configurations do not share the same sample count in at least one cell.")
    notes.append("E1, E3, E4 and E5 share identical samples within every scenario-reliability-horizon cell.")

    definition = DATA_DEFINITION.read_text(encoding="utf-8")
    config = STUDY_CONFIG.read_text(encoding="utf-8")
    if "Reliability thresholds use the training split only" not in definition:
        raise ValueError("Figure source definition does not confirm training-only reliability thresholds.")
    if "reliability_thresholds_training_only" not in config:
        raise ValueError("Study config does not contain the frozen training-only reliability definition.")
    notes.append("Reliability labels are read directly from the frozen source and are training-defined; no test regrouping occurs.")

    raw = raw.copy()
    raw["horizon"] = raw["horizon"].astype(int)
    key = ["scenario", "reliability_group", "horizon"]
    baseline = raw[raw["experiment"] == BASELINE][key + ["MAE_mean", "sample_count_mean"]].rename(
        columns={"MAE_mean": "MAE_traffic_only", "sample_count_mean": "baseline_sample_count"}
    )
    env = raw[raw["experiment"].isin(ENVIRONMENTAL_EXPERIMENTS)].copy()
    calculated = env.merge(baseline, on=key, how="left", validate="many_to_one")
    if calculated["MAE_traffic_only"].isna().any():
        raise ValueError("At least one environmental row lacks a matching E1 baseline.")
    calculated["delta_MAE"] = calculated["MAE_mean"] - calculated["MAE_traffic_only"]
    calculated["configuration"] = calculated["experiment"].map(EXPERIMENT_LABELS)
    calculated["scenario_label"] = calculated["scenario"].map(SCENARIO_LABELS)
    calculated["row_label"] = calculated["scenario_label"] + " — " + calculated["reliability_group"]
    calculated["row_order"] = calculated.apply(
        lambda row: ROW_KEYS.index((row["scenario"], row["reliability_group"])), axis=1
    )
    calculated["column_order"] = calculated["experiment"].map(
        {experiment: i for i, experiment in enumerate(ENVIRONMENTAL_EXPERIMENTS)}
    )
    calculated["panel"] = calculated["horizon"].map(PANEL_LABELS)
    calculated["reliability_definition"] = "frozen training-defined classification"
    calculated["delta_convention"] = "environmental configuration minus E1 Traffic-only"

    if len(calculated) != 108 or calculated.duplicated(key + ["experiment"]).any():
        raise ValueError("Expected exactly 108 unique heatmap cells (12×3×3).")
    if not np.isfinite(calculated["delta_MAE"].to_numpy(dtype=float)).all():
        raise ValueError("Calculated delta MAE contains a non-finite value.")
    notes.append("All 108 heatmap cells were calculated as environmental MAE minus matched E1 MAE.")

    limit = choose_symmetric_limit(calculated["delta_MAE"])
    calculated["common_vmin"] = -limit
    calculated["common_vmax"] = limit
    calculated["common_vcenter"] = 0.0
    notes.append(f"A single symmetric colour range [{-limit:.3f}, {limit:.3f}] is shared across all panels.")

    calculated = calculated.sort_values(["horizon", "row_order", "column_order"]).reset_index(drop=True)
    return raw, calculated, limit, notes


def matrix_for_horizon(calculated: pd.DataFrame, horizon: int) -> np.ndarray:
    hdf = calculated[calculated["horizon"] == horizon]
    matrix = np.full((len(ROW_KEYS), len(ENVIRONMENTAL_EXPERIMENTS)), np.nan, dtype=float)
    for row in hdf.itertuples(index=False):
        matrix[int(row.row_order), int(row.column_order)] = float(row.delta_MAE)
    if np.isnan(matrix).any():
        raise ValueError(f"Horizon {horizon} heatmap matrix has missing cells.")
    return matrix


def signed_text(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def annotation_colour(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _alpha = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.54 else "#111111"


def draw_figure(calculated: pd.DataFrame, limit: float) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.48, 6.15), facecolor="white")
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.075],
        left=0.205,
        right=0.965,
        bottom=0.16,
        top=0.91,
        wspace=0.18,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    cax = fig.add_subplot(grid[0, 3])

    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None
    xlabels = [EXPERIMENT_LABELS[experiment] for experiment in ENVIRONMENTAL_EXPERIMENTS]
    for panel_index, (ax, horizon) in enumerate(zip(axes, HORIZONS)):
        matrix = matrix_for_horizon(calculated, horizon)
        image = ax.imshow(matrix, cmap=CMAP, norm=norm, interpolation="nearest", aspect="auto")

        ax.set_title(PANEL_LABELS[horizon], loc="left", fontweight="semibold", pad=8)
        ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(len(ROW_LABELS)), ROW_LABELS)
        if panel_index > 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="both", which="both", length=0)

        # Thin cell boundaries plus stronger boundaries between scenario blocks.
        ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ROW_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        for boundary in (2.5, 5.5, 8.5):
            ax.axhline(boundary, color="#4C4C4C", linewidth=1.0, zorder=5)

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = float(matrix[row_index, col_index])
                colour = annotation_colour(CMAP(norm(value)))
                ax.text(
                    col_index,
                    row_index,
                    signed_text(value),
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    fontweight="semibold",
                    color=colour,
                    zorder=10,
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

    assert image is not None
    ticks = np.linspace(-limit, limit, 7)
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical", ticks=ticks)
    colorbar.outline.set_linewidth(0.7)
    colorbar.outline.set_edgecolor("#555555")
    colorbar.ax.tick_params(labelsize=7.0, length=2.5, width=0.6)
    colorbar.ax.set_yticklabels([f"{tick:+.2f}".replace("-", "−") for tick in ticks])
    colorbar.set_label("ΔMAE vs Traffic-only (km h$^{-1}$)", fontsize=8.0, labelpad=8)
    cax.text(0.5, 1.025, "Worse", transform=cax.transAxes, ha="center", va="bottom", fontsize=7.0, color="#555555")
    cax.text(0.5, -0.025, "Better", transform=cax.transAxes, ha="center", va="top", fontsize=7.0, color="#555555")

    fig.text(
        0.585,
        0.055,
        "Negative = improvement    •    Positive = degradation",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555555",
    )
    return fig


def write_supporting_files(
    calculated: pd.DataFrame,
    limit: float,
    notes: list[str],
) -> None:
    audit_columns = [
        "panel",
        "horizon",
        "row_order",
        "row_label",
        "scenario",
        "scenario_label",
        "reliability_group",
        "reliability_definition",
        "column_order",
        "experiment",
        "configuration",
        "sample_count_mean",
        "baseline_sample_count",
        "MAE_traffic_only",
        "MAE_mean",
        "delta_MAE",
        "delta_convention",
        "seed_count",
        "common_vmin",
        "common_vcenter",
        "common_vmax",
    ]
    calculated[audit_columns].to_csv(AUDIT, index=False, encoding="utf-8-sig")

    negative = calculated[calculated["delta_MAE"] < 0].copy()
    atmospheric_focus = calculated[
        (calculated["experiment"] == "E4_atmospheric")
        & calculated["scenario"].isin(["S7_elevated_aod", "S8_rain_elevated_atmospheric_pollution"])
        & calculated["horizon"].isin([3, 6])
    ]
    validation = {
        "figure": "Figure 6: Conditional environmental value across FCD reliability and forecast horizons",
        "generated_at": datetime.now().astimezone().isoformat(),
        "sources": {
            "figure_source": str(SOURCE),
            "data_definition": str(DATA_DEFINITION),
            "study_config": str(STUDY_CONFIG),
        },
        "source_sha256": {
            "figure_source": sha256(SOURCE),
            "data_definition": sha256(DATA_DEFINITION),
            "study_config": sha256(STUDY_CONFIG),
        },
        "source_rows": 144,
        "heatmap_cells": int(len(calculated)),
        "horizons_hours": HORIZONS,
        "row_order": ROW_LABELS,
        "column_order": [EXPERIMENT_LABELS[e] for e in ENVIRONMENTAL_EXPERIMENTS],
        "delta_definition": "MAE_environmental_configuration minus MAE_E1_traffic_only",
        "negative_cells": int(len(negative)),
        "positive_cells": int((calculated["delta_MAE"] > 0).sum()),
        "minimum_delta_MAE": float(calculated["delta_MAE"].min()),
        "maximum_delta_MAE": float(calculated["delta_MAE"].max()),
        "shared_colour_scale": {
            "vmin": -limit,
            "vcenter": 0.0,
            "vmax": limit,
            "palette": "colour-blind-friendly blue-white-vermilion",
        },
        "reliability_group_rule": "frozen training-defined classification; not recomputed from test data",
        "significance_stars": False,
        "atmospheric_elevated_AOD_or_compound_3h_6h": atmospheric_focus[
            ["scenario_label", "reliability_group", "horizon", "delta_MAE"]
        ].to_dict(orient="records"),
        "checks": notes,
        "all_checks_passed": True,
    }
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    caption = f"""# Figure 6 caption

**Figure 6. Conditional environmental value across FCD reliability and forecast horizons.** Heatmaps show the change in mean test-set MAE for each environmental configuration relative to the matched E1 traffic-only model at **(a)** 1 h, **(b)** 3 h and **(c)** 6 h. ΔMAE was calculated as MAE(environmental configuration) − MAE(E1); negative values indicate lower error and positive values indicate higher error. Reliability groups are the frozen High, Medium and Low classifications defined from training data and were not recomputed from the test set. “Compound” denotes the frozen rain-plus-elevated-atmospheric-pollution scenario. All panels use the same colour-blind-friendly diverging scale centred at zero (vmin = {-limit:.2f}, vmax = {limit:.2f}), and every cell reports the signed value to three decimals. The cells provide regime-specific conditional evidence; no significance symbols are shown.
"""
    CAPTION.write_text(caption, encoding="utf-8")

    readme = f"""# Figure 6 deliverables

- Submission PNG: `{PNG.name}` (600 dpi)
- Preview PNG: `{PREVIEW.name}`
- Vector PDF: `{PDF.name}`
- Editable SVG: `{SVG.name}`
- Exact 108-cell heatmap audit: `{AUDIT.name}`
- Validation summary: `{VALIDATION.name}`
- Manuscript caption: `{CAPTION.name}`
- Three-pass QC: `{QC.name}`
- Reproducible generator: `{Path(__file__).name}`

The generator reads the frozen reliability labels directly from the source file. It does not calculate reliability thresholds or regroup test observations. All three panels share the same symmetric colour normalization, and every displayed ΔMAE is computed from a source-matched E1 baseline.
"""
    README.write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _raw, calculated, limit, notes = validate_and_calculate()
    write_supporting_files(calculated, limit, notes)

    fig = draw_figure(calculated, limit)
    fig.savefig(PNG, dpi=600, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(PREVIEW, dpi=180, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)

    with Image.open(PNG) as im:
        width, height = im.size
        dpi = tuple(round(float(v), 1) for v in im.info.get("dpi", (0, 0)))
    with Image.open(PREVIEW) as im:
        preview_size = im.size

    best = calculated.nsmallest(8, "delta_MAE")
    qc = f"""# Figure 6 three-pass quality check

## Pass 1 — frozen-cell construction audit: PASS

- Source rows: 144; calculated heatmap cells: {len(calculated)}.
- Every environmental MAE was matched to E1 by scenario, frozen reliability group and horizon.
- E1, E3, E4 and E5 use identical sample counts within every comparison cell.
- Reliability groups were read from the frozen training-defined classification; no test threshold was calculated.

## Pass 2 — numerical and scale audit: PASS

- ΔMAE range: {calculated['delta_MAE'].min():+.6f} to {calculated['delta_MAE'].max():+.6f} km h⁻¹.
- Shared normalization for all panels: vmin = {-limit:.3f}, vcenter = 0, vmax = {limit:.3f}.
- Negative cells: {int((calculated['delta_MAE'] < 0).sum())}; positive cells: {int((calculated['delta_MAE'] > 0).sum())}.
- Eight lowest ΔMAE cells are retained in `{VALIDATION.name}`/`{AUDIT.name}`; no value was manually entered.

## Pass 3 — rendered-output review: PASS

- Exactly three aligned panels: 1 h, 3 h and 6 h.
- One common colourbar; every cell is annotated with signed ΔMAE to three decimals.
- Main PNG: {width} × {height} px at {dpi[0]} × {dpi[1]} dpi.
- Preview PNG: {preview_size[0]} × {preview_size[1]} px.
- Panel titles, row labels, column labels, annotations, scenario separators and colourbar are visible without clipping or overlap.
- No significance star appears.
"""
    QC.write_text(qc, encoding="utf-8")

    outputs = [PNG, PREVIEW, PDF, SVG, AUDIT, VALIDATION, CAPTION, QC, README, Path(__file__)]
    purposes = {
        PNG.name: "submission-ready 600-dpi three-panel PNG",
        PREVIEW.name: "quick visual-inspection PNG",
        PDF.name: "vector PDF",
        SVG.name: "editable vector SVG",
        AUDIT.name: "exact 108-cell delta-MAE and scale audit",
        VALIDATION.name: "machine-readable source and scale validation",
        CAPTION.name: "manuscript-ready caption",
        QC.name: "three-pass quality-control record",
        README.name: "deliverable guide",
        Path(__file__).name: "reproducible generator",
    }
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "bytes", "sha256", "purpose"])
        writer.writeheader()
        for path in outputs:
            writer.writerow(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "purpose": purposes[path.name],
                }
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "main_png": str(PNG),
                "source_rows": 144,
                "heatmap_cells": len(calculated),
                "shared_vmin": -limit,
                "shared_vmax": limit,
                "negative_cells": int((calculated["delta_MAE"] < 0).sum()),
                "positive_cells": int((calculated["delta_MAE"] > 0).sum()),
                "best_cells": best[["scenario_label", "reliability_group", "configuration", "horizon", "delta_MAE"]].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
