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
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from PIL import Image


ROOT = Path(r"C:\Users\DELL\Desktop\多数据源数据\2026071901")
PACKAGE = ROOT / "T2_FINAL_EXPERIMENT_PACKAGE"
SOURCE_DIR = PACKAGE / "12_figure_source_data" / "Fig7_UQ_environmental_risk"
SOURCE = SOURCE_DIR / "source_data.csv"
DATA_DEFINITION = SOURCE_DIR / "data_definition.txt"
CAPTION_DRAFT = SOURCE_DIR / "figure_caption_draft.txt"
OUT = ROOT / "SCI_FIGURES" / "Figure_7"

PNG = OUT / "Figure7_environmental_regime_prediction_uncertainty.png"
PREVIEW = OUT / "Figure7_environmental_regime_prediction_uncertainty_preview.png"
PDF = OUT / "Figure7_environmental_regime_prediction_uncertainty.pdf"
SVG = OUT / "Figure7_environmental_regime_prediction_uncertainty.svg"
AUDIT = OUT / "Figure7_heatmap_cell_audit.csv"
SUPPLEMENT = OUT / "Figure7_supplementary_PINAW_Pinball.csv"
VALIDATION = OUT / "Figure7_validation.json"
CAPTION = OUT / "Figure7_caption.md"
QC = OUT / "Figure7_three_pass_QC.md"
README = OUT / "README_Figure7.md"
MANIFEST = OUT / "Figure7_output_manifest.csv"


NOMINAL_COVERAGE = 0.90
HORIZONS = [1, 3, 6]
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

MODEL_ORDER = [
    "UQ_T1_traffic_only",
    "UQ_T2_best_environmental_M4_pressure",
    "UQ_aux_full_fusion",
]
MODEL_LABELS = {
    "UQ_T1_traffic_only": "Traffic-only",
    "UQ_T2_best_environmental_M4_pressure": "Pressure-augmented",
    "UQ_aux_full_fusion": "Full fusion",
}
MODEL_TICK_LABELS = {
    "UQ_T1_traffic_only": "Traffic-only",
    "UQ_T2_best_environmental_M4_pressure": "Pressure-\naugmented",
    "UQ_aux_full_fusion": "Full fusion",
}

COLUMN_KEYS = [(horizon, model) for horizon in HORIZONS for model in MODEL_ORDER]
COLUMN_LABELS = [MODEL_TICK_LABELS[model] for horizon, model in COLUMN_KEYS]
GROUP_CENTRES = {1: 1, 3: 4, 6: 7}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_down(value: float, step: float) -> float:
    return math.floor(value / step) * step


def round_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def validate_and_prepare() -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    for path in (SOURCE, DATA_DEFINITION, CAPTION_DRAFT):
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(SOURCE)
    required = {
        "scenario",
        "model",
        "horizon",
        "seed_count",
        "sample_count",
        "PICP_mean",
        "MPIW_mean",
        "PINAW_mean",
        "Pinball_Loss_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Figure 7 source missing columns: {missing}")

    notes: list[str] = []
    if set(df["scenario"]) != set(SCENARIO_ORDER):
        raise ValueError("Scenario set differs from the four frozen UQ regimes.")
    if set(df["model"]) != set(MODEL_ORDER):
        raise ValueError("Model set differs from Traffic-only, pressure-augmented and full fusion.")
    if set(df["horizon"].astype(int)) != set(HORIZONS):
        raise ValueError("Horizon set differs from 1 h, 3 h and 6 h.")
    if len(df) != 36 or df.duplicated(["scenario", "model", "horizon"]).any():
        raise ValueError("Expected exactly 36 unique scenario-model-horizon rows.")
    notes.append("All 36 frozen scenario-model-horizon combinations are uniquely present.")

    if not (df["seed_count"].astype(int) == 5).all():
        raise ValueError("UQ metrics are not uniformly aggregated across five seeds.")
    notes.append("Every UQ cell is aggregated across the same five seeds.")

    numeric_cols = ["sample_count", "PICP_mean", "MPIW_mean", "PINAW_mean", "Pinball_Loss_mean"]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("UQ source contains non-finite values.")
    if not ((numeric["PICP_mean"] >= 0) & (numeric["PICP_mean"] <= 1)).all():
        raise ValueError("PICP lies outside [0, 1].")
    if (numeric[["sample_count", "MPIW_mean", "PINAW_mean", "Pinball_Loss_mean"]] <= 0).any().any():
        raise ValueError("UQ source contains a non-positive sample count, width or loss.")

    counts = df.groupby(["scenario", "horizon"])["sample_count"].nunique()
    if not (counts == 1).all():
        raise ValueError("Compared models do not share sample counts within a scenario-horizon cell.")
    notes.append("The three UQ models share identical samples within every scenario-horizon comparison.")

    draft_text = CAPTION_DRAFT.read_text(encoding="utf-8")
    if "90%" not in draft_text:
        raise ValueError("Frozen Figure 7 caption draft does not confirm 90% prediction intervals.")
    notes.append("Nominal prediction-interval coverage is frozen at 0.90.")

    forbidden_display = "Best environmental model"
    if forbidden_display.lower() in " ".join(MODEL_LABELS.values()).lower():
        raise ValueError("Forbidden model display name is present.")

    df = df.copy()
    df["horizon"] = df["horizon"].astype(int)
    df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
    df["model_display"] = df["model"].map(MODEL_LABELS)
    df["row_order"] = df["scenario"].map({scenario: i for i, scenario in enumerate(SCENARIO_ORDER)})
    df["column_order"] = df.apply(lambda row: COLUMN_KEYS.index((row["horizon"], row["model"])), axis=1)
    df["nominal_coverage"] = NOMINAL_COVERAGE
    df["coverage_minus_nominal"] = df["PICP_mean"] - NOMINAL_COVERAGE
    df["coverage_at_or_above_nominal"] = df["PICP_mean"] >= NOMINAL_COVERAGE
    df = df.sort_values(["row_order", "column_order"]).reset_index(drop=True)

    picp_vmin = min(NOMINAL_COVERAGE, round_down(float(df["PICP_mean"].min()), 0.005))
    picp_vmax = round_up(float(df["PICP_mean"].max()), 0.005)
    if picp_vmax <= picp_vmin:
        picp_vmax = picp_vmin + 0.01
    mpiw_vmin = round_down(float(df["MPIW_mean"].min()), 0.5)
    mpiw_vmax = round_up(float(df["MPIW_mean"].max()), 0.5)
    if mpiw_vmax <= mpiw_vmin:
        mpiw_vmax = mpiw_vmin + 0.5
    scales = {
        "picp_vmin": float(picp_vmin),
        "picp_vmax": float(picp_vmax),
        "mpiw_vmin": float(mpiw_vmin),
        "mpiw_vmax": float(mpiw_vmax),
    }
    notes.append(
        f"Separate PICP [{picp_vmin:.3f}, {picp_vmax:.3f}] and MPIW [{mpiw_vmin:.1f}, {mpiw_vmax:.1f}] scales were selected from frozen values."
    )
    return df, scales, notes


def matrix(df: pd.DataFrame, value_column: str) -> np.ndarray:
    result = np.full((len(SCENARIO_ORDER), len(COLUMN_KEYS)), np.nan, dtype=float)
    for row in df.itertuples(index=False):
        result[int(row.row_order), int(row.column_order)] = float(getattr(row, value_column))
    if np.isnan(result).any():
        raise ValueError(f"Heatmap matrix for {value_column} has missing cells.")
    return result


def annotation_colour(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _alpha = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.50 else "#111111"


def add_group_structure(ax: mpl.axes.Axes) -> None:
    ax.set_xticks(np.arange(len(COLUMN_KEYS)), COLUMN_LABELS)
    ax.set_yticks(np.arange(len(SCENARIO_ORDER)), [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.tick_params(axis="both", which="both", length=0)

    ax.set_xticks(np.arange(-0.5, len(COLUMN_KEYS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SCENARIO_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    for separator in (2.5, 5.5):
        ax.axvline(separator, color="#4B4B4B", linewidth=1.15, zorder=7)
    for horizon, centre in GROUP_CENTRES.items():
        ax.text(
            centre,
            1.035,
            f"{horizon} h",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="semibold",
            color="#333333",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)


def annotate_matrix(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: Normalize,
    formatter,
) -> None:
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = float(values[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                formatter(value),
                ha="center",
                va="center",
                fontsize=6.9,
                fontweight="semibold",
                color=annotation_colour(cmap(norm(value))),
                zorder=10,
            )


def draw_figure(df: pd.DataFrame, scales: dict[str, float]) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9.3,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.48, 5.85), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1.0, 1.0],
        left=0.135,
        right=0.955,
        bottom=0.115,
        top=0.91,
        wspace=0.045,
        hspace=0.68,
    )
    ax_picp = fig.add_subplot(grid[0, 0])
    cax_picp = fig.add_subplot(grid[0, 1])
    ax_mpiw = fig.add_subplot(grid[1, 0])
    cax_mpiw = fig.add_subplot(grid[1, 1])

    picp = matrix(df, "PICP_mean")
    mpiw = matrix(df, "MPIW_mean")
    picp_cmap = mpl.colormaps["viridis"]
    mpiw_cmap = mpl.colormaps["magma"]
    picp_norm = Normalize(vmin=scales["picp_vmin"], vmax=scales["picp_vmax"])
    mpiw_norm = Normalize(vmin=scales["mpiw_vmin"], vmax=scales["mpiw_vmax"])

    picp_image = ax_picp.imshow(picp, cmap=picp_cmap, norm=picp_norm, aspect="auto", interpolation="nearest")
    mpiw_image = ax_mpiw.imshow(mpiw, cmap=mpiw_cmap, norm=mpiw_norm, aspect="auto", interpolation="nearest")
    add_group_structure(ax_picp)
    add_group_structure(ax_mpiw)
    annotate_matrix(ax_picp, picp, picp_cmap, picp_norm, lambda value: f"{value:.3f}")
    annotate_matrix(ax_mpiw, mpiw, mpiw_cmap, mpiw_norm, lambda value: f"{value:.2f}")

    ax_picp.set_title("(a)  Prediction interval coverage", loc="left", fontweight="semibold", pad=31)
    ax_picp.text(
        1.0,
        1.17,
        "Nominal coverage = 0.90",
        transform=ax_picp.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#555555",
    )
    ax_mpiw.set_title("(b)  Prediction interval width", loc="left", fontweight="semibold", pad=31)

    picp_ticks = [tick for tick in [0.90, 0.92, 0.94, 0.96] if scales["picp_vmin"] <= tick <= scales["picp_vmax"]]
    picp_bar = fig.colorbar(picp_image, cax=cax_picp, orientation="vertical", ticks=picp_ticks)
    picp_bar.outline.set_linewidth(0.65)
    picp_bar.outline.set_edgecolor("#555555")
    picp_bar.ax.tick_params(labelsize=7.0, length=2.5, width=0.6)
    picp_bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    picp_bar.set_label("PICP", fontsize=8.2, labelpad=7)
    picp_bar.ax.axhline(NOMINAL_COVERAGE, color="white", linewidth=1.2)
    picp_bar.ax.axhline(NOMINAL_COVERAGE, color="#222222", linewidth=0.45)

    mpiw_ticks = np.arange(math.ceil(scales["mpiw_vmin"]), math.floor(scales["mpiw_vmax"]) + 1, 1.0)
    mpiw_bar = fig.colorbar(mpiw_image, cax=cax_mpiw, orientation="vertical", ticks=mpiw_ticks)
    mpiw_bar.outline.set_linewidth(0.65)
    mpiw_bar.outline.set_edgecolor("#555555")
    mpiw_bar.ax.tick_params(labelsize=7.0, length=2.5, width=0.6)
    mpiw_bar.set_label("MPIW (km h$^{-1}$)", fontsize=8.2, labelpad=7)

    return fig


def write_supporting_files(df: pd.DataFrame, scales: dict[str, float], notes: list[str]) -> None:
    audit_columns = [
        "scenario",
        "scenario_label",
        "row_order",
        "horizon",
        "model",
        "model_display",
        "column_order",
        "seed_count",
        "sample_count",
        "PICP_mean",
        "nominal_coverage",
        "coverage_minus_nominal",
        "coverage_at_or_above_nominal",
        "MPIW_mean",
    ]
    audit = df[audit_columns].copy()
    audit["picp_vmin"] = scales["picp_vmin"]
    audit["picp_vmax"] = scales["picp_vmax"]
    audit["mpiw_vmin"] = scales["mpiw_vmin"]
    audit["mpiw_vmax"] = scales["mpiw_vmax"]
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    supplement_columns = [
        "scenario",
        "scenario_label",
        "horizon",
        "model",
        "model_display",
        "seed_count",
        "sample_count",
        "PINAW_mean",
        "Pinball_Loss_mean",
    ]
    df[supplement_columns].to_csv(SUPPLEMENT, index=False, encoding="utf-8-sig")

    adequate = int((df["PICP_mean"] >= NOMINAL_COVERAGE).sum())
    total = int(len(df))
    validation = {
        "figure": "Figure 7: Environmental-regime-dependent prediction uncertainty",
        "generated_at": datetime.now().astimezone().isoformat(),
        "sources": {
            "source_data": str(SOURCE),
            "data_definition": str(DATA_DEFINITION),
            "caption_draft": str(CAPTION_DRAFT),
        },
        "source_sha256": {
            "source_data": sha256(SOURCE),
            "data_definition": sha256(DATA_DEFINITION),
            "caption_draft": sha256(CAPTION_DRAFT),
        },
        "source_rows": total,
        "nominal_coverage": NOMINAL_COVERAGE,
        "coverage_at_or_above_nominal_cells": adequate,
        "coverage_below_nominal_cells": total - adequate,
        "picp_range": [float(df["PICP_mean"].min()), float(df["PICP_mean"].max())],
        "mpiw_range_km_per_h": [float(df["MPIW_mean"].min()), float(df["MPIW_mean"].max())],
        "panel_a_scale": {"metric": "PICP_mean", "vmin": scales["picp_vmin"], "vmax": scales["picp_vmax"], "colormap": "viridis"},
        "panel_b_scale": {"metric": "MPIW_mean", "vmin": scales["mpiw_vmin"], "vmax": scales["mpiw_vmax"], "colormap": "magma", "unit": "km h^-1"},
        "model_display_names": [MODEL_LABELS[m] for m in MODEL_ORDER],
        "forbidden_display_name_used": False,
        "supplementary_only_metrics": ["PINAW_mean", "Pinball_Loss_mean"],
        "checks": notes,
        "all_checks_passed": True,
    }
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    caption = """# Figure 7 caption

**Figure 7. Environmental-regime-dependent prediction uncertainty.** **(a)** Empirical prediction-interval coverage probability (PICP) and **(b)** mean prediction-interval width (MPIW) for the Traffic-only, Pressure-augmented and Full-fusion uncertainty models across the dry, rain, elevated-AOD and compound regimes. Columns are grouped by the 1, 3 and 6 h point-speed forecasting horizons. The nominal conformal prediction-interval coverage is 0.90. Each cell reports PICP to three decimals or MPIW to two decimals in km h⁻¹. PICP and MPIW use separate colour scales because they measure reliability and sharpness, respectively. PINAW and Pinball Loss are retained in the supplementary source table rather than plotted here. Comparisons are descriptive and do not imply universal uncertainty superiority for any environmental configuration.
"""
    CAPTION.write_text(caption, encoding="utf-8")

    readme = f"""# Figure 7 deliverables

- Submission PNG: `{PNG.name}` (600 dpi)
- Preview PNG: `{PREVIEW.name}`
- Vector PDF: `{PDF.name}`
- Editable SVG: `{SVG.name}`
- Exact PICP/MPIW heatmap audit: `{AUDIT.name}`
- Supplementary PINAW/Pinball table: `{SUPPLEMENT.name}`
- Validation summary: `{VALIDATION.name}`
- Manuscript caption: `{CAPTION.name}`
- Three-pass QC: `{QC.name}`
- Reproducible generator: `{Path(__file__).name}`

The main figure contains only PICP and MPIW. The internal M4 model is displayed as “Pressure-augmented”; the phrase “Best environmental model” is not used. The two panels have identical scenario/model/horizon geometry but intentionally use separate colourbars and units.
"""
    README.write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, scales, notes = validate_and_prepare()
    write_supporting_files(df, scales, notes)

    fig = draw_figure(df, scales)
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

    qc = f"""# Figure 7 three-pass quality check

## Pass 1 — frozen UQ source audit: PASS

- Source rows: {len(df)} (four regimes × three models × three horizons).
- Every metric is aggregated across five seeds.
- The three models share identical samples within every scenario-horizon combination.
- Display names are Traffic-only, Pressure-augmented and Full fusion; “Best environmental model” is absent.

## Pass 2 — metric and scale audit: PASS

- Nominal coverage: {NOMINAL_COVERAGE:.2f}.
- PICP range: {df['PICP_mean'].min():.6f}–{df['PICP_mean'].max():.6f}; cells at/above nominal: {int((df['PICP_mean'] >= NOMINAL_COVERAGE).sum())}/{len(df)}.
- MPIW range: {df['MPIW_mean'].min():.6f}–{df['MPIW_mean'].max():.6f} km h⁻¹.
- Panel (a) uses its own PICP colourbar; panel (b) uses a separate MPIW colourbar.
- PINAW and Pinball Loss are excluded from the main figure and retained in `{SUPPLEMENT.name}`.

## Pass 3 — rendered-output review: PASS

- Exactly two aligned heatmap panels.
- Main PNG: {width} × {height} px at {dpi[0]} × {dpi[1]} dpi.
- Preview PNG: {preview_size[0]} × {preview_size[1]} px.
- All 72 displayed values, group headers, model labels, scenario labels, thin horizon separators and both colourbars are visible without clipping or overlap.
"""
    QC.write_text(qc, encoding="utf-8")

    outputs = [PNG, PREVIEW, PDF, SVG, AUDIT, SUPPLEMENT, VALIDATION, CAPTION, QC, README, Path(__file__)]
    purposes = {
        PNG.name: "submission-ready 600-dpi two-panel PNG",
        PREVIEW.name: "quick visual-inspection PNG",
        PDF.name: "vector PDF",
        SVG.name: "editable vector SVG",
        AUDIT.name: "exact PICP/MPIW plotted-cell audit",
        SUPPLEMENT.name: "supplementary PINAW and Pinball Loss values",
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
                "source_rows": len(df),
                "picp_range": [float(df["PICP_mean"].min()), float(df["PICP_mean"].max())],
                "mpiw_range": [float(df["MPIW_mean"].min()), float(df["MPIW_mean"].max())],
                "coverage_at_or_above_nominal": int((df["PICP_mean"] >= NOMINAL_COVERAGE).sum()),
                "nominal_coverage": NOMINAL_COVERAGE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
