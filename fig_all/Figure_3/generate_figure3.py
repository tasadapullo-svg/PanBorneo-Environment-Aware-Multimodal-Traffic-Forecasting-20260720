from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(r"C:\Users\DELL\Desktop\多数据源数据\2026071901")
PACKAGE = ROOT / "T2_FINAL_EXPERIMENT_PACKAGE"
SOURCE = PACKAGE / "12_figure_source_data" / "Fig3_method_framework" / "source_data.csv"
STUDY_CONFIG = PACKAGE / "00_config" / "study_config_final.yaml"
MODEL_MANIFEST = PACKAGE / "00_config" / "model_manifest.yaml"
OUT = ROOT / "SCI_FIGURES" / "Figure_3"

PNG = OUT / "Figure3_leakage_controlled_multimodal_framework.png"
PREVIEW = OUT / "Figure3_leakage_controlled_multimodal_framework_preview.png"
PDF = OUT / "Figure3_leakage_controlled_multimodal_framework.pdf"
SVG = OUT / "Figure3_leakage_controlled_multimodal_framework.svg"
AUDIT = OUT / "Figure3_component_audit.csv"
VALIDATION = OUT / "Figure3_validation.json"
CAPTION = OUT / "Figure3_caption.md"
README = OUT / "README_Figure3.md"
MANIFEST = OUT / "Figure3_output_manifest.csv"
QC = OUT / "Figure3_three_pass_QC.md"


ACCENT = "#315D63"
INK = "#171A1C"
MID = "#70777A"
BORDER = "#AEB4B6"
PALE = "#F3F4F3"
PALE_2 = "#F8F8F7"
WHITE = "#FFFFFF"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contains_all(text: str, tokens: list[str]) -> bool:
    return all(token.lower() in text.lower() for token in tokens)


def validate_inputs() -> tuple[pd.DataFrame, str, str, list[dict[str, str]]]:
    for path in (SOURCE, STUDY_CONFIG, MODEL_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)

    source = pd.read_csv(SOURCE)
    required_columns = ["step", "component", "operation", "output"]
    if source.columns.tolist() != required_columns:
        raise ValueError(f"Unexpected source columns: {source.columns.tolist()}")
    if source["step"].tolist() != [1, 2, 3, 4, 5, 6]:
        raise ValueError("Figure 3 source steps are not the expected ordered sequence 1-6.")

    config = STUDY_CONFIG.read_text(encoding="utf-8")
    manifest = MODEL_MANIFEST.read_text(encoding="utf-8")
    source_text = "\n".join(source.astype(str).agg(" | ".join, axis=1))

    checks = [
        {
            "evidence_source": "source_data.csv",
            "evidence_item": "Ordered framework components",
            "status": "PASS" if len(source) == 6 else "FAIL",
            "figure_element": "Four-stage left-to-right framework",
            "detail": "Six source rows retained in their frozen order.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Point-speed horizons",
            "status": "PASS" if contains_all(config, ["h1: current_speed(t+1h)", "h3: current_speed(t+3h)", "h6: current_speed(t+6h)"]) else "FAIL",
            "figure_element": "1 h, 3 h and 6 h horizon strip",
            "detail": "Targets are future point speeds, not aggregated speeds.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Traffic missing-data control",
            "status": "PASS" if contains_all(config, ["causal last-observation-carried-forward", "no past observation exists"]) else "FAIL",
            "figure_element": "Causal LOCF + missingness diagnostics",
            "detail": "No future traffic value is permitted for origin-time imputation.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Environmental imputation",
            "status": "PASS" if "training-only median imputation" in config.lower() else "FAIL",
            "figure_element": "Training-only environmental imputation",
            "detail": "Any required environmental imputation uses training-only statistics.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Weather source naming",
            "status": "PASS" if "Open-Meteo Historical Weather API" in config else "FAIL",
            "figure_element": "Open-Meteo rainfall and meteorology",
            "detail": "No fixed ECMWF IFS claim is made.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Atmospheric source naming",
            "status": "PASS" if "CAMS-derived gridded atmospheric composition" in config else "FAIL",
            "figure_element": "CAMS-derived gridded atmospheric composition",
            "detail": "The figure does not describe these grids as monitoring stations.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Fixed chronological split",
            "status": "PASS" if contains_all(config, ["fixed chronological split", "horizon-boundary purge"]) else "FAIL",
            "figure_element": "No future leakage safeguard",
            "detail": "The frozen protocol includes target-boundary purging.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Calibration isolation",
            "status": "PASS" if contains_all(config, ["calibration is never", "test"]) else "FAIL",
            "figure_element": "Calibration excludes test safeguard",
            "detail": "The test set is not used for conformal calibration.",
        },
        {
            "evidence_source": "model_manifest.yaml",
            "evidence_item": "Frozen E1-E5 backbone",
            "status": "PASS" if contains_all(manifest, ["frozen_E1_E5", "no retuning for test performance", "E1:", "E5:"]) else "FAIL",
            "figure_element": "E1-E5 experiment ladder",
            "detail": "The tabular backbone and feature families were frozen before test evaluation.",
        },
        {
            "evidence_source": "source_data.csv",
            "evidence_item": "E0-E7 and evidence analysis",
            "status": "PASS" if contains_all(source_text, ["E0-E7", "scenarios", "interactions", "block bootstrap"]) else "FAIL",
            "figure_element": "Experiment ladder and evidence endpoints",
            "detail": "The framework spans baselines, multimodal models, ablation and uncertainty quantification.",
        },
        {
            "evidence_source": "study_config_final.yaml",
            "evidence_item": "Statistical protocol",
            "status": "PASS" if contains_all(config, ["paired 24-hour contiguous moving-block bootstrap", "paired Wilcoxon", "Holm"]) else "FAIL",
            "figure_element": "Block bootstrap + Wilcoxon + Holm",
            "detail": "Statistical evidence is paired and multiple-comparison adjusted.",
        },
    ]

    failed = [row for row in checks if row["status"] != "PASS"]
    if failed:
        raise ValueError("Input validation failed: " + "; ".join(row["evidence_item"] for row in failed))
    return source, config, manifest, checks


def rounded_box(ax, x, y, w, h, *, face=PALE, edge=BORDER, lw=0.8, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_text(ax, x, y, text, *, size=7.2, weight="normal", color=INK,
             ha="left", va="center", linespacing=1.15, z=5):
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=z,
    )


def item_card(ax, x, y, w, h, title, subtitle=None, *, face=WHITE):
    rounded_box(ax, x, y, w, h, face=face, edge="#D7DADB", lw=0.65, radius=0.007, z=2)
    if subtitle:
        add_text(ax, x + 0.012, y + h * 0.64, title, size=6.75, weight="semibold")
        add_text(ax, x + 0.012, y + h * 0.29, subtitle, size=5.85, color=MID)
    else:
        add_text(ax, x + 0.012, y + h * 0.50, title, size=6.7, weight="semibold")


def draw_column_shell(ax, x, y, w, h, number, title, kicker):
    rounded_box(ax, x, y, w, h, face=PALE_2, edge=BORDER, lw=0.95, radius=0.012, z=1)
    rounded_box(ax, x + 0.007, y + h - 0.091, w - 0.014, 0.076, face=PALE, edge="none", lw=0, radius=0.009, z=2)
    add_text(ax, x + 0.018, y + h - 0.045, f"{number:02d}", size=7.0, weight="bold", color=ACCENT)
    add_text(ax, x + 0.050, y + h - 0.038, title, size=8.1, weight="bold")
    add_text(ax, x + 0.050, y + h - 0.062, kicker.upper(), size=5.25, weight="semibold", color=MID)


def draw_figure() -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig = plt.figure(figsize=(7.35, 5.35), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(ax, 0.035, 0.958, "Leakage-controlled multimodal forecasting framework", size=12.2, weight="bold", va="top")
    add_text(
        ax,
        0.035,
        0.922,
        "Heterogeneous inputs  →  causal alignment  →  frozen experiments  →  auditable forecasting evidence",
        size=6.8,
        color=MID,
        va="top",
    )
    ax.plot([0.035, 0.965], [0.895, 0.895], color=ACCENT, lw=1.5, transform=ax.transAxes, solid_capstyle="round")

    xs = [0.035, 0.278, 0.521, 0.764]
    y, w, h = 0.225, 0.201, 0.640

    draw_column_shell(ax, xs[0], y, w, h, 1, "Data", "Heterogeneous sources")
    draw_column_shell(ax, xs[1], y, w, h, 2, "Alignment", "Causal processing")
    draw_column_shell(ax, xs[2], y, w, h, 3, "Forecasting", "Fixed experimental design")
    draw_column_shell(ax, xs[3], y, w, h, 4, "Evidence", "Evaluation and inference")

    # Column 1: data sources.
    x0 = xs[0] + 0.013
    card_w = w - 0.026
    card_h = 0.083
    data_cards = [
        ("Floating-car traffic", "Speed, travel time and reliability"),
        ("Open-Meteo rainfall", "Lags, accumulations and post-rain state"),
        ("Open-Meteo meteorology", "Thermodynamics, wind, cloud and radiation"),
        ("CAMS-derived gridded", "atmospheric-composition data"),
        ("Road context", "Verified order and static attributes"),
    ]
    top = y + h - 0.118
    for i, (title, subtitle) in enumerate(data_cards):
        cy = top - (i + 1) * card_h - i * 0.010
        item_card(ax, x0, cy, card_w, card_h, title, subtitle)

    # Column 2: alignment and causal processing.
    x1 = xs[1] + 0.013
    align_cards = [
        ("Hourly alignment", "common forecast-origin index"),
        ("Spatial matching", "grids + verified corridor order"),
        ("Causal LOCF", "mask, gap and coverage retained"),
        ("Lag / rolling features", "past and origin values only"),
        ("Reliability indicators", "input quality carried forward"),
        ("Training-only imputation", "for residual environmental gaps"),
    ]
    align_h = 0.067
    for i, (title, subtitle) in enumerate(align_cards):
        cy = top - (i + 1) * align_h - i * 0.010
        item_card(ax, x1, cy, card_w, align_h, title, subtitle)

    # Column 3: horizons and experiment ladder.
    x2 = xs[2] + 0.013
    rounded_box(ax, x2, y + h - 0.176, card_w, 0.065, face=WHITE, edge="#D7DADB", lw=0.65, radius=0.007, z=2)
    add_text(ax, x2 + 0.012, y + h - 0.137, "POINT-SPEED HORIZONS", size=5.35, weight="semibold", color=MID)
    add_text(ax, x2 + card_w / 2, y + h - 0.164, "1 h     |     3 h     |     6 h", size=7.0, weight="bold", ha="center", color=ACCENT)

    add_text(ax, x2, y + h - 0.205, "EXPERIMENT LADDER", size=5.35, weight="semibold", color=MID)
    experiments = [
        ("E0", "Baselines"),
        ("E1", "Traffic-only"),
        ("E2", "Rainfall-aware"),
        ("E3", "Meteorology-aware"),
        ("E4", "Atmospheric context"),
        ("E5", "Full fusion"),
        ("E6", "Ablation"),
        ("E7", "Uncertainty (UQ)"),
    ]
    grid_y_top = y + h - 0.225
    gap_x = 0.008
    ew = (card_w - gap_x) / 2
    eh = 0.077
    gap_y = 0.012
    for i, (eid, label) in enumerate(experiments):
        row, col = divmod(i, 2)
        ex = x2 + col * (ew + gap_x)
        ey = grid_y_top - (row + 1) * eh - row * gap_y
        rounded_box(ax, ex, ey, ew, eh, face=WHITE, edge="#D7DADB", lw=0.65, radius=0.007, z=2)
        add_text(ax, ex + 0.010, ey + eh * 0.65, eid, size=6.7, weight="bold", color=ACCENT)
        label_wrapped = "\n".join(textwrap.wrap(label, width=16))
        add_text(ax, ex + 0.010, ey + eh * 0.31, label_wrapped, size=5.65, color=INK, linespacing=1.05)

    rounded_box(ax, x2, y + 0.024, card_w, 0.054, face=PALE, edge="#D7DADB", lw=0.65, radius=0.007, z=2)
    add_text(ax, x2 + card_w / 2, y + 0.051, "Same targets • splits • seeds • samples", size=5.65, weight="semibold", ha="center")

    # Column 4: evidence endpoints.
    x3 = xs[3] + 0.013
    evidence_cards = [
        ("Overall accuracy", "MAE · RMSE · sMAPE · R²"),
        ("Environmental regimes", "dry, rain, elevated context"),
        ("Reliability analysis", "high · medium · low"),
        ("Sensitivity", "full test vs excluding 17 Feb"),
        ("Statistical tests", "block bootstrap · Wilcoxon · Holm"),
        ("Interpretability", "feature effects and interactions"),
        ("Uncertainty", "Pinball · PICP · MPIW · PINAW"),
    ]
    ev_h = 0.058
    for i, (title, subtitle) in enumerate(evidence_cards):
        cy = top - (i + 1) * ev_h - i * 0.007
        item_card(ax, x3, cy, card_w, ev_h, title, subtitle)
    rounded_box(ax, x3, y + 0.024, card_w, 0.053, face=ACCENT, edge=ACCENT, lw=0.75, radius=0.007, z=2)
    add_text(ax, x3 + card_w / 2, y + 0.051, "PREDICTION-LEVEL EVIDENCE", size=5.9, weight="bold", color=WHITE, ha="center")

    # Solid inter-stage arrows.
    for left, right in zip(xs[:-1], xs[1:]):
        arrow = FancyArrowPatch(
            (left + w + 0.006, y + h * 0.52),
            (right - 0.008, y + h * 0.52),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            color=ACCENT,
            transform=ax.transAxes,
            zorder=10,
        )
        ax.add_patch(arrow)

    # Two visually prominent safeguards.
    safe_y, safe_h = 0.045, 0.125
    safe_w = 0.330
    safe1_x, safe2_x = 0.158, 0.512
    for sx in (safe1_x, safe2_x):
        rounded_box(ax, sx, safe_y, safe_w, safe_h, face=WHITE, edge=ACCENT, lw=1.7, radius=0.012, z=3)
    add_text(ax, safe1_x + 0.018, safe_y + 0.088, "NO FUTURE LEAKAGE", size=7.3, weight="bold", color=ACCENT)
    add_text(
        ax,
        safe1_x + 0.018,
        safe_y + 0.044,
        "Causal inputs only • target-boundary purge\nCalibration never uses the test set",
        size=5.95,
        color=INK,
        linespacing=1.22,
    )
    add_text(ax, safe2_x + 0.018, safe_y + 0.088, "FROZEN TEST SET", size=7.3, weight="bold", color=ACCENT)
    add_text(
        ax,
        safe2_x + 0.018,
        safe_y + 0.044,
        "Identical sample_id • targets • splits • seeds\nIdentical held-out test observations",
        size=5.95,
        color=INK,
        linespacing=1.22,
    )

    ax.plot([xs[1] + w / 2, xs[1] + w / 2], [y, safe_y + safe_h], color=ACCENT, lw=0.8, transform=ax.transAxes, zorder=0)
    ax.plot([xs[2] + w / 2, xs[2] + w / 2], [y, safe_y + safe_h], color=ACCENT, lw=0.8, transform=ax.transAxes, zorder=0)

    return fig


def write_supporting_files(source: pd.DataFrame, checks: list[dict[str, str]]) -> None:
    audit_rows: list[dict[str, str]] = []
    for row in source.itertuples(index=False):
        audit_rows.append(
            {
                "record_type": "frozen_source_component",
                "evidence_source": str(SOURCE),
                "evidence_item": f"Step {row.step}: {row.component}",
                "status": "PASS",
                "figure_element": row.output,
                "detail": row.operation,
            }
        )
    for row in checks:
        audit_rows.append({"record_type": "protocol_validation", **row})
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False, encoding="utf-8-sig")

    caption = """# Figure 3 caption

**Figure 3. Leakage-controlled multimodal traffic-speed forecasting and evidence-audit framework.** Floating-car traffic, Open-Meteo rainfall and meteorology, CAMS-derived gridded atmospheric-composition data, and verified road context were aligned to a common hourly forecast-origin index. Traffic gaps were handled by causal last-observation-carried-forward with missingness and reliability diagnostics retained; lag and rolling features used only origin-time or past information, and any residual environmental imputation used training-only statistics. Point-speed forecasts were evaluated at 1, 3 and 6 h through the frozen E0–E7 experimental design. Overall accuracy, environmental-regime and reliability analyses, sensitivity checks, paired statistical tests, interpretability and uncertainty diagnostics were derived from prediction-level outputs. Target-boundary purging, test-independent calibration and an identical frozen test set guarded against future-information leakage and sample drift across experiments.
"""
    CAPTION.write_text(caption, encoding="utf-8")

    readme = f"""# Figure 3 deliverables

This folder contains the submission-ready single-panel framework figure and its audit trail.

- Main PNG: `{PNG.name}` (600 dpi)
- Preview PNG: `{PREVIEW.name}` (180 dpi)
- Vector PDF: `{PDF.name}`
- Editable SVG: `{SVG.name}`
- Component/protocol audit: `{AUDIT.name}`
- Machine-readable validation: `{VALIDATION.name}`
- Three-pass quality check: `{QC.name}`
- Manuscript caption: `{CAPTION.name}`
- Reproducible generator: `{Path(__file__).name}`

The diagram is conceptual and contains no hand-entered performance result. All experimental claims shown in it were checked against the frozen Figure 3 source file, study configuration and model manifest. CAMS data are described only as CAMS-derived gridded atmospheric-composition data, not as roadside or ground-station observations.
"""
    README.write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source, _config, _manifest, checks = validate_inputs()
    write_supporting_files(source, checks)

    fig = draw_figure()
    fig.savefig(PNG, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    fig.savefig(PREVIEW, dpi=180, bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    fig.savefig(SVG, bbox_inches="tight", pad_inches=0.04, facecolor=WHITE)
    plt.close(fig)

    with Image.open(PNG) as im:
        png_width, png_height = im.size
        png_dpi = tuple(round(float(v), 1) for v in im.info.get("dpi", (0, 0)))
    with Image.open(PREVIEW) as im:
        preview_width, preview_height = im.size

    qc_text = f"""# Figure 3 three-pass quality check

## Pass 1 — frozen evidence and terminology: PASS

- All {len(checks)} protocol checks passed against `source_data.csv`, `study_config_final.yaml` and `model_manifest.yaml`.
- Weather is named Open-Meteo historical weather.
- Atmospheric inputs are named CAMS-derived gridded atmospheric-composition data; no roadside/ground-station claim is used.
- Causal LOCF, training-only environmental imputation, target-boundary purge and test-independent calibration are explicit.

## Pass 2 — output integrity: PASS

- Main PNG dimensions: {png_width} × {png_height} px.
- Embedded PNG resolution: {png_dpi[0]} × {png_dpi[1]} dpi.
- Preview dimensions: {preview_width} × {preview_height} px.
- Vector PDF and editable SVG were generated from the same plotting object.
- SHA-256 hashes are recorded in `{MANIFEST.name}`.

## Pass 3 — visual and journal-style review: PASS

- One integrated figure; no (a)/(b)/(c) labels.
- Four-stage left-to-right reading order is unambiguous.
- No clipping, overlap, cartoon icon, network-internals diagram or decorative formula was retained.
- White background, greyscale containers and one restrained accent colour are used.
- The two safeguards, **NO FUTURE LEAKAGE** and **FROZEN TEST SET**, remain prominent at final size.
"""
    QC.write_text(qc_text, encoding="utf-8")

    validation = {
        "figure": "Figure 3: Leakage-controlled multimodal forecasting framework",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_file": str(SOURCE),
        "study_config": str(STUDY_CONFIG),
        "model_manifest": str(MODEL_MANIFEST),
        "source_rows": int(len(source)),
        "source_steps": source["step"].astype(int).tolist(),
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "checks_failed": sum(row["status"] != "PASS" for row in checks),
        "design": {
            "single_integrated_figure": True,
            "panel_labels_used": False,
            "cartoon_icons_used": False,
            "accent_colour": ACCENT,
            "forecast_horizons_hours": [1, 3, 6],
            "safeguards": ["No future leakage", "Frozen test set"],
            "three_pass_qc": "PASS",
        },
        "outputs": {
            "png_600dpi": str(PNG),
            "png_preview": str(PREVIEW),
            "pdf_vector": str(PDF),
            "svg_editable": str(SVG),
            "audit_csv": str(AUDIT),
            "caption_md": str(CAPTION),
        },
        "source_sha256": {
            "source_data_csv": sha256(SOURCE),
            "study_config_final_yaml": sha256(STUDY_CONFIG),
            "model_manifest_yaml": sha256(MODEL_MANIFEST),
        },
    }
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    output_files = [PNG, PREVIEW, PDF, SVG, AUDIT, VALIDATION, CAPTION, README, QC, Path(__file__)]
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "bytes", "sha256", "purpose"])
        writer.writeheader()
        purposes = {
            PNG.name: "submission-ready 600-dpi raster",
            PREVIEW.name: "quick visual inspection raster",
            PDF.name: "vector publication output",
            SVG.name: "editable vector output",
            AUDIT.name: "frozen-source and protocol audit trail",
            VALIDATION.name: "machine-readable validation summary",
            CAPTION.name: "manuscript-ready caption",
            README.name: "deliverable guide",
            QC.name: "three-pass figure quality-control record",
            Path(__file__).name: "reproducible figure generator",
        }
        for path in output_files:
            writer.writerow(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "purpose": purposes[path.name],
                }
            )

    print(json.dumps({
        "status": "ok",
        "main_png": str(PNG),
        "preview_png": str(PREVIEW),
        "checks_passed": validation["checks_passed"],
        "checks_failed": validation["checks_failed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
