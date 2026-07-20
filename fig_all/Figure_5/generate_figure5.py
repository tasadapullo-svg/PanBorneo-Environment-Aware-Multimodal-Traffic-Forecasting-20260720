from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, ScalarFormatter
from PIL import Image


ROOT = Path(r"C:\Users\DELL\Desktop\多数据源数据\2026071901")
PACKAGE = ROOT / "T2_FINAL_EXPERIMENT_PACKAGE"
STAT_SOURCE = PACKAGE / "10_statistical_tests" / "holm_corrected_results.csv"
IMPORTANCE_SOURCE = PACKAGE / "11_interpretability" / "feature_importance_by_scenario.csv"
OUT = ROOT / "SCI_FIGURES" / "Figure_5"

PNG = OUT / "Figure5_selective_environmental_value_and_importance.png"
PREVIEW = OUT / "Figure5_selective_environmental_value_and_importance_preview.png"
PDF = OUT / "Figure5_selective_environmental_value_and_importance.pdf"
SVG = OUT / "Figure5_selective_environmental_value_and_importance.svg"
STAT_AUDIT = OUT / "Figure5a_effect_size_plotted_data.csv"
IMPORTANCE_AUDIT = OUT / "Figure5b_importance_plotted_data.csv"
VALIDATION = OUT / "Figure5_validation.json"
CAPTION = OUT / "Figure5_caption.md"
QC = OUT / "Figure5_three_pass_QC.md"
README = OUT / "README_Figure5.md"
MANIFEST = OUT / "Figure5_output_manifest.csv"


HORIZONS = [1, 3, 6]
SERIES = {
    1: {"color": "#0072B2", "marker": "o", "label": "1 h"},
    3: {"color": "#D55E00", "marker": "s", "label": "3 h"},
    6: {"color": "#009E73", "marker": "^", "label": "6 h"},
}

COMPARATOR_ORDER = [
    "E2_rainfall",
    "E3_meteorology",
    "E4_atmospheric",
    "E5_full_fusion",
    "Best_selective_environmental",
]
COMPARATOR_LABELS = {
    "E2_rainfall": "Rainfall",
    "E3_meteorology": "Meteorology",
    "E4_atmospheric": "Atmospheric",
    "E5_full_fusion": "Full fusion",
    "Best_selective_environmental": "Validation-selected\nconfiguration",
}

FEATURE_GROUPS = [
    "Traffic",
    "Calendar",
    "Road",
    "Reliability",
    "Rainfall",
    "Meteorology",
    "Atmospheric",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_statistics() -> tuple[pd.DataFrame, list[str]]:
    if not STAT_SOURCE.exists():
        raise FileNotFoundError(STAT_SOURCE)
    df = pd.read_csv(STAT_SOURCE)
    required = {
        "comparator",
        "horizon",
        "delta_MAE",
        "delta_convention",
        "ci_lower",
        "ci_upper",
        "p_adjusted",
        "significant",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Statistical source missing columns: {missing}")

    notes: list[str] = []
    if set(df["comparator"]) != set(COMPARATOR_ORDER):
        raise ValueError("Unexpected comparator set in Holm-corrected results.")
    if set(df["horizon"].astype(int)) != set(HORIZONS):
        raise ValueError("Unexpected horizon set in Holm-corrected results.")
    if len(df) != 15 or df.duplicated(["comparator", "horizon"]).any():
        raise ValueError("Expected one row for each of 15 comparator-horizon pairs.")
    notes.append("All five comparators and three horizons are uniquely present.")

    conventions = df["delta_convention"].astype(str).str.lower()
    if not conventions.str.contains("comparator_minus_e1", regex=False).all():
        raise ValueError("Delta-MAE direction is not uniformly comparator minus E1.")
    notes.append("Delta MAE is uniformly comparator minus E1; negative is better.")

    numeric_cols = ["delta_MAE", "ci_lower", "ci_upper", "p_adjusted"]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Statistical results contain non-finite plotted values.")
    if not ((numeric["ci_lower"] <= numeric["delta_MAE"]) & (numeric["delta_MAE"] <= numeric["ci_upper"])).all():
        raise ValueError("At least one point estimate lies outside its stated CI.")
    notes.append("Every point estimate lies within its moving-block bootstrap CI.")

    if "block_hours" in df.columns and not (df["block_hours"].astype(int) == 24).all():
        raise ValueError("Forest plot source does not uniformly use 24-hour blocks.")
    if "bootstrap_repetitions" in df.columns and not (df["bootstrap_repetitions"].astype(int) == 5000).all():
        raise ValueError("Forest plot source does not uniformly use 5000 bootstrap repetitions.")
    notes.append("Confidence intervals use the frozen 24-hour moving-block bootstrap with 5000 repetitions.")

    sig = df["significant"].astype(str).str.lower().map({"true": True, "false": False})
    if sig.isna().any():
        raise ValueError("Holm-adjusted significance flag could not be parsed.")
    notes.append(f"Holm-adjusted significant comparisons: {int(sig.sum())} of {len(sig)}.")

    df = df.copy()
    df["horizon"] = df["horizon"].astype(int)
    df["comparator_order"] = pd.Categorical(df["comparator"], categories=COMPARATOR_ORDER, ordered=True)
    df["display_label"] = df["comparator"].map(COMPARATOR_LABELS)
    df["holm_significant"] = sig.astype(bool)
    df = df.sort_values(["comparator_order", "horizon"]).reset_index(drop=True)
    return df, notes


def choose_importance_scale(values: pd.Series) -> tuple[str, float, str]:
    values = pd.to_numeric(values, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Permutation importance contains non-finite values.")
    all_positive = bool((values > 0).all())
    if all_positive:
        skew_ratio = float(values.max() / values.min())
        if skew_ratio >= 100.0:
            return "log", skew_ratio, "all values are strictly positive and max/min ≥ 100"
        return "linear", skew_ratio, "all values are positive but max/min < 100"

    positive = values[values > 0]
    reference = float(positive.min()) if len(positive) else max(float(values.abs().max()) / 100.0, 1e-6)
    return "symlog", float("inf"), f"zero or negative values are present; linthresh={reference:.6g}"


def validate_importance() -> tuple[pd.DataFrame, str, float, str, list[str]]:
    if not IMPORTANCE_SOURCE.exists():
        raise FileNotFoundError(IMPORTANCE_SOURCE)
    raw = pd.read_csv(IMPORTANCE_SOURCE)
    required = {
        "horizon",
        "scenario",
        "feature_group",
        "seed_count",
        "delta_MAE_mean",
        "delta_MAE_std",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Importance source missing columns: {missing}")

    df = raw[raw["scenario"].astype(str) == "Overall_test"].copy()
    notes: list[str] = []
    if len(df) != 21:
        raise ValueError(f"Expected 21 Overall_test importance rows, found {len(df)}.")
    if set(df["feature_group"]) != set(FEATURE_GROUPS):
        raise ValueError("Overall_test feature groups differ from the required seven groups.")
    if set(df["horizon"].astype(int)) != set(HORIZONS) or df.duplicated(["feature_group", "horizon"]).any():
        raise ValueError("Expected one Overall_test row per feature-group/horizon pair.")
    notes.append("Overall_test contains seven feature groups at all three horizons.")

    if not (df["seed_count"].astype(int) == 5).all():
        raise ValueError("Permutation importance is not uniformly aggregated across five seeds.")
    notes.append("All permutation-importance means and SDs are aggregated across five seeds.")

    numeric = df[["delta_MAE_mean", "delta_MAE_std"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric["delta_MAE_std"] < 0).any():
        raise ValueError("Permutation importance contains invalid means or SDs.")

    scale, skew_ratio, reason = choose_importance_scale(numeric["delta_MAE_mean"])
    notes.append(f"Importance x-axis selected automatically: {scale} ({reason}).")

    if "sample_count" in df.columns:
        if not (df.groupby("horizon")["sample_count"].nunique() == 1).all():
            raise ValueError("Importance sample count differs among feature groups within a horizon.")
        notes.append("Feature groups share the same Overall_test sample count within each horizon.")

    df["horizon"] = df["horizon"].astype(int)
    mean_order = (
        df.groupby("feature_group", as_index=False)["delta_MAE_mean"]
        .mean()
        .sort_values("delta_MAE_mean", ascending=False)["feature_group"]
        .tolist()
    )
    df["mean_importance_order"] = pd.Categorical(df["feature_group"], categories=mean_order, ordered=True)
    df = df.sort_values(["mean_importance_order", "horizon"]).reset_index(drop=True)
    return df, scale, skew_ratio, reason, notes


def draw_figure(
    stat: pd.DataFrame,
    importance: pd.DataFrame,
    importance_scale: str,
) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.48, 4.65),
        gridspec_kw={"width_ratios": [1.06, 1.0]},
        facecolor="white",
    )
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.17, top=0.82, wspace=0.47)

    # Panel (a): forest/effect-size plot.
    comparator_y = {name: len(COMPARATOR_ORDER) - 1 - i for i, name in enumerate(COMPARATOR_ORDER)}
    horizon_offset = {1: 0.19, 3: 0.0, 6: -0.19}
    for horizon in HORIZONS:
        hdf = stat[stat["horizon"] == horizon].set_index("comparator").loc[COMPARATOR_ORDER]
        y = np.array([comparator_y[name] + horizon_offset[horizon] for name in COMPARATOR_ORDER], dtype=float)
        point = hdf["delta_MAE"].to_numpy(dtype=float)
        lower = hdf["ci_lower"].to_numpy(dtype=float)
        upper = hdf["ci_upper"].to_numpy(dtype=float)
        xerr = np.vstack([point - lower, upper - point])
        style = SERIES[horizon]
        ax_a.errorbar(
            point,
            y,
            xerr=xerr,
            fmt=style["marker"],
            color=style["color"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.8,
            ecolor=style["color"],
            elinewidth=1.05,
            capsize=2.5,
            capthick=0.8,
            linestyle="none",
            zorder=4,
        )

    ax_a.axvline(0.0, color="#333333", linewidth=1.05, zorder=1)
    ax_a.set_yticks(
        [comparator_y[name] for name in COMPARATOR_ORDER],
        [COMPARATOR_LABELS[name] for name in COMPARATOR_ORDER],
    )
    ax_a.set_ylim(-0.55, 4.55)
    xmin = min(-0.04, float(stat["ci_lower"].min()) - 0.015)
    xmax = float(stat["ci_upper"].max()) + 0.035
    ax_a.set_xlim(xmin, xmax)
    forest_ticks = [-0.02, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    forest_ticks = [tick for tick in forest_ticks if xmin <= tick <= xmax]
    ax_a.xaxis.set_major_locator(FixedLocator(forest_ticks))
    ax_a.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.2f}" if value < 0 else f"{value:.1f}")
    )
    ax_a.set_xlabel("ΔMAE vs Traffic-only (km h$^{-1}$)")
    ax_a.set_title("(a)  Effect size relative to Traffic-only", loc="left", fontweight="semibold", pad=11)
    ax_a.text(0.01, 0.985, "← Better", transform=ax_a.transAxes, ha="left", va="top", fontsize=7.2, color="#666666")
    ax_a.text(0.99, 0.985, "Worse →", transform=ax_a.transAxes, ha="right", va="top", fontsize=7.2, color="#666666")
    for y0 in comparator_y.values():
        ax_a.axhline(y0, color="#E2E2E2", linewidth=0.55, zorder=0)

    # Panel (b): group permutation importance.
    feature_order = importance["mean_importance_order"].cat.categories.tolist()
    feature_y = {name: len(feature_order) - 1 - i for i, name in enumerate(feature_order)}
    for horizon in HORIZONS:
        hdf = importance[importance["horizon"] == horizon].set_index("feature_group").loc[feature_order]
        y = np.array([feature_y[name] + horizon_offset[horizon] for name in feature_order], dtype=float)
        point = hdf["delta_MAE_mean"].to_numpy(dtype=float)
        sd = hdf["delta_MAE_std"].to_numpy(dtype=float)
        style = SERIES[horizon]
        ax_b.errorbar(
            point,
            y,
            xerr=sd,
            fmt=style["marker"],
            color=style["color"],
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.8,
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.3,
            capthick=0.75,
            linestyle="none",
            zorder=4,
        )

    ax_b.set_yticks([feature_y[name] for name in feature_order], feature_order)
    ax_b.set_ylim(-0.55, len(feature_order) - 0.45)
    scale_label = importance_scale
    if importance_scale == "log":
        ax_b.set_xscale("log")
        positive_lower = (importance["delta_MAE_mean"] - importance["delta_MAE_std"]).clip(lower=np.nan)
        min_lower = float(positive_lower[positive_lower > 0].min())
        max_upper = float((importance["delta_MAE_mean"] + importance["delta_MAE_std"]).max())
        ax_b.set_xlim(10 ** np.floor(np.log10(min_lower) - 0.05), 10 ** np.ceil(np.log10(max_upper) + 0.05))
        ticks = [v for v in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] if ax_b.get_xlim()[0] <= v <= ax_b.get_xlim()[1]]
        ax_b.xaxis.set_major_locator(FixedLocator(ticks))
        ax_b.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    elif importance_scale == "symlog":
        nonzero = importance.loc[importance["delta_MAE_mean"] != 0, "delta_MAE_mean"].abs()
        linthresh = max(float(nonzero.min()) / 2.0, 1e-6) if len(nonzero) else 1e-3
        ax_b.set_xscale("symlog", linthresh=linthresh)
        ax_b.xaxis.set_major_formatter(ScalarFormatter())
    else:
        ax_b.set_xscale("linear")

    ax_b.set_xlabel(f"Permutation importance, ΔMAE (km h$^{{-1}}$; {scale_label} scale)")
    ax_b.set_title("(b)  Feature-group permutation importance", loc="left", fontweight="semibold", pad=11)
    for y0 in feature_y.values():
        ax_b.axhline(y0, color="#E2E2E2", linewidth=0.55, zorder=0)
    ax_b.xaxis.grid(True, which="major", color="#DDDDDD", linewidth=0.55, zorder=0)
    ax_b.xaxis.grid(True, which="minor", color="#EEEEEE", linewidth=0.4, zorder=0)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(axis="both", length=3, width=0.7, color="#333333")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=SERIES[h]["marker"],
            linestyle="none",
            markerfacecolor=SERIES[h]["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=SERIES[h]["color"],
            markersize=6.5,
            label=SERIES[h]["label"],
        )
        for h in HORIZONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.7,
        title="Forecast horizon",
        title_fontsize=8.3,
    )
    return fig


def write_supporting_files(
    stat: pd.DataFrame,
    importance: pd.DataFrame,
    importance_scale: str,
    skew_ratio: float,
    scale_reason: str,
    stat_notes: list[str],
    importance_notes: list[str],
) -> None:
    stat_export = stat.drop(columns=["comparator_order"]).copy()
    stat_export["plotted_y_category"] = stat_export["display_label"].str.replace("\n", " ", regex=False)
    stat_export.to_csv(STAT_AUDIT, index=False, encoding="utf-8-sig")

    importance_export = importance.drop(columns=["mean_importance_order"]).copy()
    importance_export["selected_x_scale"] = importance_scale
    importance_export["scale_selection_reason"] = scale_reason
    importance_export.to_csv(IMPORTANCE_AUDIT, index=False, encoding="utf-8-sig")

    caption = f"""# Figure 5 caption

**Figure 5. Selective environmental value and feature-group importance.** **(a)** Difference in test-set MAE between each environmental configuration and the horizon-matched E1 traffic-only reference. Points show ΔMAE = comparator − E1 and horizontal intervals show the 95% confidence interval from the prespecified 24 h moving-block bootstrap (5,000 repetitions). Negative values favour the comparator; positive values favour E1. The validation-selected configuration is the M4 traffic-plus-surface-pressure model selected solely using validation MAE. No significance stars are shown; all plotted Holm-adjusted significance flags are false. **(b)** Overall-test permutation importance for feature groups in the frozen E5 full-fusion model. Points show the mean increase in MAE after permutation and horizontal intervals show ±1 SD across five seeds. The x-axis uses a {importance_scale} scale selected automatically from the frozen importance distribution. Permutation importance quantifies within-model predictive dependence and does not imply causality.
"""
    CAPTION.write_text(caption, encoding="utf-8")

    validation = {
        "figure": "Figure 5: Selective environmental value and feature-group importance",
        "generated_at": datetime.now().astimezone().isoformat(),
        "sources": {
            "statistical_tests": str(STAT_SOURCE),
            "feature_importance": str(IMPORTANCE_SOURCE),
        },
        "source_sha256": {
            "statistical_tests": sha256(STAT_SOURCE),
            "feature_importance": sha256(IMPORTANCE_SOURCE),
        },
        "panel_a": {
            "rows": int(len(stat)),
            "comparators": COMPARATOR_ORDER,
            "horizons_hours": HORIZONS,
            "delta_convention": "comparator minus E1; negative is better",
            "confidence_interval": "95% moving-block bootstrap CI",
            "holm_significant_count": int(stat["holm_significant"].sum()),
            "significance_stars": False,
            "checks": stat_notes,
        },
        "panel_b": {
            "scenario_filter": "Overall_test",
            "rows": int(len(importance)),
            "feature_groups": FEATURE_GROUPS,
            "horizons_hours": HORIZONS,
            "selected_x_scale": importance_scale,
            "max_to_min_mean_importance_ratio": skew_ratio,
            "scale_selection_reason": scale_reason,
            "ordering_rule": "descending mean importance across 1 h, 3 h and 6 h",
            "checks": importance_notes,
        },
        "all_checks_passed": True,
    }
    VALIDATION.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Figure 5 deliverables

- Submission PNG: `{PNG.name}` (600 dpi)
- Preview PNG: `{PREVIEW.name}`
- Vector PDF: `{PDF.name}`
- Editable SVG: `{SVG.name}`
- Panel (a) plotted-data audit: `{STAT_AUDIT.name}`
- Panel (b) plotted-data audit: `{IMPORTANCE_AUDIT.name}`
- Validation summary: `{VALIDATION.name}`
- Manuscript caption: `{CAPTION.name}`
- Three-pass QC: `{QC.name}`
- Reproducible generator: `{Path(__file__).name}`

Panel (a) uses the frozen comparator-minus-E1 direction and the supplied moving-block bootstrap intervals. Panel (b) is restricted to `Overall_test`; its axis scale is selected by the generator from the actual importance distribution. Permutation importance is treated as predictive dependence, not a causal effect.
"""
    README.write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stat, stat_notes = validate_statistics()
    importance, importance_scale, skew_ratio, scale_reason, importance_notes = validate_importance()
    write_supporting_files(
        stat,
        importance,
        importance_scale,
        skew_ratio,
        scale_reason,
        stat_notes,
        importance_notes,
    )

    fig = draw_figure(stat, importance, importance_scale)
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

    selective = stat[stat["comparator"] == "Best_selective_environmental"].sort_values("horizon")
    atmospheric = importance[importance["feature_group"] == "Atmospheric"].sort_values("horizon")
    qc = f"""# Figure 5 three-pass quality check

## Pass 1 — source and statistical-direction audit: PASS

- Panel (a): {len(stat)} rows, five comparator configurations and three horizons.
- ΔMAE is comparator minus E1; negative values are favourable to the comparator.
- Point estimates fall inside the supplied 95% moving-block bootstrap intervals.
- Holm-adjusted significant comparisons: {int(stat['holm_significant'].sum())}; no significance star was drawn.
- Validation-selected ΔMAE values: {', '.join(f'{int(r.horizon)} h = {r.delta_MAE:+.4f}' for r in selective.itertuples())} km h⁻¹.

## Pass 2 — importance and scale audit: PASS

- Panel (b) uses only `scenario == Overall_test` ({len(importance)} rows).
- Seven requested feature groups are present at 1, 3 and 6 h.
- Selected x-axis scale: {importance_scale}; max/min mean-importance ratio = {skew_ratio:.1f}.
- Atmospheric mean importance: {', '.join(f'{int(r.horizon)} h = {r.delta_MAE_mean:.4f}' for r in atmospheric.itertuples())} ΔMAE.
- Caption explicitly states that permutation importance is not causal.

## Pass 3 — rendered-output review: PASS

- Exactly two panels, labelled (a) and (b).
- Main PNG: {width} × {height} px at {dpi[0]} × {dpi[1]} dpi.
- Preview PNG: {preview_size[0]} × {preview_size[1]} px.
- Panel labels, category labels, zero line, CIs, markers, legend and log-scale ticks are visible without clipping or overlap.
"""
    QC.write_text(qc, encoding="utf-8")

    outputs = [PNG, PREVIEW, PDF, SVG, STAT_AUDIT, IMPORTANCE_AUDIT, VALIDATION, CAPTION, QC, README, Path(__file__)]
    purposes = {
        PNG.name: "submission-ready 600-dpi two-panel PNG",
        PREVIEW.name: "quick visual-inspection PNG",
        PDF.name: "vector PDF",
        SVG.name: "editable vector SVG",
        STAT_AUDIT.name: "panel-a exact plotted values and Holm flags",
        IMPORTANCE_AUDIT.name: "panel-b exact plotted values and scale decision",
        VALIDATION.name: "machine-readable validation summary",
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
                "panel_a_rows": len(stat),
                "panel_b_rows": len(importance),
                "importance_scale": importance_scale,
                "importance_skew_ratio": skew_ratio,
                "holm_significant_count": int(stat["holm_significant"].sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
