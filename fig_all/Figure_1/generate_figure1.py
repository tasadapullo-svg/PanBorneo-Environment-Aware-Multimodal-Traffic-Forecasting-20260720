from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter


PROJECT = Path(r"C:\Users\DELL\Desktop\多数据源数据\2026071901")
PACKAGE = PROJECT / "T2_FINAL_EXPERIMENT_PACKAGE"
SOURCE = PACKAGE / "12_figure_source_data" / "Fig1_corridor_environmental_grids" / "source_data.csv"
MASTER = PACKAGE / "01_final_data" / "01_fcd_reference" / "node_master_final.csv"
ORDER = PACKAGE / "01_final_data" / "01_fcd_reference" / "node_order_verified.csv"
OUT = PROJECT / "SCI_FIGURES" / "Figure_1"
ASSETS = OUT / "assets"
TILES = ASSETS / "osm_tiles"

STEM = "Figure1_study_corridor_multisource_mapping"

NATURAL_EARTH_ADMIN0 = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_50m_admin_0_countries.geojson"
)
NATURAL_EARTH_ADMIN1 = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_50m_admin_1_states_provinces.geojson"
)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "T2-SCI-Figure/1.0 (academic static map)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def prop(props: dict, *names: str) -> str:
    lower = {str(k).lower(): v for k, v in props.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None:
            return str(value)
    return ""


def polygon_rings(geometry: dict):
    if not geometry:
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        for polygon in [coordinates]:
            if polygon:
                yield polygon[0]
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if polygon:
                yield polygon[0]


def validate_sources() -> tuple[pd.DataFrame, dict, list[dict]]:
    for path in (SOURCE, MASTER, ORDER):
        if not path.exists():
            raise FileNotFoundError(path)

    data = pd.read_csv(SOURCE)
    master = pd.read_csv(MASTER)
    order = pd.read_csv(ORDER)

    numeric = [
        "node_order_verified",
        "corridor_km",
        "latitude",
        "longitude",
        "returned_grid_latitude",
        "returned_grid_longitude",
        "distance_to_grid_km",
        "returned_grid_latitude_aq",
        "returned_grid_longitude_aq",
        "distance_to_aq_grid_km",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data = data.sort_values("node_order_verified").reset_index(drop=True)
    order = order.sort_values("node_order_verified").reset_index(drop=True)
    master = master.sort_values("node_order_verified").reset_index(drop=True)

    if len(data) != 51 or data["node_id"].nunique() != 51:
        raise AssertionError("The frozen Figure 1 source must contain 51 unique FCD nodes")
    if not np.array_equal(data["node_id"].to_numpy(), order["node_id"].to_numpy()):
        raise AssertionError("Figure source and node_order_verified.csv disagree")
    if not np.array_equal(data["node_id"].to_numpy(), master["node_id"].to_numpy()):
        raise AssertionError("Figure source and node_master_final.csv disagree on node order")
    if not np.allclose(data["latitude"], master["latitude"], atol=1e-10, rtol=0):
        raise AssertionError("Latitude mismatch against node_master_final.csv")
    if not np.allclose(data["longitude"], master["longitude"], atol=1e-10, rtol=0):
        raise AssertionError("Longitude mismatch against node_master_final.csv")
    if not data["corridor_km"].is_monotonic_increasing:
        raise AssertionError("corridor_km is not monotonic in verified order")
    if "row_status" in data and not data["row_status"].eq("PASS").all():
        raise AssertionError("At least one Figure 1 source row is not PASS")

    representative = ["N001", "N010", "N020", "N030", "N040", "N051"]
    missing_labels = sorted(set(representative) - set(data["node_id"]))
    if missing_labels:
        raise AssertionError(f"Representative labels missing: {missing_labels}")

    summary = {
        "status": "PASS",
        "node_count": int(len(data)),
        "unique_node_count": int(data["node_id"].nunique()),
        "corridor_km_min": float(data["corridor_km"].min()),
        "corridor_km_max": float(data["corridor_km"].max()),
        "corridor_km_span": float(data["corridor_km"].max() - data["corridor_km"].min()),
        "unique_weather_grids": int(data["weather_grid_id"].nunique()),
        "unique_cams_grids": int(data["aq_grid_id"].nunique()),
        "node_order_mismatches": 0,
        "coordinate_mismatches_vs_master": 0,
        "failed_source_rows": 0,
        "representative_labels": representative,
        "source_sha256": sha256(SOURCE),
        "node_master_sha256": sha256(MASTER),
        "node_order_sha256": sha256(ORDER),
    }
    checks = [
        {"check": "51 unique FCD nodes", "status": "PASS", "detail": "rows=51; unique node_id=51"},
        {"check": "verified corridor order", "status": "PASS", "detail": "order mismatches=0"},
        {"check": "coordinates against node master", "status": "PASS", "detail": "coordinate mismatches=0"},
        {
            "check": "frozen corridor coverage",
            "status": "PASS",
            "detail": f"{summary['corridor_km_min']:.3f} to {summary['corridor_km_max']:.3f} km",
        },
        {
            "check": "environmental grid counts",
            "status": "PASS",
            "detail": f"weather={summary['unique_weather_grids']}; CAMS={summary['unique_cams_grids']}",
        },
        {"check": "source row status", "status": "PASS", "detail": "51/51 PASS"},
    ]
    return data, summary, checks


def lon_to_tile_x(lon: float, zoom: int) -> int:
    return int((lon + 180.0) / 360.0 * (2**zoom))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2**zoom))


def tile_x_to_lon(x: int, zoom: int) -> float:
    return x / (2**zoom) * 360.0 - 180.0


def tile_y_to_lat(y: int, zoom: int) -> float:
    n = math.pi - 2.0 * math.pi * y / (2**zoom)
    return math.degrees(math.atan(math.sinh(n)))


def osm_tile(zoom: int, x: int, y: int) -> Path:
    path = TILES / str(zoom) / str(x) / f"{y}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "T2-SCI-Figure/1.0 (academic static map; OpenStreetMap attribution included)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
        time.sleep(0.08)
    return path


def draw_osm(ax, bounds: tuple[float, float, float, float], zoom: int = 11) -> int:
    min_lon, max_lon, min_lat, max_lat = bounds
    x0, x1 = lon_to_tile_x(min_lon, zoom), lon_to_tile_x(max_lon, zoom)
    y0, y1 = lat_to_tile_y(max_lat, zoom), lat_to_tile_y(min_lat, zoom)
    count = 0
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile_path = osm_tile(zoom, x, y)
            image = np.asarray(Image.open(tile_path).convert("RGB"), dtype=float) / 255.0
            gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
            light = 0.30 * gray + 0.70
            rgb = np.dstack([light, light * 0.995, light * 0.985])
            left, right = tile_x_to_lon(x, zoom), tile_x_to_lon(x + 1, zoom)
            top, bottom = tile_y_to_lat(y, zoom), tile_y_to_lat(y + 1, zoom)
            ax.imshow(
                rgb,
                extent=[left, right, bottom, top],
                origin="upper",
                interpolation="bilinear",
                zorder=0,
            )
            count += 1
    return count


def draw_scale_bar(ax, bounds: tuple[float, float, float, float], latitude: float, length_km: float = 20.0) -> None:
    min_lon, max_lon, min_lat, max_lat = bounds
    km_per_degree_lon = 111.320 * math.cos(math.radians(latitude))
    total_width = length_km / km_per_degree_lon
    segment_width = total_width / 2.0
    x0 = min_lon + 0.035 * (max_lon - min_lon)
    y0 = min_lat + 0.055 * (max_lat - min_lat)
    height = 0.014 * (max_lat - min_lat)
    pad_x = 0.012 * (max_lon - min_lon)
    pad_y = 0.020 * (max_lat - min_lat)
    ax.add_patch(
        Rectangle(
            (x0 - pad_x, y0 - pad_y),
            total_width + 2 * pad_x,
            height + 3.1 * pad_y,
            facecolor="white",
            edgecolor="0.55",
            linewidth=0.45,
            alpha=0.88,
            zorder=7,
        )
    )
    for index, face in enumerate(("#1F2937", "white")):
        ax.add_patch(
            Rectangle(
                (x0 + index * segment_width, y0),
                segment_width,
                height,
                facecolor=face,
                edgecolor="#1F2937",
                linewidth=0.65,
                zorder=8,
            )
        )
    ax.text(x0, y0 - 0.006, "0", ha="center", va="top", fontsize=6.5, zorder=9)
    ax.text(x0 + segment_width, y0 - 0.006, "10", ha="center", va="top", fontsize=6.5, zorder=9)
    ax.text(x0 + total_width, y0 - 0.006, "20 km", ha="center", va="top", fontsize=6.5, zorder=9)


def draw_north_arrow(ax, bounds: tuple[float, float, float, float]) -> None:
    min_lon, max_lon, min_lat, max_lat = bounds
    x = max_lon - 0.045 * (max_lon - min_lon)
    y_base = max_lat - 0.175 * (max_lat - min_lat)
    y_tip = max_lat - 0.050 * (max_lat - min_lat)
    ax.annotate(
        "",
        xy=(x, y_tip),
        xytext=(x, y_base),
        arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.15, mutation_scale=12),
        zorder=9,
    )
    ax.text(x, y_tip + 0.012 * (max_lat - min_lat), "N", ha="center", va="bottom", weight="bold", fontsize=8, zorder=9)


def draw_geojson_feature(ax, feature: dict, facecolor: str, edgecolor: str, linewidth: float, zorder: int) -> None:
    for ring in polygon_rings(feature.get("geometry", {})):
        coords = np.asarray(ring, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2:
            continue
        ax.add_patch(
            Polygon(
                coords[:, :2],
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=zorder,
            )
        )


def draw_inset(ax, corridor_mid: tuple[float, float], admin0: dict, admin1: dict) -> None:
    inset = ax.inset_axes([0.018, 0.735, 0.235, 0.245], zorder=20)
    inset.set_facecolor("#EEF4F7")
    bbox = (98.0, 120.0, -5.0, 9.0)
    for feature in admin0.get("features", []):
        geometry = feature.get("geometry", {})
        all_points = [np.asarray(ring) for ring in polygon_rings(geometry)]
        if not all_points:
            continue
        coords = np.vstack(all_points)
        if coords[:, 0].max() < bbox[0] or coords[:, 0].min() > bbox[1] or coords[:, 1].max() < bbox[2] or coords[:, 1].min() > bbox[3]:
            continue
        country = prop(feature.get("properties", {}), "ADMIN", "NAME", "SOVEREIGNT")
        is_malaysia = country.lower() == "malaysia"
        draw_geojson_feature(
            inset,
            feature,
            facecolor="#D8D5CE" if not is_malaysia else "#C8D8E6",
            edgecolor="#7A7A78" if not is_malaysia else "#36576E",
            linewidth=0.28 if not is_malaysia else 0.65,
            zorder=2 if not is_malaysia else 3,
        )

    sarawak_feature = None
    for feature in admin1.get("features", []):
        props = feature.get("properties", {})
        name = prop(props, "name", "name_en", "gn_name")
        country = prop(props, "admin", "geonunit", "sovereignt")
        if name.lower() == "sarawak" and (not country or country.lower() == "malaysia"):
            sarawak_feature = feature
            break
    if sarawak_feature is not None:
        draw_geojson_feature(inset, sarawak_feature, "#E69F00", "#7A4C00", 0.65, 5)

    lon, lat = corridor_mid
    inset.scatter([lon], [lat], s=20, marker="*", c="#B2182B", edgecolors="white", linewidths=0.45, zorder=8)
    inset.annotate(
        "Study corridor",
        xy=(lon, lat),
        xytext=(112.0, 4.4),
        fontsize=5.8,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", lw=0.55, color="#B2182B"),
        color="#7F1D1D",
        zorder=9,
    )
    inset.text(102.3, 5.6, "MALAYSIA", fontsize=5.6, color="#36576E", weight="bold")
    inset.text(111.0, 1.5, "Sarawak", fontsize=5.5, color="#5A3900", weight="bold")
    inset.set_xlim(bbox[0], bbox[1])
    inset.set_ylim(bbox[2], bbox[3])
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#6B7280")
        spine.set_linewidth(0.55)


def make_figure(data: pd.DataFrame, summary: dict) -> tuple[plt.Figure, int]:
    weather = (
        data[["weather_grid_id", "returned_grid_longitude", "returned_grid_latitude"]]
        .drop_duplicates("weather_grid_id")
        .sort_values("weather_grid_id")
    )
    cams = (
        data[["aq_grid_id", "returned_grid_longitude_aq", "returned_grid_latitude_aq"]]
        .drop_duplicates("aq_grid_id")
        .sort_values("aq_grid_id")
    )

    lon_min = min(data["longitude"].min(), weather["returned_grid_longitude"].min(), cams["returned_grid_longitude_aq"].min())
    lon_max = max(data["longitude"].max(), weather["returned_grid_longitude"].max(), cams["returned_grid_longitude_aq"].max())
    lat_min = min(data["latitude"].min(), weather["returned_grid_latitude"].min(), cams["returned_grid_latitude_aq"].min())
    lat_max = max(data["latitude"].max(), weather["returned_grid_latitude"].max(), cams["returned_grid_latitude_aq"].max())
    bounds = (lon_min - 0.29, lon_max + 0.07, lat_min - 0.09, lat_max + 0.09)

    admin0_path = ASSETS / "ne_50m_admin_0_countries.geojson"
    admin1_path = ASSETS / "ne_50m_admin_1_states_provinces.geojson"
    ensure_download(NATURAL_EARTH_ADMIN0, admin0_path)
    ensure_download(NATURAL_EARTH_ADMIN1, admin1_path)
    admin0 = json.loads(admin0_path.read_text(encoding="utf-8"))
    admin1 = json.loads(admin1_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(7.08, 5.25))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.985)
    tile_count = draw_osm(ax, bounds, zoom=11)

    corridor_color = "#1F2937"
    node_color = "#0072B2"
    weather_color = "#D17C00"
    cams_color = "#00866A"

    ax.plot(
        data["longitude"],
        data["latitude"],
        color="white",
        linewidth=4.2,
        alpha=0.92,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        data["longitude"],
        data["latitude"],
        color=corridor_color,
        linewidth=2.45,
        solid_capstyle="round",
        label="Study corridor",
        zorder=4,
    )
    ax.scatter(
        data["longitude"],
        data["latitude"],
        s=17,
        marker="o",
        c=node_color,
        edgecolors="white",
        linewidths=0.45,
        zorder=6,
    )
    ax.scatter(
        weather["returned_grid_longitude"],
        weather["returned_grid_latitude"],
        s=52,
        marker="o",
        facecolors="none",
        edgecolors=weather_color,
        linewidths=1.15,
        zorder=5,
    )
    ax.scatter(
        cams["returned_grid_longitude_aq"],
        cams["returned_grid_latitude_aq"],
        s=48,
        marker="s",
        facecolors="none",
        edgecolors=cams_color,
        linewidths=1.15,
        zorder=5,
    )

    label_offsets = {
        "N001": (7, -13),
        "N010": (-6, 8),
        "N020": (-7, 8),
        "N030": (-9, -14),
        "N040": (7, 8),
        "N051": (-7, 9),
    }
    for node_id, offset in label_offsets.items():
        row = data.loc[data["node_id"].eq(node_id)].iloc[0]
        label = node_id
        if node_id == "N001":
            label = "N001  Start"
        elif node_id == "N051":
            label = "N051  End"
        ax.annotate(
            label,
            xy=(row["longitude"], row["latitude"]),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=6.9,
            weight="bold" if node_id in {"N001", "N051"} else "normal",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor="none", alpha=0.78),
            zorder=10,
        )

    direction_start = data.iloc[30]
    direction_end = data.iloc[35]
    arrow = FancyArrowPatch(
        (direction_start["longitude"], direction_start["latitude"]),
        (direction_end["longitude"], direction_end["latitude"]),
        arrowstyle="-|>",
        mutation_scale=12,
        color="#B2182B",
        linewidth=1.35,
        zorder=8,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    mid_x = (direction_start["longitude"] + direction_end["longitude"]) / 2
    mid_y = (direction_start["latitude"] + direction_end["latitude"]) / 2
    ax.text(
        mid_x,
        mid_y - 0.020,
        "Corridor direction",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#7F1D1D",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.1),
        zorder=9,
    )

    handles = [
        Line2D([0], [0], color=corridor_color, lw=2.4, label="Study corridor"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=node_color, markeredgecolor="white", markersize=5.5, label="FCD node (n=51)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=weather_color, markeredgewidth=1.1, markersize=6.5, label=f"Open-Meteo weather grid (n={summary['unique_weather_grids']})"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor=cams_color, markeredgewidth=1.1, markersize=6.2, label=f"CAMS-derived atmospheric-composition grid (n={summary['unique_cams_grids']})"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.805),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#9CA3AF",
        borderpad=0.65,
        handlelength=2.4,
        labelspacing=0.55,
    )
    legend.get_frame().set_linewidth(0.55)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    mean_lat = float(data["latitude"].mean())
    ax.set_aspect(1.0 / math.cos(math.radians(mean_lat)), adjustable="box")
    first_tick = math.ceil(bounds[0] * 5.0) / 5.0
    ax.set_xticks(np.arange(first_tick, bounds[1] - 0.05, 0.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.1f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.1f}°N"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="white", linewidth=0.55, alpha=0.62, zorder=1)

    draw_scale_bar(ax, bounds, mean_lat, length_km=20.0)
    draw_north_arrow(ax, bounds)
    draw_inset(
        ax,
        (float(data["longitude"].mean()), float(data["latitude"].mean())),
        admin0,
        admin1,
    )

    fig.text(
        0.985,
        0.025,
        "Basemap © OpenStreetMap contributors (ODbL); inset boundaries: Natural Earth.",
        ha="right",
        va="bottom",
        fontsize=6.0,
        color="#4B5563",
    )
    fig.text(
        0.075,
        0.025,
        f"Frozen corridor coverage: {summary['corridor_km_min']:.3f}–{summary['corridor_km_max']:.3f} km.",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color="#4B5563",
    )
    return fig, tile_count


def save_outputs(fig: plt.Figure) -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [
        (OUT / f"{STEM}.pdf", {"format": "pdf"}),
        (OUT / f"{STEM}.svg", {"format": "svg"}),
        (OUT / f"{STEM}.png", {"format": "png", "dpi": 600}),
        (
            OUT / f"{STEM}.tiff",
            {"format": "tiff", "dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        ),
        (OUT / f"{STEM}_preview.png", {"format": "png", "dpi": 200}),
    ]
    for path, kwargs in outputs:
        fig.savefig(path, facecolor="white", **kwargs)
    records = []
    for path, _ in outputs:
        record = {
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix.lower() in {".png", ".tif", ".tiff"}:
            with Image.open(path) as image:
                record["pixel_width"] = image.width
                record["pixel_height"] = image.height
                dpi = image.info.get("dpi")
                if isinstance(dpi, tuple):
                    dpi = tuple(float(value) for value in dpi)
                record["dpi"] = dpi
        records.append(record)
    return records


def write_documentation(summary: dict, checks: list[dict], outputs: list[dict], tile_count: int) -> None:
    (OUT / "Figure1_source_validation.json").write_text(
        json.dumps({**summary, "osm_tiles_used": tile_count}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (OUT / "Figure1_data_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    with (OUT / "Figure1_output_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["file", "size_bytes", "sha256", "pixel_width", "pixel_height", "dpi"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(outputs)

    caption = f"""# Figure 1 caption

**Figure 1. Study corridor and multisource environmental data mapping.** The verified tropical highway corridor contains 51 floating-car-data (FCD) nodes ordered by the frozen corridor definition ({summary['corridor_km_min']:.3f}–{summary['corridor_km_max']:.3f} km). The map shows the FCD nodes and corridor direction together with {summary['unique_weather_grids']} unique Open-Meteo Historical Weather API grid locations and {summary['unique_cams_grids']} unique CAMS-derived atmospheric-composition grid locations used for spatial matching. Only representative nodes are labelled. The inset locates the study corridor within Sarawak, Malaysia. Basemap © OpenStreetMap contributors (ODbL); inset boundaries from Natural Earth.

The map describes spatial alignment only and does not present forecasting performance.
"""
    (OUT / "Figure1_caption.md").write_text(caption, encoding="utf-8")

    readme = f"""# Figure 1 delivery

This folder contains one integrated GIS-style figure; it is not a multi-panel figure.

## Frozen-data facts

- FCD nodes: {summary['node_count']}
- Frozen corridor coverage: {summary['corridor_km_min']:.3f}–{summary['corridor_km_max']:.3f} km
- Unique Open-Meteo weather grids: {summary['unique_weather_grids']}
- Unique CAMS-derived atmospheric-composition grids: {summary['unique_cams_grids']}
- Node-order mismatches: 0
- Coordinate mismatches against node_master_final.csv: 0

## Formats

- PDF and SVG: editable vector versions.
- PNG and TIFF: 600 dpi publication versions.
- Preview PNG: 200 dpi review version.

## Map conventions

- Filled circles: FCD nodes.
- Open circles: Open-Meteo weather grids.
- Open squares: CAMS-derived atmospheric-composition grids.
- No claim is made that CAMS grids are roadside or ground-station observations.
- No 131 km label is used; the figure reports the frozen 95.690 km coverage.

## Reproduction

Run `generate_figure1.py` with Python plus pandas, numpy, Pillow, and matplotlib. Cached OpenStreetMap tiles and Natural Earth GeoJSON are retained under `assets` for exact rerendering.
"""
    (OUT / "README_Figure1.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    data, summary, checks = validate_sources()
    figure, tile_count = make_figure(data, summary)
    outputs = save_outputs(figure)
    plt.close(figure)
    write_documentation(summary, checks, outputs, tile_count)
    print(json.dumps({"status": "PASS", **summary, "osm_tiles_used": tile_count, "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
