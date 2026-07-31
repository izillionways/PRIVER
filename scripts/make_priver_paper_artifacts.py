from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper_figures" / "priver"
TABLE_DIR = ROOT / "paper_tables" / "priver"
SOURCE_TABLE_DIR = ROOT / "results" / "tables"

DOTA = ROOT / "outputs" / "dota_v15_val_458" / "priver_paper"
SODA = ROOT / "outputs" / "soda_a_val_holdout_526" / "priver_paper"
DEV = ROOT / "outputs" / "dota_v15_train_1411"

COLORS = {
    "semantic": "#4B5563",
    "prompt": "#C58A18",
    "graph": "#2F6B9A",
    "priver": "#168267",
    "accent": "#B23A48",
    "light": "#E8EEF2",
    "ink": "#1F2933",
}

LABELS = {
    "semantic": "Semantic",
    "prompt": "Prompt only",
    "graph": "Graph only",
    "priver": "PRIVER",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
    }.items():
        fig.savefig(
            FIG_DIR / f"{name}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
            **kwargs,
        )
    plt.close(fig)


def read_summary(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / "ablation_comparison" / "summary.csv")


def metric_value(
    frame: pd.DataFrame,
    method: str,
    metric: str,
) -> float:
    row = frame[(frame["method"] == method) & (frame["metric"] == metric)]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {method}/{metric}")
    return float(row.iloc[0]["mean"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + height - 0.18),
            width,
            0.18,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.text(
        x + width / 2,
        y + height - 0.09,
        title,
        ha="center",
        va="center",
        color="white",
        fontsize=5.8,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + (height - 0.18) / 2,
        body,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=5.4,
        linespacing=1.25,
    )


def make_overview() -> None:
    data_root = Path(os.environ.get("RS_DATA_ROOT", "/data/rsdata"))
    image_path = (
        data_root
        / "DOTA-v1.5"
        / "DOTA_V1.5"
        / "val"
        / "images"
        / "P1373.png"
    )
    if not image_path.exists():
        raise FileNotFoundError(
            "Fig. 1 requires DOTA-v1.5 val image P1373.png. "
            "Set RS_DATA_ROOT to the mounted dataset root."
        )

    image = plt.imread(image_path)
    x0, y0 = 0, 1500
    x1 = min(4000, image.shape[1])
    y1 = min(3500, image.shape[0])
    scene = image[y0:y1, x0:x1]

    fig = plt.figure(figsize=(7.15, 2.55), dpi=300)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(0.92, 1.72, 1.08),
        left=0.025,
        right=0.985,
        top=0.90,
        bottom=0.13,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def step_title(ax: plt.Axes, label: str, title: str) -> None:
        ax.text(
            0.00,
            1.02,
            label,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            0.10,
            1.02,
            title,
            ha="left",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
            color=COLORS["ink"],
        )

    # (a) Four controlled prompts are aggregated into one stable score.
    ax = axes[0]
    step_title(ax, "a", "Prompt-stable scoring")
    ax.add_patch(
        FancyBboxPatch(
            (0.13, 0.83),
            0.74,
            0.10,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F4F6F8",
            edgecolor="#C9D0D6",
            linewidth=0.7,
        )
    )
    ax.text(
        0.50,
        0.88,
        'class query  "planes"',
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["ink"],
    )

    prompt_labels = ("locate", "find", "retrieve", "which regions")
    score_labels = (r"$s_{i1}$", r"$s_{i2}$", r"$s_{i3}$", r"$s_{i4}$")
    prompt_y = (0.70, 0.57, 0.44, 0.31)
    for index, (label, score, y) in enumerate(
        zip(prompt_labels, score_labels, prompt_y),
        start=1,
    ):
        ax.add_patch(
            Circle(
                (0.13, y),
                0.025,
                facecolor=COLORS["prompt"],
                edgecolor="none",
            )
        )
        ax.text(
            0.13,
            y,
            str(index),
            ha="center",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color="white",
        )
        ax.text(
            0.20,
            y,
            label,
            ha="left",
            va="center",
            fontsize=5.9,
            color=COLORS["ink"],
        )
        ax.text(
            0.62,
            y,
            score,
            ha="center",
            va="center",
            fontsize=6.0,
            color=COLORS["prompt"],
        )
        ax.add_patch(
            FancyArrowPatch(
                (0.69, y),
                (0.80, 0.505),
                arrowstyle="-",
                connectionstyle=f"arc3,rad={0.08 * (2.5 - index):.2f}",
                linewidth=0.7,
                color="#C4CBD1",
            )
        )

    ax.add_patch(
        Circle(
            (0.84, 0.505),
            0.085,
            facecolor="#FFF7E6",
            edgecolor=COLORS["prompt"],
            linewidth=1.0,
        )
    )
    ax.text(
        0.84,
        0.52,
        "stable",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["ink"],
    )
    ax.text(
        0.84,
        0.475,
        r"$n_i$",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["prompt"],
    )
    ax.text(
        0.50,
        0.13,
        r"$\bar{s}_i=\mathrm{mean}_t(s_{it})"
        r"-0.5\,\mathrm{std}_t(s_{it})$"
        "\n"
        r"$\mathcal{C}=\operatorname{TopK}_{100}(\bar{\mathbf{s}}),\quad"
        r"n_i=\mathcal{N}_{\mathcal{C}}(\bar{s}_i)$",
        ha="center",
        va="center",
        fontsize=5.5,
        linespacing=1.35,
        color=COLORS["ink"],
    )

    # (b) A real aerial scene anchors the same- and inter-scale geometry.
    ax = axes[1]
    step_title(ax, "b", "Spatial corroboration")
    scene_width = x1 - x0
    scene_height = y1 - y0
    panel_position = ax.get_position()
    panel_aspect = (
        panel_position.width
        * fig.get_figwidth()
        / (panel_position.height * fig.get_figheight())
    )
    image_height = panel_aspect / (scene_width / scene_height)
    image_top = 0.92
    image_bottom = image_top - image_height
    ax.imshow(
        scene,
        cmap="gray",
        extent=(0, 1, image_bottom, image_top),
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )
    ax.add_patch(
        Rectangle(
            (0, image_bottom),
            1,
            image_top - image_bottom,
            facecolor="none",
            edgecolor="#AEB7BF",
            linewidth=0.8,
            zorder=8,
        )
    )

    def scene_bbox(
        bbox: tuple[int, int, int, int],
    ) -> tuple[float, float, float, float]:
        bx0, by0, bx1, by1 = bbox
        left = (bx0 - x0) / scene_width
        width = (bx1 - bx0) / scene_width
        top = image_top - (by0 - y0) / scene_height * (
            image_top - image_bottom
        )
        bottom = image_top - (by1 - y0) / scene_height * (
            image_top - image_bottom
        )
        return left, bottom, width, top - bottom

    patch_specs = (
        ("L", (768, 2304, 1792, 3328), COLORS["priver"], "-", 1.6),
        ("S1", (384, 2304, 896, 2816), COLORS["graph"], "-", 1.2),
        ("S2", (768, 2304, 1280, 2816), COLORS["graph"], "--", 1.2),
        ("U", (2304, 2304, 3328, 3328), COLORS["accent"], "-", 1.3),
    )
    centers: dict[str, tuple[float, float]] = {}
    for name, bbox, color, linestyle, linewidth in patch_specs:
        left, bottom, width, height = scene_bbox(bbox)
        centers[name] = (left + width / 2, bottom + height / 2)
        ax.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor="none",
                edgecolor=color,
                linestyle=linestyle,
                linewidth=linewidth,
                clip_on=False,
                zorder=5,
            )
        )
        label = {
            "L": "L · 1024 px",
            "S1": "S1 · 512 px",
            "S2": "S2 · 512 px",
            "U": "unsupported",
        }[name]
        if name == "L":
            label_x, label_y = left + width - 0.008, bottom + height - 0.010
            label_ha, label_va = "right", "top"
        elif name == "S2":
            label_x, label_y = left + 0.008, bottom + 0.010
            label_ha, label_va = "left", "bottom"
        else:
            label_x, label_y = left + 0.008, bottom + height - 0.010
            label_ha, label_va = "left", "top"
        ax.text(
            label_x,
            label_y,
            label,
            ha=label_ha,
            va=label_va,
            fontsize=4.9,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.13",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.94,
            },
            zorder=7,
        )

    ax.add_patch(
        FancyArrowPatch(
            centers["S1"],
            centers["S2"],
            arrowstyle="<->",
            mutation_scale=7,
            linewidth=1.1,
            color=COLORS["graph"],
            connectionstyle="arc3,rad=-0.12",
            zorder=7,
        )
    )
    for name in ("S2",):
        ax.add_patch(
            FancyArrowPatch(
                centers[name],
                centers["L"],
                arrowstyle="<->",
                mutation_scale=7,
                linewidth=1.1,
                color=COLORS["priver"],
                connectionstyle="arc3,rad=0.16",
                zorder=7,
            )
        )
    # A small visual-only inset makes the queried object recognizable without
    # drawing annotation boxes or implying that labels enter the method.
    detail_bbox = (980, 2640, 1140, 2800)
    dx0, dy0, dx1, dy1 = detail_bbox
    detail_ax = ax.inset_axes([0.505, 0.675, 0.205, 0.215], zorder=10)
    detail_ax.imshow(
        image[dy0:dy1, dx0:dx1],
        cmap="gray",
        interpolation="nearest",
    )
    detail_ax.set_xticks([])
    detail_ax.set_yticks([])
    for spine in detail_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(COLORS["prompt"])
        spine.set_linewidth(1.0)
    ax.text(
        0.607,
        0.655,
        "plane detail",
        ha="center",
        va="bottom",
        fontsize=4.8,
        fontweight="bold",
        color=COLORS["prompt"],
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=11,
    )
    detail_center = (
        ((dx0 + dx1) / 2 - x0) / scene_width,
        image_top
        - ((dy0 + dy1) / 2 - y0)
        / scene_height
        * (image_top - image_bottom),
    )
    ax.add_patch(
        FancyArrowPatch(
            detail_center,
            (0.505, 0.700),
            arrowstyle="-",
            linewidth=0.8,
            color=COLORS["prompt"],
            connectionstyle="arc3,rad=0.08",
            zorder=11,
        )
    )

    # (c) The score decomposition explains relative-rank changes.
    ax = axes[2]
    step_title(ax, "c", "Support-aware reranking")
    ax.text(
        0.50,
        0.89,
        r"$R_i=n_i+0.25S_{\mathrm{same}}(i)"
        r"+1.0S_{\mathrm{inter}}(i)$",
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["ink"],
    )
    legend_items = (
        ("semantic", COLORS["semantic"]),
        ("same scale", COLORS["graph"]),
        ("inter-scale", COLORS["priver"]),
    )
    legend_x = (0.12, 0.45, 0.78)
    for x, (label, color) in zip(legend_x, legend_items):
        ax.add_patch(
            Circle((x - 0.045, 0.79), 0.012, facecolor=color, edgecolor="none")
        )
        ax.text(
            x,
            0.79,
            label,
            ha="left",
            va="center",
            fontsize=5.1,
            color=COLORS["ink"],
        )

    supported_bbox = (768, 2304, 1280, 2816)
    unsupported_bbox = (2304, 2304, 3328, 3328)

    def patch_crop(bbox: tuple[int, int, int, int]) -> np.ndarray:
        bx0, by0, bx1, by1 = bbox
        return image[by0:by1, bx0:bx1]

    rows = (
        (
            0.49,
            patch_crop(supported_bbox),
            COLORS["priver"],
            "corroborated candidate",
            ((COLORS["semantic"], 0.17), (COLORS["graph"], 0.10),
             (COLORS["priver"], 0.22)),
            "rank rises",
            r"$\uparrow$",
        ),
        (
            0.17,
            patch_crop(unsupported_bbox),
            COLORS["accent"],
            "unsupported candidate",
            ((COLORS["semantic"], 0.24),),
            "rank may fall",
            r"$\downarrow$",
        ),
    )
    for y, thumbnail, border, label, segments, action, arrow in rows:
        thumb_ax = ax.inset_axes([0.02, y, 0.25, 0.23])
        thumb_ax.imshow(thumbnail, cmap="gray", interpolation="bilinear")
        thumb_ax.set_xticks([])
        thumb_ax.set_yticks([])
        for spine in thumb_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border)
            spine.set_linewidth(1.2)

        ax.text(
            0.31,
            y + 0.16,
            label,
            ha="left",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=border,
        )
        bar_x = 0.31
        bar_y = y + 0.07
        for color, width in segments:
            ax.add_patch(
                Rectangle(
                    (bar_x, bar_y),
                    width,
                    0.055,
                    facecolor=color,
                    edgecolor="none",
                )
            )
            bar_x += width + 0.007
        ax.text(
            0.93,
            y + 0.11,
            arrow,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=border,
        )
        ax.text(
            0.93,
            y + 0.035,
            action,
            ha="center",
            va="center",
            fontsize=5.2,
            color=border,
        )

    ax.text(
        0.50,
        0.025,
        "red border = no geometric support; score lengths are schematic",
        ha="center",
        va="bottom",
        fontsize=4.7,
        color="#6C7680",
    )

    save_figure(fig, "fig1_priver_overview")


def make_main_results(
    dota: pd.DataFrame,
    soda: pd.DataFrame,
) -> None:
    metrics = [
        ("hit_at_1", "Hit@1"),
        ("precision_at_10", "P@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("object_recall_at_10", "Object R@10"),
    ]
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    for ax, (dataset_name, frame), label in zip(
        axes,
        [("DOTA-v1.5", dota), ("SODA-A", soda)],
        ["a", "b"],
    ):
        x = np.arange(len(metrics))
        width = 0.34
        semantic = [
            metric_value(frame, "semantic", metric) for metric, _ in metrics
        ]
        priver = [
            metric_value(frame, "priver", metric) for metric, _ in metrics
        ]
        ax.bar(
            x - width / 2,
            semantic,
            width,
            color=COLORS["semantic"],
            label="Semantic",
        )
        ax.bar(
            x + width / 2,
            priver,
            width,
            color=COLORS["priver"],
            label="PRIVER",
        )
        for index, (base, method) in enumerate(zip(semantic, priver)):
            ax.text(
                index + width / 2,
                method + 0.018,
                f"+{method - base:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color=COLORS["priver"],
            )
            source_rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metrics[index][0],
                    "semantic": base,
                    "priver": method,
                    "delta": method - base,
                }
            )
        ax.set_xticks(x, [display for _, display in metrics])
        ax.set_ylim(0, 1.02)
        ax.set_title(dataset_name)
        ax.grid(axis="y", color="#D8DEE4", linewidth=0.5, alpha=0.8)
        panel_label(ax, label)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="upper left", ncol=2)
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "fig2_main_results")
    pd.DataFrame(source_rows).to_csv(
        TABLE_DIR / "fig2_main_results_source.csv",
        index=False,
    )


def make_ablation(
    dota: pd.DataFrame,
    soda: pd.DataFrame,
) -> None:
    metrics = [
        ("hit_at_1", "Hit@1"),
        ("precision_at_10", "P@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("object_recall_at_10", "Object R@10"),
    ]
    methods = ["semantic", "prompt", "graph", "priver"]
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), sharey=True)
    for ax, (dataset_name, frame), label in zip(
        axes,
        [("DOTA-v1.5", dota), ("SODA-A", soda)],
        ["a", "b"],
    ):
        x = np.arange(len(metrics))
        width = 0.19
        for method_index, method in enumerate(methods):
            values = [
                metric_value(frame, method, metric) for metric, _ in metrics
            ]
            offsets = (method_index - 1.5) * width
            ax.bar(
                x + offsets,
                values,
                width,
                color=COLORS[method],
                label=LABELS[method],
            )
            for (metric, _), value in zip(metrics, values):
                source_rows.append(
                    {
                        "dataset": dataset_name,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
        ax.set_xticks(x, [display for _, display in metrics])
        ax.set_ylim(0, 1.02)
        ax.set_title(dataset_name)
        ax.grid(axis="y", color="#D8DEE4", linewidth=0.5, alpha=0.8)
        panel_label(ax, label)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="upper left", ncol=2)
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "fig3_component_ablation")
    pd.DataFrame(source_rows).to_csv(
        TABLE_DIR / "fig3_component_ablation_source.csv",
        index=False,
    )


def make_risk_coverage() -> None:
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), sharey=True)
    for ax, dataset_name, root, label in [
        (axes[0], "DOTA-v1.5", DOTA, "a"),
        (axes[1], "SODA-A", SODA, "b"),
    ]:
        frame = pd.read_csv(root / "risk_coverage" / "risk_coverage.csv")
        selected = frame[
            frame["confidence"] == "consistency_confidence"
        ].copy()
        ax.plot(
            100 * selected["coverage"],
            selected["precision_at_10"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=COLORS["priver"],
            label="P@10",
        )
        ax.plot(
            100 * selected["coverage"],
            selected["hit_at_1"],
            marker="s",
            markersize=3,
            linewidth=1.2,
            color=COLORS["graph"],
            label="Hit@1",
        )
        ax.set_xlim(8, 102)
        ax.set_ylim(0.45, 1.02)
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_xlabel("Coverage (%)")
        ax.set_title(dataset_name)
        ax.grid(color="#D8DEE4", linewidth=0.5, alpha=0.8)
        panel_label(ax, label)
        source_rows.append(selected.assign(dataset=dataset_name))
    axes[0].set_ylabel("Score on retained queries")
    axes[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "fig4_risk_coverage")
    pd.concat(source_rows, ignore_index=True).to_csv(
        TABLE_DIR / "fig4_risk_coverage_source.csv",
        index=False,
    )


def make_hard_queries() -> None:
    subsets = [
        ("hard_composite", "Hard"),
        ("baseline_low_precision10", "Low P@10"),
        ("few_positive_patches", "Few positives"),
        ("easy_composite", "Easy"),
    ]
    source_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.6), sharey=True)
    for ax, dataset_name, root, label in [
        (axes[0], "DOTA-v1.5", DOTA, "a"),
        (axes[1], "SODA-A", SODA, "b"),
    ]:
        frame = pd.read_csv(root / "hard_query" / "hard_query_summary.csv")
        selected = frame.set_index("subset").loc[
            [name for name, _ in subsets]
        ]
        x = np.arange(len(subsets))
        width = 0.34
        baseline = selected["semantic_precision_at_10"].to_numpy()
        priver = selected["priver_precision_at_10"].to_numpy()
        baseline_hit1 = selected["semantic_hit_at_1"].to_numpy()
        priver_hit1 = selected["priver_hit_at_1"].to_numpy()
        ax.bar(
            x - width / 2,
            baseline,
            width,
            color=COLORS["semantic"],
            label="Semantic",
        )
        ax.bar(
            x + width / 2,
            priver,
            width,
            color=COLORS["priver"],
            label="PRIVER",
        )
        for index, (base, method) in enumerate(zip(baseline, priver)):
            ax.text(
                index + width / 2,
                method + 0.015,
                f"{method - base:+.3f}",
                ha="center",
                fontsize=6,
                color=(
                    COLORS["priver"]
                    if method >= base
                    else COLORS["accent"]
                ),
            )
            source_rows.append(
                {
                    "dataset": dataset_name,
                    "subset": subsets[index][0],
                    "num_queries": int(selected.iloc[index]["num_queries"]),
                    "semantic_precision_at_10": base,
                    "priver_precision_at_10": method,
                    "delta_precision_at_10": method - base,
                    "semantic_hit_at_1": baseline_hit1[index],
                    "priver_hit_at_1": priver_hit1[index],
                    "delta_hit_at_1": (
                        priver_hit1[index] - baseline_hit1[index]
                    ),
                }
            )
        ax.set_xticks(
            x,
            [display for _, display in subsets],
            rotation=18,
            ha="right",
        )
        ax.set_ylim(0, 1.02)
        ax.set_title(dataset_name)
        ax.grid(axis="y", color="#D8DEE4", linewidth=0.5, alpha=0.8)
        panel_label(ax, label)
    axes[0].set_ylabel("Precision@10")
    axes[1].legend(loc="upper left")
    fig.tight_layout(w_pad=1.3)
    save_figure(fig, "fig5_hard_queries")
    pd.DataFrame(source_rows).to_csv(
        TABLE_DIR / "fig5_hard_queries_source.csv",
        index=False,
    )


def make_class_gains() -> None:
    frame = pd.read_csv(DOTA / "subgroup" / "by_class_name.csv")
    eligible = frame[frame["num_queries"] >= 32].sort_values(
        "delta_precision_at_10"
    )
    colors = [
        COLORS["priver"] if value >= 0 else COLORS["accent"]
        for value in eligible["delta_precision_at_10"]
    ]
    fig, ax = plt.subplots(figsize=(3.55, 4.2))
    y = np.arange(len(eligible))
    ax.barh(y, eligible["delta_precision_at_10"], color=colors)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_yticks(y, eligible["group"].str.replace("-", " "))
    ax.set_xlabel(r"$\Delta$ Precision@10")
    ax.set_title("DOTA-v1.5 class-level gains")
    ax.grid(axis="x", color="#D8DEE4", linewidth=0.5, alpha=0.8)
    for index, value in enumerate(eligible["delta_precision_at_10"]):
        ax.text(
            value + 0.002,
            index,
            f"{value:+.3f}",
            va="center",
            fontsize=5.8,
            color=COLORS["ink"],
        )
    fig.tight_layout()
    save_figure(fig, "fig6_class_gains")
    eligible.to_csv(TABLE_DIR / "fig6_class_gains_source.csv", index=False)

    soda_metrics = pd.read_csv(
        SODA / "hard_query" / "query_metrics_with_difficulty.csv"
    )
    soda_rows = []
    for class_name, class_frame in soda_metrics.groupby("class_name"):
        semantic = class_frame[
            class_frame["method"] == "semantic"
        ]["precision_at_10"]
        priver = class_frame[
            class_frame["method"] == "priver"
        ]["precision_at_10"]
        if len(semantic) != len(priver):
            raise ValueError(
                f"Mismatched SODA-A class rows for {class_name}"
            )
        semantic_mean = float(semantic.mean())
        priver_mean = float(priver.mean())
        soda_rows.append(
            {
                "group_type": "class_name",
                "group": class_name,
                "num_queries": int(len(priver)),
                "semantic_precision_at_10": semantic_mean,
                "priver_precision_at_10": priver_mean,
                "delta_precision_at_10": priver_mean - semantic_mean,
            }
        )
    pd.DataFrame(soda_rows).to_csv(
        TABLE_DIR / "fig6_soda_class_gains_source.csv",
        index=False,
    )


def make_failure_analyses() -> None:
    hard = pd.read_csv(TABLE_DIR / "fig5_hard_queries_source.csv")
    dota_classes = pd.read_csv(
        TABLE_DIR / "fig6_class_gains_source.csv"
    )
    soda_classes = pd.read_csv(
        TABLE_DIR / "fig6_soda_class_gains_source.csv"
    )
    subset_order = [
        ("hard_composite", "Hard"),
        ("baseline_low_precision10", "Low P@10"),
        ("few_positive_patches", "Few positives"),
        ("easy_composite", "Easy"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.25),
        gridspec_kw={
            "height_ratios": [1.0, 1.65],
            "hspace": 0.62,
            "wspace": 0.55,
        },
    )
    width = 0.34

    for ax, dataset_name, label in [
        (axes[0, 0], "DOTA-v1.5", "a"),
        (axes[0, 1], "SODA-A", "b"),
    ]:
        selected = hard[hard["dataset"] == dataset_name].set_index("subset")
        selected = selected.loc[[name for name, _ in subset_order]]
        x = np.arange(len(subset_order))
        precision_delta = selected["delta_precision_at_10"].to_numpy()
        hit1_delta = selected["delta_hit_at_1"].to_numpy()
        precision_bars = ax.bar(
            x - width / 2,
            precision_delta,
            width,
            color=COLORS["priver"],
            label=r"$\Delta$ P@10",
        )
        hit1_bars = ax.bar(
            x + width / 2,
            hit1_delta,
            width,
            color=COLORS["graph"],
            label=r"$\Delta$ Hit@1",
        )
        for bars in (precision_bars, hit1_bars):
            for bar in bars:
                value = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.006 if value >= 0 else -0.008),
                    f"{value:+.3f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=5.4,
                    color=COLORS["ink"],
                )
        ax.set_xticks(
            x,
            [display for _, display in subset_order],
            rotation=20,
            ha="right",
        )
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.set_ylim(-0.085, 0.23)
        ax.set_title(dataset_name)
        ax.grid(axis="y", color="#D8DEE4", linewidth=0.5, alpha=0.8)
        panel_label(ax, label)

    axes[0, 0].set_ylabel("Change from semantic baseline")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
    )

    class_panels = [
        (
            axes[1, 0],
            "DOTA-v1.5 class gains",
            dota_classes,
            "c",
        ),
        (
            axes[1, 1],
            "SODA-A class gains",
            soda_classes,
            "d",
        ),
    ]
    for class_ax, title, classes, label in class_panels:
        classes = classes[classes["num_queries"] >= 32].sort_values(
            "delta_precision_at_10"
        )
        y = np.arange(len(classes))
        colors = [
            COLORS["priver"] if value >= 0 else COLORS["accent"]
            for value in classes["delta_precision_at_10"]
        ]
        class_ax.barh(
            y,
            classes["delta_precision_at_10"],
            color=colors,
        )
        class_ax.set_yticks(
            y,
            classes["group"].str.replace("-", " "),
        )
        class_ax.set_xlim(0, 0.15)
        class_ax.set_xlabel(r"$\Delta$ Precision@10")
        class_ax.set_title(title)
        class_ax.grid(
            axis="x",
            color="#D8DEE4",
            linewidth=0.5,
            alpha=0.8,
        )
        class_ax.tick_params(axis="y", labelsize=6.2)
        for index, value in enumerate(
            classes["delta_precision_at_10"]
        ):
            class_ax.text(
                value + 0.002,
                index,
                f"{value:+.3f}",
                va="center",
                fontsize=5.6,
                color=COLORS["ink"],
            )
        panel_label(class_ax, label)
    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        bottom=0.09,
        top=0.91,
    )
    save_figure(fig, "fig5_failure_analyses")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_candidate_pool_sensitivity() -> None:
    frame = pd.read_csv(
        SOURCE_TABLE_DIR / "fig_s1_candidate_pool_source.csv"
    )
    fig, ax = plt.subplots(figsize=(3.55, 2.35))
    for metric, label, marker, color in [
        ("hit_at_1", "Hit@1", "o", COLORS["graph"]),
        ("precision_at_10", "P@10", "s", COLORS["priver"]),
        ("ndcg_at_10", "nDCG@10", "^", COLORS["prompt"]),
    ]:
        ax.plot(
            frame["candidate_k"],
            frame[metric],
            marker=marker,
            color=color,
            linewidth=1.3,
            markersize=4,
            label=label,
        )
    ax.axvline(100, color=COLORS["accent"], linestyle="--", linewidth=0.9)
    ax.text(103, 0.78, "selected", color=COLORS["accent"], fontsize=6)
    ax.set_xticks([50, 100, 200])
    ax.set_xlabel(r"Candidate-pool size $K_c$")
    ax.set_ylabel("Development score")
    ax.set_ylim(0.60, 0.80)
    ax.grid(color="#D8DEE4", linewidth=0.5, alpha=0.8)
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig_s1_candidate_pool")
    frame.to_csv(
        TABLE_DIR / "fig_s1_candidate_pool_source.csv",
        index=False,
    )


def export_tables(
    dota: pd.DataFrame,
    soda: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            dota.assign(dataset="DOTA-v1.5"),
            soda.assign(dataset="SODA-A"),
        ],
        ignore_index=True,
    ).to_csv(TABLE_DIR / "all_ablation_metrics.csv", index=False)

    encoder = pd.read_csv(
        DOTA / "encoder_comparison" / "summary.csv"
    )
    encoder.to_csv(TABLE_DIR / "encoder_transfer.csv", index=False)

    nms = pd.concat(
        [
            pd.read_csv(DOTA / "diversity_nms" / "summary.csv").assign(
                dataset="DOTA-v1.5"
            ),
            pd.read_csv(SODA / "diversity_nms" / "summary.csv").assign(
                dataset="SODA-A"
            ),
        ],
        ignore_index=True,
    )
    nms.to_csv(TABLE_DIR / "nms_object_coverage.csv", index=False)

    templates = pd.concat(
        [
            pd.read_csv(
                DOTA
                / "query_templates"
                / "query_template_robustness.csv"
            ).assign(dataset="DOTA-v1.5"),
            pd.read_csv(
                SODA
                / "query_templates"
                / "query_template_robustness.csv"
            ).assign(dataset="SODA-A"),
        ],
        ignore_index=True,
    )
    templates.to_csv(TABLE_DIR / "query_template_robustness.csv", index=False)


def main() -> None:
    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    dota = read_summary(DOTA)
    soda = read_summary(SODA)
    export_tables(dota, soda)
    make_overview()
    make_main_results(dota, soda)
    make_ablation(dota, soda)
    make_risk_coverage()
    make_hard_queries()
    make_class_gains()
    make_failure_analyses()
    make_candidate_pool_sensitivity()
    print(
        json.dumps(
            {
                "figure_dir": str(FIG_DIR),
                "table_dir": str(TABLE_DIR),
                "figures": 8,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
