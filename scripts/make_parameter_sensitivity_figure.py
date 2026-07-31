from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from priver.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot parameter sensitivity heatmaps.")
    parser.add_argument(
        "--sensitivity-dir",
        default="outputs/dota_v15_100/parameter_sensitivity",
        help="Directory containing parameter sensitivity CSV files.",
    )
    parser.add_argument("--fig-dir", default="paper_figures", help="Output figure directory.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "figure.dpi": 150,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str, xlabel: str, ylabel: str) -> None:
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.60, vmax=0.66)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    for y, row in enumerate(pivot.index):
        for x, col in enumerate(pivot.columns):
            value = pivot.loc[row, col]
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", color="white", fontsize=5.6)
    return image


def main() -> None:
    args = parse_args()
    sensitivity_dir = Path(args.sensitivity_dir)
    fig_dir = ensure_dir(args.fig_dir)
    configure_matplotlib()

    full = pd.read_csv(sensitivity_dir / "full_parameter_grid.csv")
    adaptive = pd.read_csv(sensitivity_dir / "adaptive_parameter_grid.csv")

    full_pivot = full.pivot(index="alpha", columns="beta", values="precision_at_10").sort_index(ascending=True)
    adaptive_pivot = (
        adaptive.pivot(index="gamma", columns="beta", values="precision_at_10").sort_index(ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    image = draw_heatmap(
        axes[0],
        full_pivot,
        "Full rerank Precision@10",
        "beta: cross-scale weight",
        "alpha: spatial weight",
    )
    draw_heatmap(
        axes[1],
        adaptive_pivot,
        "Adaptive rerank Precision@10",
        "beta: cross-scale weight",
        "gamma: semantic gate",
    )
    fig.colorbar(image, ax=axes, shrink=0.82, label="Precision@10")
    save_figure(fig, fig_dir / "fig_s_parameter_sensitivity")
    print({"figure": str(fig_dir / "fig_s_parameter_sensitivity")})


if __name__ == "__main__":
    main()
