from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from priver.io import ensure_dir


METRICS = ["hit_at_1", "hit_at_10", "precision_at_10", "recall_at_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reranking parameter sensitivity grids.")
    parser.add_argument(
        "--full-grid",
        required=True,
        help="Path to full-rerank grid_search_summary.json.",
    )
    parser.add_argument(
        "--adaptive-grid",
        required=True,
        help="Path to adaptive grid_search_summary.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/dota_v15_100/parameter_sensitivity",
        help="Output directory for summary tables.",
    )
    return parser.parse_args()


def read_grid(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def metric_stability(df: pd.DataFrame, method: str) -> dict:
    row = {"method": method, "num_settings": int(len(df))}
    for metric in METRICS:
        row[f"{metric}_min"] = float(df[metric].min())
        row[f"{metric}_max"] = float(df[metric].max())
        row[f"{metric}_mean"] = float(df[metric].mean())
        row[f"{metric}_std"] = float(df[metric].std(ddof=0))
    return row


def top_rows(df: pd.DataFrame, method: str, n: int = 12) -> pd.DataFrame:
    cols = ["method", "alpha", "beta", "gamma", *METRICS]
    out = df.sort_values(
        ["precision_at_10", "hit_at_1", "recall_at_10"],
        ascending=False,
    ).head(n)
    out = out.copy()
    out["source_grid"] = method
    return out[["source_grid", *cols]]


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    full_grid = read_grid(Path(args.full_grid))
    adaptive_grid = read_grid(Path(args.adaptive_grid))

    full_df = full_grid[full_grid["method"] == "full"].copy()
    adaptive_df = adaptive_grid[adaptive_grid["method"] == "adaptive"].copy()

    # In the full grid, gamma does not affect the full mode; keep one row per alpha/beta.
    full_df = full_df.sort_values("gamma").drop_duplicates(["alpha", "beta"], keep="first")

    full_df.to_csv(out_dir / "full_parameter_grid.csv", index=False)
    adaptive_df.to_csv(out_dir / "adaptive_parameter_grid.csv", index=False)

    best = pd.concat(
        [
            top_rows(full_df, "full"),
            top_rows(adaptive_df, "adaptive"),
        ],
        ignore_index=True,
    )
    best.to_csv(out_dir / "top_parameter_settings.csv", index=False)

    stability = pd.DataFrame(
        [
            metric_stability(full_df, "full"),
            metric_stability(adaptive_df, "adaptive"),
        ]
    )
    stability.to_csv(out_dir / "parameter_stability.csv", index=False)

    full_default = full_df[(full_df["alpha"] == 0.5) & (full_df["beta"] == 0.5)]
    adaptive_default = adaptive_df[
        (adaptive_df["alpha"] == 0.5)
        & (adaptive_df["beta"] == 0.15)
        & (adaptive_df["gamma"] == 0.5)
    ]
    summary = {
        "full_grid": str(args.full_grid),
        "adaptive_grid": str(args.adaptive_grid),
        "num_full_settings": int(len(full_df)),
        "num_adaptive_settings": int(len(adaptive_df)),
        "best_full": top_rows(full_df, "full", n=1).to_dict(orient="records")[0],
        "best_adaptive": top_rows(adaptive_df, "adaptive", n=1).to_dict(orient="records")[0],
        "default_full": full_default.to_dict(orient="records"),
        "default_adaptive": adaptive_default.to_dict(orient="records"),
        "stability": stability.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
