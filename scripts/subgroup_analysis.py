from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from priver.geometry import box_area, polygon_area
from priver.io import ensure_dir, read_jsonl, read_yaml


METRICS = [
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "precision_at_5",
    "precision_at_10",
    "recall_at_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval methods by class, size, and density.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Semantic-baseline retrieval directory.",
    )
    parser.add_argument(
        "--priver-dir",
        required=True,
        help="PRIVER retrieval directory.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Optional output directory. Defaults to METHOD_DIR/subgroup_analysis.",
    )
    return parser.parse_args()


def load_metrics(path: Path, method: str) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path / "retrieval_metrics.jsonl"))
    df["method"] = method
    return df


def object_area(obj: dict) -> float:
    if obj.get("polygon"):
        return polygon_area(obj["polygon"])
    return box_area(obj["bbox"])


def query_attributes(out_dir: Path) -> pd.DataFrame:
    objects = {row["object_id"]: row for row in read_jsonl(out_dir / "object_index.jsonl")}
    rows = []
    for query in read_jsonl(out_dir / "query_index.jsonl"):
        areas = []
        for object_id in query.get("positive_object_ids", []):
            obj = objects.get(object_id)
            if obj:
                areas.append(object_area(obj))
        rows.append(
            {
                "query_id": query["query_id"],
                "class_name": query["class_name"],
                "num_positive_objects": int(query.get("num_positive_objects", len(areas))),
                "mean_object_area": float(sum(areas) / len(areas)) if areas else 0.0,
                "median_object_area": float(pd.Series(areas).median()) if areas else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df["size_bin"] = pd.qcut(
        df["mean_object_area"],
        q=min(3, df["mean_object_area"].nunique()),
        labels=["small", "medium", "large"][: min(3, df["mean_object_area"].nunique())],
        duplicates="drop",
    ).astype(str)
    df["density_bin"] = pd.qcut(
        df["num_positive_objects"],
        q=min(3, df["num_positive_objects"].nunique()),
        labels=["sparse", "medium", "dense"][: min(3, df["num_positive_objects"].nunique())],
        duplicates="drop",
    ).astype(str)
    return df


def summarize_group(df: pd.DataFrame, group_col: str, baseline_name: str) -> pd.DataFrame:
    rows = []
    for group_value, group in df.groupby(group_col, dropna=False):
        row = {
            "group_type": group_col,
            "group": str(group_value),
            "num_queries": int(group["query_id"].nunique()),
        }
        for method, method_group in group.groupby("method"):
            row[f"{method}_num_queries"] = int(len(method_group))
            for metric in METRICS:
                row[f"{method}_{metric}"] = float(method_group[metric].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    method_names = sorted(df["method"].unique().tolist())
    if len(method_names) == 2:
        baseline, method = method_names
        if baseline != baseline_name:
            baseline, method = method, baseline
        for metric in METRICS:
            base_col = f"{baseline}_{metric}"
            method_col = f"{method}_{metric}"
            if base_col in out.columns and method_col in out.columns:
                out[f"delta_{metric}"] = out[method_col] - out[base_col]
    return out.sort_values(["group_type", "group"]).reset_index(drop=True)


def top_changes(
    df: pd.DataFrame,
    group_col: str,
    metric: str,
    baseline_name: str,
    method_name: str,
    n: int = 8,
) -> list[dict]:
    summary = summarize_group(df, group_col, baseline_name)
    delta_col = f"delta_{metric}"
    if delta_col not in summary.columns:
        return []
    cols = ["group_type", "group", "num_queries", delta_col]
    for col in [f"{baseline_name}_{metric}", f"{method_name}_{metric}"]:
        if col in summary.columns:
            cols.append(col)
    return summary.sort_values(delta_col, ascending=False)[cols].head(n).to_dict(orient="records")


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = Path(cfg["experiment"]["output_dir"])
    baseline_dir = Path(args.baseline_dir)
    priver_dir = Path(args.priver_dir)
    analysis_dir = ensure_dir(
        Path(args.analysis_dir)
        if args.analysis_dir
        else priver_dir / "subgroup_analysis"
    )

    attrs = query_attributes(out_dir)
    baseline = load_metrics(baseline_dir, "semantic")
    method = load_metrics(priver_dir, "priver")
    df = pd.concat([baseline, method], ignore_index=True).merge(attrs, on=["query_id", "class_name"], how="left")

    summaries = []
    for group_col in ["class_name", "size_bin", "density_bin"]:
        summary = summarize_group(df, group_col, "semantic")
        summary.to_csv(analysis_dir / f"by_{group_col}.csv", index=False)
        summaries.append(summary)

    all_groups = pd.concat(summaries, ignore_index=True)
    all_groups.to_csv(analysis_dir / "subgroup_summary.csv", index=False)
    summary = {
        "baseline_dir": str(baseline_dir),
        "priver_dir": str(priver_dir),
        "num_queries": int(attrs["query_id"].nunique()),
        "best_precision_at_10_gains": top_changes(
            df,
            "class_name",
            "precision_at_10",
            "semantic",
            "priver",
        ),
        "best_hit_at_1_gains": top_changes(
            df,
            "class_name",
            "hit_at_1",
            "semantic",
            "priver",
        ),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
