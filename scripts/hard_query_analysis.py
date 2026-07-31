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
    parser = argparse.ArgumentParser(description="Analyze retrieval gains on hard-query subsets.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Baseline retrieval directory. Defaults to output_dir/retrieval_openclip.",
    )
    parser.add_argument(
        "--method-dirs",
        default=None,
        help="Comma-separated method directories to compare.",
    )
    parser.add_argument(
        "--method-names",
        default=None,
        help="Comma-separated method names matching --method-dirs.",
    )
    parser.add_argument(
        "--baseline-name",
        default="openclip",
        help="Name assigned to the baseline metrics in output tables.",
    )
    parser.add_argument(
        "--analysis-subdir",
        default="hard_query_analysis",
        help="Output subdirectory under experiment.output_dir.",
    )
    return parser.parse_args()


def split_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_metrics(path: Path, method: str) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path / "retrieval_metrics.jsonl"))
    df["method"] = method
    return df


def object_area(obj: dict) -> float:
    if obj.get("polygon"):
        return polygon_area(obj["polygon"])
    return box_area(obj["bbox"])


def add_query_attributes(out_dir: Path, baseline_df: pd.DataFrame) -> pd.DataFrame:
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
    attrs = pd.DataFrame(rows)
    df = baseline_df.merge(attrs, on=["query_id", "class_name"], how="left")
    df["baseline_hit1_failure"] = df["hit_at_1"] < 1.0
    df["baseline_hit5_failure"] = df["hit_at_5"] < 1.0
    df["baseline_low_precision10"] = df["precision_at_10"] < 0.5
    df["baseline_very_low_precision10"] = df["precision_at_10"] < 0.25
    df["sparse_objects"] = df["num_positive_objects"] <= df["num_positive_objects"].quantile(0.33)
    df["dense_objects"] = df["num_positive_objects"] >= df["num_positive_objects"].quantile(0.67)
    df["few_positive_patches"] = df["num_positive_patches"] <= df["num_positive_patches"].quantile(0.33)
    df["many_positive_patches"] = df["num_positive_patches"] >= df["num_positive_patches"].quantile(0.67)
    df["small_objects"] = df["mean_object_area"] <= df["mean_object_area"].quantile(0.33)
    df["large_objects"] = df["mean_object_area"] >= df["mean_object_area"].quantile(0.67)
    df["hard_composite"] = (
        df["baseline_low_precision10"]
        | df["baseline_hit1_failure"]
        | df["sparse_objects"]
        | df["few_positive_patches"]
    )
    df["easy_composite"] = (
        (~df["baseline_low_precision10"])
        & (~df["baseline_hit1_failure"])
        & (~df["sparse_objects"])
        & (~df["few_positive_patches"])
    )
    return df


def summarize_subset(df: pd.DataFrame, subset_name: str, query_ids: set[str], baseline_name: str) -> dict:
    subset = df[df["query_id"].isin(query_ids)]
    row = {"subset": subset_name, "num_queries": int(len(query_ids))}
    for method, group in subset.groupby("method"):
        row[f"{method}_num_queries"] = int(len(group))
        for metric in METRICS:
            row[f"{method}_{metric}"] = float(group[metric].mean()) if len(group) else None
    if baseline_name in df["method"].unique():
        for method in sorted(name for name in df["method"].unique() if name != baseline_name):
            for metric in METRICS:
                base_col = f"{baseline_name}_{metric}"
                method_col = f"{method}_{metric}"
                if base_col in row and method_col in row:
                    row[f"{method}_delta_{metric}"] = row[method_col] - row[base_col]
    return row


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = Path(cfg["experiment"]["output_dir"])
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else out_dir / "retrieval_openclip"
    method_dirs = split_csv(args.method_dirs) or [
        str(out_dir / "retrieval_openclip" / "consistency_rerank" / "full"),
        str(out_dir / "retrieval_openclip" / "consistency_rerank_adaptive" / "adaptive"),
    ]
    method_names = split_csv(args.method_names) or ["consistency_full", "adaptive"]
    if len(method_dirs) != len(method_names):
        raise ValueError("--method-dirs and --method-names must have the same length")

    analysis_dir = ensure_dir(out_dir / args.analysis_subdir)
    baseline = load_metrics(baseline_dir, args.baseline_name)
    baseline_attrs = add_query_attributes(out_dir, baseline)

    all_metrics = [baseline]
    for method_dir, method_name in zip(method_dirs, method_names):
        all_metrics.append(load_metrics(Path(method_dir), method_name))
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df = metrics_df.merge(
        baseline_attrs[
            [
                "query_id",
                "num_positive_objects",
                "mean_object_area",
                "median_object_area",
                "baseline_hit1_failure",
                "baseline_hit5_failure",
                "baseline_low_precision10",
                "baseline_very_low_precision10",
                "sparse_objects",
                "dense_objects",
                "few_positive_patches",
                "many_positive_patches",
                "small_objects",
                "large_objects",
                "hard_composite",
                "easy_composite",
            ]
        ],
        on="query_id",
        how="left",
    )

    subset_defs = {
        "all": set(baseline_attrs["query_id"]),
        "hard_composite": set(baseline_attrs.loc[baseline_attrs["hard_composite"], "query_id"]),
        "easy_composite": set(baseline_attrs.loc[baseline_attrs["easy_composite"], "query_id"]),
        "baseline_hit1_failure": set(baseline_attrs.loc[baseline_attrs["baseline_hit1_failure"], "query_id"]),
        "baseline_hit5_failure": set(baseline_attrs.loc[baseline_attrs["baseline_hit5_failure"], "query_id"]),
        "baseline_low_precision10": set(baseline_attrs.loc[baseline_attrs["baseline_low_precision10"], "query_id"]),
        "baseline_very_low_precision10": set(
            baseline_attrs.loc[baseline_attrs["baseline_very_low_precision10"], "query_id"]
        ),
        "sparse_objects": set(baseline_attrs.loc[baseline_attrs["sparse_objects"], "query_id"]),
        "dense_objects": set(baseline_attrs.loc[baseline_attrs["dense_objects"], "query_id"]),
        "few_positive_patches": set(baseline_attrs.loc[baseline_attrs["few_positive_patches"], "query_id"]),
        "many_positive_patches": set(baseline_attrs.loc[baseline_attrs["many_positive_patches"], "query_id"]),
        "small_objects": set(baseline_attrs.loc[baseline_attrs["small_objects"], "query_id"]),
        "large_objects": set(baseline_attrs.loc[baseline_attrs["large_objects"], "query_id"]),
    }

    rows = [
        summarize_subset(metrics_df, name, query_ids, args.baseline_name)
        for name, query_ids in subset_defs.items()
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(analysis_dir / "hard_query_summary.csv", index=False)
    baseline_attrs.to_csv(analysis_dir / "query_difficulty_labels.csv", index=False)
    metrics_df.to_csv(analysis_dir / "query_metrics_with_difficulty.csv", index=False)

    selected = summary_df[
        summary_df["subset"].isin(
            [
                "all",
                "hard_composite",
                "easy_composite",
                "baseline_low_precision10",
                "sparse_objects",
                "few_positive_patches",
            ]
        )
    ]
    summary = {
        "baseline_dir": str(baseline_dir),
        "method_dirs": method_dirs,
        "num_queries": int(baseline_attrs["query_id"].nunique()),
        "selected_subsets": selected.to_dict(orient="records"),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
