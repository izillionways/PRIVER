#!/usr/bin/env python3
"""Summarize retrieval metrics by query template."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def template_id(query_id: str) -> str:
    if "_t" not in query_id:
        return "unknown"
    return "t" + query_id.rsplit("_t", 1)[1]


def metric_value(row: dict, metric: str) -> float:
    if metric != "recall_at_5" or metric in row:
        return float(row[metric])
    num_positive = int(row["num_positive_patches"])
    retrieved_positive = float(row["precision_at_5"]) * int(row["returned_at_5"])
    return retrieved_positive / num_positive if num_positive else 0.0


def add_returned_counts(metric_rows: list[dict], result_path: Path) -> list[dict]:
    result_rows = read_jsonl(result_path)
    returned_at_5 = {
        row["query"]["query_id"]: min(5, len(row.get("retrieved", [])))
        for row in result_rows
    }
    for row in metric_rows:
        row["returned_at_5"] = returned_at_5[row["query_id"]]
    return metric_rows


def summarize(rows: list[dict], method: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(template_id(row["query_id"]), []).append(row)

    summary = []
    for tmpl, group in sorted(groups.items()):
        out = {"method": method, "template_id": tmpl, "num_queries": len(group)}
        for metric in METRICS:
            out[metric] = sum(metric_value(row, metric) for row in group) / len(group)
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize semantic and PRIVER metrics by controlled query template."
        )
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        required=True,
        help="Directory containing semantic retrieval outputs.",
    )
    parser.add_argument(
        "--priver-dir",
        type=Path,
        required=True,
        help="Directory containing PRIVER retrieval outputs.",
    )
    parser.add_argument("--analysis-dir", type=Path, default=None)
    args = parser.parse_args()

    analysis_dir = args.analysis_dir or args.priver_dir / "query_template_robustness"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    method_paths = {
        "semantic": args.semantic_dir / "retrieval_metrics.jsonl",
        "priver": args.priver_dir / "retrieval_metrics.jsonl",
    }

    rows: list[dict] = []
    for method, path in method_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        metric_rows = add_returned_counts(read_jsonl(path), path.with_name("retrieval_results.jsonl"))
        rows.extend(summarize(metric_rows, method))

    csv_path = analysis_dir / "query_template_robustness.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "template_id", "num_queries", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    by_method: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["template_id"]] = row

    deltas = []
    for method, group in by_method.items():
        if "t0" not in group:
            continue
        for template in sorted(group):
            if template == "t0":
                continue
            delta = {"method": method, "comparison": f"{template}_minus_t0"}
            for metric in METRICS:
                delta[metric] = group[template][metric] - group["t0"][metric]
            deltas.append(delta)

    delta_path = analysis_dir / "query_template_deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "comparison", *METRICS])
        writer.writeheader()
        writer.writerows(deltas)

    summary = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "methods": sorted(method_paths),
        "num_rows": len(rows),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
