from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_METRICS = (
    "hit_at_1",
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-image cluster-bootstrap intervals for semantic "
            "and PRIVER retrieval metrics."
        )
    )
    parser.add_argument("--semantic-metrics", required=True)
    parser.add_argument("--priver-metrics", required=True)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric fields present in both JSONL files.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def align_by_query(
    method_rows: dict[str, list[dict]],
    metrics: tuple[str, ...],
) -> tuple[list[str], list[str], dict[str, dict[str, np.ndarray]]]:
    query_sets = [set(row["query_id"] for row in rows) for rows in method_rows.values()]
    shared_query_ids = sorted(set.intersection(*query_sets))
    if not shared_query_ids:
        raise ValueError("No shared query IDs across methods.")

    reference_by_query = {row["query_id"]: row for row in next(iter(method_rows.values()))}
    image_ids = [reference_by_query[q]["image_id"] for q in shared_query_ids]

    aligned: dict[str, dict[str, np.ndarray]] = {}
    for method, rows in method_rows.items():
        by_query = {row["query_id"]: row for row in rows}
        aligned[method] = {
            metric: np.asarray([float(by_query[q][metric]) for q in shared_query_ids], dtype=float)
            for metric in metrics
        }
    return shared_query_ids, image_ids, aligned


def bootstrap_means(
    values: np.ndarray,
    image_ids: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_images = sorted(set(image_ids))
    image_array = np.asarray(image_ids)
    image_sums = np.asarray(
        [values[image_array == image_id].sum() for image_id in unique_images],
        dtype=float,
    )
    image_counts = np.asarray(
        [(image_array == image_id).sum() for image_id in unique_images],
        dtype=float,
    )
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled_indices = rng.integers(
            0, len(unique_images), size=len(unique_images)
        )
        boot[i] = (
            image_sums[sampled_indices].sum()
            / image_counts[sampled_indices].sum()
        )
    return boot


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    if not metrics:
        raise ValueError("--metrics must contain at least one field")
    method_paths = {
        "semantic": Path(args.semantic_metrics),
        "priver": Path(args.priver_metrics),
    }
    method_rows = {
        method: read_jsonl(path) for method, path in method_paths.items()
    }
    query_ids, image_ids, aligned = align_by_query(method_rows, metrics)
    n = len(query_ids)
    rng = np.random.default_rng(args.seed)

    rows = []
    for method in method_paths:
        for metric in metrics:
            values = aligned[method][metric]
            boot = bootstrap_means(
                values, image_ids, args.n_bootstrap, rng
            )
            low, high = ci(boot)
            rows.append({
                "method": method,
                "metric": metric,
                "num_queries": n,
                "mean": float(values.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "delta_vs_baseline": "",
                "delta_ci95_low": "",
                "delta_ci95_high": "",
            })

            if method != "semantic":
                delta_values = values - aligned["semantic"][metric]
                delta_boot = bootstrap_means(
                    delta_values, image_ids, args.n_bootstrap, rng
                )
                delta_low, delta_high = ci(delta_boot)
                rows.append({
                    "method": method,
                    "metric": f"delta_{metric}_vs_semantic",
                    "num_queries": n,
                    "mean": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "delta_vs_baseline": float(delta_values.mean()),
                    "delta_ci95_low": delta_low,
                    "delta_ci95_high": delta_high,
                })

    csv_path = out_dir / "bootstrap_ci.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "bootstrap_ci_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "methods": {
                    method: str(path) for method, path in method_paths.items()
                },
                "baseline": "semantic",
                "metrics": metrics,
                "num_queries": n,
                "num_images": len(set(image_ids)),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "resample_unit": "source_image",
                "csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "csv": str(csv_path),
        "num_queries": n,
        "num_images": len(set(image_ids)),
        "n_bootstrap": args.n_bootstrap,
        "resample_unit": "source_image",
    }, indent=2))


if __name__ == "__main__":
    main()
