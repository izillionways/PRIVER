from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("hit_at_1", "hit_at_10", "precision_at_10", "recall_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for retrieval metrics.")
    parser.add_argument("--output-dir", default="outputs/dota_v15_100", help="Experiment output directory.")
    parser.add_argument("--encoder-dir", default="retrieval_openclip", help="Retrieval directory under output-dir.")
    parser.add_argument("--methods", default="openclip:retrieval_metrics.jsonl,full:consistency_rerank/full/retrieval_metrics.jsonl,adaptive:consistency_rerank_adaptive/adaptive/retrieval_metrics.jsonl")
    parser.add_argument("--baseline", default="openclip", help="Method used for paired delta CIs.")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="Number of bootstrap samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resample-unit",
        choices=("query", "image"),
        default="query",
        help="Resample individual queries or source-image clusters.",
    )
    parser.add_argument("--out-subdir", default="bootstrap_ci", help="Output subdirectory under output-dir.")
    parser.add_argument("--out-dir", default=None, help="Explicit output directory. Overrides --out-subdir.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_methods(raw: str) -> dict[str, str]:
    methods = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, rel_path = item.split(":", 1)
        methods[name.strip()] = rel_path.strip()
    return methods


def align_by_query(
    method_rows: dict[str, list[dict]],
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
            for metric in METRICS
        }
    return shared_query_ids, image_ids, aligned


def bootstrap_means(
    values: np.ndarray,
    image_ids: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
    resample_unit: str,
) -> np.ndarray:
    if resample_unit == "query":
        boot = np.empty(n_bootstrap, dtype=float)
        for index in range(n_bootstrap):
            sampled_indices = rng.integers(0, len(values), size=len(values))
            boot[index] = values[sampled_indices].mean()
        return boot

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
    output_dir = Path(args.output_dir)
    retrieval_dir = output_dir / args.encoder_dir
    out_dir = Path(args.out_dir) if args.out_dir else output_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    method_paths = parse_methods(args.methods)
    method_rows = {
        method: read_jsonl(retrieval_dir / rel_path)
        for method, rel_path in method_paths.items()
    }
    query_ids, image_ids, aligned = align_by_query(method_rows)
    n = len(query_ids)
    rng = np.random.default_rng(args.seed)

    rows = []
    for method in method_paths:
        for metric in METRICS:
            values = aligned[method][metric]
            boot = bootstrap_means(
                values, image_ids, args.n_bootstrap, rng, args.resample_unit
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

            if method != args.baseline:
                delta_values = values - aligned[args.baseline][metric]
                delta_boot = bootstrap_means(
                    delta_values, image_ids, args.n_bootstrap, rng, args.resample_unit
                )
                delta_low, delta_high = ci(delta_boot)
                rows.append({
                    "method": method,
                    "metric": f"delta_{metric}_vs_{args.baseline}",
                    "num_queries": n,
                    "mean": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "delta_vs_baseline": float(delta_values.mean()),
                    "delta_ci95_low": delta_low,
                    "delta_ci95_high": delta_high,
                })

    experiment_name = output_dir.name
    csv_path = out_dir / f"{experiment_name}_{args.encoder_dir}_bootstrap_ci.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / f"{experiment_name}_{args.encoder_dir}_bootstrap_ci_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "encoder_dir": args.encoder_dir,
                "experiment_name": experiment_name,
                "methods": method_paths,
                "baseline": args.baseline,
                "num_queries": n,
                "num_images": len(set(image_ids)),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
                "resample_unit": args.resample_unit,
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
        "resample_unit": args.resample_unit,
    }, indent=2))


if __name__ == "__main__":
    main()
