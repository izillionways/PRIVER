from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from priver.consistency import build_positive_patch_map
from priver.io import read_jsonl
from priver.metrics import precision_at_k, recall_at_k, topk_hit
from priver.ranking_evaluation import bootstrap_cluster_means, ndcg_at_k


METRICS = (
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "object_recall_at_5",
    "object_recall_at_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved rankings with source-image cluster bootstrap "
            "confidence intervals."
        )
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        required=True,
        help="Comma-separated name=path entries; paths are relative to output-dir.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-batch-size", type=int, default=500)
    return parser.parse_args()


def parse_methods(raw: str) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for item in raw.split(","):
        name, path = item.split("=", 1)
        methods[name.strip()] = Path(path.strip())
    if not methods:
        raise ValueError("At least one method is required")
    return methods


def evaluate_ranking(
    query: dict,
    retrieved: list[dict],
    positive_patch_ids: set[str],
    patch_objects: dict[str, set[str]],
    top_k: int,
) -> dict:
    ids = [item["patch_id"] for item in retrieved[:top_k]]
    positive_objects = set(query["positive_object_ids"])

    def object_recall(k: int) -> float:
        covered: set[str] = set()
        for patch_id in ids[:k]:
            covered.update(patch_objects.get(patch_id, set()) & positive_objects)
        return len(covered) / len(positive_objects) if positive_objects else 0.0

    return {
        "query_id": query["query_id"],
        "image_id": query["image_id"],
        "class_name": query["class_name"],
        "hit_at_1": topk_hit(ids, positive_patch_ids, 1),
        "hit_at_5": topk_hit(ids, positive_patch_ids, min(5, top_k)),
        "hit_at_10": topk_hit(ids, positive_patch_ids, min(10, top_k)),
        "precision_at_5": precision_at_k(
            ids, positive_patch_ids, min(5, top_k)
        ),
        "precision_at_10": precision_at_k(
            ids, positive_patch_ids, min(10, top_k)
        ),
        "recall_at_5": recall_at_k(
            ids, positive_patch_ids, min(5, top_k)
        ),
        "recall_at_10": recall_at_k(
            ids, positive_patch_ids, min(10, top_k)
        ),
        "ndcg_at_5": ndcg_at_k(ids, positive_patch_ids, min(5, top_k)),
        "ndcg_at_10": ndcg_at_k(ids, positive_patch_ids, min(10, top_k)),
        "object_recall_at_5": object_recall(min(5, top_k)),
        "object_recall_at_10": object_recall(min(10, top_k)),
    }


def load_method_metrics(
    path: Path,
    positive_patch_map: dict[tuple[str, str], set[str]],
    patch_objects: dict[str, set[str]],
    top_k: int,
) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for row in read_jsonl(path):
        query = row["query"]
        positives = positive_patch_map.get(
            (query["image_id"], query["class_name"]), set()
        )
        metric = evaluate_ranking(
            query,
            row["retrieved"],
            positives,
            patch_objects,
            top_k,
        )
        metrics[query["query_id"]] = metric
    return metrics


def cluster_sums(
    method_metrics: dict[str, dict[str, dict]],
    query_ids: list[str],
    image_ids: list[str],
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    unique_images = sorted(set(image_ids))
    image_index = {image_id: index for index, image_id in enumerate(unique_images)}
    counts = np.zeros(len(unique_images), dtype=float)
    sums = {
        method: np.zeros((len(unique_images), len(METRICS)), dtype=float)
        for method in method_metrics
    }
    for query_id, image_id in zip(query_ids, image_ids, strict=True):
        index = image_index[image_id]
        counts[index] += 1.0
        for method, rows in method_metrics.items():
            sums[method][index] += np.asarray(
                [float(rows[query_id][metric]) for metric in METRICS],
                dtype=float,
            )
    return unique_images, counts, sums


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    method_paths = parse_methods(args.methods)
    if args.baseline not in method_paths:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    patch_rows = read_jsonl(output_dir / "patch_index.jsonl")
    positive_patch_map = build_positive_patch_map(patch_rows)
    patch_objects: dict[str, set[str]] = defaultdict(set)
    for link in read_jsonl(output_dir / "patch_object_links.jsonl"):
        patch_objects[link["patch_id"]].add(link["object_id"])

    method_metrics = {
        method: load_method_metrics(
            output_dir / path,
            positive_patch_map,
            patch_objects,
            args.top_k,
        )
        for method, path in method_paths.items()
    }
    query_sets = [set(rows) for rows in method_metrics.values()]
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("Methods do not contain identical query IDs")
    query_ids = sorted(query_sets[0])
    baseline_rows = method_metrics[args.baseline]
    image_ids = [baseline_rows[query_id]["image_id"] for query_id in query_ids]

    unique_images, counts, sums = cluster_sums(
        method_metrics, query_ids, image_ids
    )
    boot = bootstrap_cluster_means(
        counts,
        sums,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        batch_size=args.bootstrap_batch_size,
    )
    total_queries = len(query_ids)
    summary_rows: list[dict] = []
    for method, method_sums in sums.items():
        means = method_sums.sum(axis=0) / total_queries
        lows, highs = np.percentile(boot[method], [2.5, 97.5], axis=0)
        for index, metric in enumerate(METRICS):
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "num_queries": total_queries,
                    "num_images": len(unique_images),
                    "mean": float(means[index]),
                    "ci95_low": float(lows[index]),
                    "ci95_high": float(highs[index]),
                }
            )

    delta_rows: list[dict] = []
    baseline_boot = boot[args.baseline]
    for method in method_paths:
        if method == args.baseline:
            continue
        method_boot = boot[method]
        for index, metric in enumerate(METRICS):
            differences = np.asarray(
                [
                    float(method_metrics[method][query_id][metric])
                    - float(baseline_rows[query_id][metric])
                    for query_id in query_ids
                ],
                dtype=float,
            )
            delta_samples = method_boot[:, index] - baseline_boot[:, index]
            low, high = np.percentile(delta_samples, [2.5, 97.5])
            delta_rows.append(
                {
                    "method": method,
                    "baseline": args.baseline,
                    "metric": metric,
                    "delta": float(differences.mean()),
                    "delta_ci95_low": float(low),
                    "delta_ci95_high": float(high),
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "query_wins": int((differences > 0.0).sum()),
                    "query_ties": int((differences == 0.0).sum()),
                    "query_losses": int((differences < 0.0).sum()),
                }
            )

    write_csv(result_dir / "summary.csv", summary_rows)
    write_csv(result_dir / "paired_deltas.csv", delta_rows)
    with gzip.open(
        result_dir / "per_query_metrics.csv.gz",
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method",) + tuple(next(iter(baseline_rows.values()))),
        )
        writer.writeheader()
        for method, rows in method_metrics.items():
            for query_id in query_ids:
                writer.writerow({"method": method, **rows[query_id]})

    run_config = {
        "output_dir": str(output_dir),
        "method_paths": {key: str(value) for key, value in method_paths.items()},
        "baseline": args.baseline,
        "top_k": args.top_k,
        "num_queries": total_queries,
        "num_images": len(unique_images),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "resample_unit": "source_image",
        "metrics": list(METRICS),
    }
    (result_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                **run_config,
                "summary": str(result_dir / "summary.csv"),
                "paired_deltas": str(result_dir / "paired_deltas.csv"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
