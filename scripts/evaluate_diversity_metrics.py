from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from priver.baselines import rerank_consistency_baseline
from priver.consistency import build_positive_patch_map
from priver.geometry import box_area, intersection_area
from priver.io import read_jsonl
from priver.metrics import mean, precision_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ranking quality, unique-object coverage, and semantic NMS baselines."
    )
    parser.add_argument("--output-dir", required=True, help="Dataset output directory.")
    parser.add_argument("--retrieval-dir", required=True, help="Retrieval directory with Top-N candidates.")
    parser.add_argument("--result-dir", required=True, help="Directory for supplementary evaluation outputs.")
    parser.add_argument(
        "--reranked-dir",
        default=None,
        help=(
            "Optional directory containing saved proposed-method rankings. "
            "When supplied, these rankings replace the built-in consistency reranker."
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--rerank-mode",
        choices=("full", "adaptive"),
        default="adaptive",
        help="Consistency reranking mode used for the proposed method.",
    )
    parser.add_argument(
        "--method-name",
        default="adaptive",
        help="Method label written to result tables.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Optional candidate-pool size applied before reranking and semantic NMS.",
    )
    parser.add_argument("--nms-thresholds", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iou(a: list[float], b: list[float]) -> float:
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def semantic_nms(candidates: list[dict], threshold: float, top_k: int) -> list[dict]:
    selected: list[dict] = []
    for candidate in candidates:
        if all(iou(candidate["bbox"], kept["bbox"]) <= threshold for kept in selected):
            selected.append(candidate)
            if len(selected) == top_k:
                break
    return selected


def ndcg_at_k(ids: list[str], positives: set[str], k: int) -> float:
    gains = [1.0 if patch_id in positives else 0.0 for patch_id in ids[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_count = min(k, len(positives))
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / idcg


def evaluate(
    query: dict,
    selected: list[dict],
    positive_patches: set[str],
    patch_objects: dict[str, set[str]],
    top_k: int,
) -> dict:
    ids = [item["patch_id"] for item in selected]
    positive_objects = set(query["positive_object_ids"])

    def covered_objects(k: int) -> set[str]:
        covered: set[str] = set()
        for patch_id in ids[:k]:
            covered.update(patch_objects.get(patch_id, set()) & positive_objects)
        return covered

    covered_5 = covered_objects(min(5, top_k))
    covered_10 = covered_objects(min(10, top_k))
    denominator = len(positive_objects)
    return {
        "query_id": query["query_id"],
        "image_id": query["image_id"],
        "class_name": query["class_name"],
        "num_positive_patches": len(positive_patches),
        "num_positive_objects": denominator,
        "precision_at_5": precision_at_k(ids, positive_patches, min(5, top_k)),
        "precision_at_10": precision_at_k(ids, positive_patches, min(10, top_k)),
        "ndcg_at_5": ndcg_at_k(ids, positive_patches, min(5, top_k)),
        "ndcg_at_10": ndcg_at_k(ids, positive_patches, min(10, top_k)),
        "unique_objects_at_5": len(covered_5),
        "unique_objects_at_10": len(covered_10),
        "object_recall_at_5": len(covered_5) / denominator if denominator else 0.0,
        "object_recall_at_10": len(covered_10) / denominator if denominator else 0.0,
    }


def summarize(method: str, rows: list[dict]) -> dict:
    metric_names = [
        "precision_at_5",
        "precision_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "unique_objects_at_5",
        "unique_objects_at_10",
        "object_recall_at_5",
        "object_recall_at_10",
    ]
    summary = {"method": method, "num_queries": len(rows)}
    summary.update({name: mean([float(row[name]) for row in rows]) for name in metric_names})
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_cluster_bootstrap(
    baseline_rows: list[dict],
    method_rows: list[dict],
    metric: str,
    n_bootstrap: int,
    seed: int,
    method_name: str = "adaptive",
) -> dict:
    baseline_by_query = {row["query_id"]: row for row in baseline_rows}
    method_by_query = {row["query_id"]: row for row in method_rows}
    query_ids = sorted(set(baseline_by_query) & set(method_by_query))
    differences_by_image: dict[str, list[float]] = defaultdict(list)
    for query_id in query_ids:
        image_id = baseline_by_query[query_id]["image_id"]
        differences_by_image[image_id].append(
            float(method_by_query[query_id][metric])
            - float(baseline_by_query[query_id][metric])
        )
    unique_images = sorted(differences_by_image)
    image_sums = np.asarray(
        [sum(differences_by_image[image_id]) for image_id in unique_images],
        dtype=float,
    )
    image_counts = np.asarray(
        [len(differences_by_image[image_id]) for image_id in unique_images],
        dtype=float,
    )
    differences = np.concatenate(
        [
            np.asarray(differences_by_image[image_id], dtype=float)
            for image_id in unique_images
        ]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_indices = rng.integers(
            0, len(unique_images), size=len(unique_images)
        )
        samples[index] = (
            image_sums[sampled_indices].sum()
            / image_counts[sampled_indices].sum()
        )
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "method": method_name,
        "baseline": "semantic",
        "metric": metric,
        "num_queries": len(query_ids),
        "num_images": len(unique_images),
        "delta": float(differences.mean()),
        "delta_ci95_low": float(low),
        "delta_ci95_high": float(high),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    retrieval_dir = Path(args.retrieval_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    reranked_by_query = None
    if args.reranked_dir:
        reranked_by_query = {
            row["query"]["query_id"]: row["retrieved"]
            for row in read_jsonl(
                Path(args.reranked_dir) / "retrieval_results.jsonl"
            )
        }

    patch_rows = read_jsonl(output_dir / "patch_index.jsonl")
    positive_patch_map = build_positive_patch_map(patch_rows)
    links = read_jsonl(output_dir / "patch_object_links.jsonl")
    patch_objects: dict[str, set[str]] = defaultdict(set)
    for link in links:
        patch_objects[link["patch_id"]].add(link["object_id"])

    thresholds = [float(value) for value in args.nms_thresholds.split(",")]
    methods = ["semantic", args.method_name] + [
        f"semantic_nms_{value:g}" for value in thresholds
    ]
    per_query: dict[str, list[dict]] = {method: [] for method in methods}

    for row in read_jsonl(retrieval_dir / "retrieval_results.jsonl"):
        query = row["query"]
        candidates = (
            row["retrieved"][: args.candidate_k]
            if args.candidate_k is not None
            else row["retrieved"]
        )
        positive_patches = positive_patch_map.get(
            (query["image_id"], query["class_name"]), set()
        )

        if reranked_by_query is None:
            proposed = rerank_consistency_baseline(
                candidates,
                mode=args.rerank_mode,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )[: args.top_k]
        else:
            proposed = reranked_by_query[query["query_id"]][: args.top_k]
        rankings = {
            "semantic": candidates[: args.top_k],
            args.method_name: proposed,
        }
        for threshold in thresholds:
            rankings[f"semantic_nms_{threshold:g}"] = semantic_nms(
                candidates, threshold=threshold, top_k=args.top_k
            )

        for method, selected in rankings.items():
            metric = evaluate(
                query,
                selected,
                positive_patches=positive_patches,
                patch_objects=patch_objects,
                top_k=args.top_k,
            )
            metric["method"] = method
            per_query[method].append(metric)

    summaries = [summarize(method, per_query[method]) for method in methods]
    (result_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(result_dir / "summary.csv", summaries)
    write_csv(
        result_dir / "per_query_metrics.csv",
        [row for method in methods for row in per_query[method]],
    )
    bootstrap_rows = [
        paired_cluster_bootstrap(
            per_query["semantic"],
            per_query[args.method_name],
            method_name=args.method_name,
            metric=metric,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        for metric in ("precision_at_10", "ndcg_at_10", "object_recall_at_10")
    ]
    write_csv(result_dir / "rerank_vs_semantic_bootstrap.csv", bootstrap_rows)
    (result_dir / "rerank_vs_semantic_bootstrap.json").write_text(
        json.dumps(bootstrap_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "candidate_k": args.candidate_k,
                "top_k": args.top_k,
                "reranked_dir": args.reranked_dir,
                "rerank_mode": args.rerank_mode,
                "method_name": args.method_name,
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "nms_thresholds": thresholds,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(json.dumps(bootstrap_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
