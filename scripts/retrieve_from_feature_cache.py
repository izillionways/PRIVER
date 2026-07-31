from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from priver.metrics import mean, precision_at_k, recall_at_k, topk_hit
from priver.retrieval import rank_queries_within_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval from precomputed feature cache files.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--patch-features", required=True, help="Patch feature .npz file.")
    parser.add_argument("--query-features", required=True, help="Query feature .npz file.")
    parser.add_argument("--output-name", required=True, help="Retrieval output directory name under experiment output_dir.")
    parser.add_argument("--candidate-k", type=int, required=True, help="Number of candidates to store per query.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k metrics to evaluate.")
    return parser.parse_args()


def load_features(path: Path, expected_ids: list[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(str).tolist()
    if ids != expected_ids:
        raise ValueError(f"Feature ID mismatch for {path}")
    return data["features"].astype(np.float32)


def build_positive_patch_map(patch_rows: list[dict]) -> dict[tuple[str, str], set[str]]:
    positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for patch in patch_rows:
        for class_name, count in patch.get("class_counts", {}).items():
            if count > 0:
                positives[(patch["image_id"], class_name)].add(patch["patch_id"])
    return positives


def score_uncertainty(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"score_entropy": 0.0, "top1_top2_margin": 0.0, "score_std": 0.0}
    shifted = scores - scores.max()
    probs = np.exp(shifted)
    probs = probs / max(probs.sum(), 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs))) if len(probs) > 1 else 0.0
    margin = float(scores[0] - scores[1]) if len(scores) > 1 else float(scores[0])
    return {"score_entropy": entropy, "top1_top2_margin": margin, "score_std": float(scores.std())}


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = Path(cfg["experiment"]["output_dir"])
    retrieval_dir = ensure_dir(out_dir / args.output_name)

    patch_rows = read_jsonl(out_dir / "patch_index.jsonl")
    query_rows = read_jsonl(out_dir / "query_index.jsonl")
    patch_ids = [row["patch_id"] for row in patch_rows]
    query_ids = [row["query_id"] for row in query_rows]
    patch_features = load_features(Path(args.patch_features), patch_ids)
    query_features = load_features(Path(args.query_features), query_ids)

    positives = build_positive_patch_map(patch_rows)

    rankings = rank_queries_within_images(
        query_features,
        patch_features,
        query_rows,
        patch_rows,
        args.candidate_k,
    )
    results: list[dict] = []
    metric_rows: list[dict] = []

    for query_idx, query in enumerate(tqdm(query_rows, desc="Retrieving from cache")):
        top_indices, top_scores = rankings[query_idx]
        retrieved = [
            {
                "patch_id": patch_ids[idx],
                "score": float(score),
                "bbox": patch_rows[idx]["bbox"],
                "patch_size": patch_rows[idx]["patch_size"],
                "class_counts": patch_rows[idx].get("class_counts", {}),
            }
            for idx, score in zip(top_indices, top_scores)
        ]
        retrieved_patch_ids = [row["patch_id"] for row in retrieved]
        positive_patch_ids = positives.get((query["image_id"], query["class_name"]), set())
        metrics = {
            "query_id": query["query_id"],
            "image_id": query["image_id"],
            "class_name": query["class_name"],
            "num_positive_patches": len(positive_patch_ids),
            "hit_at_1": topk_hit(retrieved_patch_ids, positive_patch_ids, 1),
            "hit_at_5": topk_hit(retrieved_patch_ids, positive_patch_ids, min(5, args.top_k)),
            "hit_at_10": topk_hit(retrieved_patch_ids, positive_patch_ids, min(10, args.top_k)),
            "precision_at_5": precision_at_k(retrieved_patch_ids, positive_patch_ids, min(5, args.top_k)),
            "precision_at_10": precision_at_k(retrieved_patch_ids, positive_patch_ids, min(10, args.top_k)),
            "recall_at_5": recall_at_k(retrieved_patch_ids, positive_patch_ids, min(5, args.top_k)),
            "recall_at_10": recall_at_k(retrieved_patch_ids, positive_patch_ids, min(10, args.top_k)),
            **score_uncertainty(top_scores),
        }
        metric_rows.append(metrics)
        results.append({"query": query, "metrics": metrics, "retrieved": retrieved})

    summary = {
        "encoder": cfg.get("retrieval", {}),
        "retrieval_name": args.output_name,
        "patch_feature_cache": str(args.patch_features),
        "query_feature_cache": str(args.query_features),
        "num_patches": len(patch_rows),
        "num_queries": len(query_rows),
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "hit_at_1": mean([row["hit_at_1"] for row in metric_rows]),
        "hit_at_5": mean([row["hit_at_5"] for row in metric_rows]),
        "hit_at_10": mean([row["hit_at_10"] for row in metric_rows]),
        "precision_at_5": mean([row["precision_at_5"] for row in metric_rows]),
        "precision_at_10": mean([row["precision_at_10"] for row in metric_rows]),
        "recall_at_5": mean([row["recall_at_5"] for row in metric_rows]),
        "recall_at_10": mean([row["recall_at_10"] for row in metric_rows]),
        "score_entropy": mean([row["score_entropy"] for row in metric_rows]),
        "top1_top2_margin": mean([row["top1_top2_margin"] for row in metric_rows]),
    }
    write_jsonl(retrieval_dir / "retrieval_results.jsonl", results)
    write_jsonl(retrieval_dir / "retrieval_metrics.jsonl", metric_rows)
    (retrieval_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
