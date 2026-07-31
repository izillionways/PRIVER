from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from priver.metrics import mean, precision_at_k, recall_at_k, topk_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run label-derived oracle/random retrieval baselines.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def positive_patch_map(patch_rows: list[dict]) -> dict[tuple[str, str], set[str]]:
    positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for patch in patch_rows:
        for class_name, count in patch.get("class_counts", {}).items():
            if count > 0:
                positives[(patch["image_id"], class_name)].add(patch["patch_id"])
    return positives


def evaluate(
    name: str,
    query_rows: list[dict],
    patch_rows: list[dict],
    positives: dict[tuple[str, str], set[str]],
    top_k: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    patches_by_image: dict[str, list[dict]] = defaultdict(list)
    for patch in patch_rows:
        patches_by_image[patch["image_id"]].append(patch)

    results: list[dict] = []
    metrics: list[dict] = []

    for query in query_rows:
        candidates = list(patches_by_image[query["image_id"]])
        pos_ids = positives.get((query["image_id"], query["class_name"]), set())
        if name == "oracle":
            candidates.sort(
                key=lambda p: (
                    p["patch_id"] not in pos_ids,
                    -int(p.get("class_counts", {}).get(query["class_name"], 0)),
                    p["patch_size"],
                    p["patch_id"],
                )
            )
        elif name == "random":
            rng.shuffle(candidates)
        else:
            raise ValueError(f"Unknown baseline: {name}")

        retrieved = candidates[:top_k]
        retrieved_ids = [row["patch_id"] for row in retrieved]
        metric = {
            "query_id": query["query_id"],
            "image_id": query["image_id"],
            "class_name": query["class_name"],
            "num_positive_patches": len(pos_ids),
            "hit_at_1": topk_hit(retrieved_ids, pos_ids, 1),
            "hit_at_5": topk_hit(retrieved_ids, pos_ids, min(5, top_k)),
            "hit_at_10": topk_hit(retrieved_ids, pos_ids, min(10, top_k)),
            "precision_at_5": precision_at_k(retrieved_ids, pos_ids, min(5, top_k)),
            "precision_at_10": precision_at_k(retrieved_ids, pos_ids, min(10, top_k)),
            "recall_at_5": recall_at_k(retrieved_ids, pos_ids, min(5, top_k)),
            "recall_at_10": recall_at_k(retrieved_ids, pos_ids, min(10, top_k)),
        }
        metrics.append(metric)
        results.append(
            {
                "query": query,
                "metrics": metric,
                "retrieved": [
                    {
                        "patch_id": row["patch_id"],
                        "bbox": row["bbox"],
                        "patch_size": row["patch_size"],
                        "class_counts": row.get("class_counts", {}),
                    }
                    for row in retrieved
                ],
            }
        )

    summary = {
        "baseline": name,
        "num_queries": len(query_rows),
        "num_patches": len(patch_rows),
        "hit_at_1": mean([row["hit_at_1"] for row in metrics]),
        "hit_at_5": mean([row["hit_at_5"] for row in metrics]),
        "hit_at_10": mean([row["hit_at_10"] for row in metrics]),
        "precision_at_5": mean([row["precision_at_5"] for row in metrics]),
        "precision_at_10": mean([row["precision_at_10"] for row in metrics]),
        "recall_at_5": mean([row["recall_at_5"] for row in metrics]),
        "recall_at_10": mean([row["recall_at_10"] for row in metrics]),
    }
    return results, metrics, summary


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    patch_rows = read_jsonl(out_dir / "patch_index.jsonl")
    query_rows = read_jsonl(out_dir / "query_index.jsonl")
    positives = positive_patch_map(patch_rows)
    top_k = int(cfg.get("retrieval", {}).get("top_k", 10))
    seed = int(cfg["experiment"].get("seed", 42))

    summaries = []
    for baseline in ["random", "oracle"]:
        baseline_dir = ensure_dir(out_dir / f"retrieval_{baseline}")
        results, metrics, summary = evaluate(
            baseline, query_rows, patch_rows, positives, top_k=top_k, seed=seed
        )
        write_jsonl(baseline_dir / "retrieval_results.jsonl", results)
        write_jsonl(baseline_dir / "retrieval_metrics.jsonl", metrics)
        (baseline_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries.append(summary)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
