from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from priver.config import load_priver_config
from priver.consistency import (
    build_positive_patch_map,
    positive_patch_ids_from_query,
)
from priver.evaluation import METRIC_NAMES, evaluate_scores, metrics_dict
from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from priver.reranking import rerank_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the frozen PRIVER reranker to semantic candidates."
    )
    parser.add_argument(
        "--dataset-config",
        required=True,
        help="Dataset YAML used to build the patch index.",
    )
    parser.add_argument(
        "--retrieval-dir",
        required=True,
        help="Directory containing retrieval_results.jsonl.",
    )
    parser.add_argument(
        "--priver-config",
        default="configs/priver_frozen.json",
        help="Frozen PRIVER JSON configuration.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = read_yaml(args.dataset_config)
    experiment_dir = Path(dataset_config["experiment"]["output_dir"])
    retrieval_dir = Path(args.retrieval_dir)
    output_dir = ensure_dir(args.out_dir)
    config = load_priver_config(args.priver_config)

    patch_rows = read_jsonl(experiment_dir / "patch_index.jsonl")
    positive_patch_map = build_positive_patch_map(patch_rows)
    retrieval_rows = read_jsonl(retrieval_dir / "retrieval_results.jsonl")
    if not retrieval_rows:
        raise ValueError("No retrieval rows were found")

    results: list[dict] = []
    metric_rows: list[dict] = []
    metric_sums = np.zeros(len(METRIC_NAMES), dtype=float)
    started = time.perf_counter()
    for row in tqdm(retrieval_rows, desc="Applying PRIVER"):
        query = row["query"]
        candidates = row["retrieved"][: config.candidate_k]
        reranked = rerank_candidates(
            candidates,
            config.same_scale_support,
            config.inter_scale_support,
            config.same_scale_weight,
            config.inter_scale_weight,
        )
        positive_ids = positive_patch_ids_from_query(query, positive_patch_map)
        relevant = np.asarray(
            [
                candidate["patch_id"] in positive_ids
                for candidate in reranked
            ],
            dtype=bool,
        )
        patch_sizes = np.asarray(
            [candidate["patch_size"] for candidate in reranked],
            dtype=int,
        )
        scores = np.asarray(
            [candidate["priver_score"] for candidate in reranked],
            dtype=float,
        )
        values = evaluate_scores(
            scores[None, :],
            relevant,
            len(positive_ids),
            patch_sizes,
            config.top_k,
        )[0]
        metric_sums += values
        metric = {
            "query_id": query["query_id"],
            "image_id": query["image_id"],
            "class_name": query["class_name"],
            "num_positive_patches": len(positive_ids),
            **metrics_dict(values),
        }
        metric_rows.append(metric)
        results.append(
            {
                "query": query,
                "metrics": metric,
                "retrieved": reranked[: config.top_k],
                "method": "PRIVER",
            }
        )
    elapsed = time.perf_counter() - started

    write_jsonl(output_dir / "retrieval_results.jsonl", results)
    write_jsonl(output_dir / "retrieval_metrics.jsonl", metric_rows)
    summary = {
        "method": "PRIVER",
        "full_name": config.full_name,
        "source_retrieval_dir": str(retrieval_dir),
        "priver_config": str(args.priver_config),
        "candidate_k": config.candidate_k,
        "top_k": config.top_k,
        "num_queries": len(metric_rows),
        **metrics_dict(metric_sums / len(metric_rows)),
        "runtime": {
            "elapsed_seconds": elapsed,
            "queries_per_second": len(metric_rows) / elapsed,
            "milliseconds_per_query": 1000.0 * elapsed / len(metric_rows),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
