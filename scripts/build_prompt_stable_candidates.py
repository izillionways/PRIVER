from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from priver.config import load_priver_config
from priver.consistency import build_positive_patch_map
from priver.evaluation import METRIC_NAMES, evaluate_scores, metrics_dict
from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from priver.prompting import aggregate_prompt_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build semantic candidate lists using PRIVER's frozen "
            "prompt-stable score."
        )
    )
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--patch-features", required=True)
    parser.add_argument("--query-features", required=True)
    parser.add_argument(
        "--priver-config",
        default="configs/priver_frozen.json",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_features(path: str | Path, expected_ids: list[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(str).tolist()
    if ids != expected_ids:
        raise ValueError(f"Feature ID mismatch for {path}")
    return data["features"].astype(np.float32)


def main() -> None:
    args = parse_args()
    dataset_config = read_yaml(args.dataset_config)
    experiment_dir = Path(dataset_config["experiment"]["output_dir"])
    output_dir = ensure_dir(args.out_dir)
    config = load_priver_config(args.priver_config)

    patch_rows = read_jsonl(experiment_dir / "patch_index.jsonl")
    query_rows = read_jsonl(experiment_dir / "query_index.jsonl")
    patch_features = load_features(
        args.patch_features,
        [row["patch_id"] for row in patch_rows],
    )
    query_features = load_features(
        args.query_features,
        [row["query_id"] for row in query_rows],
    )
    positive_patch_map = build_positive_patch_map(patch_rows)

    patch_indices_by_image: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(patch_rows):
        patch_indices_by_image[row["image_id"]].append(index)
    query_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(query_rows):
        query_groups[(row["image_id"], row["class_name"])].append(index)

    results: list[dict] = []
    metric_rows: list[dict] = []
    metric_sums = np.zeros(len(METRIC_NAMES), dtype=float)
    for (image_id, class_name), query_indices_list in tqdm(
        query_groups.items(),
        desc="Building prompt-stable candidates",
    ):
        query_indices = np.asarray(query_indices_list, dtype=int)
        if len(query_indices) != len(config.prompt_scoring.templates):
            raise ValueError(
                f"{image_id}/{class_name} has {len(query_indices)} prompts; "
                f"expected {len(config.prompt_scoring.templates)}"
            )
        patch_indices = np.asarray(
            patch_indices_by_image[image_id],
            dtype=int,
        )
        template_scores = (
            query_features[query_indices] @ patch_features[patch_indices].T
        )
        scores = aggregate_prompt_scores(
            template_scores,
            config.prompt_scoring.standard_deviation_weight,
        )
        keep = min(config.candidate_k, len(patch_indices))
        local_order = np.argsort(-scores, kind="stable")[:keep]
        ranked_patch_indices = patch_indices[local_order]
        relevant_ids = positive_patch_map.get((image_id, class_name), set())

        retrieved = [
            {
                "patch_id": patch_rows[index]["patch_id"],
                "score": float(scores[local_index]),
                "bbox": patch_rows[index]["bbox"],
                "patch_size": patch_rows[index]["patch_size"],
                "class_counts": patch_rows[index].get("class_counts", {}),
            }
            for local_index, index in zip(
                local_order,
                ranked_patch_indices,
                strict=True,
            )
        ]
        relevant = np.asarray(
            [
                patch_rows[index]["patch_id"] in relevant_ids
                for index in ranked_patch_indices
            ],
            dtype=bool,
        )
        patch_sizes = np.asarray(
            [patch_rows[index]["patch_size"] for index in ranked_patch_indices],
            dtype=int,
        )
        metric_values = evaluate_scores(
            scores[local_order][None, :],
            relevant,
            len(relevant_ids),
            patch_sizes,
            config.top_k,
        )[0]

        for query_index in query_indices:
            query = query_rows[int(query_index)]
            metric = {
                "query_id": query["query_id"],
                "image_id": image_id,
                "class_name": class_name,
                "num_positive_patches": len(relevant_ids),
                **metrics_dict(metric_values),
            }
            metric_rows.append(metric)
            metric_sums += metric_values
            results.append(
                {
                    "query": query,
                    "metrics": metric,
                    "retrieved": retrieved,
                    "prompt_scoring": {
                        "aggregation": config.prompt_scoring.aggregation,
                        "standard_deviation_weight": (
                            config.prompt_scoring.standard_deviation_weight
                        ),
                        "num_templates": len(query_indices),
                    },
                }
            )

    if not metric_rows:
        raise ValueError("No query groups were found")
    write_jsonl(output_dir / "retrieval_results.jsonl", results)
    write_jsonl(output_dir / "retrieval_metrics.jsonl", metric_rows)
    summary = {
        "method": "prompt_stable_semantic_retrieval",
        "num_queries": len(metric_rows),
        "num_query_groups": len(query_groups),
        "candidate_k": config.candidate_k,
        "top_k": config.top_k,
        "prompt_scoring": config.prompt_scoring.to_dict(),
        **metrics_dict(metric_sums / len(metric_rows)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
