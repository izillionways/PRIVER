from __future__ import annotations

import argparse
import json
from pathlib import Path

from priver.baselines import rerank_consistency_baseline
from priver.consistency import (
    build_positive_patch_map,
    positive_patch_ids_from_query,
)
from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl
from priver.metrics import mean, precision_at_k, recall_at_k, topk_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank retrieval candidates using consistency scores.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--retrieval-dir", default=None, help="Directory containing retrieval_results.jsonl.")
    parser.add_argument("--alpha", type=float, default=0.15, help="Spatial support weight.")
    parser.add_argument("--beta", type=float, default=0.15, help="Cross-scale support weight.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Semantic preservation exponent for adaptive mode.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Optional candidate-pool size for manually specified parameters.",
    )
    parser.add_argument("--grid-search", action="store_true", help="Scan alpha/beta values without writing per-query outputs.")
    parser.add_argument("--alpha-values", default="0,0.05,0.1,0.15,0.2,0.3,0.5", help="Comma-separated alpha values for grid search.")
    parser.add_argument("--beta-values", default="0,0.05,0.1,0.15,0.2,0.3,0.5", help="Comma-separated beta values for grid search.")
    parser.add_argument("--gamma-values", default="0.5,1,2", help="Comma-separated gamma values for adaptive grid search.")
    parser.add_argument("--top-k", type=int, default=None, help="Final top-k to evaluate.")
    parser.add_argument("--modes", default="clip,spatial,cross_scale,full,adaptive", help="Comma-separated modes to write.")
    parser.add_argument("--rerank-subdir", default="consistency_rerank", help="Output subdirectory under retrieval-dir.")
    parser.add_argument(
        "--selection-report",
        default=None,
        help="Optional parameter_selection.json; supplies frozen per-mode parameters.",
    )
    return parser.parse_args()


def evaluate_result(query: dict, retrieved: list[dict], positive_patch_ids: set[str], top_k: int) -> dict:
    ids = [row["patch_id"] for row in retrieved]
    return {
        "query_id": query["query_id"],
        "image_id": query["image_id"],
        "class_name": query["class_name"],
        "num_positive_patches": len(positive_patch_ids),
        "hit_at_1": topk_hit(ids, positive_patch_ids, 1),
        "hit_at_5": topk_hit(ids, positive_patch_ids, min(5, top_k)),
        "hit_at_10": topk_hit(ids, positive_patch_ids, min(10, top_k)),
        "precision_at_5": precision_at_k(ids, positive_patch_ids, min(5, top_k)),
        "precision_at_10": precision_at_k(ids, positive_patch_ids, min(10, top_k)),
        "recall_at_5": recall_at_k(ids, positive_patch_ids, min(5, top_k)),
        "recall_at_10": recall_at_k(ids, positive_patch_ids, min(10, top_k)),
    }


def summarize(method: str, metrics: list[dict]) -> dict:
    return {
        "method": method,
        "num_queries": len(metrics),
        "hit_at_1": mean([row["hit_at_1"] for row in metrics]),
        "hit_at_5": mean([row["hit_at_5"] for row in metrics]),
        "hit_at_10": mean([row["hit_at_10"] for row in metrics]),
        "precision_at_5": mean([row["precision_at_5"] for row in metrics]),
        "precision_at_10": mean([row["precision_at_10"] for row in metrics]),
        "recall_at_5": mean([row["recall_at_5"] for row in metrics]),
        "recall_at_10": mean([row["recall_at_10"] for row in metrics]),
    }


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def rerank(candidates: list[dict], mode: str, alpha: float, beta: float, gamma: float = 1.0) -> list[dict]:
    return rerank_consistency_baseline(
        candidates,
        mode=mode,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = Path(cfg["experiment"]["output_dir"])
    retrieval_dir = Path(args.retrieval_dir) if args.retrieval_dir else out_dir / "retrieval_openclip"
    rerank_dir = ensure_dir(retrieval_dir / args.rerank_subdir)
    top_k = args.top_k or int(cfg.get("retrieval", {}).get("top_k", 10))

    retrieval_rows = read_jsonl(retrieval_dir / "retrieval_results.jsonl")
    patch_rows = read_jsonl(out_dir / "patch_index.jsonl")
    positive_patch_map = build_positive_patch_map(patch_rows)

    if args.grid_search:
        grid_summaries = []
        for alpha in parse_float_list(args.alpha_values):
            for beta in parse_float_list(args.beta_values):
                for gamma in parse_float_list(args.gamma_values):
                    for mode in ["clip", "spatial", "cross_scale", "full", "adaptive"]:
                        metrics = []
                        for row in retrieval_rows:
                            query = row["query"]
                            reranked = rerank(row["retrieved"], mode=mode, alpha=alpha, beta=beta, gamma=gamma)
                            selected = reranked[:top_k]
                            positive_patch_ids = positive_patch_ids_from_query(query, positive_patch_map)
                            metrics.append(evaluate_result(query, selected, positive_patch_ids, top_k))
                        summary = summarize(mode, metrics)
                        summary.update({"alpha": alpha, "beta": beta, "gamma": gamma, "top_k": top_k})
                        grid_summaries.append(summary)

        grid_summaries.sort(
            key=lambda row: (
                row["precision_at_10"],
                row["precision_at_5"],
                row["hit_at_10"],
                row["hit_at_1"],
            ),
            reverse=True,
        )
        grid_path = rerank_dir / "grid_search_summary.json"
        grid_path.write_text(json.dumps(grid_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(grid_summaries[:10], indent=2, ensure_ascii=False))
        return

    all_summaries = []
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    selected_settings = None
    if args.selection_report:
        selection_data = json.loads(
            Path(args.selection_report).read_text(encoding="utf-8")
        )
        selected_settings = {
            mode: values["selected_setting"]
            for mode, values in selection_data["modes"].items()
        }
    for mode in modes:
        if selected_settings is not None:
            if mode not in selected_settings:
                raise ValueError(
                    f"Selection report has no frozen setting for mode {mode}"
                )
            mode_alpha = float(selected_settings[mode]["alpha"])
            mode_beta = float(selected_settings[mode]["beta"])
            mode_gamma = float(selected_settings[mode]["gamma"])
            mode_candidate_k = int(selected_settings[mode]["candidate_k"])
        else:
            mode_alpha = args.alpha
            mode_beta = args.beta
            mode_gamma = args.gamma
            mode_candidate_k = args.candidate_k
        results = []
        metrics = []
        for row in retrieval_rows:
            query = row["query"]
            reranked = rerank(
                (
                    row["retrieved"][:mode_candidate_k]
                    if mode_candidate_k is not None
                    else row["retrieved"]
                ),
                mode=mode,
                alpha=mode_alpha,
                beta=mode_beta,
                gamma=mode_gamma,
            )
            selected = reranked[:top_k]
            positive_patch_ids = positive_patch_ids_from_query(query, positive_patch_map)
            metric = evaluate_result(query, selected, positive_patch_ids, top_k)
            metrics.append(metric)
            results.append({"query": query, "metrics": metric, "retrieved": selected})

        method_dir = ensure_dir(rerank_dir / mode)
        write_jsonl(method_dir / "retrieval_results.jsonl", results)
        write_jsonl(method_dir / "retrieval_metrics.jsonl", metrics)
        summary = summarize(mode, metrics)
        summary.update(
            {
                "alpha": mode_alpha,
                "beta": mode_beta,
                "gamma": mode_gamma,
                "candidate_k": mode_candidate_k,
                "top_k": top_k,
                "selection_report": args.selection_report,
            }
        )
        (method_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        all_summaries.append(summary)

    (rerank_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
