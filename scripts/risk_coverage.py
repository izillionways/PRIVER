from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from priver.geometry import box_area, intersection_area
from priver.io import ensure_dir, read_jsonl, read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build risk-coverage tables for retrieval confidence.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--retrieval-dir",
        default=None,
        help="Directory containing retrieval_metrics.jsonl and retrieval_results.jsonl.",
    )
    parser.add_argument(
        "--coverages",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated coverage levels.",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help="Optional output directory. Defaults to RETRIEVAL_DIR/risk_coverage.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    arr = arr - arr.max()
    probs = np.exp(arr) / np.exp(arr).sum()
    if len(probs) <= 1:
        return 0.0
    return float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs)))


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)
    denom = box_area(a) + box_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def retrieval_confidence_features(row: dict, top_k: int) -> dict:
    retrieved = row.get("retrieved", [])[:top_k]
    if not retrieved:
        return {
            "semantic_margin": 0.0,
            "neg_score_entropy": 0.0,
            "neg_spatial_center_std": 0.0,
            "cross_scale_iou": 0.0,
            "consistency_confidence": 0.0,
        }

    score_key = next(
        (
            key
            for key in (
                "priver_score",
                "rerank_score_v3",
                "rerank_score",
                "score",
            )
            if key in retrieved[0]
        ),
        "score",
    )
    scores = [float(item.get(score_key, item.get("score", 0.0))) for item in retrieved]
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    boxes = [item["bbox"] for item in retrieved]
    patch_sizes = [int(item["patch_size"]) for item in retrieved]
    centers = np.asarray([center(box) for box in boxes], dtype=float)
    center_std = float(np.linalg.norm(centers.std(axis=0))) if len(centers) else 0.0

    cross_scale_ious = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if patch_sizes[i] != patch_sizes[j]:
                cross_scale_ious.append(iou(boxes[i], boxes[j]))
    cross_scale = float(np.mean(cross_scale_ious)) if cross_scale_ious else 0.0

    return {
        "semantic_margin": margin,
        "neg_score_entropy": -entropy(scores),
        "neg_spatial_center_std": -center_std,
        "cross_scale_iou": cross_scale,
        "consistency_confidence": -center_std + 1000.0 * cross_scale,
    }


def risk_coverage_rows(df: pd.DataFrame, confidence_col: str, coverages: list[float]) -> list[dict]:
    ranked = df.sort_values(confidence_col, ascending=False).reset_index(drop=True)
    rows = []
    for coverage in coverages:
        keep_n = max(1, int(math.ceil(len(ranked) * coverage)))
        kept = ranked.iloc[:keep_n]
        rows.append(
            {
                "confidence": confidence_col,
                "coverage": coverage,
                "num_queries": int(len(kept)),
                "risk_at_1": float((1.0 - kept["hit_at_1"]).mean()),
                "risk_at_5": float((1.0 - kept["hit_at_5"]).mean()),
                "risk_at_10": float((1.0 - kept["hit_at_10"]).mean()),
                "low_precision_at_10_rate": float((kept["precision_at_10"] < 0.5).mean()),
                "hit_at_1": float(kept["hit_at_1"].mean()),
                "hit_at_5": float(kept["hit_at_5"].mean()),
                "hit_at_10": float(kept["hit_at_10"].mean()),
                "precision_at_10": float(kept["precision_at_10"].mean()),
                "recall_at_10": float(kept["recall_at_10"].mean()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = Path(cfg["experiment"]["output_dir"])
    top_k = int(cfg.get("retrieval", {}).get("top_k", 10))
    retrieval_dir = Path(args.retrieval_dir) if args.retrieval_dir else out_dir / "retrieval_openclip"
    analysis_dir = ensure_dir(
        Path(args.analysis_dir)
        if args.analysis_dir
        else retrieval_dir / "risk_coverage"
    )

    metrics = pd.DataFrame(read_jsonl(retrieval_dir / "retrieval_metrics.jsonl"))
    results = read_jsonl(retrieval_dir / "retrieval_results.jsonl")
    features = pd.DataFrame(
        [
            {
                "query_id": row["query"]["query_id"],
                **retrieval_confidence_features(row, top_k=top_k),
            }
            for row in results
        ]
    )
    df = metrics.merge(features, on="query_id", how="left")

    confidence_cols = [
        "semantic_margin",
        "neg_score_entropy",
        "neg_spatial_center_std",
        "cross_scale_iou",
        "consistency_confidence",
    ]
    rows = []
    for confidence_col in confidence_cols:
        rows.extend(risk_coverage_rows(df, confidence_col, parse_float_list(args.coverages)))

    risk_df = pd.DataFrame(rows)
    risk_df.to_csv(analysis_dir / "risk_coverage.csv", index=False)
    best_rows = (
        risk_df[risk_df["coverage"].isin([0.5, 0.8, 1.0])]
        .sort_values(["coverage", "risk_at_5", "low_precision_at_10_rate"])
        .groupby("coverage", as_index=False)
        .first()
    )
    summary = {
        "retrieval_dir": str(retrieval_dir),
        "num_queries": int(len(df)),
        "feature_top_k": top_k,
        "confidence_columns": confidence_cols,
        "best_by_selected_coverage": best_rows.to_dict(orient="records"),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
