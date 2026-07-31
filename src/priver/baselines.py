from __future__ import annotations

from typing import Any

from .consistency import normalize, support_scores


def rerank_consistency_baseline(
    candidates: list[dict[str, Any]],
    mode: str,
    alpha: float,
    beta: float,
    gamma: float = 1.0,
) -> list[dict[str, Any]]:
    """Apply the pre-PRIVER consistency baseline used in ablations."""
    spatial, cross_scale = support_scores(candidates)
    base_scores = normalize([float(row["score"]) for row in candidates])
    reranked: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        if mode == "spatial":
            final_score = base_scores[index] + alpha * spatial[index]
        elif mode == "cross_scale":
            final_score = base_scores[index] + beta * cross_scale[index]
        elif mode == "full":
            final_score = (
                base_scores[index]
                + alpha * spatial[index]
                + beta * cross_scale[index]
            )
        elif mode == "adaptive":
            semantic_gate = max(0.0, base_scores[index]) ** gamma
            final_score = base_scores[index] + semantic_gate * (
                alpha * spatial[index] + beta * cross_scale[index]
            )
        elif mode == "clip":
            final_score = base_scores[index]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        item = dict(row)
        item["base_score_norm"] = base_scores[index]
        item["spatial_support"] = spatial[index]
        item["cross_scale_support"] = cross_scale[index]
        item["semantic_gate"] = float(
            max(0.0, base_scores[index]) ** gamma
        )
        item["rerank_score"] = float(final_score)
        reranked.append(item)
    reranked.sort(key=lambda row: row["rerank_score"], reverse=True)
    return reranked
