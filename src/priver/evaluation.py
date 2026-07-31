from __future__ import annotations

import numpy as np


METRIC_NAMES = (
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "precision_at_5",
    "precision_at_10",
    "recall_at_5",
    "recall_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "share_1024_at_10",
    "scale_entropy_at_10",
)


def evaluate_scores(
    scores: np.ndarray,
    relevant: np.ndarray,
    num_positive: int,
    patch_sizes: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Evaluate one or more score rows against a shared candidate list."""
    score_rows = np.atleast_2d(np.asarray(scores, dtype=float))
    relevant = np.asarray(relevant, dtype=bool)
    patch_sizes = np.asarray(patch_sizes, dtype=int)
    if score_rows.shape[1] != len(relevant):
        raise ValueError("scores and relevant must describe the same candidates")
    if len(patch_sizes) != len(relevant):
        raise ValueError("patch_sizes and relevant must have equal length")

    order = np.argsort(-score_rows, axis=1, kind="stable")
    keep = min(top_k, len(relevant))
    top_order = order[:, :keep]
    ranked_relevance = relevant[top_order]

    values: list[np.ndarray] = []
    for k in (1, 5, 10):
        actual_k = min(k, keep)
        values.append(
            ranked_relevance[:, :actual_k].any(axis=1).astype(float)
            if actual_k
            else np.zeros(len(score_rows), dtype=float)
        )
    for k in (5, 10):
        actual_k = min(k, keep)
        values.append(
            ranked_relevance[:, :actual_k].sum(axis=1) / actual_k
            if actual_k
            else np.zeros(len(score_rows), dtype=float)
        )
    for k in (5, 10):
        actual_k = min(k, keep)
        values.append(
            ranked_relevance[:, :actual_k].sum(axis=1) / num_positive
            if actual_k and num_positive
            else np.zeros(len(score_rows), dtype=float)
        )
    for k in (5, 10):
        actual_k = min(k, keep)
        if not actual_k:
            values.append(np.zeros(len(score_rows), dtype=float))
            continue
        discounts = 1.0 / np.log2(np.arange(2, actual_k + 2))
        dcg = (
            ranked_relevance[:, :actual_k] * discounts[None, :]
        ).sum(axis=1)
        ideal_count = min(num_positive, actual_k)
        ideal = float(discounts[:ideal_count].sum()) if ideal_count else 0.0
        values.append(
            dcg / ideal if ideal else np.zeros(len(score_rows), dtype=float)
        )

    top_sizes = patch_sizes[top_order]
    share_1024 = (
        (top_sizes == 1024).mean(axis=1)
        if keep
        else np.zeros(len(score_rows), dtype=float)
    )
    entropy = np.zeros_like(share_1024)
    for probabilities in (share_1024, 1.0 - share_1024):
        positive = probabilities > 0
        entropy[positive] -= (
            probabilities[positive] * np.log2(probabilities[positive])
        )
    values.extend((share_1024, entropy))
    return np.column_stack(values)


def metrics_dict(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(METRIC_NAMES, values, strict=True)
    }
