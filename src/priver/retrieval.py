from __future__ import annotations

from collections import defaultdict

import numpy as np


def rank_queries_within_images(
    query_features: np.ndarray,
    patch_features: np.ndarray,
    query_rows: list[dict],
    patch_rows: list[dict],
    candidate_k: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if query_features.shape[0] != len(query_rows):
        raise ValueError("Query feature and metadata counts differ")
    if patch_features.shape[0] != len(patch_rows):
        raise ValueError("Patch feature and metadata counts differ")
    if query_features.shape[1] != patch_features.shape[1]:
        raise ValueError("Query and patch feature dimensions differ")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    query_indices_by_image: dict[str, list[int]] = defaultdict(list)
    patch_indices_by_image: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(query_rows):
        query_indices_by_image[row["image_id"]].append(index)
    for index, row in enumerate(patch_rows):
        patch_indices_by_image[row["image_id"]].append(index)

    rankings: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(query_rows)
    for image_id, query_indices_list in query_indices_by_image.items():
        patch_indices_list = patch_indices_by_image.get(image_id)
        if not patch_indices_list:
            raise ValueError(f"No candidate patches found for query image {image_id}")

        query_indices = np.asarray(query_indices_list, dtype=np.int64)
        patch_indices = np.asarray(patch_indices_list, dtype=np.int64)
        local_scores = query_features[query_indices] @ patch_features[patch_indices].T
        keep = min(candidate_k, len(patch_indices))

        for local_query_index, global_query_index in enumerate(query_indices):
            order = np.argsort(-local_scores[local_query_index])[:keep]
            rankings[int(global_query_index)] = (
                patch_indices[order],
                local_scores[local_query_index, order],
            )

    if any(ranking is None for ranking in rankings):
        raise RuntimeError("Failed to rank one or more queries")
    return [ranking for ranking in rankings if ranking is not None]
