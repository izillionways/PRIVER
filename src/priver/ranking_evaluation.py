from __future__ import annotations

import math

import numpy as np


def ndcg_at_k(ids: list[str], positives: set[str], k: int) -> float:
    gains = [1.0 if patch_id in positives else 0.0 for patch_id in ids[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_count = min(k, len(positives))
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / idcg


def bootstrap_cluster_means(
    counts: np.ndarray,
    sums: dict[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    num_images = len(counts)
    probabilities = np.full(num_images, 1.0 / num_images)
    metric_count = next(iter(sums.values())).shape[1]
    samples = {
        method: np.empty((n_bootstrap, metric_count), dtype=float)
        for method in sums
    }
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        weights = rng.multinomial(
            num_images, probabilities, size=stop - start
        ).astype(float)
        denominators = weights @ counts
        for method, method_sums in sums.items():
            samples[method][start:stop] = (
                weights @ method_sums
            ) / denominators[:, None]
    return samples
