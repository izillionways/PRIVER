from __future__ import annotations

import numpy as np
import pytest

from priver.ranking_evaluation import bootstrap_cluster_means, ndcg_at_k


def test_ndcg_rewards_earlier_positive() -> None:
    positives = {"p1"}
    assert ndcg_at_k(["p1", "p2"], positives, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["p2", "p1"], positives, 2) < 1.0


def test_cluster_bootstrap_is_seeded_and_preserves_constant_mean() -> None:
    counts = np.asarray([2.0, 2.0])
    sums = {"method": np.asarray([[1.0, 2.0], [1.0, 2.0]])}
    first = bootstrap_cluster_means(counts, sums, 20, 42, 5)["method"]
    second = bootstrap_cluster_means(counts, sums, 20, 42, 5)["method"]
    assert np.array_equal(first, second)
    assert np.allclose(first, np.asarray([0.5, 1.0]))
