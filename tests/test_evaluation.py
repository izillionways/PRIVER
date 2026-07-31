import numpy as np
import pytest

from priver.evaluation import METRIC_NAMES, evaluate_scores, metrics_dict


def test_evaluate_scores_returns_expected_binary_ranking_metrics() -> None:
    values = evaluate_scores(
        np.asarray([[0.9, 0.8, 0.7]]),
        np.asarray([True, False, True]),
        num_positive=2,
        patch_sizes=np.asarray([512, 1024, 512]),
        top_k=3,
    )[0]
    metrics = metrics_dict(values)

    assert tuple(metrics) == METRIC_NAMES
    assert metrics["hit_at_1"] == 1.0
    assert metrics["precision_at_5"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5"] == 1.0


def test_evaluate_scores_rejects_misaligned_candidates() -> None:
    with pytest.raises(ValueError, match="same candidates"):
        evaluate_scores(
            np.asarray([[0.5, 0.4]]),
            np.asarray([True]),
            num_positive=1,
            patch_sizes=np.asarray([512]),
            top_k=1,
        )
