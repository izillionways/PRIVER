import numpy as np
import pytest

from priver.prompting import aggregate_prompt_scores


def test_prompt_aggregation_matches_frozen_rule() -> None:
    scores = np.asarray(
        [
            [0.8, 0.6],
            [0.8, 0.4],
            [0.8, 0.8],
            [0.8, 0.6],
        ]
    )
    aggregated = aggregate_prompt_scores(scores, 0.5)

    assert aggregated[0] == pytest.approx(0.8)
    assert aggregated[1] == pytest.approx(
        scores[:, 1].mean() - 0.5 * scores[:, 1].std()
    )


def test_prompt_aggregation_requires_a_matrix() -> None:
    with pytest.raises(ValueError, match="shape"):
        aggregate_prompt_scores(np.asarray([0.8, 0.7]))
