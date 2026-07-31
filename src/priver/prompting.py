from __future__ import annotations

import numpy as np


def aggregate_prompt_scores(
    template_scores: np.ndarray,
    standard_deviation_weight: float = 0.5,
) -> np.ndarray:
    """Aggregate equivalent prompt scores using the frozen PRIVER rule."""
    scores = np.asarray(template_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("template_scores must have shape [templates, candidates]")
    if scores.shape[0] == 0:
        raise ValueError("At least one prompt template is required")
    if standard_deviation_weight < 0:
        raise ValueError("standard_deviation_weight must be nonnegative")
    return scores.mean(axis=0) - standard_deviation_weight * scores.std(axis=0)
