"""PRIVER: training-free reranking for remote-sensing evidence patches."""

from .config import PRIVERConfig, PromptScoringConfig, load_priver_config
from .prompting import aggregate_prompt_scores
from .reranking import (
    InterScaleSupportSpec,
    SameScaleSupportSpec,
    rerank_candidates,
)

__all__ = [
    "InterScaleSupportSpec",
    "PRIVERConfig",
    "PromptScoringConfig",
    "SameScaleSupportSpec",
    "aggregate_prompt_scores",
    "load_priver_config",
    "rerank_candidates",
]

__version__ = "0.1.0"
