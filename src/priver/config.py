from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reranking import InterScaleSupportSpec, SameScaleSupportSpec


@dataclass(frozen=True)
class PromptScoringConfig:
    templates: tuple[str, ...]
    aggregation: str = "mean_minus_standard_deviation"
    standard_deviation_weight: float = 0.5

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PromptScoringConfig":
        templates = tuple(str(item) for item in values.get("templates", ()))
        if not templates:
            raise ValueError("prompt_scoring.templates must not be empty")
        aggregation = str(
            values.get("aggregation", "mean_minus_standard_deviation")
        )
        if aggregation != "mean_minus_standard_deviation":
            raise ValueError(
                "Only mean_minus_standard_deviation is supported by the "
                "frozen PRIVER release"
            )
        weight = float(values.get("standard_deviation_weight", 0.5))
        if weight < 0:
            raise ValueError("standard_deviation_weight must be nonnegative")
        return cls(
            templates=templates,
            aggregation=aggregation,
            standard_deviation_weight=weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "templates": list(self.templates),
            "aggregation": self.aggregation,
            "standard_deviation_weight": self.standard_deviation_weight,
        }


@dataclass(frozen=True)
class PRIVERConfig:
    full_name: str
    candidate_k: int
    top_k: int
    prompt_scoring: PromptScoringConfig
    same_scale_support: SameScaleSupportSpec
    inter_scale_support: InterScaleSupportSpec
    same_scale_weight: float
    inter_scale_weight: float
    selection_protocol: dict[str, Any]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PRIVERConfig":
        if values.get("method") != "PRIVER":
            raise ValueError("Configuration method must be PRIVER")
        candidate_k = int(values["candidate_k"])
        top_k = int(values["top_k"])
        if candidate_k <= 0 or top_k <= 0:
            raise ValueError("candidate_k and top_k must be positive")
        if top_k > candidate_k:
            raise ValueError("top_k cannot exceed candidate_k")

        same_weight = float(values["same_scale_weight"])
        inter_weight = float(values["inter_scale_weight"])
        if same_weight < 0 or inter_weight < 0:
            raise ValueError("PRIVER support weights must be nonnegative")

        return cls(
            full_name=str(values["full_name"]),
            candidate_k=candidate_k,
            top_k=top_k,
            prompt_scoring=PromptScoringConfig.from_dict(
                values["prompt_scoring"]
            ),
            same_scale_support=SameScaleSupportSpec.from_dict(
                values["same_scale_support"]
            ),
            inter_scale_support=InterScaleSupportSpec.from_dict(
                values["inter_scale_support"]
            ),
            same_scale_weight=same_weight,
            inter_scale_weight=inter_weight,
            selection_protocol=dict(values.get("selection_protocol", {})),
        )


def load_priver_config(path: str | Path) -> PRIVERConfig:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return PRIVERConfig.from_dict(document)
