from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .consistency import normalize


@dataclass(frozen=True)
class SameScaleSupportSpec:
    """Configuration for support from neighboring patches at the same scale."""

    kernel: str = "iom"
    aggregation: str = "top_m"
    top_m: int = 3
    supporter_floor: float = 0.25
    score_weight: str = "normalized"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SameScaleSupportSpec":
        return cls(
            kernel=str(values.get("kernel", "iom")),
            aggregation=str(values.get("aggregation", "top_m")),
            top_m=int(values.get("top_m", 3)),
            supporter_floor=float(values.get("supporter_floor", 0.25)),
            score_weight=str(values.get("score_weight", "normalized")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "aggregation": self.aggregation,
            "top_m": self.top_m,
            "supporter_floor": self.supporter_floor,
            "score_weight": self.score_weight,
        }


@dataclass(frozen=True)
class InterScaleSupportSpec:
    """Configuration for reciprocal support between patches at different scales."""

    kernel: str = "iom"
    aggregation: str = "top_m"
    top_m: int = 8
    supporter_floor: float = 0.25
    score_weight: str = "normalized"
    reciprocal_k: int = 12
    row_degree_power: float = 0.5
    column_degree_power: float = 0.5
    consensus: str = "additive"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "InterScaleSupportSpec":
        return cls(
            kernel=str(values.get("kernel", "iom")),
            aggregation=str(values.get("aggregation", "top_m")),
            top_m=int(values.get("top_m", 8)),
            supporter_floor=float(values.get("supporter_floor", 0.25)),
            score_weight=str(values.get("score_weight", "normalized")),
            reciprocal_k=int(values.get("reciprocal_k", 12)),
            row_degree_power=float(values.get("row_degree_power", 0.5)),
            column_degree_power=float(values.get("column_degree_power", 0.5)),
            consensus=str(values.get("consensus", "additive")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "aggregation": self.aggregation,
            "top_m": self.top_m,
            "supporter_floor": self.supporter_floor,
            "score_weight": self.score_weight,
            "reciprocal_k": self.reciprocal_k,
            "row_degree_power": self.row_degree_power,
            "column_degree_power": self.column_degree_power,
            "consensus": self.consensus,
        }


@dataclass(frozen=True)
class CandidateContext:
    base: np.ndarray
    raw_scores: np.ndarray
    iou: np.ndarray
    iom: np.ndarray
    same_scale_mask: np.ndarray
    inter_scale_mask: np.ndarray


def build_candidate_context(candidates: list[dict[str, Any]]) -> CandidateContext:
    """Precompute normalized semantics, overlap, and scale relationships."""
    if not candidates:
        empty = np.asarray([], dtype=float)
        empty_matrix = np.empty((0, 0), dtype=float)
        empty_mask = np.empty((0, 0), dtype=bool)
        return CandidateContext(
            base=empty,
            raw_scores=empty,
            iou=empty_matrix,
            iom=empty_matrix,
            same_scale_mask=empty_mask,
            inter_scale_mask=empty_mask,
        )

    boxes = np.asarray([candidate["bbox"] for candidate in candidates], dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Each candidate bbox must contain [x1, y1, x2, y2]")
    patch_sizes = np.asarray(
        [int(candidate["patch_size"]) for candidate in candidates],
        dtype=int,
    )
    raw_scores = np.maximum(
        0.0,
        np.asarray([float(candidate["score"]) for candidate in candidates]),
    )
    base = np.asarray(normalize(raw_scores.tolist()), dtype=float)

    left = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    top = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    right = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersections = np.maximum(0.0, right - left) * np.maximum(
        0.0, bottom - top
    )
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    unions = areas[:, None] + areas[None, :] - intersections
    minimum_areas = np.minimum(areas[:, None], areas[None, :])
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    iom = np.divide(
        intersections,
        minimum_areas,
        out=np.zeros_like(intersections),
        where=minimum_areas > 0,
    )

    diagonal = np.eye(len(candidates), dtype=bool)
    same_scale_mask = (
        patch_sizes[:, None] == patch_sizes[None, :]
    ) & ~diagonal
    inter_scale_mask = patch_sizes[:, None] != patch_sizes[None, :]
    return CandidateContext(
        base=base,
        raw_scores=raw_scores,
        iou=iou,
        iom=iom,
        same_scale_mask=same_scale_mask,
        inter_scale_mask=inter_scale_mask,
    )


def _overlap_matrix(context: CandidateContext, kernel: str) -> np.ndarray:
    if kernel == "iou":
        return context.iou
    if kernel == "iom":
        return context.iom
    raise ValueError(f"Unknown overlap kernel: {kernel}")


def _score_weights(context: CandidateContext, mode: str) -> np.ndarray:
    if mode == "raw":
        return context.raw_scores
    if mode == "normalized":
        return context.base
    if mode == "sqrt_normalized":
        return np.sqrt(np.maximum(0.0, context.base))
    raise ValueError(f"Unknown score weight: {mode}")


def _aggregate_support(
    weighted: np.ndarray,
    aggregation: str,
    top_m: int,
) -> np.ndarray:
    if weighted.shape[1] == 0:
        return np.zeros(weighted.shape[0], dtype=float)
    if aggregation != "top_m":
        raise ValueError(f"Unknown support aggregation: {aggregation}")
    if top_m <= 0:
        raise ValueError("top_m aggregation requires top_m > 0")

    keep = min(top_m, weighted.shape[1])
    top_values = np.partition(
        weighted,
        weighted.shape[1] - keep,
        axis=1,
    )[:, -keep:]
    return top_values.sum(axis=1) / top_m


def compute_same_scale_support(
    context: CandidateContext,
    spec: SameScaleSupportSpec,
) -> np.ndarray:
    """Compute normalized support from semantically strong same-scale patches."""
    if len(context.base) == 0:
        return np.asarray([], dtype=float)

    supporter_mask = context.base >= spec.supporter_floor
    eligible = context.same_scale_mask & supporter_mask[None, :]
    weighted = (
        _overlap_matrix(context, spec.kernel)
        * _score_weights(context, spec.score_weight)[None, :]
        * eligible
    )
    raw = _aggregate_support(weighted, spec.aggregation, spec.top_m)
    return np.asarray(normalize(raw.tolist()), dtype=float)


def _mutual_top_k_mask(topology: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or topology.size == 0:
        return topology > 0

    keep = min(k, topology.shape[1])
    order = np.argsort(-topology, axis=1, kind="stable")[:, :keep]
    directed = np.zeros_like(topology, dtype=bool)
    rows = np.arange(topology.shape[0])[:, None]
    directed[rows, order] = topology[rows, order] > 0
    return directed & directed.T


def _combine_consensus(
    base: np.ndarray,
    support: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "additive":
        combined = support
    elif mode == "geometric":
        combined = np.sqrt(np.maximum(0.0, base * support))
    elif mode == "harmonic":
        denominator = base + support
        combined = np.divide(
            2.0 * base * support,
            denominator,
            out=np.zeros_like(base),
            where=denominator > 0,
        )
    elif mode == "product":
        combined = base * support
    elif mode == "minimum":
        combined = np.minimum(base, support)
    else:
        raise ValueError(f"Unknown consensus mode: {mode}")
    return np.asarray(normalize(combined.tolist()), dtype=float)


def compute_inter_scale_support(
    context: CandidateContext,
    spec: InterScaleSupportSpec,
) -> np.ndarray:
    """Compute reciprocal, degree-normalized support across patch scales."""
    if len(context.base) == 0:
        return np.asarray([], dtype=float)

    topology = _overlap_matrix(context, spec.kernel) * context.inter_scale_mask
    if spec.reciprocal_k > 0:
        topology = topology * _mutual_top_k_mask(topology, spec.reciprocal_k)

    row_degree = topology.sum(axis=1)
    column_degree = topology.sum(axis=0)
    denominator = np.ones_like(topology)
    epsilon = np.finfo(float).eps
    if spec.row_degree_power > 0:
        denominator *= np.power(
            np.maximum(row_degree, epsilon),
            spec.row_degree_power,
        )[:, None]
    if spec.column_degree_power > 0:
        denominator *= np.power(
            np.maximum(column_degree, epsilon),
            spec.column_degree_power,
        )[None, :]

    supporter_mask = context.base >= spec.supporter_floor
    weighted = (
        topology
        * _score_weights(context, spec.score_weight)[None, :]
        * supporter_mask[None, :]
    )
    weighted = np.divide(
        weighted,
        denominator,
        out=np.zeros_like(weighted),
        where=denominator > 0,
    )
    raw = _aggregate_support(weighted, spec.aggregation, spec.top_m)
    support = np.asarray(normalize(raw.tolist()), dtype=float)
    return _combine_consensus(context.base, support, spec.consensus)


def stable_rank(scores: np.ndarray) -> np.ndarray:
    """Return a descending stable order so tied candidates retain input order."""
    return np.argsort(-scores, kind="stable")


def rerank_candidates(
    candidates: list[dict[str, Any]],
    same_scale_spec: SameScaleSupportSpec,
    inter_scale_spec: InterScaleSupportSpec,
    same_scale_weight: float,
    inter_scale_weight: float,
) -> list[dict[str, Any]]:
    """Apply the frozen PRIVER scoring rule to one within-image candidate list."""
    context = build_candidate_context(candidates)
    same_support = compute_same_scale_support(context, same_scale_spec)
    inter_support = compute_inter_scale_support(context, inter_scale_spec)
    scores = (
        context.base
        + float(same_scale_weight) * same_support
        + float(inter_scale_weight) * inter_support
    )

    reranked: list[dict[str, Any]] = []
    for index in stable_rank(scores):
        item = dict(candidates[int(index)])
        item["base_score_norm"] = float(context.base[index])
        item["same_scale_support"] = float(same_support[index])
        item["inter_scale_support"] = float(inter_support[index])
        item["priver_score"] = float(scores[index])
        reranked.append(item)
    return reranked
