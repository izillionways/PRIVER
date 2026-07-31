from __future__ import annotations

from collections import defaultdict

import numpy as np

from .geometry import box_area, intersection_area


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)
    denominator = box_area(a) + box_area(b) - inter
    return inter / denominator if denominator > 0 else 0.0


def normalize(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return []
    minimum = float(array.min())
    maximum = float(array.max())
    if abs(maximum - minimum) < 1e-12:
        return [0.0 for _ in values]
    return ((array - minimum) / (maximum - minimum)).tolist()


def support_scores(candidates: list[dict]) -> tuple[list[float], list[float]]:
    if not candidates:
        return [], []

    boxes = np.asarray([candidate["bbox"] for candidate in candidates], dtype=float)
    patch_sizes = np.asarray([int(candidate["patch_size"]) for candidate in candidates])
    positive_scores = np.maximum(
        0.0,
        np.asarray([float(candidate["score"]) for candidate in candidates], dtype=float),
    )

    left = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    top = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    right = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersections = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    unions = areas[:, None] + areas[None, :] - intersections
    overlaps = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    weighted = overlaps * positive_scores[None, :]

    diagonal = np.eye(len(candidates), dtype=bool)
    same_scale_mask = (patch_sizes[:, None] == patch_sizes[None, :]) & ~diagonal
    cross_scale_mask = patch_sizes[:, None] != patch_sizes[None, :]
    same_counts = same_scale_mask.sum(axis=1)
    cross_counts = cross_scale_mask.sum(axis=1)

    spatial = np.divide(
        (weighted * same_scale_mask).sum(axis=1),
        same_counts,
        out=np.zeros(len(candidates), dtype=float),
        where=same_counts > 0,
    )
    cross_scale = np.divide(
        (weighted * cross_scale_mask).sum(axis=1),
        cross_counts,
        out=np.zeros(len(candidates), dtype=float),
        where=cross_counts > 0,
    )
    return normalize(spatial.tolist()), normalize(cross_scale.tolist())


def build_positive_patch_map(
    patch_rows: list[dict],
) -> dict[tuple[str, str], set[str]]:
    positives: dict[tuple[str, str], set[str]] = defaultdict(set)
    for patch in patch_rows:
        for class_name, count in patch.get("class_counts", {}).items():
            if count > 0:
                positives[(patch["image_id"], class_name)].add(patch["patch_id"])
    return positives


def positive_patch_ids_from_query(
    query: dict,
    positive_patch_map: dict[tuple[str, str], set[str]],
) -> set[str]:
    return positive_patch_map.get((query["image_id"], query["class_name"]), set())
