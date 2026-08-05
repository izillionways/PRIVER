from __future__ import annotations

from collections import defaultdict

import numpy as np


def normalize(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return []
    minimum = float(array.min())
    maximum = float(array.max())
    if abs(maximum - minimum) < 1e-12:
        return [0.0 for _ in values]
    return ((array - minimum) / (maximum - minimum)).tolist()


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
