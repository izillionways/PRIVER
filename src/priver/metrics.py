from __future__ import annotations


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def topk_hit(retrieved_patch_ids: list[str], positive_patch_ids: set[str], k: int) -> float:
    return float(any(pid in positive_patch_ids for pid in retrieved_patch_ids[:k]))


def precision_at_k(retrieved_patch_ids: list[str], positive_patch_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_patch_ids[:k]
    if not top:
        return 0.0
    return sum(1 for pid in top if pid in positive_patch_ids) / len(top)


def recall_at_k(retrieved_patch_ids: list[str], positive_patch_ids: set[str], k: int) -> float:
    if not positive_patch_ids:
        return 0.0
    top = set(retrieved_patch_ids[:k])
    return len(top & positive_patch_ids) / len(positive_patch_ids)
