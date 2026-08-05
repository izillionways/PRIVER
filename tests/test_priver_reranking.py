from priver.reranking import (
    InterScaleSupportSpec,
    SameScaleSupportSpec,
    build_candidate_context,
    rerank_candidates,
)


def test_reranking_returns_a_complete_stable_candidate_list() -> None:
    candidates = [
        {
            "patch_id": "large",
            "score": 0.8,
            "bbox": [0, 0, 1024, 1024],
            "patch_size": 1024,
        },
        {
            "patch_id": "small",
            "score": 0.7,
            "bbox": [0, 0, 512, 512],
            "patch_size": 512,
        },
        {
            "patch_id": "isolated",
            "score": 0.75,
            "bbox": [1200, 1200, 1712, 1712],
            "patch_size": 512,
        },
    ]

    ranked = rerank_candidates(
        candidates,
        SameScaleSupportSpec(),
        InterScaleSupportSpec(),
        same_scale_weight=0.25,
        inter_scale_weight=1.0,
    )

    assert sorted(item["patch_id"] for item in ranked) == [
        "isolated",
        "large",
        "small",
    ]
    assert all("priver_score" in item for item in ranked)
    assert all("same_scale_support" in item for item in ranked)
    assert all("inter_scale_support" in item for item in ranked)


def test_tied_semantic_candidates_retain_input_order_without_support() -> None:
    candidates = [
        {
            "patch_id": "first",
            "score": 0.5,
            "bbox": [0, 0, 512, 512],
            "patch_size": 512,
        },
        {
            "patch_id": "second",
            "score": 0.5,
            "bbox": [1000, 1000, 1512, 1512],
            "patch_size": 512,
        },
    ]
    ranked = rerank_candidates(
        candidates,
        SameScaleSupportSpec(),
        InterScaleSupportSpec(),
        same_scale_weight=0.25,
        inter_scale_weight=1.0,
    )

    assert [item["patch_id"] for item in ranked] == ["first", "second"]


def test_candidate_scores_are_clipped_before_minmax_normalization() -> None:
    candidates = [
        {
            "patch_id": "negative",
            "score": -0.2,
            "bbox": [0, 0, 512, 512],
            "patch_size": 512,
        },
        {
            "patch_id": "zero",
            "score": 0.0,
            "bbox": [600, 0, 1112, 512],
            "patch_size": 512,
        },
        {
            "patch_id": "positive",
            "score": 0.4,
            "bbox": [1200, 0, 1712, 512],
            "patch_size": 512,
        },
    ]

    context = build_candidate_context(candidates)

    assert context.clipped_scores.tolist() == [0.0, 0.0, 0.4]
    assert context.base.tolist() == [0.0, 0.0, 1.0]
