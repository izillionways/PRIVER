import numpy as np

from priver.retrieval import rank_queries_within_images


def test_within_image_ranking_matches_global_score_slicing() -> None:
    query_rows = [
        {"query_id": "q0", "image_id": "a"},
        {"query_id": "q1", "image_id": "b"},
        {"query_id": "q2", "image_id": "a"},
    ]
    patch_rows = [
        {"patch_id": "p0", "image_id": "a"},
        {"patch_id": "p1", "image_id": "b"},
        {"patch_id": "p2", "image_id": "a"},
        {"patch_id": "p3", "image_id": "b"},
    ]
    query_features = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
    )
    patch_features = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=np.float32
    )

    rankings = rank_queries_within_images(
        query_features,
        patch_features,
        query_rows,
        patch_rows,
        candidate_k=2,
    )
    global_scores = query_features @ patch_features.T

    for query_index, image_id in enumerate(["a", "b", "a"]):
        image_indices = np.asarray(
            [index for index, row in enumerate(patch_rows) if row["image_id"] == image_id]
        )
        order = np.argsort(-global_scores[query_index, image_indices])[:2]
        expected_indices = image_indices[order]
        expected_scores = global_scores[query_index, expected_indices]
        actual_indices, actual_scores = rankings[query_index]
        np.testing.assert_array_equal(actual_indices, expected_indices)
        np.testing.assert_allclose(actual_scores, expected_scores, rtol=0, atol=1e-7)


def test_candidate_count_is_capped_by_image_patch_count() -> None:
    rankings = rank_queries_within_images(
        np.ones((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        [{"query_id": "q0", "image_id": "a"}],
        [{"patch_id": "p0", "image_id": "a"}],
        candidate_k=200,
    )

    assert rankings[0][0].tolist() == [0]
