import numpy as np

from priver.consistency import iou, normalize, support_scores


def reference_support_scores(candidates: list[dict]) -> tuple[list[float], list[float]]:
    spatial = []
    cross_scale = []
    for index, candidate in enumerate(candidates):
        same_scale_support = []
        cross_scale_support = []
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            weighted = iou(candidate["bbox"], other["bbox"]) * max(
                0.0, float(other["score"])
            )
            if int(candidate["patch_size"]) == int(other["patch_size"]):
                same_scale_support.append(weighted)
            else:
                cross_scale_support.append(weighted)
        spatial.append(
            float(np.mean(same_scale_support)) if same_scale_support else 0.0
        )
        cross_scale.append(
            float(np.mean(cross_scale_support)) if cross_scale_support else 0.0
        )
    return normalize(spatial), normalize(cross_scale)


def test_vectorized_support_matches_reference_implementation() -> None:
    candidates = [
        {"bbox": [0, 0, 10, 10], "patch_size": 512, "score": 0.8},
        {"bbox": [5, 0, 15, 10], "patch_size": 512, "score": 0.4},
        {"bbox": [0, 0, 20, 20], "patch_size": 1024, "score": 0.6},
        {"bbox": [20, 20, 30, 30], "patch_size": 1024, "score": -0.1},
    ]

    expected_spatial, expected_cross_scale = reference_support_scores(candidates)
    actual_spatial, actual_cross_scale = support_scores(candidates)

    np.testing.assert_allclose(actual_spatial, expected_spatial, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        actual_cross_scale, expected_cross_scale, rtol=0, atol=1e-12
    )
