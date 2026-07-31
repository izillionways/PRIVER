from __future__ import annotations

import pytest

from priver.geometry import clip_polygon_to_box, polygon_area, polygon_coverage


def test_polygon_area_is_orientation_invariant() -> None:
    clockwise = [0, 0, 0, 4, 4, 4, 4, 0]
    counterclockwise = list(
        coordinate
        for point in reversed(list(zip(clockwise[0::2], clockwise[1::2])))
        for coordinate in point
    )
    assert polygon_area(clockwise) == pytest.approx(16.0)
    assert polygon_area(counterclockwise) == pytest.approx(16.0)


def test_axis_aligned_polygon_coverage() -> None:
    square = [0, 0, 4, 0, 4, 4, 0, 4]
    assert polygon_coverage(square, [0, 0, 4, 4]) == pytest.approx(1.0)
    assert polygon_coverage(square, [0, 0, 2, 4]) == pytest.approx(0.5)
    assert polygon_coverage(square, [5, 5, 6, 6]) == pytest.approx(0.0)


def test_rotated_polygon_coverage_uses_polygon_area() -> None:
    diamond = [0, 2, 2, 0, 4, 2, 2, 4]
    assert polygon_area(diamond) == pytest.approx(8.0)
    assert polygon_coverage(diamond, [0, 0, 2, 4]) == pytest.approx(0.5)


def test_clipped_polygon_stays_inside_box() -> None:
    polygon = [-2, 1, 2, -2, 6, 1, 2, 6]
    box = [0, 0, 4, 4]
    clipped = clip_polygon_to_box(polygon, box)
    assert len(clipped) >= 3
    assert all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in clipped)
