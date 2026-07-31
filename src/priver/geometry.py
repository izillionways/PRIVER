from __future__ import annotations


def polygon_to_hbb(poly: list[float]) -> list[float]:
    xs = poly[0::2]
    ys = poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return box_area([x1, y1, x2, y2])


def coverage(inner: list[float], outer: list[float]) -> float:
    denom = box_area(inner)
    if denom <= 0:
        return 0.0
    return intersection_area(inner, outer) / denom


def polygon_area(poly: list[float]) -> float:
    if len(poly) < 6 or len(poly) % 2:
        return 0.0
    points = list(zip(poly[0::2], poly[1::2]))
    signed_area = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )
    return abs(signed_area) * 0.5


def _clip_polygon(
    points: list[tuple[float, float]],
    inside,
    intersection,
) -> list[tuple[float, float]]:
    if not points:
        return []
    clipped: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                clipped.append(intersection(previous, current))
            clipped.append(current)
        elif previous_inside:
            clipped.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return clipped


def clip_polygon_to_box(
    poly: list[float],
    box: list[float],
) -> list[tuple[float, float]]:
    if len(poly) < 6 or len(poly) % 2:
        return []
    x_min, y_min, x_max, y_max = box
    points = list(zip(poly[0::2], poly[1::2]))

    def vertical_intersection(
        start: tuple[float, float],
        end: tuple[float, float],
        x_value: float,
    ) -> tuple[float, float]:
        x1, y1 = start
        x2, y2 = end
        if abs(x2 - x1) < 1e-12:
            return x_value, y1
        ratio = (x_value - x1) / (x2 - x1)
        return x_value, y1 + ratio * (y2 - y1)

    def horizontal_intersection(
        start: tuple[float, float],
        end: tuple[float, float],
        y_value: float,
    ) -> tuple[float, float]:
        x1, y1 = start
        x2, y2 = end
        if abs(y2 - y1) < 1e-12:
            return x1, y_value
        ratio = (y_value - y1) / (y2 - y1)
        return x1 + ratio * (x2 - x1), y_value

    points = _clip_polygon(
        points,
        lambda point: point[0] >= x_min,
        lambda start, end: vertical_intersection(start, end, x_min),
    )
    points = _clip_polygon(
        points,
        lambda point: point[0] <= x_max,
        lambda start, end: vertical_intersection(start, end, x_max),
    )
    points = _clip_polygon(
        points,
        lambda point: point[1] >= y_min,
        lambda start, end: horizontal_intersection(start, end, y_min),
    )
    return _clip_polygon(
        points,
        lambda point: point[1] <= y_max,
        lambda start, end: horizontal_intersection(start, end, y_max),
    )


def polygon_coverage(poly: list[float], box: list[float]) -> float:
    denominator = polygon_area(poly)
    if denominator <= 0:
        return 0.0
    clipped = clip_polygon_to_box(poly, box)
    clipped_flat = [coordinate for point in clipped for coordinate in point]
    return min(1.0, polygon_area(clipped_flat) / denominator)


def make_windows(width: int, height: int, patch_size: int, overlap: float) -> list[list[int]]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    stride = max(1, int(round(patch_size * (1 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= patch_size:
            return [0]
        values = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if values[-1] != last:
            values.append(last)
        return values

    windows: list[list[int]] = []
    for y in starts(height):
        for x in starts(width):
            x2 = min(width, x + patch_size)
            y2 = min(height, y + patch_size)
            windows.append([x, y, x2, y2])
    return windows
