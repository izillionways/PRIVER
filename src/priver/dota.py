from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image

from .dataset_split import select_paths_by_manifest
from .geometry import polygon_to_hbb
from .queries import format_query


def parse_dota_annotation(path: str | Path, include_difficult: bool = True) -> list[dict]:
    objects: list[dict] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("imagesource:") or line.startswith("gsd:"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            poly = [float(v) for v in parts[:8]]
            class_name = parts[8]
            difficult = int(parts[9]) if parts[9].isdigit() else 0
            if difficult and not include_difficult:
                continue
            objects.append(
                {
                    "class_name": class_name,
                    "polygon": poly,
                    "bbox": polygon_to_hbb(poly),
                    "difficult": difficult,
                }
            )
    return objects


def collect_dota_records(cfg: dict) -> tuple[list[dict], list[dict]]:
    ds = cfg["dataset"]
    root = Path(ds["root"])
    split_root = root / ds["split"]
    image_dir = split_root / ds["image_dir"]
    annotation_dir = split_root / ds["annotation_dir"]
    include_difficult = bool(ds.get("include_difficult", True))
    max_images = ds.get("max_images")

    image_paths = select_paths_by_manifest(
        image_dir.glob("*.png"),
        include_ids_file=ds.get("include_ids_file"),
        exclude_ids_file=ds.get("exclude_ids_file"),
        max_items=int(max_images) if max_images is not None else None,
    )

    image_rows: list[dict] = []
    object_rows: list[dict] = []

    for image_idx, image_path in enumerate(image_paths):
        image_id = image_path.stem
        ann_path = annotation_dir / f"{image_id}.txt"
        if not ann_path.exists():
            continue
        with Image.open(image_path) as im:
            width, height = im.size

        objects = parse_dota_annotation(ann_path, include_difficult=include_difficult)
        image_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "annotation_path": str(ann_path),
                "width": width,
                "height": height,
                "num_objects": len(objects),
            }
        )
        for obj_idx, obj in enumerate(objects):
            object_rows.append(
                {
                    "object_id": f"{image_id}_{obj_idx:04d}",
                    "image_id": image_id,
                    **obj,
                }
            )

    return image_rows, object_rows


def build_class_queries(
    image_rows: list[dict],
    object_rows: list[dict],
    templates: list[str],
    min_objects_per_query: int = 1,
) -> list[dict]:
    image_lookup = {row["image_id"]: row for row in image_rows}
    by_image_class: dict[tuple[str, str], list[str]] = defaultdict(list)
    for obj in object_rows:
        by_image_class[(obj["image_id"], obj["class_name"])].append(obj["object_id"])

    queries: list[dict] = []
    for (image_id, class_name), object_ids in sorted(by_image_class.items()):
        if len(object_ids) < min_objects_per_query:
            continue
        for template_idx, template in enumerate(templates):
            query_id = f"{image_id}_{class_name}_t{template_idx}"
            text = format_query(template, class_name)
            queries.append(
                {
                    "query_id": query_id,
                    "image_id": image_id,
                    "image_path": image_lookup[image_id]["image_path"],
                    "class_name": class_name,
                    "text": text,
                    "positive_object_ids": object_ids,
                    "num_positive_objects": len(object_ids),
                }
            )
    return queries
