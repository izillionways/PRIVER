from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .dataset_split import select_paths_by_manifest
from .geometry import polygon_to_hbb
from .queries import format_query


def parse_soda_annotation(path: str | Path, ignored_classes: set[str] | None = None) -> tuple[dict, list[dict]]:
    ignored_classes = ignored_classes or set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    image = data["images"]
    categories = {int(row["id"]): row["name"] for row in data.get("categories", [])}
    objects: list[dict] = []
    for ann in data.get("annotations", []):
        class_name = categories.get(int(ann["category_id"]), str(ann["category_id"]))
        if class_name in ignored_classes:
            continue
        poly = [float(v) for v in ann["poly"]]
        objects.append(
            {
                "class_name": class_name,
                "polygon": poly,
                "bbox": polygon_to_hbb(poly),
                "difficult": 0,
                "area": float(ann.get("area", 0.0)),
            }
        )
    return image, objects


def collect_soda_records(cfg: dict) -> tuple[list[dict], list[dict]]:
    ds = cfg["dataset"]
    root = Path(ds["root"])
    split = ds.get("split", "val")
    image_dir = root / ds.get("image_dir", "Images/Images")
    annotation_dir = root / ds.get("annotation_dir", f"Annotations/{split}")
    max_images = ds.get("max_images")
    classes = set(ds.get("classes", []))
    ignored_classes = set(ds.get("ignored_classes", ["ignore"]))

    ann_paths = select_paths_by_manifest(
        annotation_dir.glob("*.json"),
        include_ids_file=ds.get("include_ids_file"),
        exclude_ids_file=ds.get("exclude_ids_file"),
        max_items=int(max_images) if max_images is not None else None,
    )

    image_rows: list[dict] = []
    object_rows: list[dict] = []

    for ann_path in ann_paths:
        image, objects = parse_soda_annotation(ann_path, ignored_classes=ignored_classes)
        image_path = image_dir / image["file_name"]
        if not image_path.exists():
            continue
        image_id = Path(image["file_name"]).stem
        if classes:
            objects = [obj for obj in objects if obj["class_name"] in classes]
        image_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "annotation_path": str(ann_path),
                "width": int(image["width"]),
                "height": int(image["height"]),
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
