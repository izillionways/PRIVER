from __future__ import annotations

import argparse
import json
from collections import defaultdict

from priver.geometry import coverage, make_windows, polygon_coverage
from priver.io import ensure_dir, read_jsonl, read_yaml, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-scale patch metadata.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    image_rows = read_jsonl(out_dir / "image_index.jsonl")
    object_rows = read_jsonl(out_dir / "object_index.jsonl")

    objects_by_image: dict[str, list[dict]] = defaultdict(list)
    for obj in object_rows:
        objects_by_image[obj["image_id"]].append(obj)

    tile_cfg = cfg["tiling"]
    patch_sizes = [int(v) for v in tile_cfg["patch_sizes"]]
    overlap = float(tile_cfg.get("overlap", 0.0))
    min_valid_ratio = float(tile_cfg.get("min_valid_ratio", 0.0))
    min_object_coverage = float(tile_cfg.get("min_object_coverage", 0.3))
    object_coverage_geometry = str(
        tile_cfg.get("object_coverage_geometry", "bbox")
    ).lower()
    if object_coverage_geometry not in {"bbox", "polygon"}:
        raise ValueError(
            "tiling.object_coverage_geometry must be 'bbox' or 'polygon'"
        )

    patch_rows: list[dict] = []
    positive_links: list[dict] = []

    for image in image_rows:
        width = int(image["width"])
        height = int(image["height"])
        image_area = float(width * height)
        for patch_size in patch_sizes:
            for patch_idx, window in enumerate(make_windows(width, height, patch_size, overlap)):
                valid_ratio = ((window[2] - window[0]) * (window[3] - window[1])) / min(
                    image_area, float(patch_size * patch_size)
                )
                if valid_ratio < min_valid_ratio:
                    continue

                patch_id = f"{image['image_id']}_s{patch_size}_{patch_idx:06d}"
                class_hits: dict[str, int] = defaultdict(int)
                object_hits: list[str] = []
                for obj in objects_by_image[image["image_id"]]:
                    if object_coverage_geometry == "polygon":
                        if "polygon" not in obj:
                            raise ValueError(
                                f"Object {obj['object_id']} has no polygon annotation"
                            )
                        cov = polygon_coverage(obj["polygon"], window)
                    else:
                        cov = coverage(obj["bbox"], window)
                    if cov >= min_object_coverage:
                        object_hits.append(obj["object_id"])
                        class_hits[obj["class_name"]] += 1
                        positive_links.append(
                            {
                                "patch_id": patch_id,
                                "image_id": image["image_id"],
                                "object_id": obj["object_id"],
                                "class_name": obj["class_name"],
                                "coverage": cov,
                                "coverage_geometry": object_coverage_geometry,
                            }
                        )

                patch_rows.append(
                    {
                        "patch_id": patch_id,
                        "image_id": image["image_id"],
                        "image_path": image["image_path"],
                        "patch_size": patch_size,
                        "bbox": window,
                        "num_positive_objects": len(object_hits),
                        "positive_object_ids": object_hits,
                        "class_counts": dict(sorted(class_hits.items())),
                    }
                )

    write_jsonl(out_dir / "patch_index.jsonl", patch_rows)
    write_jsonl(out_dir / "patch_object_links.jsonl", positive_links)

    summary = {
        "num_images": len(image_rows),
        "num_patches": len(patch_rows),
        "num_patch_object_links": len(positive_links),
        "patch_sizes": patch_sizes,
        "overlap": overlap,
        "min_object_coverage": min_object_coverage,
        "object_coverage_geometry": object_coverage_geometry,
    }
    (out_dir / "tiling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
