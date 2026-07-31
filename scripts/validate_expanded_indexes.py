from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from priver.dataset_split import read_id_manifest
from priver.io import read_jsonl


DATASETS = {
    "dota_development": {
        "manifest": "splits/dota_v15_dev_train_1411.txt",
        "output_dir": "outputs/dota_v15_train_1411",
    },
    "dota_internal_test": {
        "manifest": "splits/dota_v15_test_val_458.txt",
        "output_dir": "outputs/dota_v15_val_458",
    },
    "soda_external_holdout": {
        "manifest": "splits/soda_a_val_holdout_526.txt",
        "output_dir": "outputs/soda_a_val_holdout_526",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate expanded experiment indexes and tiling outputs.")
    parser.add_argument(
        "--output",
        default="notes/expanded_index_validation_2026-07-28.json",
        help="Validation report path.",
    )
    return parser.parse_args()


def validate_dataset(name: str, manifest_path: Path, output_dir: Path) -> dict:
    expected_ids = read_id_manifest(manifest_path)
    image_rows = read_jsonl(output_dir / "image_index.jsonl")
    object_rows = read_jsonl(output_dir / "object_index.jsonl")
    query_rows = read_jsonl(output_dir / "query_index.jsonl")
    patch_rows = read_jsonl(output_dir / "patch_index.jsonl")
    image_ids = [row["image_id"] for row in image_rows]
    if image_ids != expected_ids:
        raise ValueError(f"{name}: image index does not exactly match its frozen manifest")

    expected_set = set(expected_ids)
    for row_type, rows in (
        ("object", object_rows),
        ("query", query_rows),
        ("patch", patch_rows),
    ):
        unknown_ids = sorted({row["image_id"] for row in rows}.difference(expected_set))
        if unknown_ids:
            raise ValueError(f"{name}: {row_type} rows contain unknown source image IDs")

    patches_per_image = Counter(row["image_id"] for row in patch_rows)
    missing_patch_images = sorted(expected_set.difference(patches_per_image))
    if missing_patch_images:
        raise ValueError(f"{name}: source images without candidate patches")

    query_variants = Counter(
        (row["image_id"], row["class_name"]) for row in query_rows
    )
    invalid_variant_counts = {
        f"{image_id}:{class_name}": count
        for (image_id, class_name), count in query_variants.items()
        if count != 4
    }
    if invalid_variant_counts:
        raise ValueError(f"{name}: image-class pairs without exactly four query templates")

    return {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_images": len(image_rows),
        "num_objects": len(object_rows),
        "num_image_class_pairs": len(query_variants),
        "num_queries": len(query_rows),
        "num_patches": len(patch_rows),
        "minimum_patches_per_image": min(patches_per_image.values()),
        "maximum_patches_per_image": max(patches_per_image.values()),
        "manifest_exact_match": True,
        "four_templates_per_image_class_pair": True,
    }


def main() -> None:
    args = parse_args()
    report = {
        name: validate_dataset(
            name,
            Path(paths["manifest"]),
            Path(paths["output_dir"]),
        )
        for name, paths in DATASETS.items()
    }
    report["all_checks_passed"] = True
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
