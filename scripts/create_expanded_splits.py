from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from priver.io import ensure_dir, read_jsonl


EXPECTED_COUNTS = {
    "dota_v15_dev_train_1411": 1411,
    "dota_v15_test_val_458": 458,
    "soda_a_val_pilot_50": 50,
    "soda_a_val_holdout_526": 526,
    "soda_a_test_reserve_870": 870,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze image-level manifests for the expanded experiment.")
    parser.add_argument(
        "--dota-root",
        default="/data/rsdata/DOTA-v1.5/DOTA_V1.5",
        help="DOTA-v1.5 root containing train/ and val/.",
    )
    parser.add_argument(
        "--soda-root",
        default="/data/rsdata/SODA-A",
        help="SODA-A root containing Annotations/.",
    )
    parser.add_argument(
        "--pilot-index",
        default="outputs/soda_a_val_50/image_index.jsonl",
        help="Existing 50-image SODA-A pilot index.",
    )
    parser.add_argument("--output-dir", default="splits", help="Manifest output directory.")
    return parser.parse_args()


def sorted_stems(directory: Path, pattern: str) -> list[str]:
    return sorted(path.stem for path in directory.glob(pattern))


def assert_unique(name: str, ids: list[str]) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate IDs")


def write_manifest(path: Path, ids: list[str]) -> dict:
    payload = "".join(f"{image_id}\n" for image_id in ids)
    path.write_text(payload, encoding="utf-8")
    return {
        "path": str(path),
        "count": len(ids),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
    }


def assert_disjoint(name_a: str, ids_a: list[str], name_b: str, ids_b: list[str]) -> None:
    overlap = sorted(set(ids_a).intersection(ids_b))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"{name_a} and {name_b} overlap: {preview}")


def main() -> None:
    args = parse_args()
    dota_root = Path(args.dota_root)
    soda_root = Path(args.soda_root)
    output_dir = ensure_dir(args.output_dir)

    splits = {
        "dota_v15_dev_train_1411": sorted_stems(dota_root / "train/images", "*.png"),
        "dota_v15_test_val_458": sorted_stems(dota_root / "val/images", "*.png"),
        "soda_a_val_pilot_50": [
            row["image_id"] for row in read_jsonl(args.pilot_index)
        ],
        "soda_a_test_reserve_870": sorted_stems(
            soda_root / "Annotations/test", "*.json"
        ),
    }
    soda_val_ids = sorted_stems(soda_root / "Annotations/val", "*.json")
    pilot_ids = set(splits["soda_a_val_pilot_50"])
    missing_pilot_ids = sorted(pilot_ids.difference(soda_val_ids))
    if missing_pilot_ids:
        preview = ", ".join(missing_pilot_ids[:5])
        raise ValueError(f"Pilot IDs not found in SODA-A validation annotations: {preview}")
    splits["soda_a_val_holdout_526"] = [
        image_id for image_id in soda_val_ids if image_id not in pilot_ids
    ]

    for name, ids in splits.items():
        assert_unique(name, ids)
        expected = EXPECTED_COUNTS[name]
        if len(ids) != expected:
            raise ValueError(f"{name}: expected {expected} IDs, found {len(ids)}")

    assert_disjoint(
        "DOTA train development",
        splits["dota_v15_dev_train_1411"],
        "DOTA validation test",
        splits["dota_v15_test_val_458"],
    )
    assert_disjoint(
        "SODA-A validation pilot",
        splits["soda_a_val_pilot_50"],
        "SODA-A validation holdout",
        splits["soda_a_val_holdout_526"],
    )
    assert_disjoint(
        "SODA-A validation",
        soda_val_ids,
        "SODA-A test reserve",
        splits["soda_a_test_reserve_870"],
    )
    if sorted(splits["soda_a_val_pilot_50"] + splits["soda_a_val_holdout_526"]) != soda_val_ids:
        raise ValueError("SODA-A pilot and holdout do not exactly partition the validation split")

    manifest_summaries = {}
    for name, ids in splits.items():
        manifest_summaries[name] = write_manifest(output_dir / f"{name}.txt", ids)

    summary = {
        "protocol": {
            "development": "All 1,411 DOTA-v1.5 train source images",
            "internal_test": "All 458 DOTA-v1.5 validation source images",
            "external_pilot": "Previously used first 50 SODA-A validation source images",
            "external_holdout": "Remaining 526 SODA-A validation source images",
            "untouched_reserve": "All 870 SODA-A test source images",
            "unit_of_separation": "source image",
        },
        "manifests": manifest_summaries,
        "checks": {
            "dota_development_test_overlap": 0,
            "soda_pilot_holdout_overlap": 0,
            "soda_validation_test_overlap": 0,
            "soda_pilot_plus_holdout_equals_validation": True,
        },
    }
    summary_path = output_dir / "expanded_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
