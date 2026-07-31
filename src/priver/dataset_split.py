from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable


def read_id_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    ids = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    duplicates = sorted(image_id for image_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate image IDs in {manifest_path}: {preview}")
    return ids


def select_paths_by_manifest(
    paths: Iterable[Path],
    include_ids_file: str | Path | None = None,
    exclude_ids_file: str | Path | None = None,
    max_items: int | None = None,
) -> list[Path]:
    ordered_paths = sorted(paths, key=lambda path: path.stem)
    paths_by_id: dict[str, Path] = {}
    for path in ordered_paths:
        if path.stem in paths_by_id:
            raise ValueError(f"Duplicate candidate image ID: {path.stem}")
        paths_by_id[path.stem] = path

    include_ids = read_id_manifest(include_ids_file) if include_ids_file else None
    exclude_ids = set(read_id_manifest(exclude_ids_file)) if exclude_ids_file else set()

    if include_ids is not None:
        missing = [image_id for image_id in include_ids if image_id not in paths_by_id]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Manifest IDs not found in dataset: {preview}")
        overlap = [image_id for image_id in include_ids if image_id in exclude_ids]
        if overlap:
            preview = ", ".join(overlap[:5])
            raise ValueError(f"Image IDs occur in both include and exclude manifests: {preview}")
        selected = [paths_by_id[image_id] for image_id in include_ids]
    else:
        unknown_exclusions = sorted(exclude_ids.difference(paths_by_id))
        if unknown_exclusions:
            preview = ", ".join(unknown_exclusions[:5])
            raise ValueError(f"Excluded IDs not found in dataset: {preview}")
        selected = [path for path in ordered_paths if path.stem not in exclude_ids]

    if max_items is not None:
        selected = selected[: int(max_items)]
    return selected
