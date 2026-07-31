from pathlib import Path

import pytest

from priver.dataset_split import read_id_manifest, select_paths_by_manifest


def write_manifest(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_order_controls_selected_paths(tmp_path: Path) -> None:
    paths = [tmp_path / "b.png", tmp_path / "a.png", tmp_path / "c.png"]
    manifest = tmp_path / "include.txt"
    write_manifest(manifest, ["# frozen order", "c", "a"])

    selected = select_paths_by_manifest(paths, include_ids_file=manifest)

    assert [path.stem for path in selected] == ["c", "a"]


def test_exclusion_and_limit_are_applied_after_sorting(tmp_path: Path) -> None:
    paths = [tmp_path / "c.json", tmp_path / "a.json", tmp_path / "b.json"]
    excluded = tmp_path / "exclude.txt"
    write_manifest(excluded, ["b"])

    selected = select_paths_by_manifest(paths, exclude_ids_file=excluded, max_items=1)

    assert [path.stem for path in selected] == ["a"]


def test_missing_manifest_id_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "include.txt"
    write_manifest(manifest, ["missing"])

    with pytest.raises(ValueError, match="not found"):
        select_paths_by_manifest([tmp_path / "a.png"], include_ids_file=manifest)


def test_duplicate_manifest_id_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "include.txt"
    write_manifest(manifest, ["a", "a"])

    with pytest.raises(ValueError, match="Duplicate"):
        read_id_manifest(manifest)
