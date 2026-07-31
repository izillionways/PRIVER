from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTROLS = load_script("run_priver_controls")
THRESHOLDS = load_script("evaluate_relevance_threshold_sensitivity")


def frozen_config() -> dict:
    return {
        "same_scale_support": {
            "kernel": "iom",
            "aggregation": "top_m",
            "top_m": 3,
            "supporter_floor": 0.25,
            "score_weight": "normalized",
        },
        "inter_scale_support": {
            "kernel": "iom",
            "aggregation": "top_m",
            "top_m": 8,
            "supporter_floor": 0.25,
            "score_weight": "normalized",
            "reciprocal_k": 12,
            "row_degree_power": 0.5,
            "column_degree_power": 0.5,
            "consensus": "additive",
        },
        "same_scale_weight": 0.25,
        "inter_scale_weight": 1.0,
    }


def test_control_specs_isolate_reciprocity_and_degree_normalization() -> None:
    _, graphs = CONTROLS.build_specs(frozen_config())
    all_overlap = graphs["all_overlap"]
    reciprocal_no_degree = graphs["reciprocal_no_degree"]
    full = graphs["full"]

    assert all_overlap.reciprocal_k == 0
    assert all_overlap.row_degree_power == 0.0
    assert all_overlap.column_degree_power == 0.0
    assert reciprocal_no_degree.reciprocal_k == 12
    assert reciprocal_no_degree.row_degree_power == 0.0
    assert reciprocal_no_degree.column_degree_power == 0.0
    assert full.reciprocal_k == 12
    assert full.row_degree_power == 0.5
    assert full.column_degree_power == 0.5


def test_control_weights_disable_only_named_support() -> None:
    conditions = {
        item.name: item
        for item in CONTROLS.build_control_conditions(frozen_config())
    }
    assert conditions["same_scale_only"].alpha == 0.25
    assert conditions["same_scale_only"].beta == 0.0
    assert conditions["inter_scale_only"].alpha == 0.0
    assert conditions["inter_scale_only"].beta == 1.0
    assert conditions["priver"].alpha == 0.25
    assert conditions["priver"].beta == 1.0


def test_candidate_scale_filter_applies_budget_after_filtering() -> None:
    candidates = [
        {"patch_id": "a", "patch_size": 1024},
        {"patch_id": "b", "patch_size": 512},
        {"patch_id": "c", "patch_size": 512},
    ]
    selected = CONTROLS.filter_candidates(candidates, 1, 512)
    assert [item["patch_id"] for item in selected] == ["b"]


def test_relevance_maps_follow_polygon_coverage_threshold() -> None:
    links = [
        {
            "patch_id": "p1",
            "image_id": "i1",
            "object_id": "o1",
            "class_name": "ship",
            "coverage": 0.2,
        },
        {
            "patch_id": "p2",
            "image_id": "i1",
            "object_id": "o2",
            "class_name": "ship",
            "coverage": 0.4,
        },
    ]
    positives, patch_objects, count = (
        THRESHOLDS.accumulate_relevance_maps(links, 0.3)
    )
    assert count == 1
    assert positives[("i1", "ship")] == {"p2"}
    assert patch_objects == {"p2": {"o2"}}
