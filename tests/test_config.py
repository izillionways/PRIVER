from pathlib import Path

from priver.config import load_priver_config


ROOT = Path(__file__).parents[1]


def test_frozen_config_loads_expected_protocol() -> None:
    config = load_priver_config(ROOT / "configs" / "priver_frozen.json")

    assert config.candidate_k == 100
    assert config.top_k == 10
    assert len(config.prompt_scoring.templates) == 4
    assert config.prompt_scoring.standard_deviation_weight == 0.5
    assert config.same_scale_support.top_m == 3
    assert config.inter_scale_support.reciprocal_k == 12
    assert config.same_scale_weight == 0.25
    assert config.inter_scale_weight == 1.0
