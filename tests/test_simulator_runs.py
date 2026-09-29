import math

import pytest

from experimentos.runs import metric_log_bytes, run_experiment
from experimentos.simulator import base_value, normalize_config
from conftest import make_config


def test_same_config_has_byte_identical_metric_jsonl(home):
    config = make_config(steps=15, seed=811, noise_std=0.025)
    first = run_experiment(config, root=home)
    second = run_experiment(config, root=home)
    assert metric_log_bytes(first["id"], home) == metric_log_bytes(second["id"], home)


def test_exponential_decay_formula_spot_checks():
    config = normalize_config(make_config(steps=5, profile="exponential_decay", initial_loss=2, noise_floor=0.2, decay=0.5))
    simulator = config["simulator"]
    assert base_value(simulator, 1) == pytest.approx(2)
    assert base_value(simulator, 3) == pytest.approx(0.2 + 1.8 * math.exp(-1))


def test_plateau_then_drop_formula_spot_checks():
    config = normalize_config(make_config(steps=8, profile="plateau_then_drop", initial_loss=1.2, noise_floor=0.2, decay=0.3, plateau_steps=3))
    simulator = config["simulator"]
    assert base_value(simulator, 1) == pytest.approx(1.2)
    assert base_value(simulator, 3) == pytest.approx(1.2)
    assert base_value(simulator, 4) == pytest.approx(0.2 + 1.0 * math.exp(-0.3))


def test_linear_formula_spot_checks():
    config = normalize_config(make_config(steps=5, profile="linear", initial_loss=2, slope=-0.25))
    assert base_value(config["simulator"], 1) == pytest.approx(2)
    assert base_value(config["simulator"], 5) == pytest.approx(1)


def test_json_config_file_is_loaded_and_normalized(home, tmp_path):
    from experimentos.runs import load_config
    import json

    path = tmp_path / "run.json"
    path.write_text(json.dumps(make_config()), encoding="utf-8")
    assert load_config(path)["simulator"]["profile"] == "exponential_decay"
