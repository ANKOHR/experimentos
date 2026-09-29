from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experimentos.datasets import register_dataset
from experimentos.runs import run_experiment


@pytest.fixture
def home(tmp_path: Path) -> Path:
    register_dataset("toy", seed=19, rows=12, features=2, root=tmp_path)
    return tmp_path


def make_config(dataset: str = "toy", steps: int = 8, seed: int = 71, profile: str = "exponential_decay", **params):
    simulator = {"profile": profile, "noise_std": 0.0}
    simulator.update(params)
    return {"name": "fixture", "dataset": dataset, "simulator": simulator, "steps": steps, "seed": seed}


@pytest.fixture
def make_run(home):
    def create(config=None):
        return run_experiment(config or make_config(), root=home)
    return create


def cli(home: Path, *args: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "experimentos", "--home", str(home), *args], cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True)
