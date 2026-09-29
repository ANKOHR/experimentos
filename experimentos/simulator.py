"""Seeded synthetic metric curve functions. This module performs no training."""

from __future__ import annotations

import math
import random
from typing import Any

from .errors import InvalidConfigError

PROFILES = ("exponential_decay", "plateau_then_drop", "linear")


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ("dataset", "simulator", "steps", "seed")
    missing = [key for key in required if key not in config]
    if missing:
        raise InvalidConfigError(f"config is missing required field(s): {', '.join(missing)}")
    if not isinstance(config["dataset"], str) or not config["dataset"]:
        raise InvalidConfigError("dataset must be a non-empty dataset ID")
    if not isinstance(config["steps"], int) or isinstance(config["steps"], bool) or config["steps"] < 1:
        raise InvalidConfigError("steps must be a positive integer")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise InvalidConfigError("seed must be an integer")
    simulator = config["simulator"]
    if isinstance(simulator, str):
        simulator = {"profile": simulator}
    if not isinstance(simulator, dict):
        raise InvalidConfigError("simulator must be a profile name or an object")
    profile = simulator.get("profile")
    if profile not in PROFILES:
        raise InvalidConfigError(f"simulator.profile must be one of: {', '.join(PROFILES)}")
    normalized = dict(config)
    normalized["simulator"] = dict(simulator)
    normalized["name"] = str(config.get("name", "synthetic-run"))
    defaults = {
        "initial_loss": 1.0,
        "noise_std": 0.0,
        "noise_floor": 0.05,
        "decay": 0.05,
        "plateau_steps": max(1, config["steps"] // 3),
        "slope": -0.01,
    }
    for key, value in defaults.items():
        normalized["simulator"].setdefault(key, value)
    for key in ("initial_loss", "noise_std", "noise_floor", "decay", "slope"):
        try:
            normalized["simulator"][key] = float(normalized["simulator"][key])
        except (TypeError, ValueError):
            raise InvalidConfigError(f"simulator.{key} must be numeric") from None
        if not math.isfinite(normalized["simulator"][key]):
            raise InvalidConfigError(f"simulator.{key} must be finite")
    if normalized["simulator"]["noise_std"] < 0:
        raise InvalidConfigError("simulator.noise_std cannot be negative")
    try:
        normalized["simulator"]["plateau_steps"] = int(normalized["simulator"]["plateau_steps"])
    except (TypeError, ValueError):
        raise InvalidConfigError("simulator.plateau_steps must be an integer") from None
    return normalized


def base_value(simulator: dict[str, Any], step: int) -> float:
    """Compute a zero-index-independent value for one-based step numbers."""
    profile = simulator["profile"]
    initial = simulator["initial_loss"]
    if profile == "exponential_decay":
        return simulator["noise_floor"] + (initial - simulator["noise_floor"]) * math.exp(-simulator["decay"] * (step - 1))
    if profile == "plateau_then_drop":
        plateau_steps = simulator["plateau_steps"]
        if step <= plateau_steps:
            return initial
        return simulator["noise_floor"] + (initial - simulator["noise_floor"]) * math.exp(-simulator["decay"] * (step - plateau_steps))
    if profile == "linear":
        return initial + simulator["slope"] * (step - 1)
    raise InvalidConfigError(f"unsupported simulator profile: {profile}")


def new_rng(seed: int) -> random.Random:
    return random.Random(seed)


def next_metric(simulator: dict[str, Any], step: int, rng: random.Random) -> dict[str, Any]:
    noise = rng.gauss(0.0, simulator["noise_std"])
    loss = base_value(simulator, step) + noise
    return {"step": step, "loss": loss}


def state_to_json(state: tuple[Any, ...]) -> Any:
    if isinstance(state, tuple):
        return [state_to_json(item) for item in state]
    return state


def state_from_json(state: Any) -> Any:
    if isinstance(state, list):
        return tuple(state_from_json(item) for item in state)
    return state
