"""Run management and resumable deterministic metric generation."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .datasets import get_dataset
from .errors import InvalidConfigError, NotFoundError
from .simulator import new_rng, next_metric, normalize_config, state_from_json, state_to_json
from .storage import ensure_layout, read_json, read_jsonl, write_json, workspace


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidConfigError(f"could not read config JSON: {error}") from error
    if not isinstance(config, dict):
        raise InvalidConfigError("config root must be a JSON object")
    return normalize_config(config)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _run_dir(run_id: str, root: str | Path | None = None) -> Path:
    path = workspace(root) / "runs" / run_id
    if not path.exists():
        raise NotFoundError(f"run '{run_id}' does not exist")
    return path


def get_run(run_id: str, root: str | Path | None = None) -> dict[str, Any]:
    run_dir = _run_dir(run_id, root)
    meta = read_json(run_dir / "meta.json")
    meta["metrics"] = read_jsonl(run_dir / "metrics.jsonl")
    return meta


def list_runs(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = ensure_layout(root)
    result = []
    for run_dir in sorted((base / "runs").iterdir()):
        if run_dir.is_dir() and (run_dir / "meta.json").exists():
            result.append(get_run(run_dir.name, root))
    return result


def _append_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def run_experiment(config: dict[str, Any] | None = None, root: str | Path | None = None, resume: str | None = None, steps: int | None = None) -> dict[str, Any]:
    normalized = normalize_config(config) if config is not None else None
    base = ensure_layout(root)
    if resume:
        checkpoint_path = base / "checkpoints" / f"{resume}.json"
        if not checkpoint_path.exists():
            raise NotFoundError(f"checkpoint '{resume}' does not exist")
        checkpoint = read_json(checkpoint_path)
        config_from_checkpoint = normalize_config(checkpoint["config"])
        if normalized is not None:
            raise InvalidConfigError("provide either --config or --resume, not both")
        if steps is not None:
            if not isinstance(steps, int) or steps < 1:
                raise InvalidConfigError("steps override must be a positive integer")
            config_from_checkpoint["steps"] = steps
        normalized = config_from_checkpoint
        if normalized["steps"] < checkpoint["step"]:
            raise InvalidConfigError("target steps cannot be less than checkpoint step")
        rng = new_rng(normalized["seed"])
        rng.setstate(state_from_json(checkpoint["rng_state"]))
        prefix = list(checkpoint["metrics"])
        start_step = checkpoint["step"] + 1
        source_checkpoint = resume
    else:
        if normalized is None:
            raise InvalidConfigError("a config is required unless --resume is used")
        get_dataset(normalized["dataset"], root)
        if steps is not None:
            raise InvalidConfigError("--steps can only be used with --resume")
        rng = new_rng(normalized["seed"])
        prefix = []
        start_step = 1
        source_checkpoint = None

    run_id = _new_id("run")
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", normalized)
    metric_path = run_dir / "metrics.jsonl"
    _append_metrics(metric_path, prefix)
    continued = []
    for step in range(start_step, normalized["steps"] + 1):
        continued.append(next_metric(normalized["simulator"], step, rng))
    _append_metrics(metric_path, continued)
    meta = {
        "id": run_id,
        "name": normalized["name"],
        "dataset": normalized["dataset"],
        "profile": normalized["simulator"]["profile"],
        "seed": normalized["seed"],
        "steps": normalized["steps"],
        "resumed_from": source_checkpoint,
        "metric_log": f"runs/{run_id}/metrics.jsonl",
        "config_path": f"runs/{run_id}/config.json",
    }
    write_json(run_dir / "meta.json", meta)
    meta["metrics"] = prefix + continued
    return meta


def save_checkpoint(run_id: str, step: int | None = None, root: str | Path | None = None) -> dict[str, Any]:
    base = ensure_layout(root)
    run_dir = _run_dir(run_id, root)
    meta = read_json(run_dir / "meta.json")
    config = normalize_config(read_json(run_dir / "config.json"))
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    if not metrics:
        raise InvalidConfigError(f"run '{run_id}' has no metrics to checkpoint")
    selected_step = step if step is not None else metrics[-1]["step"]
    if selected_step < 1 or selected_step > metrics[-1]["step"]:
        raise InvalidConfigError(f"checkpoint step must be between 1 and {metrics[-1]['step']}")
    prefix = [row for row in metrics if row["step"] <= selected_step]
    if len(prefix) != selected_step:
        raise InvalidConfigError("run metric log does not contain a contiguous prefix through checkpoint step")
    rng = new_rng(config["seed"])
    for current_step in range(1, selected_step + 1):
        next_metric(config["simulator"], current_step, rng)
    checkpoint_id = _new_id("ckpt")
    checkpoint = {
        "id": checkpoint_id,
        "source_run": run_id,
        "config": config,
        "step": selected_step,
        "metrics": prefix,
        "rng_state": state_to_json(rng.getstate()),
    }
    write_json(base / "checkpoints" / f"{checkpoint_id}.json", checkpoint)
    return {key: value for key, value in checkpoint.items() if key != "rng_state"}


def list_checkpoints(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = ensure_layout(root)
    result = []
    for path in sorted((base / "checkpoints").glob("*.json")):
        item = read_json(path)
        result.append({key: value for key, value in item.items() if key not in ("rng_state", "metrics")})
    return result


def metric_log_bytes(run_id: str, root: str | Path | None = None) -> bytes:
    return (_run_dir(run_id, root) / "metrics.jsonl").read_bytes()
