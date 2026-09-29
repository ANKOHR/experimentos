"""Synthetic CSV dataset registry with deterministic generation and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .errors import InvalidConfigError, NotFoundError
from .storage import ensure_layout, read_json, write_json, workspace


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-_").lower()
    if not slug:
        raise InvalidConfigError("dataset name must contain letters or numbers")
    return slug


def _dataset_csv(seed: int, rows: int, features: int) -> bytes:
    rng = random.Random(seed)
    output: list[str] = []
    header = ["sample", *[f"feature_{index}" for index in range(1, features + 1)], "target"]
    output.append(",".join(header))
    for sample in range(rows):
        values = [sample]
        feature_values = []
        for column in range(features):
            value = round(rng.uniform(-1.0, 1.0), 8)
            feature_values.append(value)
            values.append(f"{value:.8f}")
        target = sum((index + 1) * value for index, value in enumerate(feature_values))
        target = round(target + rng.uniform(-0.05, 0.05), 8)
        values.append(f"{target:.8f}")
        output.append(",".join(str(value) for value in values))
    return ("\n".join(output) + "\n").encode("utf-8")


def register_dataset(name: str, seed: int, rows: int = 100, features: int = 3, root: str | Path | None = None) -> dict[str, Any]:
    if rows < 1 or features < 1:
        raise InvalidConfigError("rows and features must both be positive integers")
    base = ensure_layout(root)
    slug = _slug(name)
    rel_path = f"datasets/{slug}.csv"
    file_path = base / rel_path
    csv_bytes = _dataset_csv(seed, rows, features)
    file_path.write_bytes(csv_bytes)
    schema = ["sample", *[f"feature_{index}" for index in range(1, features + 1)], "target"]
    entry = {
        "id": slug,
        "name": name,
        "path": rel_path,
        "format": "csv",
        "schema": schema,
        "row_count": rows,
        "generation": {
            "method": "uniform feature values in [-1, 1] with weighted-sum target and uniform additive noise",
            "seed": seed,
            "features": features,
            "target_noise_range": [-0.05, 0.05],
            "feature_range": [-1.0, 1.0],
            "random_generator": "Python random.Random (MT19937)",
        },
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    registry_path = base / "datasets.json"
    registry = read_json(registry_path) if registry_path.exists() else {}
    registry[slug] = entry
    write_json(registry_path, registry)
    return entry


def list_datasets(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = ensure_layout(root)
    return sorted(read_json(base / "datasets.json").values(), key=lambda item: item["id"])


def get_dataset(dataset_id: str, root: str | Path | None = None) -> dict[str, Any]:
    registry = read_json(ensure_layout(root) / "datasets.json")
    if dataset_id not in registry:
        raise NotFoundError(f"dataset '{dataset_id}' is not registered")
    return registry[dataset_id]
