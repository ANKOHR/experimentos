"""File-backed storage helpers for the local ExperimentOS workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def workspace(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("EXPERIMENTOS_HOME", ".experimentos")).resolve()


def ensure_layout(root: str | Path | None = None) -> Path:
    base = workspace(root)
    for folder in ("datasets", "runs", "checkpoints", "reports"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    for file_name in ("datasets.json", "hypotheses.jsonl", "findings.jsonl"):
        path = base / file_name
        if not path.exists():
            path.write_text("", encoding="utf-8") if file_name.endswith(".jsonl") else path.write_text("{}\n", encoding="utf-8")
    return base


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
