"""Findings with mandatory evidence pointers to existing runs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .errors import UncitedFindingError
from .runs import get_run
from .storage import append_jsonl, ensure_layout, read_jsonl


def add_finding(text: str, run_ids: list[str] | None, root: str | Path | None = None) -> dict[str, Any]:
    citations = [run_id for run_id in (run_ids or []) if run_id]
    if not citations:
        raise UncitedFindingError("findings require at least one --run citation")
    for run_id in citations:
        get_run(run_id, root)
    record = {"id": f"finding-{uuid.uuid4().hex[:12]}", "text": text, "run_ids": citations}
    append_jsonl(ensure_layout(root) / "findings.jsonl", record)
    return record


def list_findings(root: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(ensure_layout(root) / "findings.jsonl")
