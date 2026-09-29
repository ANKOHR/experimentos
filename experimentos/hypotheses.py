"""Hypothesis records with explicit prediction evaluation against run metrics."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from .errors import InvalidPredictionError
from .runs import get_run
from .storage import append_jsonl, ensure_layout, read_jsonl

_PREDICTION = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")


def _evaluate(value: float, operator: str, threshold: float) -> bool:
    return {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }[operator]


def add_hypothesis(text: str, predicts: str, run_id: str, root: str | Path | None = None) -> dict[str, Any]:
    match = _PREDICTION.match(predicts)
    if not match:
        raise InvalidPredictionError("prediction must use the form 'metric <= number', such as 'loss <= 0.2'")
    metric, operator, raw_threshold = match.groups()
    run = get_run(run_id, root)
    if not run["metrics"]:
        raise InvalidPredictionError(f"run '{run_id}' has no metric observations")
    observed_metrics = [row[metric] for row in run["metrics"] if metric in row]
    if not observed_metrics:
        raise InvalidPredictionError(f"metric '{metric}' is not present in run '{run_id}'")
    threshold = float(raw_threshold)
    best_observed = min(observed_metrics)
    observed = _evaluate(best_observed, operator, threshold)
    record = {
        "id": f"hyp-{uuid.uuid4().hex[:12]}",
        "text": text,
        "prediction": {"metric": metric, "operator": operator, "threshold": threshold, "expected": True},
        "run_id": run_id,
        "observed": {"metric": metric, "value": best_observed, "predicate_holds": observed},
        "verdict": "match" if observed else "mismatch",
        "evaluation_rule": "Evaluate the predicate against the minimum observed value for the chosen metric. A hypothesis predicts the predicate will hold, so it matches exactly when the predicate is true.",
    }
    append_jsonl(ensure_layout(root) / "hypotheses.jsonl", record)
    return record


def list_hypotheses(root: str | Path | None = None) -> list[dict[str, Any]]:
    return read_jsonl(ensure_layout(root) / "hypotheses.jsonl")
