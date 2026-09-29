"""Comparison and report rendering computed from stored metric logs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .findings import list_findings
from .hypotheses import list_hypotheses
from .runs import get_run, list_runs
from .storage import ensure_layout, workspace


def best_metric(metrics: list[dict[str, Any]], metric: str = "loss") -> dict[str, Any]:
    if not metrics:
        raise ValueError("cannot select a best step from an empty metric log")
    return min(metrics, key=lambda item: (item[metric], item["step"]))


def compare_runs(run_ids: list[str], root: str | Path | None = None) -> str:
    if not run_ids:
        raise ValueError("at least one run ID is required")
    runs = [get_run(run_id, root) for run_id in run_ids]
    table = ["| Step | " + " | ".join(f"{run['id']} loss" for run in runs) + " |", "| ---: | " + " | ".join("---:" for _ in runs) + " |"]
    max_step = max((len(run["metrics"]) for run in runs), default=0)
    for step in range(1, max_step + 1):
        cells = []
        for run in runs:
            row = next((item for item in run["metrics"] if item["step"] == step), None)
            cells.append(f"{row['loss']:.8f}" if row else "")
        table.append(f"| {step} | " + " | ".join(cells) + " |")
    table.extend(["", "| Run | Best step | Best loss | Final loss |", "| --- | ---: | ---: | ---: |"])
    for run in runs:
        best = best_metric(run["metrics"])
        final = run["metrics"][-1]
        table.append(f"| {run['id']} | {best['step']} | {best['loss']:.8f} | {final['loss']:.8f} |")
    return "\n".join(table)


def render_report(format_name: str, root: str | Path | None = None) -> str:
    if format_name not in ("md", "html"):
        raise ValueError("format must be 'md' or 'html'")
    runs = list_runs(root)
    hypotheses = list_hypotheses(root)
    findings = list_findings(root)
    compare = compare_runs([run["id"] for run in runs], root) if runs else "No runs registered."
    scoreboard = ["| Hypothesis | Run | Predicted condition | Observed best value | Verdict |", "| --- | --- | --- | ---: | --- |"]
    for item in hypotheses:
        pred = item["prediction"]
        scoreboard.append(f"| {item['text']} | {item['run_id']} | {pred['metric']} {pred['operator']} {pred['threshold']:g} | {item['observed']['value']:.8f} | {item['verdict']} |")
    if len(scoreboard) == 2:
        scoreboard.append("| No hypotheses recorded |  |  |  |  |")
    finding_rows = [f"- {item['text']} (evidence: {', '.join(item['run_ids'])})" for item in findings]
    if not finding_rows:
        finding_rows = ["- No findings recorded."]

    if format_name == "md":
        return "\n".join([
            "# ExperimentOS report",
            "",
            "> Synthetic curve simulations only. These results are not from real model training.",
            "",
            f"Runs recorded: {len(runs)}",
            "",
            "## Run comparison",
            "",
            compare,
            "",
            "## Hypothesis scoreboard",
            "",
            *scoreboard,
            "",
            "## Findings",
            "",
            *finding_rows,
            "",
        ])

    def md_table_to_html(markdown: str) -> str:
        rows = [line.strip().strip("|").split("|") for line in markdown.splitlines() if line.startswith("|")]
        if not rows:
            return f"<p>{html.escape(markdown)}</p>"
        headers = rows[0]
        body = [row for row in rows[2:] if not all(set(cell.strip()) <= {"-", ":"} for cell in row)]
        th = "".join(f"<th>{html.escape(cell.strip())}</th>" for cell in headers)
        trs = "".join("<tr>" + "".join(f"<td>{html.escape(cell.strip())}</td>" for cell in row) + "</tr>" for row in body)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"

    compare_sections = compare.split("\n\n")
    compare_html = "".join(md_table_to_html(section) for section in compare_sections)
    hypotheses_html = md_table_to_html("\n".join(scoreboard))
    finding_html = "<ul>" + "".join(f"<li>{html.escape(item['text'])} (evidence: {html.escape(', '.join(item['run_ids']))})</li>" for item in findings) + ("" if findings else "<li>No findings recorded.</li>") + "</ul>"
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ExperimentOS report</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}table{border-collapse:collapse;margin:1rem 0 2rem;width:100%}th,td{border:1px solid #ccd3da;padding:.45rem;text-align:left}th{background:#f1f4f7}.banner{padding:1rem;background:#fff3cd;border:1px solid #e5c65b;font-weight:700}</style></head>
<body><h1>ExperimentOS report</h1><p class="banner">Synthetic curve simulations only. These results are not from real model training.</p>
<p>Runs recorded: RUN_COUNT</p><h2>Run comparison</h2>COMPARE_TABLES
<h2>Hypothesis scoreboard</h2>HYPOTHESIS_TABLE<h2>Findings</h2>FINDING_LIST</body></html>
""".replace("RUN_COUNT", str(len(runs))).replace("COMPARE_TABLES", compare_html).replace("HYPOTHESIS_TABLE", hypotheses_html).replace("FINDING_LIST", finding_html)


def write_report(format_name: str, output: str | Path | None = None, root: str | Path | None = None) -> Path:
    base = ensure_layout(root)
    destination = Path(output) if output else base / "reports" / f"report.{format_name}"
    if not destination.is_absolute():
        destination = workspace(root) / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(format_name, root), encoding="utf-8", newline="\n")
    return destination
