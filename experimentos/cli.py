"""Command line interface for ExperimentOS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import compare_runs, write_report
from .datasets import list_datasets, register_dataset
from .errors import ExperimentOSError
from .findings import add_finding, list_findings
from .hypotheses import add_hypothesis, list_hypotheses
from .runs import list_checkpoints, list_runs, load_config, run_experiment, save_checkpoint

HONESTY_NOTICE = "Synthetic curve simulation only. No model training was performed."


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo", description="ExperimentOS local research control centre. Runs are seeded synthetic curve simulations, not model training.")
    parser.add_argument("--home", help="workspace directory (default: .experimentos or EXPERIMENTOS_HOME)")
    parser.add_argument("--version", action="version", version="exo 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    datasets = commands.add_parser("datasets", help="register and list synthetic CSV datasets")
    dataset_commands = datasets.add_subparsers(dest="datasets_command", required=True)
    register = dataset_commands.add_parser("register", help="generate and register a seeded synthetic CSV")
    register.add_argument("--name", required=True)
    register.add_argument("--seed", required=True, type=int)
    register.add_argument("--rows", type=int, default=100)
    register.add_argument("--features", type=int, default=3)
    dataset_commands.add_parser("list", help="list registered datasets")

    run = commands.add_parser("run", help="run or resume a deterministic synthetic curve")
    run_source = run.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--config", help="path to experiment config JSON")
    run_source.add_argument("--resume", metavar="CHECKPOINT_ID", help="resume from a saved checkpoint")
    run.add_argument("--steps", type=int, help="target total steps when resuming")

    checkpoints = commands.add_parser("checkpoints", help="save and list run checkpoints")
    checkpoint_commands = checkpoints.add_subparsers(dest="checkpoints_command", required=True)
    save = checkpoint_commands.add_parser("save", help="save a run snapshot at a selected step")
    save.add_argument("run_id")
    save.add_argument("--step", type=int, help="checkpoint after this one-based step; defaults to the last step")
    checkpoint_commands.add_parser("list", help="list saved checkpoints")

    hypothesis = commands.add_parser("hypothesize", help="record a prediction and evaluate it against a run")
    hypothesis.add_argument("--text", required=True)
    hypothesis.add_argument("--predicts", required=True, help="predicate such as 'loss <= 0.2'")
    hypothesis.add_argument("--run", required=True, dest="run_id", help="run ID that supplies observed evidence")

    findings = commands.add_parser("findings", help="record conclusions with run evidence")
    finding_commands = findings.add_subparsers(dest="findings_command", required=True)
    finding_add = finding_commands.add_parser("add", help="add a finding with one or more run citations")
    finding_add.add_argument("--text", required=True)
    finding_add.add_argument("--run", dest="run_ids", action="append", help="cited run ID, repeat for multiple citations")
    finding_commands.add_parser("list", help="list findings")

    compare = commands.add_parser("compare", help="compare stored metric logs side by side")
    compare.add_argument("run_ids", nargs="+", metavar="RUN_ID")

    report = commands.add_parser("report", help="write a Markdown or HTML research report")
    report.add_argument("--format", choices=("md", "html"), required=True)
    report.add_argument("--output", help="destination path, default is under the workspace")

    commands.add_parser("runs", help="list recorded runs")
    return parser


def _run(args: argparse.Namespace, root: str | None) -> int:
    config = load_config(args.config) if args.config else None
    record = run_experiment(config, root=root, resume=args.resume, steps=args.steps)
    print(f"{HONESTY_NOTICE} Run ID: {record['id']} ({len(record['metrics'])} steps).")
    print(f"Metrics: {record['metric_log']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.home
    try:
        if args.command == "datasets":
            if args.datasets_command == "register":
                item = register_dataset(args.name, args.seed, args.rows, args.features, root)
                print("Synthetic CSV dataset registered.")
                print(_json(item))
            else:
                entries = list_datasets(root)
                print(_json(entries))
        elif args.command == "run":
            return _run(args, root)
        elif args.command == "checkpoints":
            if args.checkpoints_command == "save":
                print(_json(save_checkpoint(args.run_id, args.step, root)))
            else:
                print(_json(list_checkpoints(root)))
        elif args.command == "hypothesize":
            print(_json(add_hypothesis(args.text, args.predicts, args.run_id, root)))
        elif args.command == "findings":
            if args.findings_command == "add":
                print(_json(add_finding(args.text, args.run_ids, root)))
            else:
                print(_json(list_findings(root)))
        elif args.command == "compare":
            print("Synthetic metric comparison. No model training was performed.")
            print(compare_runs(args.run_ids, root))
        elif args.command == "report":
            path = write_report(args.format, args.output, root)
            print(f"Report written to {path}")
            if args.format == "html":
                print(HONESTY_NOTICE)
        elif args.command == "runs":
            print(_json(list_runs(root)))
        return 0
    except ExperimentOSError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
