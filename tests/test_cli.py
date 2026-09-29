import json

import pytest

from experimentos.runs import run_experiment, save_checkpoint
from conftest import cli, make_config


def _config_file(tmp_path, config=None):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config or make_config()), encoding="utf-8")
    return path


@pytest.mark.parametrize("command", ["datasets register", "datasets list"])
def test_cli_dataset_subcommands_smoke(home, command):
    args = command.split()
    if args[-1] == "register":
        args += ["--name", "cli-toy", "--seed", "12", "--rows", "4", "--features", "1"]
    result = cli(home, *args)
    assert result.returncode == 0, result.stderr
    assert "synthetic" in result.stdout.lower() or "toy" in result.stdout.lower()


def test_cli_run_smoke(home, tmp_path):
    result = cli(home, "run", "--config", str(_config_file(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert "No model training was performed" in result.stdout


@pytest.mark.parametrize("command", ["checkpoints save", "checkpoints list"])
def test_cli_checkpoint_subcommands_smoke(home, command):
    run = run_experiment(make_config(steps=6), root=home)
    args = command.split()
    if args[-1] == "save":
        args += [run["id"], "--step", "3"]
    result = cli(home, *args)
    assert result.returncode == 0, result.stderr
    assert "ckpt-" in result.stdout or result.stdout.strip() == "[]"


def test_cli_resume_smoke(home):
    run = run_experiment(make_config(steps=6, noise_std=0.02), root=home)
    checkpoint = save_checkpoint(run["id"], step=3, root=home)
    result = cli(home, "run", "--resume", checkpoint["id"])
    assert result.returncode == 0, result.stderr
    assert "No model training was performed" in result.stdout


def test_cli_hypothesize_smoke(home):
    run = run_experiment(make_config(), root=home)
    result = cli(home, "hypothesize", "--text", "loss is small", "--predicts", "loss <= 2", "--run", run["id"])
    assert result.returncode == 0, result.stderr
    assert '"verdict": "match"' in result.stdout


@pytest.mark.parametrize("command", ["findings add", "findings list"])
def test_cli_findings_subcommands_smoke(home, command):
    run = run_experiment(make_config(), root=home)
    args = command.split()
    if args[-1] == "add":
        args += ["--text", "CLI finding", "--run", run["id"]]
    result = cli(home, *args)
    assert result.returncode == 0, result.stderr
    assert "CLI finding" in result.stdout or result.stdout.strip() == "[]"


def test_cli_uncited_finding_has_named_error(home):
    result = cli(home, "findings", "add", "--text", "No evidence")
    assert result.returncode == 2
    assert "UncitedFindingError" in result.stderr


def test_cli_compare_smoke(home):
    one = run_experiment(make_config(steps=3), root=home)
    two = run_experiment(make_config(steps=4, profile="linear"), root=home)
    result = cli(home, "compare", one["id"], two["id"])
    assert result.returncode == 0, result.stderr
    assert "| Step |" in result.stdout
    assert one["id"] in result.stdout and two["id"] in result.stdout


@pytest.mark.parametrize("format_name", ["md", "html"])
def test_cli_report_formats_smoke(home, format_name):
    result = cli(home, "report", "--format", format_name)
    assert result.returncode == 0, result.stderr
    assert "Report written to" in result.stdout
    report = home / "reports" / f"report.{format_name}"
    assert report.exists()
    if format_name == "html":
        assert "Synthetic curve simulations only" in report.read_text(encoding="utf-8")


def test_cli_runs_listing_smoke(home):
    run = run_experiment(make_config(), root=home)
    result = cli(home, "runs")
    assert result.returncode == 0, result.stderr
    assert run["id"] in result.stdout


def test_cli_end_to_end_register_run_checkpoint_resume_hypothesis_findings_report(tmp_path):
    reg = cli(tmp_path, "datasets", "register", "--name", "e2e", "--seed", "31", "--rows", "10", "--features", "2")
    assert reg.returncode == 0, reg.stderr
    config = make_config(dataset="e2e", steps=10, seed=9, profile="plateau_then_drop", plateau_steps=3, noise_std=0.02)
    config_path = _config_file(tmp_path, config)
    original = cli(tmp_path, "run", "--config", str(config_path))
    assert original.returncode == 0, original.stderr
    original_id = original.stdout.split("Run ID: ", 1)[1].split(" ", 1)[0]
    saved = cli(tmp_path, "checkpoints", "save", original_id, "--step", "4")
    assert saved.returncode == 0, saved.stderr
    checkpoint_id = json.loads(saved.stdout)["id"]
    resumed = cli(tmp_path, "run", "--resume", checkpoint_id)
    assert resumed.returncode == 0, resumed.stderr
    resumed_id = resumed.stdout.split("Run ID: ", 1)[1].split(" ", 1)[0]
    hypothesis = cli(tmp_path, "hypothesize", "--text", "loss reaches a low value", "--predicts", "loss <= 1", "--run", resumed_id)
    assert hypothesis.returncode == 0, hypothesis.stderr
    finding = cli(tmp_path, "findings", "add", "--text", "The seeded curve was reproduced.", "--run", resumed_id)
    assert finding.returncode == 0, finding.stderr
    report = cli(tmp_path, "report", "--format", "html")
    assert report.returncode == 0, report.stderr
    html = (tmp_path / "reports" / "report.html").read_text(encoding="utf-8")
    assert "Synthetic curve simulations only. These results are not from real model training." in html
    assert "The seeded curve was reproduced." in html
    assert "loss reaches a low value" in html
