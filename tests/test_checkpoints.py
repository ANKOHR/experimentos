import json

from experimentos.runs import metric_log_bytes, run_experiment, save_checkpoint
from conftest import make_config


def test_checkpoint_resume_is_identical_to_uninterrupted_run(home):
    config = make_config(steps=24, seed=555, noise_std=0.03)
    uninterrupted = run_experiment(config, root=home)
    partial = run_experiment(config, root=home)
    checkpoint = save_checkpoint(partial["id"], step=9, root=home)
    checkpoint_data = json.loads((home / "checkpoints" / f"{checkpoint['id']}.json").read_text(encoding="utf-8"))
    assert checkpoint_data["step"] == 9
    assert checkpoint_data["rng_state"]
    resumed = run_experiment(root=home, resume=checkpoint["id"])
    assert metric_log_bytes(uninterrupted["id"], home) == metric_log_bytes(resumed["id"], home)


def test_resume_can_extend_saved_total_steps(home):
    config = make_config(steps=12, seed=8, noise_std=0.02)
    source = run_experiment(config, root=home)
    checkpoint = save_checkpoint(source["id"], step=5, root=home)
    extended = run_experiment(root=home, resume=checkpoint["id"], steps=17)
    assert len(extended["metrics"]) == 17
    assert extended["metrics"][-1]["step"] == 17


def test_checkpoint_default_uses_last_step(home, make_run):
    run = make_run(make_config(steps=4))
    checkpoint = save_checkpoint(run["id"], root=home)
    assert checkpoint["step"] == 4
