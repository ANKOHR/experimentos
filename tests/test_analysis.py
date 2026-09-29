from experimentos.analysis import best_metric, compare_runs, render_report
from experimentos.findings import add_finding
from experimentos.hypotheses import add_hypothesis
from conftest import make_config


def test_comparison_best_step_matches_hand_computed_fixture(home, make_run):
    run = make_run(make_config(steps=5, profile="linear", initial_loss=1.0, slope=-0.2))
    assert best_metric(run["metrics"]) == {"step": 5, "loss": 0.19999999999999996}
    table = compare_runs([run["id"]], home)
    assert f"| {run['id']} | 5 | 0.20000000 | 0.20000000 |" in table
    assert "| Step |" in table


def test_report_contains_hypothesis_scoreboard_and_cited_finding(home, make_run):
    run = make_run()
    add_hypothesis("loss stays under 2", "loss <= 2", run["id"], home)
    add_finding("Synthetic curve is decreasing.", [run["id"]], home)
    markdown = render_report("md", home)
    html = render_report("html", home)
    assert "Hypothesis scoreboard" in markdown
    assert "Synthetic curve is decreasing." in markdown
    assert "Synthetic curve simulations only. These results are not from real model training." in html
    assert "loss stays under 2" in html
