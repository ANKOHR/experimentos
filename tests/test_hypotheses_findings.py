import pytest

from experimentos.errors import InvalidPredictionError, UncitedFindingError
from experimentos.findings import add_finding, list_findings
from experimentos.hypotheses import add_hypothesis
from conftest import make_config


def test_hypothesis_match_and_mismatch_follow_predicate_rule(home, make_run):
    run = make_run(make_config(steps=5, profile="linear", initial_loss=1, slope=-0.1))
    match = add_hypothesis("loss should be below 0.7", "loss <= 0.7", run["id"], home)
    mismatch = add_hypothesis("loss should be below 0.5", "loss <= 0.5", run["id"], home)
    assert match["observed"]["value"] == pytest.approx(0.6)
    assert match["verdict"] == "match"
    assert mismatch["verdict"] == "mismatch"


def test_hypothesis_requires_supported_predicate(home, make_run):
    run = make_run()
    with pytest.raises(InvalidPredictionError, match="prediction must use"):
        add_hypothesis("bad syntax", "loss around 0.2", run["id"], home)


def test_uncited_finding_is_rejected_with_named_error(home):
    with pytest.raises(UncitedFindingError, match="at least one --run citation"):
        add_finding("uncited conclusion", [], home)


def test_cited_finding_links_to_existing_run(home, make_run):
    run = make_run()
    item = add_finding("The curve decreased.", [run["id"]], home)
    assert item["run_ids"] == [run["id"]]
    assert list_findings(home)[0]["run_ids"] == [run["id"]]
