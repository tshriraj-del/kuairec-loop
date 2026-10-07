"""Permanent rows appear in every results table (ADR-0029); the ADR-0027 acceptance logic."""
import importlib.util
from pathlib import Path

import pytest

from kuairec_eval.data import REPO_ROOT
from kuairec_eval.env import EnvConfig
from policies.reference import PERMANENT_ROWS

PERMANENT_KEYS = {"rotation_random", "shortest_first", "longest_first", "rotation_shortest", "rotation_longest"}


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, REPO_ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_permanent_rows_are_the_four_duration_probes_and_their_comparator():
    assert {k for k, _, _ in PERMANENT_ROWS} == PERMANENT_KEYS


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "scripts").glob("*.py")), ids=lambda p: p.name)
def test_every_results_script_includes_the_permanent_rows(path: Path):
    src = path.read_text()
    if "harness.run(" in src:
        assert "PERMANENT_ROWS" in src, f"{path.name} produces a results table without the permanent rows"


def test_run_baselines_always_appends_the_permanent_rows():
    rb = _load("run_baselines")
    for chosen in (["random"], ["popularity"], ["random", "popularity"]):
        keys = [k for k, _ in rb.table_rows(chosen)]
        assert PERMANENT_KEYS <= set(keys) and keys[:len(chosen)] == chosen


def test_audit_rewards_tables_include_the_permanent_rows():
    ar = _load("audit_rewards")
    for reward in ("raw", "capped5", "duration_pct"):
        keys = {k for k, _, _ in ar.policy_specs(EnvConfig.from_toml(reward=reward))}
        assert PERMANENT_KEYS <= keys


def _fake_cmp(diffs: dict):
    def cmp(a, b, metric):
        mean, lo, hi = diffs[a]
        return {"a": a, "b": b, "metric": metric, "mean": mean, "ci": [lo, hi]}
    return cmp


def test_acceptance_needs_both_probe_intervals_inside_the_margin():
    ar = _load("audit_rewards")
    base = {"oracle_personalized": (10.0, 9.5, 10.5), "shortest_first": (0.1, -0.1, 0.3)}
    ok = ar.acceptance(_fake_cmp({**base, "rotation_shortest": (0.1, -0.3, 0.45), "rotation_longest": (0.2, 0.1, 0.49)}))
    assert ok["T1_duration_probes_within_tolerance"]["margin"] == pytest.approx(0.5)
    assert ok["T1_duration_probes_within_tolerance"]["pass"] and ok["both_pass"]
    # point estimate inside the margin but the interval crosses it: fails
    edge = ar.acceptance(_fake_cmp({**base, "rotation_shortest": (0.4, 0.2, 0.6), "rotation_longest": (0.0, -0.1, 0.1)}))
    assert not edge["T1_duration_probes_within_tolerance"]["pass"] and not edge["both_pass"]
    # two-sided: a large negative difference fails too
    neg = ar.acceptance(_fake_cmp({**base, "rotation_shortest": (-0.7, -0.9, -0.55), "rotation_longest": (0.0, -0.1, 0.1)}))
    assert not neg["T1_duration_probes_within_tolerance"]["pass"]
    # T2 must exclude zero
    no_gap = ar.acceptance(_fake_cmp({**base, "oracle_personalized": (0.2, -0.1, 0.5),
                                      "rotation_shortest": (0.0, -0.001, 0.001), "rotation_longest": (0.0, -0.001, 0.001)}))
    assert not no_gap["T2_personalized_beats_nonpersonalized"]["pass"] and not no_gap["both_pass"]
