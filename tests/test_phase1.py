"""Phase 1: slate rule fidelity, scorer correctness on toy data, feature sets, grid and ledger."""
import importlib.util
import json

import numpy as np
import pytest

from kuairec_eval import harness
from kuairec_eval.data import REPO_ROOT
from kuairec_eval.env import EnvConfig, boredom_quit
from policies.phase1 import scorers as S
from policies.phase1.features import DURATION_COLUMNS, VARIANTS, columns
from policies.phase1.pairs import History, Pairs
from policies.phase1.slate import RuleAwareSlate, would_quit


def _load_script(name):
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_slate_rule_copy_matches_the_frozen_rule(env):
    cfg = EnvConfig.from_toml()
    rng = np.random.default_rng(0)
    cats = env.categories
    for _ in range(3000):
        seq = [int(v) for v in rng.choice(env.catalog, size=rng.integers(0, 15), replace=False)]
        v = int(rng.choice(env.catalog))
        assert would_quit(seq, v, cats, cfg.quit_window, cfg.quit_threshold) == \
            boredom_quit(seq, v, cats, cfg.quit_window, cfg.quit_threshold, "intended")


def test_slate_never_triggers_the_primary_rule(env):
    n = len(env.catalog)
    factory = lambda c, s: RuleAwareSlate(c, s, lambda u: np.random.default_rng(u).random(n), 9, 1)  # noqa: E731
    res = harness.run(factory, "search", env=env)
    assert not res["stochastic"]
    assert res["summary"]["boredom_quit_rate"]["mean"] == 0.0
    assert res["summary"]["session_length"]["mean"] == env.config.max_turn


def test_slate_shows_the_best_safe_video_first(env):
    scores = np.zeros(len(env.catalog))
    scores[5] = 10.0
    p = RuleAwareSlate(harness.context_for(env), 0, lambda u: scores, 9, 1)
    p.begin_session(env.users[0], frozenset())
    assert p.act() == env.catalog[5]
    p.begin_session(env.users[0], frozenset({env.catalog[5]}))
    assert p.act() != env.catalog[5]


def _toy_pairs():
    u = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3], np.int32)
    v = np.array([1, 2, 3, 1, 2, 2, 3, 4, 4], np.int32)
    y = np.array([0.9, 0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.5, 0.4], np.float32)
    return Pairs(u, v, np.zeros(len(u)), y, np.ones(len(u), np.int32))


def test_item_knn_matches_a_brute_force_computation():
    p = _toy_pairs()
    cat = np.array([1, 2, 3, 4])
    knn = S.ItemKNN("centered", k=2, shrink=0.0).fit(p, None, cat)
    x = np.zeros((4, S.N_VIDEOS))
    x[p.user, p.video] = 1
    n = x.sum(0)
    co = x.T @ x
    hv, hl = np.array([3, 4]), np.array([0.9, 0.1], np.float32)
    for i, item in enumerate(cat):
        s = np.array([co[item, j] / np.sqrt(n[item] * n[j]) if item != j else 0.0 for j in hv])
        expected = (s * (hl - 0.5)).sum() / (np.abs(s).sum() + 1e-6)
        assert knn.score(hv, hl)[i] == pytest.approx(expected, abs=1e-5)


def test_affinity_shrinks_toward_the_global_category_mean():
    p = _toy_pairs()
    m = np.zeros((S.N_VIDEOS, 2), np.float32)
    m[[1, 2], 0] = 1
    m[[3, 4], 1] = 1
    a = S.Affinity(shrinkage=5.0).fit(p, m, np.array([1, 2, 3, 4]))
    aff_empty, _ = a.user_affinity(np.array([], int), np.array([], np.float32))
    assert np.allclose(aff_empty, a.global_mean)
    aff_big, _ = S.Affinity(shrinkage=1e-9).fit(p, m, np.array([1, 2, 3, 4])).user_affinity(np.array([1, 2]), np.array([1.0, 0.0]))
    assert aff_big[0] == pytest.approx(0.5)


def test_history_store_returns_each_users_pairs():
    h = History(_toy_pairs())
    v, y = h.get(2)
    assert v.tolist() == [2, 3, 4] and np.allclose(y, [0.3, 0.7, 0.5])
    assert len(h.get(99)[0]) == 0


def test_no_duration_variant_has_no_duration_derived_column():
    import pandas as pd
    vt = pd.DataFrame(columns=["perday_show_cnt", "rate_like_cnt", *DURATION_COLUMNS])
    ut = pd.DataFrame(columns=["u_register_days"])
    cols = columns("no_duration", vt, ut, None)
    assert not set(cols) & set(DURATION_COLUMNS)
    banned = ("video_duration", "play_duration", "complete_play", "valid_play", "long_time_play",
              "short_time_play", "play_progress", "duration")
    assert not [c for c in cols if any(b in c for b in banned)]
    assert set(columns("no_duration", vt, ut, None)) < set(columns("duration", vt, ut, None))
    assert VARIANTS == ("no_duration", "duration", "text")


def test_grid_refuses_undeclared_and_repeated_shadow_runs(tmp_path, monkeypatch):
    c = _load_script("phase1_common")
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"runs": [{"name": "x", "partition": "shadow", "config_hash": "h"}]}))
    monkeypatch.setattr(c, "GRID", grid)
    monkeypatch.setattr(c, "LEDGER", tmp_path / "ledger.jsonl")
    with pytest.raises(PermissionError, match="not in the declared grid"):
        c.run(None, "y", None, "search", "h")
    with pytest.raises(PermissionError, match="differs"):
        c.run(None, "x", None, "shadow", "other")
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"policy": "x", "partition": "shadow"}) + "\n")
    with pytest.raises(PermissionError, match="used once"):
        c.run(None, "x", None, "shadow", "h")
    grid.write_text(json.dumps({"runs": [{"name": "a", "partition": "shadow", "config_hash": "h", "condition": "best"},
                                         {"name": "b", "partition": "shadow", "config_hash": "h", "condition": "best"}]}))
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"policy": "a", "partition": "shadow"}) + "\n")
    with pytest.raises(PermissionError, match="exactly one scorer"):
        c.run(None, "b", None, "shadow", "h")


def test_ratio_bootstrap():
    c = _load_script("phase1_common")
    rng = np.random.default_rng(1)
    a, b = rng.normal(2, 1, 500), rng.normal(1, 1, 500)
    p, q = rng.normal(5, 1, 500), rng.normal(1, 1, 500)
    r = c.ratio_bootstrap(a, b, p, q)
    assert r["point"] == pytest.approx((a - b).mean() / (p - q).mean())
    assert r["ci"][0] < r["point"] < r["ci"][1]
