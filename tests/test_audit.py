"""Audit components: runner == frozen harness, reward overlay, labels, reference policies, oracles."""
import numpy as np
import pandas as pd
import pytest

from audit.reward_env import RewardEnv
from audit.runner import compare, paired_bootstrap, run_policy
from kuairec_eval import harness, labels, partitions
from kuairec_eval.data import load_small_matrix
from kuairec_eval.diagnostics import oracles
from kuairec_eval.rewards import RewardConfig, reward_matrix
from policies.baselines import UniformRandom
from policies.reference import BigMeanLabel, CategoryRotation, ShortestFirst

K = RewardConfig.from_toml().duration_buckets


@pytest.fixture(scope="module")
def small():
    return load_small_matrix()


@pytest.fixture(scope="module")
def envs(env):
    return {name: RewardEnv(env, RewardConfig.from_toml(active=name)) for name in ("raw", "capped5", "duration_pct")}


def test_runner_summary_equals_frozen_harness(env, envs):
    ref = harness.run(UniformRandom, "search", seeds=[0, 1], env=env)
    for e in (env, envs["raw"]):
        run = run_policy(e, UniformRandom, "search", [0, 1], "random", True)
        assert run.summary == ref["summary"] and run.per_seed == ref["per_seed"]


def test_runner_per_user_values_add_up(env):
    run = run_policy(env, UniformRandom, "search", [0, 1], "random", True)
    assert np.isclose(run.cum_reward.mean(), run.summary["cum_reward"]["mean"])
    assert np.isclose(run.session_length.mean(), run.summary["session_length"]["mean"])


def test_reward_candidates_match_their_definitions(envs, small):
    rows = small.sample(300, random_state=1)
    pct = labels.duration_buckets(K).percentile(rows.video_id.to_numpy(), rows.play_duration.to_numpy())
    for (u, v, wr), p in zip(zip(rows.user_id, rows.video_id, rows.watch_ratio), pct):
        got = {}
        for name, e in envs.items():
            e.reset(int(u))
            got[name] = e.step(int(v)).reward
        assert got["raw"] == wr
        assert got["capped5"] == min(wr, 5.0)
        assert got["duration_pct"] == p and 0.0 <= p <= 1.0


def test_unrecorded_pairs_keep_v0_semantics(envs, small):
    seen = set(zip(small.user_id.tolist(), small.video_id.tolist()))
    e = envs["duration_pct"]
    user = e.users[0]
    missing = next(v for v in e.catalog if (user, v) not in seen)
    e.reset(user)
    r = e.step(missing)
    assert not r.observed and r.reward == 0.0


def test_reward_env_holds_no_arrays(envs):
    for e in envs.values():
        for value in vars(e).values():
            assert not isinstance(value, (np.ndarray, pd.DataFrame, pd.Series))


def test_duration_percentile_is_midrank_and_centred():
    b = labels.duration_buckets(K)
    for ref in b.reference:
        sample = ref[:: max(1, len(ref) // 5000)]
        lo = np.searchsorted(ref, sample, side="left")
        hi = np.searchsorted(ref, sample, side="right")
        mid = (lo + hi) / (2 * len(ref))
        assert mid.min() >= 0 and mid.max() <= 1
    # mean mid-rank percentile over a full reference set is exactly 0.5
    ref = b.reference[0]
    lo = np.searchsorted(ref, ref, side="left")
    hi = np.searchsorted(ref, ref, side="right")
    assert abs(((lo + hi) / (2 * len(ref))).mean() - 0.5) < 1e-9


@pytest.mark.parametrize("pick", ["random", "shortest"])
def test_rotation_never_triggers_the_quit_rule(env, pick):
    run = run_policy(env, lambda c, s: CategoryRotation(c, s, pick), "search", [0], pick, pick == "random")
    assert run.summary["boredom_quit_rate"]["mean"] == 0.0
    assert run.summary["session_length"]["mean"] == env.config.max_turn


def test_rotation_reads_no_quit_parameters():
    import inspect
    from policies import reference
    src = inspect.getsource(reference)
    for name in ("quit_window", "quit_threshold", "max_turn", "evaluator.toml"):
        assert name not in src


def test_reference_policies_are_deterministic_where_declared(env):
    # Compare per-user values: pooled per-seed means differ in the last float bit because the
    # harness visits users in a seed-dependent order and float summation is order-dependent.
    for factory in (ShortestFirst, lambda c, s: CategoryRotation(c, s, "shortest"), BigMeanLabel):
        a = run_policy(env, factory, "search", [0], "x", False)
        b = run_policy(env, factory, "search", [5], "x", False)
        assert np.array_equal(a.cum_reward, b.cum_reward)
        assert np.array_equal(a.reward_sum, b.reward_sum) and np.array_equal(a.reward_n, b.reward_n)


def test_oracles_never_trigger_and_personalized_first_pick_is_argmax(envs):
    e = envs["capped5"]
    cfg = RewardConfig.from_toml(active="capped5")
    mat = reward_matrix(cfg, e.users, e.catalog)
    users, videos = np.asarray(e.users), np.asarray(e.catalog)
    search = partitions.load("search")
    p_fac = oracles.personalized_factory(mat, users, videos, e.config)
    np_fac = oracles.nonpersonalized_factory(mat, users, videos, e.config, search, e.categories)
    for fac in (p_fac, np_fac):
        run = run_policy(e, fac, "search", [0], "oracle", False)
        assert run.summary["boredom_quit_rate"]["mean"] == 0.0
        assert run.summary["session_length"]["mean"] == e.config.max_turn
    ctx = harness.PolicyContext(catalog=e.catalog, categories=e.categories)
    user = search[0]
    o = p_fac(ctx, 0)
    o.begin_session(user)
    row = np.where(np.isnan(mat[list(users).index(user)]), 0.0, mat[list(users).index(user)])
    assert row[list(videos).index(o.act())] == row.max()


def test_paired_bootstrap_basics():
    a = np.arange(100, dtype=float)
    same = paired_bootstrap(a, a)
    assert same["mean"] == 0 and same["ci"] == [0.0, 0.0]
    shifted = paired_bootstrap(a + 1, a)
    assert shifted["mean"] == 1 and shifted["ci"] == [1.0, 1.0]
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=500), rng.normal(size=500)
    r1, r2 = paired_bootstrap(x, y), paired_bootstrap(x, y)
    assert r1 == r2 and r1["ci"][0] < r1["mean"] < r1["ci"][1]


def test_compare_requires_same_users(env):
    a = run_policy(env, UniformRandom, "search", [0], "a", True)
    b = run_policy(env, UniformRandom, "shadow", [0], "b", True)
    with pytest.raises(ValueError):
        compare(a, b, "cum_reward")
