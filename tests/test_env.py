"""Quit rule, quit profiles, action set, repeat ban, session lifecycle and reward fidelity (v1)."""
import numpy as np
import pytest

from kuairec_eval.data import load_small_matrix
from kuairec_eval.env import (EnvConfig, KuaiEnv, RepeatRecommendationError, SessionClosedError,
                              UnavailableActionError, boredom_quit)

CATS = {1: (0,), 2: (0,), 3: (1,), 4: (1, 2), 5: (2,), 6: (3,), 7: (0, 3), 8: (4,), 9: (5,)}


@pytest.fixture(scope="module")
def raw_env():
    return KuaiEnv(EnvConfig.from_toml(reward="raw"))


@pytest.fixture(scope="module")
def small():
    return load_small_matrix()


def test_no_quit_at_t0():
    assert not boredom_quit([], 1, CATS, 5, 0, "intended")


def test_quit_needs_more_than_threshold_items_sharing_a_category():
    assert not boredom_quit([1], 2, CATS, 5, 1, "intended")
    assert boredom_quit([1, 7], 2, CATS, 5, 1, "intended")
    assert boredom_quit([1], 2, CATS, 5, 0, "intended")


def test_window_only_looks_back_N_items():
    seq = [1, 7, 3, 5, 6]
    assert boredom_quit(seq, 2, CATS, 5, 1, "intended")
    assert not boredom_quit(seq, 2, CATS, 3, 1, "intended")


def test_easyrl4rec_slice_drops_early_items_for_t_between_N_over_2_and_N():
    seq = [1, 7, 3, 6]
    assert boredom_quit(seq, 2, CATS, 5, 1, "intended")
    assert not boredom_quit(seq, 2, CATS, 5, 1, "easyrl4rec_slice")
    seq5 = [1, 7, 3, 6, 8]
    assert boredom_quit(seq5, 2, CATS, 5, 1, "intended") == boredom_quit(seq5, 2, CATS, 5, 1, "easyrl4rec_slice")


def test_config_and_quit_profiles():
    c = EnvConfig.from_toml()
    assert (c.quit_profile, c.quit_window, c.quit_threshold, c.max_turn, c.window_mode, c.reward) == \
        ("primary", 9, 1, 30, "intended", "duration_pct")
    assert (EnvConfig.from_toml(quit_profile="looser").quit_threshold,
            EnvConfig.from_toml(quit_profile="tighter").quit_threshold) == (2, 0)
    with pytest.raises(ValueError):
        EnvConfig.from_toml(reward="watch_time")


def test_reset_reports_exactly_the_unrecorded_pairs(raw_env, small):
    recorded = small.groupby("user_id").video_id.apply(set)
    for user in raw_env.users[:50]:
        excluded = raw_env.reset(user)["excluded"]
        assert excluded == frozenset(raw_env.catalog) - recorded[user]


def test_unrecorded_pair_is_rejected_without_ending_the_session(raw_env):
    user = next(u for u in raw_env.users if raw_env.reset(u)["excluded"])
    excluded = raw_env.reset(user)["excluded"]
    with pytest.raises(UnavailableActionError):
        raw_env.step(next(iter(excluded)))
    ok = next(v for v in raw_env.catalog if v not in excluded)
    assert raw_env.step(ok).t == 0          # the rejected call did not consume a step


def test_repeat_is_forbidden_and_closed_session_rejects_steps(raw_env):
    user = raw_env.users[0]
    excluded = raw_env.reset(user)["excluded"]
    avail = [v for v in raw_env.catalog if v not in excluded]
    raw_env.step(avail[0])
    with pytest.raises(RepeatRecommendationError):
        raw_env.step(avail[0])
    raw_env.reset(user)
    raw_env.step(avail[0])
    for w in avail[1:]:
        if raw_env.step(w).done:
            break
    with pytest.raises(SessionClosedError):
        raw_env.step(avail[-1])


def test_raw_rewards_equal_recorded_watch_ratio_and_quit_step_is_paid(raw_env, small):
    rows = small.sample(200, random_state=0)
    for u, v, wr in zip(rows.user_id, rows.video_id, rows.watch_ratio):
        raw_env.reset(int(u))
        assert raw_env.step(int(v)).reward == wr
    lookup = dict(zip(zip(small.user_id, small.video_id), small.watch_ratio))
    user = raw_env.users[0]
    excluded = raw_env.reset(user)["excluded"]
    for v in raw_env.catalog:
        if v in excluded:
            continue
        r = raw_env.step(v)
        if r.done:
            assert r.quit_reason == "boredom" and r.reward == lookup[(user, v)]
            break


def test_reward_switch(small):
    from kuairec_eval import labels
    rows = small.sample(100, random_state=3)
    envs = {name: KuaiEnv(EnvConfig.from_toml(reward=name)) for name in ("capped5", "duration_pct")}
    k = EnvConfig.from_toml().duration_buckets
    pct = labels.duration_buckets(k).percentile(rows.video_id.to_numpy(), rows.play_duration.to_numpy())
    for (u, v, wr), p in zip(zip(rows.user_id, rows.video_id, rows.watch_ratio), pct):
        for name, e in envs.items():
            e.reset(int(u))
            got = e.step(int(v)).reward
            assert got == (min(wr, 5.0) if name == "capped5" else p)


def test_session_never_exceeds_max_turn(raw_env):
    cats = raw_env.categories
    user = raw_env.users[1]
    excluded = raw_env.reset(user)["excluded"]
    hist, n = [], 0
    rng = np.random.default_rng(0)
    for v in rng.permutation(np.asarray(raw_env.catalog)).tolist():
        recent = set().union(*hist[-raw_env.config.quit_window:]) if hist else set()
        if v in excluded or recent & set(cats[v]):
            continue
        r = raw_env.step(v)
        hist.append(set(cats[v]))
        n += 1
        if r.done:
            break
    assert n == raw_env.config.max_turn and r.quit_reason == "max_turn"
