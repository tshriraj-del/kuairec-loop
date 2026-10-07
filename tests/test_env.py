"""Quit rule, repeat ban, session lifecycle and reward fidelity."""
import numpy as np
import pytest

from kuairec_eval.data import load_small_matrix
from kuairec_eval.env import (EnvConfig, RepeatRecommendationError, SessionClosedError,
                              boredom_quit)

CATS = {1: (0,), 2: (0,), 3: (1,), 4: (1, 2), 5: (2,), 6: (3,), 7: (0, 3), 8: (4,), 9: (5,)}


def test_no_quit_at_t0():
    assert not boredom_quit([], 1, CATS, 5, 0, "intended")


def test_quit_needs_more_than_threshold_items_sharing_a_category():
    # category 0 appears in one window item -> count 1, not > 1
    assert not boredom_quit([1], 2, CATS, 5, 1, "intended")
    # two window items carry category 0 -> count 2 > 1
    assert boredom_quit([1, 7], 2, CATS, 5, 1, "intended")
    # threshold 0: any shared category quits
    assert boredom_quit([1], 2, CATS, 5, 0, "intended")


def test_window_only_looks_back_N_items():
    seq = [1, 7, 3, 5, 6]          # category 0 in items at t=0,1
    assert boredom_quit(seq, 2, CATS, 5, 1, "intended")
    assert not boredom_quit(seq, 2, CATS, 3, 1, "intended")  # window = [3,5,6]


def test_easyrl4rec_slice_drops_early_items_for_t_between_N_over_2_and_N():
    # N=5, t=4: seq[-1:4] = [seq[3]] in EasyRL4Rec, versus seq[0:4] intended
    seq = [1, 7, 3, 6]
    assert boredom_quit(seq, 2, CATS, 5, 1, "intended")
    assert not boredom_quit(seq, 2, CATS, 5, 1, "easyrl4rec_slice")
    # t = N: both windows are seq[0:N]
    seq5 = [1, 7, 3, 6, 8]
    assert boredom_quit(seq5, 2, CATS, 5, 1, "intended") == boredom_quit(seq5, 2, CATS, 5, 1, "easyrl4rec_slice")


def test_config_is_read_from_toml():
    c = EnvConfig.from_toml()
    assert (c.quit_window, c.quit_threshold, c.max_turn, c.window_mode) == (9, 1, 30, "intended")


def test_repeat_is_forbidden_and_closed_session_rejects_steps(env):
    user = env.users[0]
    env.reset(user)
    v = env.catalog[0]
    env.step(v)
    with pytest.raises(RepeatRecommendationError):
        env.step(v)
    env.reset(user)
    env.step(v)                     # a new session may show it again
    for w in env.catalog[1:]:
        if env.step(w).done:
            break
    with pytest.raises(SessionClosedError):
        env.step(env.catalog[-1])


def test_rewards_equal_recorded_watch_ratio_and_quit_step_is_paid(env):
    small = load_small_matrix()
    rows = small.sample(200, random_state=0)
    for u, v, wr in zip(rows.user_id, rows.video_id, rows.watch_ratio):
        env.reset(int(u))
        r = env.step(int(v))
        assert r.reward == wr and r.observed
    # find a session that quits by boredom and check its last reward is still the recorded value
    lookup = dict(zip(zip(small.user_id, small.video_id), small.watch_ratio))
    user = env.users[0]
    env.reset(user)
    for v in env.catalog:
        r = env.step(v)
        if r.done:
            assert r.quit_reason == "boredom"
            assert r.reward == lookup.get((user, v), 0.0)
            break


def test_unrecorded_pair_returns_configured_reward(env):
    small = load_small_matrix()
    seen = set(zip(small.user_id.tolist(), small.video_id.tolist()))
    user = env.users[0]
    missing = next(v for v in env.catalog if (user, v) not in seen)
    env.reset(user)
    r = env.step(missing)
    assert not r.observed and r.reward == env.config.missing_pair_reward


def test_session_never_exceeds_max_turn(env):
    cats = env.categories
    user = env.users[1]
    env.reset(user)
    hist, n = [], 0
    rng = np.random.default_rng(0)
    for v in rng.permutation(np.asarray(env.catalog)).tolist():
        recent = set().union(*hist[-env.config.quit_window:]) if hist else set()
        if recent & set(cats[v]):
            continue
        r = env.step(v)
        hist.append(set(cats[v]))
        n += 1
        if r.done:
            break
    assert n == env.config.max_turn and r.quit_reason == "max_turn"
