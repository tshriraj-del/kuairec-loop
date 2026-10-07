"""v1 harness protocol, paired comparisons, reference policies and oracle ceilings."""
import numpy as np
import pytest

from kuairec_eval import harness, labels, partitions
from kuairec_eval.diagnostics import oracles
from kuairec_eval.env import EnvConfig, KuaiEnv
from kuairec_eval.rewards import reward_matrix
from policies.baselines import GlobalPopularity, UniformRandom
from policies.reference import BigMeanLabel, CategoryRotation, ShortestFirst


def test_harness_refuses_the_easyrl4rec_slice():
    slice_env = KuaiEnv(EnvConfig.from_toml(window_mode="easyrl4rec_slice"))
    with pytest.raises(ValueError, match="intended"):
        harness.run(UniformRandom, "search", seeds=[0], env=slice_env)


def test_policies_must_declare_stochastic(env):
    class Undeclared(UniformRandom):
        stochastic = None
    with pytest.raises(TypeError):
        harness.run(Undeclared, "search", seeds=[0], env=env)


def test_deterministic_declaration_is_checked(env):
    class LearnsAcrossUsers:
        """No internal randomness, but carries state across users, so user order matters."""
        stochastic = False

        def __init__(self, context, seed):
            self._catalog, self._offset = list(context.catalog), 0

        def begin_session(self, user, excluded):
            self._queue = [v for v in self._catalog[self._offset:] + self._catalog[:self._offset] if v not in excluded]
            self._offset = (self._offset + 7) % len(self._catalog)

        def act(self):
            return self._queue.pop(0)

        def observe(self, video, result):
            pass
    with pytest.raises(ValueError, match="declare stochastic=True"):
        harness.run(LearnsAcrossUsers, "search", env=env)


def test_seed_intervals_only_for_stochastic_policies(env):
    r = harness.run(UniformRandom, "search", seeds=[0, 1], env=env)
    p = harness.run(GlobalPopularity, "search", seeds=[0, 1], env=env)
    assert r["stochastic"] and r["seeds"] == [0, 1] and r["summary"]["cum_reward"]["ci_seeds"] is not None
    assert not p["stochastic"] and p["seeds"] == [0] and p["summary"]["cum_reward"]["ci_seeds"] is None
    assert p["determinism_check_seed"] == 1


def test_deterministic_policies_give_identical_per_user_values_for_any_seed(env):
    for factory in (GlobalPopularity, ShortestFirst, BigMeanLabel, lambda c, s: CategoryRotation(c, s, "shortest")):
        a = harness.run(factory, "search", seeds=[0], env=env)
        b = harness.run(factory, "search", seeds=[7], env=env)
        assert a["per_user"] == b["per_user"]


def test_per_user_values_reconcile_with_summary(env):
    r = harness.run(UniformRandom, "search", seeds=[0, 1], env=env)
    assert np.isclose(np.mean(r["per_user"]["cum_reward"]), r["summary"]["cum_reward"]["mean"])


def test_compare_is_paired_and_deterministic(env):
    a = harness.run(UniformRandom, "search", seeds=[0, 1], env=env)
    same = harness.compare(a, a)
    assert same["mean"] == 0 and same["ci"] == [0.0, 0.0]
    b = harness.run(GlobalPopularity, "search", env=env)
    assert harness.compare(a, b) == harness.compare(a, b)
    shadow = harness.run(UniformRandom, "shadow", seeds=[0], env=env)
    with pytest.raises(ValueError):
        harness.compare(a, shadow)


@pytest.mark.parametrize("pick", ["random", "shortest", "longest"])
def test_rotation_never_triggers_under_any_profile(pick):
    for profile in ("primary", "looser", "tighter"):
        e = KuaiEnv(EnvConfig.from_toml(quit_profile=profile))
        r = harness.run(lambda c, s: CategoryRotation(c, s, pick), "search", seeds=[0], env=e)
        assert r["summary"]["boredom_quit_rate"]["mean"] == 0.0
        assert r["summary"]["session_length"]["mean"] == e.config.max_turn


def test_policies_stay_inside_the_action_set(env):
    # UnavailableActionError would propagate out of harness.run
    for factory in (UniformRandom, GlobalPopularity, ShortestFirst, BigMeanLabel,
                    lambda c, s: CategoryRotation(c, s, "random")):
        harness.run(factory, "search", seeds=[0], env=env)


def test_oracles_never_trigger_and_stay_inside_the_action_set(env):
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    mat = reward_matrix(env.config.reward_config(), users, videos)
    search = partitions.load("search")
    for fac in (oracles.personalized_factory(mat, users, videos, env.config),
                oracles.nonpersonalized_factory(mat, users, videos, env.config, search)):
        r = harness.run(fac, "search", env=env)
        assert r["summary"]["boredom_quit_rate"]["mean"] == 0.0
        assert r["summary"]["session_length"]["mean"] == env.config.max_turn


def test_personalized_first_pick_is_the_users_best_available_video(env):
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    mat = reward_matrix(env.config.reward_config(), users, videos)
    ctx = harness.context_for(env)
    o = oracles.personalized_factory(mat, users, videos, env.config)(ctx, 0)
    user = partitions.load("search")[0]
    row = mat[list(users).index(user)]
    o.begin_session(user, env.reset(user)["excluded"])
    assert row[list(videos).index(o.act())] == np.nanmax(row)


def test_duration_percentile_is_a_centred_midrank():
    b = labels.duration_buckets(EnvConfig.from_toml().duration_buckets)
    ref = b.reference[0]
    lo, hi = np.searchsorted(ref, ref, side="left"), np.searchsorted(ref, ref, side="right")
    mid = (lo + hi) / (2 * len(ref))
    assert mid.min() >= 0 and mid.max() <= 1 and abs(mid.mean() - 0.5) < 1e-9


def _affinity(env, mode):
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    mat = reward_matrix(env.config.reward_config(), users, videos)
    return mat, oracles.category_affinity_factory(mat, users, videos, env.categories, env.config, mode)


@pytest.mark.parametrize("profile", ["looser", "primary", "tighter"])
def test_category_affinity_oracle_never_triggers(profile):
    e = KuaiEnv(EnvConfig.from_toml(quit_profile=profile))
    for mode in oracles.AFFINITY_MODES:
        r = harness.run(_affinity(e, mode)[1], "search", seeds=[0], env=e)
        assert r["stochastic"]
        assert r["summary"]["boredom_quit_rate"]["mean"] == 0.0
        assert r["summary"]["session_length"]["mean"] == e.config.max_turn


@pytest.mark.parametrize("profile", ["looser", "primary"])
def test_affinity_modes_respect_their_repetition_limits(profile):
    e = KuaiEnv(EnvConfig.from_toml(quit_profile=profile))
    n, q = e.config.quit_window, e.config.quit_threshold
    ctx = harness.context_for(e)
    for mode, limit in (("strict", 0), ("free", q)):
        fac = _affinity(e, mode)[1]
        for user in partitions.load("search")[:40]:
            o = fac(ctx, 0)
            o.begin_session(user, e.reset(user)["excluded"])
            cats = [e.categories[o.act()][0] for _ in range(e.config.max_turn)]
            for t in range(1, len(cats)):
                assert cats[max(0, t - n):t].count(cats[t]) <= limit
    # free actually uses its freedom somewhere under these profiles
    o = _affinity(e, "free")[1](ctx, 0)
    user = partitions.load("search")[0]
    o.begin_session(user, e.reset(user)["excluded"])
    cats = [e.categories[o.act()][0] for _ in range(e.config.max_turn)]
    assert any(cats[max(0, t - n):t].count(cats[t]) > 0 for t in range(1, len(cats)))


def test_free_equals_strict_under_tighter():
    e = KuaiEnv(EnvConfig.from_toml(quit_profile="tighter"))
    a = harness.run(_affinity(e, "free")[1], "search", seeds=[0, 1], env=e)
    b = harness.run(_affinity(e, "strict")[1], "search", seeds=[0, 1], env=e)
    assert a["per_user"] == b["per_user"]


def test_affinity_is_the_mean_over_recorded_single_tag_videos(env):
    mat, fac = _affinity(env, "strict")
    users, videos = list(env.users), list(env.catalog)
    user = partitions.load("search")[3]
    for k, c in enumerate(fac.categories[:5]):
        cols = [j for j, v in enumerate(videos) if env.categories[v] == (c,)]
        expected = np.nanmean(mat[users.index(user), cols])
        assert fac.affinity[users.index(user), k] == pytest.approx(expected)
