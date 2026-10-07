"""Same seed, identical numbers."""
from kuairec_eval import harness
from policies.baselines import REGISTRY


def _run(env, name, seeds):
    return harness.run(REGISTRY[name], "search", seeds=seeds, env=env, policy_name=name)


def test_same_seed_identical_numbers(env):
    for name in REGISTRY:
        a, b = _run(env, name, [3, 7]), _run(env, name, [3, 7])
        assert a["per_seed"] == b["per_seed"]
        assert a["summary"] == b["summary"]


def test_different_seeds_change_random_policy(env):
    a, b = _run(env, "random", [0]), _run(env, "random", [1])
    assert a["per_seed"] != b["per_seed"]
