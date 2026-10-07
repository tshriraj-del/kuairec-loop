"""Small-matrix values are reachable only through step()."""
import numpy as np
import pandas as pd

from kuairec_eval.env import StepResult


def _walk(obj, seen, depth=0):
    if id(obj) in seen or depth > 6:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, dict) or hasattr(obj, "items") and not isinstance(obj, (str, bytes)):
        try:
            for k, v in obj.items():
                yield from _walk(k, seen, depth + 1)
                yield from _walk(v, seen, depth + 1)
        except TypeError:
            pass
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _walk(v, seen, depth + 1)
    elif hasattr(obj, "__dict__") and not callable(obj):
        yield from _walk(vars(obj), seen, depth + 1)


def test_env_attributes_hold_no_reward_data(env):
    config_floats = {v for v in vars(env.config).values() if isinstance(v, float)}
    for o in _walk({k: v for k, v in vars(env).items() if k != "config"}, set()):
        assert not isinstance(o, (np.ndarray, pd.DataFrame, pd.Series)), type(o)
        assert not isinstance(o, float), o
    assert config_floats == {env.config.reward_cap}


def test_public_surface_has_no_reward_accessor(env):
    public = {n for n in dir(env) if not n.startswith("_")}
    assert public == {"catalog", "categories", "config", "history", "manifest_digest",
                      "reset", "step", "users"}


def test_reset_and_step_reveal_only_the_requested_pair(env):
    obs = env.reset(env.users[0])
    assert set(obs) == {"user", "t", "excluded"} and obs["user"] == env.users[0] and obs["t"] == 0
    assert all(isinstance(v, int) for v in obs["excluded"])
    r = env.step(next(v for v in env.catalog if v not in obs["excluded"]))
    assert isinstance(r, StepResult)
    assert set(vars(r)) == {"reward", "done", "quit_reason", "t"}
    assert isinstance(r.reward, float)


def test_policy_context_carries_no_rewards(env):
    from kuairec_eval.harness import PolicyContext
    assert set(PolicyContext.__dataclass_fields__) == {"catalog", "categories"}
