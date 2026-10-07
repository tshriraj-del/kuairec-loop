"""Run a policy over a user partition; compare policies by paired bootstrap over users (v1).

Protocol (ADR-0008, ADR-0021, ADR-0022):
  * policy_factory(context, seed) returns a policy with a boolean attribute `stochastic`
    and methods begin_session(user, excluded), act() -> video, observe(video, StepResult);
  * `stochastic` means "per-user results may depend on the seed": internal randomness OR
    state carried across users (the seed sets the user order). A stochastic policy is run
    once per seed in config seeds. A policy declaring stochastic=False is run on seeds[0]
    and checked on seeds[1]; if any user's values differ, run() raises (ADR-0021);
  * per run: a fresh policy, every partition user visited once in default_rng(seed)
    order, one session per user; the policy may keep state across users within the run;
  * run() refuses window_mode != "intended" (the EasyRL4Rec slice is for the differential
    check only).

Metrics per run, pooled over the run's sessions:
  mean_reward         mean reward per recommendation
  cum_reward          mean over sessions of the summed reward
  session_length      mean number of recommendations per session
  top_category_share  max over categories c of (#recommendations carrying c) / #recommendations
  catalog_coverage    distinct videos recommended / catalog size (3,327)
  boredom_quit_rate, max_turn_rate

Reported per policy: mean over runs; ci_seeds (t-interval over runs) for stochastic policies
only; ci_users (user-cluster bootstrap) as a descriptive interval; per-user values (mean over
runs). Decisions between policies use compare(): per-user differences on the same users,
users resampled with replacement, percentile interval of the mean difference.
"""
from __future__ import annotations

import dataclasses
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np
from scipy import stats

from kuairec_eval import _integrity, partitions
from kuairec_eval.data import REPO_ROOT
from kuairec_eval.env import KuaiEnv, StepResult

CONFIG_PATH = REPO_ROOT / "config" / "evaluator.toml"
METRICS = ("mean_reward", "cum_reward", "session_length", "top_category_share",
           "catalog_coverage", "boredom_quit_rate", "max_turn_rate")
PER_USER = ("cum_reward", "session_length", "mean_reward")


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy may know about the world besides its own step() feedback."""
    catalog: tuple[int, ...]
    categories: Mapping[int, tuple[int, ...]]


class Policy(Protocol):
    stochastic: bool
    def begin_session(self, user: int, excluded: frozenset[int]) -> None: ...
    def act(self) -> int: ...
    def observe(self, video: int, result: StepResult) -> None: ...


PolicyFactory = Callable[[PolicyContext, int], Policy]


@dataclass
class SeedLog:
    seed: int
    rec_user: list[int] = field(default_factory=list)
    rec_video: list[int] = field(default_factory=list)
    rec_reward: list[float] = field(default_factory=list)
    sess_user: list[int] = field(default_factory=list)
    sess_len: list[int] = field(default_factory=list)
    sess_reward: list[float] = field(default_factory=list)
    sess_reason: list[str] = field(default_factory=list)


def harness_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)["harness"]


def context_for(env: KuaiEnv) -> PolicyContext:
    return PolicyContext(catalog=env.catalog, categories=MappingProxyType(dict(env.categories)))


def run_seed(env: KuaiEnv, policy: Policy, users: list[int], seed: int) -> SeedLog:
    order = np.random.default_rng(seed).permutation(np.asarray(users, dtype=np.int64))
    log = SeedLog(seed=seed)
    for user in order.tolist():
        obs = env.reset(user)
        policy.begin_session(user, obs["excluded"])
        total, n, result = 0.0, 0, None
        while result is None or not result.done:
            video = policy.act()
            result = env.step(video)
            policy.observe(video, result)
            log.rec_user.append(user)
            log.rec_video.append(int(video))
            log.rec_reward.append(result.reward)
            total += result.reward
            n += 1
        log.sess_user.append(user)
        log.sess_len.append(n)
        log.sess_reward.append(total)
        log.sess_reason.append(result.quit_reason)
    return log


def seed_metrics(log: SeedLog, env: KuaiEnv) -> dict[str, float]:
    cat_counts = Counter(c for v in log.rec_video for c in env.categories[v])
    n_rec = len(log.rec_video)
    reasons = Counter(log.sess_reason)
    return {
        "mean_reward": float(np.mean(log.rec_reward)),
        "cum_reward": float(np.mean(log.sess_reward)),
        "session_length": float(np.mean(log.sess_len)),
        "top_category_share": max(cat_counts.values()) / n_rec,
        "catalog_coverage": len(set(log.rec_video)) / len(env.catalog),
        "boredom_quit_rate": reasons["boredom"] / len(log.sess_len),
        "max_turn_rate": reasons["max_turn"] / len(log.sess_len),
    }


def t_interval(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = float(stats.t.ppf(0.5 + confidence / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def per_user_values(logs: list[SeedLog]) -> dict[str, np.ndarray]:
    """Per-user values, each the mean over runs; mean_reward pools a user's recommendations over runs."""
    users = np.array(sorted(set(logs[0].sess_user)))
    index = {u: i for i, u in enumerate(users)}
    n = len(users)
    cum, length, rsum, rn = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for log in logs:
        for u, ln, r in zip(log.sess_user, log.sess_len, log.sess_reward):
            cum[index[u]] += r
            length[index[u]] += ln
        for u, r in zip(log.rec_user, log.rec_reward):
            rsum[index[u]] += r
            rn[index[u]] += 1
    return {"users": users, "cum_reward": cum / len(logs), "session_length": length / len(logs),
            "mean_reward": rsum / rn, "reward_sum": rsum, "reward_n": rn}


def user_bootstrap(per_user: dict[str, np.ndarray], confidence: float, resamples: int) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(0)
    n = len(per_user["users"])
    draws = rng.integers(0, n, size=(resamples, n))
    lo_q, hi_q = 50 * (1 - confidence), 50 * (1 + confidence)
    out = {}
    for name in ("cum_reward", "session_length"):
        boot = per_user[name][draws].mean(1)
        out[name] = (float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q)))
    boot = per_user["reward_sum"][draws].sum(1) / per_user["reward_n"][draws].sum(1)
    out["mean_reward"] = (float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q)))
    return out


def run(factory: PolicyFactory, partition: str, seeds: list[int] | None = None,
        env: KuaiEnv | None = None, policy_name: str = "") -> dict:
    digest = _integrity.verify()
    cfg = harness_config()
    env = env or KuaiEnv()
    if env.config.window_mode != "intended":
        raise ValueError("harness.run evaluates only window_mode='intended'; the EasyRL4Rec slice is for "
                         "scripts/differential.py")
    seeds = list(cfg["seeds"] if seeds is None else seeds)
    users = partitions.load(partition)
    ctx = context_for(env)
    first = factory(ctx, seeds[0])
    if not isinstance(getattr(first, "stochastic", None), bool):
        raise TypeError("policies must declare a boolean `stochastic` attribute")
    stochastic = first.stochastic
    check_seed = None
    if not stochastic:
        seeds, check_seed = seeds[:1], (seeds[1] if len(seeds) > 1 else None)
    started = time.time()
    logs = [run_seed(env, first if i == 0 else factory(ctx, s), users, s) for i, s in enumerate(seeds)]
    per_seed = [seed_metrics(log, env) for log in logs]
    pu = per_user_values(logs)
    if check_seed is not None:
        check = per_user_values([run_seed(env, factory(ctx, check_seed), users, check_seed)])
        if any(not np.array_equal(pu[k], check[k]) for k in ("cum_reward", "session_length", "reward_sum")):
            raise ValueError(f"policy {policy_name!r} declares stochastic=False but its per-user results differ "
                             f"between seeds {seeds[0]} and {check_seed}; declare stochastic=True")
    summary = {}
    for m in METRICS:
        mean, lo, hi = t_interval(np.array([p[m] for p in per_seed]), cfg["confidence"])
        summary[m] = {"mean": mean, "ci_seeds": [lo, hi] if stochastic else None}
    for m, ci in user_bootstrap(pu, cfg["confidence"], cfg["user_bootstrap_resamples"]).items():
        summary[m]["ci_users"] = list(ci)
    return {
        "policy": policy_name,
        "stochastic": stochastic,
        "partition": partition,
        "n_users": len(users),
        "seeds": seeds,
        "determinism_check_seed": check_seed,
        "confidence": cfg["confidence"],
        "env_config": dataclasses.asdict(env.config),
        "manifest_digest": digest,
        "summary": summary,
        "per_seed": per_seed,
        "per_user": {k: pu[k].tolist() for k in ("users",) + PER_USER},
        "wall_seconds": round(time.time() - started, 2),
    }


def compare(a: dict, b: dict, metric: str = "cum_reward") -> dict:
    """Paired bootstrap over users of mean_u[a(u) - b(u)] for a per-user metric."""
    if metric not in PER_USER:
        raise KeyError(f"metric must be one of {PER_USER}")
    if a["per_user"]["users"] != b["per_user"]["users"]:
        raise ValueError("paired comparison needs the same users")
    cfg = harness_config()
    d = np.asarray(a["per_user"][metric]) - np.asarray(b["per_user"][metric])
    rng = np.random.default_rng(cfg["paired_bootstrap_seed"])
    resamples = cfg["paired_bootstrap_resamples"]
    boot = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        boot[start:stop] = d[rng.integers(0, len(d), size=(stop - start, len(d)))].mean(axis=1)
    conf = cfg["confidence"]
    lo, hi = np.percentile(boot, [50 * (1 - conf), 50 * (1 + conf)])
    return {"a": a["policy"], "b": b["policy"], "metric": metric, "mean": float(d.mean()),
            "ci": [float(lo), float(hi)], "n_users": int(len(d)), "resamples": resamples}
