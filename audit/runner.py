"""Run policies through the frozen harness pieces and compare them by paired bootstrap.

run_policy() executes exactly what kuairec_eval.harness.run does (verify, partitions.load,
run_seed per seed, seed_metrics, t_interval, user_bootstrap), so its `summary` equals
harness.run's; tests/test_audit.py checks that. It additionally keeps per-user values,
averaged over seeds, which harness.run does not return and paired comparisons need.

paired_bootstrap(): per-user differences d_u = A(u) - B(u) on the same users; resample
users with replacement (numpy default_rng(seed)); percentile interval of mean(d). ADR-0018.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kuairec_eval import _integrity, harness, partitions


@dataclass
class PolicyRun:
    name: str
    stochastic: bool
    seeds: list[int]
    summary: dict
    per_seed: list[dict]
    users: np.ndarray
    cum_reward: np.ndarray        # per user, mean over seeds
    session_length: np.ndarray    # per user, mean over seeds
    reward_sum: np.ndarray        # per user, summed over seeds, recorded pairs only
    reward_n: np.ndarray          # per user, count over seeds, recorded pairs only
    extra: dict = field(default_factory=dict)

    @property
    def mean_reward(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.reward_sum / self.reward_n


def run_policy(env, factory, partition: str, seeds: list[int], name: str, stochastic: bool) -> PolicyRun:
    _integrity.verify()
    cfg = harness.harness_config()
    users = partitions.load(partition)
    logs = [harness.run_seed(env, factory, users, s) for s in seeds]
    per_seed = [harness.seed_metrics(log, env) for log in logs]
    summary = {}
    for m in harness.METRICS:
        mean, lo, hi = harness.t_interval(np.array([p[m] for p in per_seed]), cfg["confidence"])
        summary[m] = {"mean": mean, "ci_seeds": [lo, hi]}
    for m, ci in harness.user_bootstrap(logs, cfg["confidence"], cfg["user_bootstrap_resamples"]).items():
        summary[m]["ci_users"] = list(ci)

    order = np.array(sorted(users))
    index = {u: i for i, u in enumerate(order)}
    n = len(order)
    cum, length = np.zeros(n), np.zeros(n)
    rsum, rn = np.zeros(n), np.zeros(n)
    for log in logs:
        for u, ln, r in zip(log.sess_user, log.sess_len, log.sess_reward):
            cum[index[u]] += r
            length[index[u]] += ln
        for u, r, ok in zip(log.rec_user, log.rec_reward, log.rec_observed):
            if ok:
                rsum[index[u]] += r
                rn[index[u]] += 1
    return PolicyRun(name=name, stochastic=stochastic, seeds=list(seeds), summary=summary, per_seed=per_seed,
                     users=order, cum_reward=cum / len(logs), session_length=length / len(logs),
                     reward_sum=rsum, reward_n=rn)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = 10_000, seed: int = 0,
                     confidence: float = 0.95) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        boot[start:stop] = d[rng.integers(0, len(d), size=(stop - start, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [50 * (1 - confidence), 50 * (1 + confidence)])
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "n_users": int(keep.sum()),
            "dropped_users": int((~keep).sum()), "resamples": resamples, "seed": seed}


def compare(a: PolicyRun, b: PolicyRun, metric: str, **kw) -> dict:
    if not np.array_equal(a.users, b.users):
        raise ValueError("paired comparison needs the same users")
    get = {"cum_reward": lambda r: r.cum_reward, "session_length": lambda r: r.session_length,
           "mean_reward": lambda r: r.mean_reward}[metric]
    return {"a": a.name, "b": b.name, "metric": metric, **paired_bootstrap(get(a), get(b), **kw)}
