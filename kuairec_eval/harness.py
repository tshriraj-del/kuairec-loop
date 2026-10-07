"""Run a policy over a user partition for S seeds and summarise five metrics.

Per seed s (ADR-0008, ADR-0009):
  * users of the partition are visited once each, in numpy default_rng(s) order;
  * the policy is built fresh by policy_factory(context, s);
  * each user gets exactly one session (reset -> act/step until done).

Metrics per seed, all pooled over the seed's sessions:
  mean_watch_ratio      mean reward over recommendations whose pair is recorded
                        (unrecorded pairs are excluded here, counted in missing_pair_rate)
  cum_reward            mean over sessions of the summed reward (unrecorded pairs add
                        config.missing_pair_reward, i.e. 0)
  session_length        mean number of recommendations per session
  top_category_share    max over categories c of (#recommendations whose video carries c)
                        / #recommendations; a video with k tags counts toward k categories
  catalog_coverage      distinct videos recommended / catalog size (3,327)
plus diagnostics: missing_pair_rate, boredom_quit_rate, max_turn_rate.

Across seeds: mean and a two-sided t-interval at `confidence` with S-1 dof.
Because seeds only change policy randomness and user order, a deterministic policy
has a zero-width seed interval. For session-level metrics we therefore also report a
user-cluster bootstrap interval (resampling users, seed-averaged), labelled ci_users.
"""
from __future__ import annotations

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
METRICS = ("mean_watch_ratio", "cum_reward", "session_length", "top_category_share",
           "catalog_coverage", "missing_pair_rate", "boredom_quit_rate", "max_turn_rate")


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy may know about the world besides its own step() feedback."""
    catalog: tuple[int, ...]
    categories: Mapping[int, tuple[int, ...]]


class Policy(Protocol):
    def begin_session(self, user: int) -> None: ...
    def act(self) -> int: ...
    def observe(self, video: int, result: StepResult) -> None: ...


PolicyFactory = Callable[[PolicyContext, int], Policy]


@dataclass
class SeedLog:
    seed: int
    rec_user: list[int] = field(default_factory=list)
    rec_video: list[int] = field(default_factory=list)
    rec_reward: list[float] = field(default_factory=list)
    rec_observed: list[bool] = field(default_factory=list)
    sess_user: list[int] = field(default_factory=list)
    sess_len: list[int] = field(default_factory=list)
    sess_reward: list[float] = field(default_factory=list)
    sess_reason: list[str] = field(default_factory=list)


def harness_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)["harness"]


def run_seed(env: KuaiEnv, factory: PolicyFactory, users: list[int], seed: int) -> SeedLog:
    ctx = PolicyContext(catalog=env.catalog, categories=MappingProxyType(dict(env.categories)))
    policy = factory(ctx, seed)
    order = np.random.default_rng(seed).permutation(np.asarray(users, dtype=np.int64))
    log = SeedLog(seed=seed)
    for user in order.tolist():
        env.reset(user)
        policy.begin_session(user)
        total, n, result = 0.0, 0, None
        while result is None or not result.done:
            video = policy.act()
            result = env.step(video)
            policy.observe(video, result)
            log.rec_user.append(user)
            log.rec_video.append(int(video))
            log.rec_reward.append(result.reward)
            log.rec_observed.append(result.observed)
            total += result.reward
            n += 1
        log.sess_user.append(user)
        log.sess_len.append(n)
        log.sess_reward.append(total)
        log.sess_reason.append(result.quit_reason)
    return log


def seed_metrics(log: SeedLog, env: KuaiEnv) -> dict[str, float]:
    reward = np.asarray(log.rec_reward)
    observed = np.asarray(log.rec_observed)
    cat_counts = Counter(c for v in log.rec_video for c in env.categories[v])
    n_rec = len(log.rec_video)
    reasons = Counter(log.sess_reason)
    return {
        "mean_watch_ratio": float(reward[observed].mean()) if observed.any() else float("nan"),
        "cum_reward": float(np.mean(log.sess_reward)),
        "session_length": float(np.mean(log.sess_len)),
        "top_category_share": max(cat_counts.values()) / n_rec,
        "catalog_coverage": len(set(log.rec_video)) / len(env.catalog),
        "missing_pair_rate": float(1.0 - observed.mean()),
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


def user_bootstrap(logs: list[SeedLog], confidence: float, resamples: int) -> dict[str, tuple[float, float]]:
    """Percentile CI from resampling users; each user's value is averaged over seeds."""
    users = sorted(set(logs[0].sess_user))
    index = {u: i for i, u in enumerate(users)}
    n_u, n_s = len(users), len(logs)
    sess_r, sess_l = np.zeros((n_s, n_u)), np.zeros((n_s, n_u))
    wr_sum, wr_cnt = np.zeros((n_s, n_u)), np.zeros((n_s, n_u))
    for s, log in enumerate(logs):
        for u, length, r in zip(log.sess_user, log.sess_len, log.sess_reward):
            sess_r[s, index[u]], sess_l[s, index[u]] = r, length
        for u, r, ok in zip(log.rec_user, log.rec_reward, log.rec_observed):
            if ok:
                wr_sum[s, index[u]] += r
                wr_cnt[s, index[u]] += 1
    per_user = {"cum_reward": sess_r.mean(0), "session_length": sess_l.mean(0)}
    wr_sum_u, wr_cnt_u = wr_sum.sum(0), wr_cnt.sum(0)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, n_u, size=(resamples, n_u))
    lo_q, hi_q = 50 * (1 - confidence), 50 * (1 + confidence)
    out = {}
    for name, vals in per_user.items():
        boot = vals[draws].mean(1)
        out[name] = (float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q)))
    boot = wr_sum_u[draws].sum(1) / wr_cnt_u[draws].sum(1)
    out["mean_watch_ratio"] = (float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q)))
    return out


def run(factory: PolicyFactory, partition: str, seeds: list[int] | None = None,
        env: KuaiEnv | None = None, policy_name: str = "") -> dict:
    digest = _integrity.verify()
    cfg = harness_config()
    seeds = list(cfg["seeds"] if seeds is None else seeds)
    users = partitions.load(partition)
    env = env or KuaiEnv()
    started = time.time()
    logs = [run_seed(env, factory, users, s) for s in seeds]
    per_seed = [seed_metrics(log, env) for log in logs]
    summary = {}
    for m in METRICS:
        mean, lo, hi = t_interval(np.array([p[m] for p in per_seed]), cfg["confidence"])
        summary[m] = {"mean": mean, "ci_seeds": [lo, hi]}
    for m, ci in user_bootstrap(logs, cfg["confidence"], cfg["user_bootstrap_resamples"]).items():
        summary[m]["ci_users"] = list(ci)
    return {
        "policy": policy_name,
        "partition": partition,
        "n_users": len(users),
        "seeds": seeds,
        "confidence": cfg["confidence"],
        "env_config": env.config.__dict__,
        "manifest_digest": digest,
        "summary": summary,
        "per_seed": per_seed,
        "wall_seconds": round(time.time() - started, 2),
    }
