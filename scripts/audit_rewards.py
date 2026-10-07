"""Items 2, 3, 6 and 7 of the evaluator audit, on the SEARCH partition only.

For each reward candidate (raw, capped5, duration_pct):
  * run random, popularity, reference policies (a)-(d), the extra (e) and two probes
    through the frozen v0 harness pieces (audit/runner.py) on RewardEnv (audit/reward_env.py);
    stochastic policies use the harness seeds 0-9, deterministic ones seed 0;
  * run both oracle ceilings (kuairec_eval/diagnostics/oracles.py) and the quit-free top-30 bounds;
  * per-video rank correlation of mean reward with canonical duration (search users);
  * paired bootstrap over users (10,000 resamples, seed 0) for the acceptance tests:
      T1 shortest-first must not score above random:  pass iff the 95% interval of
         mean_u[(a)(u) - random(u)] cumulative reward does NOT lie entirely above 0
      T2 personalized oracle must beat non-personalized: pass iff the 95% interval of
         mean_u[P(u) - NP(u)] cumulative reward lies entirely above 0

Usage: .venv/bin/python scripts/audit_rewards.py [--rewards raw capped5 duration_pct]
Writes runs/audit/rewards_<reward>.{json,md}, runs/audit/per_user_<reward>.csv, runs/audit/acceptance.json
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from audit.reward_env import RewardEnv  # noqa: E402
from audit.runner import compare, run_policy  # noqa: E402
from kuairec_eval import harness, labels, partitions  # noqa: E402
from kuairec_eval.diagnostics import oracles  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from kuairec_eval.rewards import REWARDS, RewardConfig, reward_matrix  # noqa: E402
from policies.baselines import GlobalPopularity, UniformRandom  # noqa: E402
from policies.reference import BigMeanLabel, CategoryRotation, LongestFirst, ShortestFirst  # noqa: E402

OUT = ROOT / "runs" / "audit"
RESAMPLES, BOOT_SEED = 10_000, 0


def policy_specs(cfg: RewardConfig):
    """(key, label, factory, stochastic, requested)"""
    specs = [
        ("random", "random", UniformRandom, True, True),
        ("popularity", "popularity (big-matrix distinct users)", GlobalPopularity, False, True),
        ("a_shortest_first", "(a) shortest first", ShortestFirst, False, True),
        ("b_rotation_random", "(b) category rotation, random pick", lambda c, s: CategoryRotation(c, s, "random"), True, True),
        ("c_rotation_shortest", "(c) category rotation, shortest pick", lambda c, s: CategoryRotation(c, s, "shortest"), False, True),
        ("d_big_mean_watch_ratio", "(d) big-matrix mean watch_ratio, >=100 rows", BigMeanLabel, False, True),
    ]
    if cfg.active != "raw":
        specs.append(("e_big_mean_label", f"(e) big-matrix mean {cfg.active} label, >=100 rows [extra]",
                      lambda c, s: BigMeanLabel(c, s, cfg.active, cfg.duration_buckets), False, False))
    specs += [
        ("probe_longest_first", "probe: longest first [extra]", LongestFirst, False, False),
        ("probe_rotation_longest", "probe: category rotation, longest pick [extra]",
         lambda c, s: CategoryRotation(c, s, "longest"), False, False),
    ]
    return specs


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def run_candidate(base: KuaiEnv, name: str, search: list[int], seeds: list[int]) -> dict:
    cfg = RewardConfig.from_toml(active=name)
    users, videos = np.asarray(base.users), np.asarray(base.catalog)
    mat = reward_matrix(cfg, users, videos)
    env = RewardEnv(base, cfg, matrix=mat)
    runs, labels_of, timing = {}, {}, {}
    for key, label, factory, stochastic, requested in policy_specs(cfg):
        t0 = time.time()
        runs[key] = run_policy(env, factory, "search", seeds if stochastic else [seeds[0]], key, stochastic)
        runs[key].extra["requested"] = requested
        labels_of[key], timing[key] = label, round(time.time() - t0, 1)
    t0 = time.time()
    np_fac = oracles.nonpersonalized_factory(mat, users, videos, base.config, search, base.categories)
    runs["oracle_nonpersonalized"] = run_policy(env, np_fac, "search", [seeds[0]], "oracle_nonpersonalized", False)
    runs["oracle_personalized"] = run_policy(env, oracles.personalized_factory(mat, users, videos, base.config),
                                             "search", [seeds[0]], "oracle_personalized", False)
    for k in ("oracle_nonpersonalized", "oracle_personalized"):
        runs[k].extra["requested"] = True
    labels_of["oracle_nonpersonalized"] = "oracle: non-personalized fixed sequence (greedy, search-user means)"
    labels_of["oracle_personalized"] = "oracle: personalized greedy"
    timing["oracles"] = round(time.time() - t0, 1)
    bound = oracles.top_k_bound(mat, users, search, base.config.max_turn, base.config.missing_pair_reward)

    rows = np.searchsorted(users, np.asarray(sorted(search)))
    per_video = np.nanmean(mat[rows], axis=0)
    dur = labels.canonical_durations().reindex(videos).to_numpy()
    rho = float(stats.spearmanr(dur, per_video).statistic)

    c = lambda a, b, m: compare(runs[a], runs[b], m, resamples=RESAMPLES, seed=BOOT_SEED)  # noqa: E731
    t1 = c("a_shortest_first", "random", "cum_reward")
    t2 = c("oracle_personalized", "oracle_nonpersonalized", "cum_reward")
    acceptance = {
        "T1_shortest_not_above_random": {**t1, "pass": not (t1["ci"][0] > 0)},
        "T2_personalized_beats_nonpersonalized": {**t2, "pass": t2["ci"][0] > 0},
    }
    acceptance["both_pass"] = acceptance["T1_shortest_not_above_random"]["pass"] and \
        acceptance["T2_personalized_beats_nonpersonalized"]["pass"]
    supplementary = [
        c("a_shortest_first", "random", "mean_reward"),
        c("c_rotation_shortest", "b_rotation_random", "cum_reward"),
        c("c_rotation_shortest", "b_rotation_random", "mean_reward"),
        c("oracle_personalized", "oracle_nonpersonalized", "mean_reward"),
        c("probe_longest_first", "random", "cum_reward"),
        c("probe_rotation_longest", "b_rotation_random", "cum_reward"),
        c("probe_rotation_longest", "b_rotation_random", "mean_reward"),
        c("b_rotation_random", "random", "cum_reward"),
    ]
    if "e_big_mean_label" in runs:
        supplementary.append(c("e_big_mean_label", "random", "cum_reward"))
    supplementary.append(c("d_big_mean_watch_ratio", "random", "cum_reward"))

    pd.DataFrame({"user_id": runs["random"].users,
                  **{k: r.cum_reward for k, r in runs.items()}}).to_csv(OUT / f"per_user_{name}.csv", index=False)
    return {"reward": name, "reward_config": cfg.__dict__, "rho_per_video_mean_reward_vs_duration": rho,
            "bounds_top30": {"personalized": bound["personalized"], "nonpersonalized": bound["nonpersonalized"]},
            "np_oracle_sequence": list(np_fac.sequence),
            "policies": {k: {"label": labels_of[k], "stochastic": r.stochastic, "requested": r.extra["requested"],
                             "seeds": r.seeds, "summary": r.summary} for k, r in runs.items()},
            "acceptance": acceptance, "supplementary": supplementary, "timing_seconds": timing}


def markdown(res: dict, command: str) -> str:
    lines = [f"### Reward: `{res['reward']}`", "", f"Command: `{command}`  ",
             f"Per-video Spearman(mean reward over search users, canonical duration) = "
             f"{res['rho_per_video_mean_reward_vs_duration']:+.4f}", "",
             "Cum reward: mean per session over 847 search users {95% user bootstrap}"
             " [95% seed t-interval, stochastic policies only].", "",
             "| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate | Unrecorded-pair rate |",
             "|---|---|---|---|---|---|---|---|"]
    for key, p in res["policies"].items():
        s = p["summary"]
        cr = s["cum_reward"]
        cell = f"{fmt(cr['mean'])} {{{fmt(cr['ci_users'][0])}, {fmt(cr['ci_users'][1])}}}"
        if p["stochastic"]:
            cell += f" [{fmt(cr['ci_seeds'][0])}, {fmt(cr['ci_seeds'][1])}]"
        lines.append(f"| {p['label']} | {cell} | {fmt(s['mean_watch_ratio']['mean'], 4)} | "
                     f"{fmt(s['session_length']['mean'], 2)} | {fmt(s['top_category_share']['mean'], 3)} | "
                     f"{fmt(s['catalog_coverage']['mean'], 4)} | {fmt(s['boredom_quit_rate']['mean'], 3)} | "
                     f"{fmt(s['missing_pair_rate']['mean'], 4)} |")
    b = res["bounds_top30"]
    lines += ["", f"Quit-free top-30 bounds: personalized {fmt(b['personalized'])}, non-personalized {fmt(b['nonpersonalized'])}.", "",
              "| Paired comparison (A - B), per-user, 10,000 resamples | Metric | Mean | 95% interval |", "|---|---|---|---|"]
    a = res["acceptance"]
    for key in ("T1_shortest_not_above_random", "T2_personalized_beats_nonpersonalized"):
        t = a[key]
        lines.append(f"| **{key}** {t['a']} - {t['b']} -> {'PASS' if t['pass'] else 'FAIL'} | {t['metric']} | "
                     f"{fmt(t['mean'], 4)} | [{fmt(t['ci'][0], 4)}, {fmt(t['ci'][1], 4)}] |")
    for t in res["supplementary"]:
        lines.append(f"| {t['a']} - {t['b']} | {t['metric']} | {fmt(t['mean'], 4)} | [{fmt(t['ci'][0], 4)}, {fmt(t['ci'][1], 4)}] |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", nargs="+", default=list(REWARDS), choices=REWARDS)
    args = ap.parse_args()
    command = ".venv/bin/python scripts/audit_rewards.py " + shlex.join(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    base = KuaiEnv()
    search = partitions.load("search")
    seeds = list(harness.harness_config()["seeds"])
    acceptance = {}
    for name in args.rewards:
        t0 = time.time()
        res = run_candidate(base, name, search, seeds)
        res["command"], res["wall_seconds"] = command, round(time.time() - t0, 1)
        (OUT / f"rewards_{name}.json").write_text(json.dumps(res, indent=2))
        md = markdown(res, command)
        (OUT / f"rewards_{name}.md").write_text(md)
        print(md)
        acceptance[name] = res["acceptance"]
    path = OUT / "acceptance.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(acceptance)
    path.write_text(json.dumps(merged, indent=2))
    for name, a in merged.items():
        print(f"{name:14s} T1 {'PASS' if a['T1_shortest_not_above_random']['pass'] else 'FAIL'}  "
              f"T2 {'PASS' if a['T2_personalized_beats_nonpersonalized']['pass'] else 'FAIL'}  "
              f"both {'PASS' if a['both_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
