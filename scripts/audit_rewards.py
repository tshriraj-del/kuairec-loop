"""Reward-candidate audit under evaluator v1, on the SEARCH partition only (primary quit profile).

For each reward candidate (raw, capped5, duration_pct):
  * run random, popularity, the PERMANENT_ROWS (rotation-random and the four duration probes),
    (d) and (e) with harness.run (stochastic policies on the harness seeds, deterministic ones
    on seeds[0] plus the determinism check);
  * run the oracle ceilings: category affinity (strict, full_cycle, free), non-personalized and
    personalized, and the quit-free versions;
  * per-video rank correlation of mean reward with canonical duration (search users);
  * acceptance (ADR-0027), all comparisons with harness.compare (paired bootstrap over users):
      T1  inside category rotation, the shortest and the longest probe must each differ from
          rotation-random by less than TOLERANCE x the personalization gap: the whole 95%
          interval of mean_u[probe(u) - rotation_random(u)] inside (-margin, +margin),
          margin = TOLERANCE x point estimate of the gap (set after seeing the audit numbers)
      T2  personalized oracle beats non-personalized: interval of mean_u[P(u) - NP(u)] above 0
      T1 legacy (secondary, not part of the verdict): shortest-first minus random, interval not
          entirely above 0

Usage: .venv/bin/python scripts/audit_rewards.py [--rewards raw capped5 duration_pct]
Writes runs/audit_v1/rewards_<reward>.{json,md} and runs/audit_v1/acceptance.json
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
from scipy import stats  # noqa: E402

from kuairec_eval import harness, labels, partitions  # noqa: E402
from kuairec_eval.diagnostics import oracles  # noqa: E402
from kuairec_eval.env import EnvConfig, KuaiEnv  # noqa: E402
from kuairec_eval.rewards import REWARDS, reward_matrix  # noqa: E402
from policies.baselines import GlobalPopularity, UniformRandom  # noqa: E402
from policies.reference import PERMANENT_ROWS, BigMeanLabel  # noqa: E402

OUT = ROOT / "runs" / "audit_v1"
TOLERANCE = 0.05


def policy_specs(cfg: EnvConfig):
    """(key, label, factory) for every non-oracle row."""
    specs = [("random", "random", UniformRandom),
             ("popularity", "popularity (big-matrix distinct users)", GlobalPopularity)]
    specs += [(k, f"{label} [permanent]", f) for k, label, f in PERMANENT_ROWS]
    specs.append(("d_big_mean_watch_ratio", "(d) big-matrix mean watch_ratio, >=100 rows", BigMeanLabel))
    if cfg.reward != "raw":
        specs.append(("e_big_mean_label", f"(e) big-matrix mean {cfg.reward} label, >=100 rows",
                      lambda c, s: BigMeanLabel(c, s, cfg.reward, cfg.duration_buckets)))
    return specs


def acceptance(cmp) -> dict:
    """cmp(a, b, metric) -> harness.compare result. Returns the ADR-0027 verdict."""
    t2 = cmp("oracle_personalized", "oracle_nonpersonalized", "cum_reward")
    gap = t2["mean"]
    margin = TOLERANCE * gap
    t1 = {"tolerance": TOLERANCE, "gap": gap, "margin": margin}
    for probe in ("rotation_shortest", "rotation_longest"):
        d = cmp(probe, "rotation_random", "cum_reward")
        t1[probe] = {**d, "share_of_gap": d["mean"] / gap, "ci_share_of_gap": [x / gap for x in d["ci"]],
                     "pass": (-margin < d["ci"][0]) and (d["ci"][1] < margin)}
    t1["pass"] = t1["rotation_shortest"]["pass"] and t1["rotation_longest"]["pass"]
    legacy = cmp("shortest_first", "random", "cum_reward")
    return {"T1_duration_probes_within_tolerance": t1,
            "T2_personalized_beats_nonpersonalized": {**t2, "pass": t2["ci"][0] > 0},
            "T1_legacy_shortest_not_above_random": {**legacy, "pass": not (legacy["ci"][0] > 0)},
            "both_pass": t1["pass"] and t2["ci"][0] > 0}


def fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def run_candidate(name: str, search: list[int]) -> dict:
    env = KuaiEnv(EnvConfig.from_toml(reward=name))
    cfg = env.config
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    mat = reward_matrix(cfg.reward_config(), users, videos)
    runs, labels_of = {}, {}
    for key, label, factory in policy_specs(cfg):
        runs[key], labels_of[key] = harness.run(factory, "search", env=env, policy_name=key), label
    for mode in ("strict", "full_cycle", "free"):
        key = f"oracle_affinity_{mode}"
        fac = oracles.category_affinity_factory(mat, users, videos, env.categories, cfg, mode)
        runs[key] = harness.run(fac, "search", env=env, policy_name=key)
        labels_of[key] = f"oracle: category affinity, {mode.replace('_', ' ')}"
    np_fac = oracles.nonpersonalized_factory(mat, users, videos, cfg, search)
    runs["oracle_nonpersonalized"] = harness.run(np_fac, "search", env=env, policy_name="oracle_nonpersonalized")
    runs["oracle_personalized"] = harness.run(oracles.personalized_factory(mat, users, videos, cfg), "search",
                                              env=env, policy_name="oracle_personalized")
    labels_of["oracle_nonpersonalized"] = "oracle: non-personalized (common values, greedy)"
    labels_of["oracle_personalized"] = "oracle: personalized greedy"
    quit_free = oracles.top_k(mat, users, search, cfg.max_turn, np_fac.common_values)

    rows = np.searchsorted(users, np.asarray(sorted(search)))
    per_video = np.nanmean(mat[rows], axis=0)
    dur = labels.canonical_durations().reindex(videos).to_numpy()
    rho = float(stats.spearmanr(dur, per_video).statistic)

    cmp = lambda a, b, m: harness.compare(runs[a], runs[b], m)  # noqa: E731
    supplementary = [
        cmp("shortest_first", "random", "mean_reward"),
        cmp("longest_first", "random", "cum_reward"),
        cmp("rotation_shortest", "rotation_random", "mean_reward"),
        cmp("rotation_longest", "rotation_random", "mean_reward"),
        cmp("oracle_personalized", "oracle_nonpersonalized", "mean_reward"),
        cmp("oracle_affinity_free", "oracle_affinity_strict", "cum_reward"),
        cmp("rotation_random", "random", "cum_reward"),
    ]
    return {"reward": name, "env_config": runs["random"]["env_config"], "manifest_digest": runs["random"]["manifest_digest"],
            "rho_per_video_mean_reward_vs_duration": rho, "quit_free": quit_free,
            "policies": {k: {"label": labels_of[k], "stochastic": r["stochastic"], "seeds": r["seeds"],
                             "summary": r["summary"]} for k, r in runs.items()},
            "acceptance": acceptance(cmp), "supplementary": supplementary}


def markdown(res: dict, command: str) -> str:
    lines = [f"### Reward: `{res['reward']}` (evaluator v1, primary quit profile)", "", f"Command: `{command}`  ",
             f"Manifest digest `{res['manifest_digest'][:16]}`. Per-video Spearman(mean reward over search users, "
             f"canonical duration) = {res['rho_per_video_mean_reward_vs_duration']:+.4f}", "",
             "Cum reward: mean per session over 847 search users {95% user bootstrap}"
             " [95% seed t-interval, stochastic policies only].", "",
             "| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate |",
             "|---|---|---|---|---|---|---|"]
    for key, p in res["policies"].items():
        s = p["summary"]
        cr = s["cum_reward"]
        cell = f"{fmt(cr['mean'])} {{{fmt(cr['ci_users'][0])}, {fmt(cr['ci_users'][1])}}}"
        if cr["ci_seeds"] is not None:
            cell += f" [{fmt(cr['ci_seeds'][0])}, {fmt(cr['ci_seeds'][1])}]"
        lines.append(f"| {p['label']} | {cell} | {fmt(s['mean_reward']['mean'], 4)} | {fmt(s['session_length']['mean'], 2)} | "
                     f"{fmt(s['top_category_share']['mean'], 3)} | {fmt(s['catalog_coverage']['mean'], 4)} | "
                     f"{fmt(s['boredom_quit_rate']['mean'], 3)} |")
    q, a = res["quit_free"], res["acceptance"]
    t1, t2, leg = (a["T1_duration_probes_within_tolerance"], a["T2_personalized_beats_nonpersonalized"],
                   a["T1_legacy_shortest_not_above_random"])
    lines += ["", f"Quit-free versions: personalized bound {fmt(q['personalized_bound'])}, "
              f"non-personalized {fmt(q['nonpersonalized_quit_free'])}.", "",
              f"**Acceptance: {'PASS' if a['both_pass'] else 'FAIL'}.** Personalization gap {fmt(t1['gap'], 3)}; "
              f"T1 margin = {t1['tolerance']:.0%} of the gap = +/-{fmt(t1['margin'], 3)}.", "",
              "| Test (paired bootstrap over users) | Mean | 95% interval | Share of gap | Result |", "|---|---|---|---|---|"]
    for probe in ("rotation_shortest", "rotation_longest"):
        d = t1[probe]
        lines.append(f"| T1 {probe} - rotation_random (cum reward) | {fmt(d['mean'], 4)} | [{fmt(d['ci'][0], 4)}, "
                     f"{fmt(d['ci'][1], 4)}] | {d['share_of_gap']:+.1%} [{d['ci_share_of_gap'][0]:+.1%}, "
                     f"{d['ci_share_of_gap'][1]:+.1%}] | {'pass' if d['pass'] else 'FAIL'} |")
    lines.append(f"| T2 personalized - non-personalized oracle (cum reward) | {fmt(t2['mean'], 4)} | [{fmt(t2['ci'][0], 4)}, "
                 f"{fmt(t2['ci'][1], 4)}] | 100% | {'pass' if t2['pass'] else 'FAIL'} |")
    lines.append(f"| secondary: legacy T1 shortest_first - random (cum reward) | {fmt(leg['mean'], 4)} | "
                 f"[{fmt(leg['ci'][0], 4)}, {fmt(leg['ci'][1], 4)}] | {leg['mean'] / t1['gap']:+.1%} | "
                 f"{'pass' if leg['pass'] else 'fail'} (not in verdict) |")
    lines += ["", "| Other paired comparisons (A - B) | Metric | Mean | 95% interval |", "|---|---|---|---|"]
    for t in res["supplementary"]:
        lines.append(f"| {t['a']} - {t['b']} | {t['metric']} | {fmt(t['mean'], 4)} | [{fmt(t['ci'][0], 4)}, {fmt(t['ci'][1], 4)}] |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewards", nargs="+", default=list(REWARDS), choices=REWARDS)
    args = ap.parse_args()
    command = ".venv/bin/python scripts/audit_rewards.py " + shlex.join(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    search = partitions.load("search")
    verdicts = {}
    for name in args.rewards:
        t0 = time.time()
        res = run_candidate(name, search)
        res["command"], res["wall_seconds"] = command, round(time.time() - t0, 1)
        (OUT / f"rewards_{name}.json").write_text(json.dumps(res, indent=2))
        md = markdown(res, command)
        (OUT / f"rewards_{name}.md").write_text(md)
        print(md)
        verdicts[name] = res["acceptance"]
    path = OUT / "acceptance.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(verdicts)
    path.write_text(json.dumps(merged, indent=2))
    for name, a in merged.items():
        print(f"{name:14s} T1 {'PASS' if a['T1_duration_probes_within_tolerance']['pass'] else 'FAIL'}  "
              f"T2 {'PASS' if a['T2_personalized_beats_nonpersonalized']['pass'] else 'FAIL'}  "
              f"verdict {'PASS' if a['both_pass'] else 'FAIL'}   (legacy T1 "
              f"{'pass' if a['T1_legacy_shortest_not_above_random']['pass'] else 'fail'})")


if __name__ == "__main__":
    main()
