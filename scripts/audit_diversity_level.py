"""Does tolerating repetition pay? The category-affinity check behind claim 2 (ADR-0028).

Under each quit profile (looser, primary, tighter), with the frozen reward (duration_pct), on the
SEARCH partition only:
  * the ladder: random, the PERMANENT_ROWS, the category-affinity oracle in its three modes
    (strict, full_cycle, free), the non-personalized and the personalized oracle;
  * personalization gap = harness.compare(personalized, non-personalized), cumulative reward;
  * diversity-level value = harness.compare(affinity free, affinity strict), cumulative reward,
    also as a share of that profile's gap (point estimate and interval divided by the gap's
    point estimate). The same against full_cycle is reported as the alternative reading of
    "strict rotation".

Decision rule (yours, applied to the looser profile with the strict reading):
  share >= 10%  ->  claim 2 stands as "the system learns the diversity level from the
                    session-ended signal"
  share <  10%  ->  strict rotation is close to optimal everywhere; claim 2 returns to "a greedy
                    personalized policy narrows and the diversity layer fixes it"; the three
                    settings are robustness checks only

Usage: .venv/bin/python scripts/audit_diversity_level.py
Writes runs/audit_v1/diversity_level.{json,md}
"""
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from kuairec_eval import harness, partitions  # noqa: E402
from kuairec_eval.diagnostics import oracles  # noqa: E402
from kuairec_eval.env import EnvConfig, KuaiEnv  # noqa: E402
from kuairec_eval.rewards import reward_matrix  # noqa: E402
from policies.baselines import UniformRandom  # noqa: E402
from policies.reference import PERMANENT_ROWS  # noqa: E402

OUT = ROOT / "runs" / "audit_v1"
PROFILES = ("looser", "primary", "tighter")
THRESHOLD = 0.10
CLAIM_STANDS = "the system learns the diversity level from the session-ended signal"
CLAIM_RETURNS = "a greedy personalized policy narrows and the diversity layer fixes it"


def fmt(x, d=3):
    return f"{x:.{d}f}"


def run_profile(profile: str, search: list[int]) -> dict:
    env = KuaiEnv(EnvConfig.from_toml(quit_profile=profile))
    cfg = env.config
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    mat = reward_matrix(cfg.reward_config(), users, videos)
    specs = [("random", "random", UniformRandom)] + [(k, f"{label} [permanent]", f) for k, label, f in PERMANENT_ROWS]
    for mode in ("strict", "full_cycle", "free"):
        specs.append((f"oracle_affinity_{mode}", f"oracle: category affinity, {mode.replace('_', ' ')}",
                      oracles.category_affinity_factory(mat, users, videos, env.categories, cfg, mode)))
    specs += [("oracle_nonpersonalized", "oracle: non-personalized (common values, greedy)",
               oracles.nonpersonalized_factory(mat, users, videos, cfg, search)),
              ("oracle_personalized", "oracle: personalized greedy", oracles.personalized_factory(mat, users, videos, cfg))]
    runs = {k: harness.run(f, "search", env=env, policy_name=k) for k, _, f in specs}
    gap = harness.compare(runs["oracle_personalized"], runs["oracle_nonpersonalized"], "cum_reward")
    out = {"profile": profile, "env_config": runs["random"]["env_config"],
           "manifest_digest": runs["random"]["manifest_digest"], "gap": gap,
           "labels": {k: lbl for k, lbl, _ in specs},
           "summaries": {k: {"stochastic": r["stochastic"], "seeds": r["seeds"], "summary": r["summary"]}
                         for k, r in runs.items()}}
    for a, b in (("free", "strict"), ("free", "full_cycle"), ("strict", "full_cycle")):
        d = harness.compare(runs[f"oracle_affinity_{a}"], runs[f"oracle_affinity_{b}"], "cum_reward")
        out[f"{a}_minus_{b}"] = {**d, "share_of_gap": d["mean"] / gap["mean"],
                                 "ci_share_of_gap": [x / gap["mean"] for x in d["ci"]]}
    return out


def markdown(results: dict, decision: dict, command: str) -> str:
    lines = ["### Diversity-level check (category-affinity oracle, duration_pct, search partition)", "",
             f"Command: `{command}`", "",
             "| Profile | N, n_Q | Affinity strict | Affinity free | Free - strict [95%] | Share of gap [95%] | Gap P - NP [95%] | Free - full cycle (share) |",
             "|---|---|---|---|---|---|---|---|"]
    for p, r in results.items():
        s = r["summaries"]
        d, c, g = r["free_minus_strict"], r["free_minus_full_cycle"], r["gap"]
        lines.append(f"| {p} | {r['env_config']['quit_window']}, {r['env_config']['quit_threshold']} | "
                     f"{fmt(s['oracle_affinity_strict']['summary']['cum_reward']['mean'])} | "
                     f"{fmt(s['oracle_affinity_free']['summary']['cum_reward']['mean'])} | "
                     f"{fmt(d['mean'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}] | "
                     f"{d['share_of_gap']:.1%} [{d['ci_share_of_gap'][0]:.1%}, {d['ci_share_of_gap'][1]:.1%}] | "
                     f"{fmt(g['mean'])} [{fmt(g['ci'][0])}, {fmt(g['ci'][1])}] | {fmt(c['mean'])} ({c['share_of_gap']:.1%}) |")
    lines += ["", "Decomposition of free minus full cycle (cumulative reward, share of gap):", "",
              "| Profile | Concentrating on favourite categories: strict - full cycle | Tolerating repeats: free - strict |",
              "|---|---|---|"]
    for p, r in results.items():
        c, d = r["strict_minus_full_cycle"], r["free_minus_strict"]
        lines.append(f"| {p} | {fmt(c['mean'])} [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}] ({c['share_of_gap']:.1%}) | "
                     f"{fmt(d['mean'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}] ({d['share_of_gap']:.1%}) |")
    lines += ["", f"**Decision (looser, strict reading, threshold {THRESHOLD:.0%}):** {decision['text']}", ""]
    if decision["readings_disagree"]:
        lines += [f"**Flag:** the full-cycle reading of strict rotation gives {decision['share_full_cycle']:.1%} under "
                  "looser, which lands on the other side of the threshold.", ""]
    for p, r in results.items():
        lines += [f"#### Ladder under `{p}` (N={r['env_config']['quit_window']}, n_Q={r['env_config']['quit_threshold']})", "",
                  "| Rung | Cum reward / session {95% users} [95% seeds] | Mean reward / rec | Session length | Boredom-quit rate |",
                  "|---|---|---|---|---|"]
        for k, v in r["summaries"].items():
            s = v["summary"]
            cr = s["cum_reward"]
            cell = f"{fmt(cr['mean'])} {{{fmt(cr['ci_users'][0])}, {fmt(cr['ci_users'][1])}}}"
            if cr["ci_seeds"] is not None:
                cell += f" [{fmt(cr['ci_seeds'][0])}, {fmt(cr['ci_seeds'][1])}]"
            lines.append(f"| {r['labels'][k]} | {cell} | {fmt(s['mean_reward']['mean'], 4)} | "
                         f"{fmt(s['session_length']['mean'], 2)} | {fmt(s['boredom_quit_rate']['mean'], 3)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    command = ".venv/bin/python scripts/audit_diversity_level.py " + shlex.join(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    search = partitions.load("search")
    results = {p: run_profile(p, search) for p in PROFILES}
    share = results["looser"]["free_minus_strict"]["share_of_gap"]
    share_cycle = results["looser"]["free_minus_full_cycle"]["share_of_gap"]
    stands = share >= THRESHOLD
    text = (f"looser-setting difference is {share:.1%} of the personalization gap (>= {THRESHOLD:.0%}): claim 2 stands "
            f"as \"{CLAIM_STANDS}\"." if stands else
            f"looser-setting difference is {share:.1%} of the personalization gap (< {THRESHOLD:.0%}): strict rotation is "
            f"close to optimal everywhere; claim 2 returns to \"{CLAIM_RETURNS}\"; the three settings are robustness "
            "checks only.")
    decision = {"threshold": THRESHOLD, "share_looser_strict": share, "share_full_cycle": share_cycle,
                "claim2_stands": stands, "text": text,
                "readings_disagree": (share_cycle >= THRESHOLD) != stands}
    payload = {"command": command, "decision": decision, "profiles": results}
    (OUT / "diversity_level.json").write_text(json.dumps(payload, indent=2))
    md = markdown(results, decision, command)
    (OUT / "diversity_level.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
