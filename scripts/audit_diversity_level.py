"""Does tolerating repetition pay? The category-affinity check behind claim 2 (ADR-0028).

Under each quit profile (looser, primary, tighter), with the frozen reward (duration_pct), on the
SEARCH partition only:
  * the ladder: random, the PERMANENT_ROWS, the category-affinity oracle in its three modes
    (full_cycle, strict, free), the non-personalized and the personalized oracle;
  * personalization gap = harness.compare(personalized, non-personalized), cumulative reward;
  * both readings of "strict rotation", side by side, each as harness.compare(free, reading) and
    as a share of that profile's gap (point estimate and interval divided by the gap's point
    estimate):
      full_cycle  (the baseline, C1) no category repeated until every rotation category is shown
      strict      (window reading)   no category repeated within the quit window
  * D1, under tighter only: the free oracle built for the primary setting, run in the tighter
    environment, against the same oracle built for tighter (the loss from a wrongly learned
    repeat tolerance).

Decision rule (yours; the 10% threshold was set before the check, the reading after it):
  free - full_cycle under looser >= 10% of the gap
      ->  claim 2 stands: "from the session-ended signal alone, the system learns how soon a
          category can be revisited and how many repeats are tolerated, measured against the
          full-cycle policy"
  otherwise
      ->  strict rotation (the full cycle) is close to optimal everywhere; claim 2 returns to "a
          greedy personalized policy narrows and the diversity layer fixes it"; the three settings
          are robustness checks only

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
BASELINE = "full_cycle"
READINGS = {"full_cycle": "full cycle: no category repeated until every rotation category has been shown",
            "strict": "window: no category repeated within the quit window"}
CLAIM_STANDS = ("from the session-ended signal alone, the system learns how soon a category can be revisited and "
                "how many repeats are tolerated, measured against the full-cycle policy")
CLAIM_RETURNS = "a greedy personalized policy narrows and the diversity layer fixes it"
D1 = {"built_for": "primary", "run_under": "tighter"}


def baseline_reason(n_categories: int) -> str:
    return ("the quit parameters are hidden from policies, so a policy that has learned nothing cannot run "
            "window-based rotation; its only safe option is the full cycle through all "
            f"{n_categories} rotation categories")


def fmt(x, d=3):
    return f"{x:.{d}f}"


def _share(d: dict, gap: dict) -> dict:
    return {**d, "share_of_gap": d["mean"] / gap["mean"], "ci_share_of_gap": [x / gap["mean"] for x in d["ci"]]}


def run_profile(profile: str, search: list[int]) -> dict:
    env = KuaiEnv(EnvConfig.from_toml(quit_profile=profile))
    cfg = env.config
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    mat = reward_matrix(cfg.reward_config(), users, videos)
    specs = [("random", "random", UniformRandom)] + [(k, f"{label} [permanent]", f) for k, label, f in PERMANENT_ROWS]
    affinity = {mode: oracles.category_affinity_factory(mat, users, videos, env.categories, cfg, mode)
                for mode in ("full_cycle", "strict", "free")}
    specs += [("oracle_affinity_full_cycle", "oracle: category affinity, full cycle (baseline reading)", affinity["full_cycle"]),
              ("oracle_affinity_strict", "oracle: category affinity, strict (window reading)", affinity["strict"]),
              ("oracle_affinity_free", "oracle: category affinity, free", affinity["free"])]
    if profile == D1["run_under"]:
        built = EnvConfig.from_toml(quit_profile=D1["built_for"])
        specs.append(("oracle_affinity_free_built_for_primary", "oracle: category affinity, free, built for primary (D1)",
                      oracles.category_affinity_factory(mat, users, videos, env.categories, built, "free")))
    specs += [("oracle_nonpersonalized", "oracle: non-personalized (common values, greedy)",
               oracles.nonpersonalized_factory(mat, users, videos, cfg, search)),
              ("oracle_personalized", "oracle: personalized greedy", oracles.personalized_factory(mat, users, videos, cfg))]
    runs = {k: harness.run(f, "search", env=env, policy_name=k) for k, _, f in specs}
    gap = harness.compare(runs["oracle_personalized"], runs["oracle_nonpersonalized"], "cum_reward")
    out = {"profile": profile, "env_config": runs["random"]["env_config"],
           "manifest_digest": runs["random"]["manifest_digest"], "gap": gap,
           "n_rotation_categories": len(affinity["full_cycle"].categories),
           "labels": {k: lbl for k, lbl, _ in specs},
           "summaries": {k: {"stochastic": r["stochastic"], "seeds": r["seeds"], "summary": r["summary"]}
                         for k, r in runs.items()}}
    for a, b in (("free", "strict"), ("free", "full_cycle"), ("strict", "full_cycle")):
        out[f"{a}_minus_{b}"] = _share(harness.compare(runs[f"oracle_affinity_{a}"], runs[f"oracle_affinity_{b}"],
                                                       "cum_reward"), gap)
    if profile == D1["run_under"]:
        mis = runs["oracle_affinity_free_built_for_primary"]
        out["d1_tolerance_mismatch"] = {
            **D1,
            "loss": _share(harness.compare(runs["oracle_affinity_free"], mis, "cum_reward"), gap),
            "session_length_loss": harness.compare(runs["oracle_affinity_free"], mis, "session_length"),
            "mismatched_minus_full_cycle": _share(harness.compare(mis, runs["oracle_affinity_full_cycle"], "cum_reward"), gap),
        }
    return out


def decide(results: dict) -> dict:
    """Claim-2 decision with the full-cycle baseline; both readings kept side by side."""
    readings = {}
    for reading in ("full_cycle", "strict"):
        shares = {p: r[f"free_minus_{reading}"]["share_of_gap"] for p, r in results.items()}
        readings[reading] = {"description": READINGS[reading], "share_of_gap": shares,
                             "claim2_stands_under_rule": shares["looser"] >= THRESHOLD}
    split = {p: {"revisit_after_window": r["strict_minus_full_cycle"]["share_of_gap"],
                 "tolerate_repeats": r["free_minus_strict"]["share_of_gap"]} for p, r in results.items()}
    reason = baseline_reason(results["looser"]["n_rotation_categories"])
    share = readings[BASELINE]["share_of_gap"]["looser"]
    stands = readings[BASELINE]["claim2_stands_under_rule"]
    head = f"Baseline: the full-cycle policy, because {reason}. Under looser, free minus full cycle is {share:.1%} of the personalization gap"
    if stands:
        text = (f"{head} (>= {THRESHOLD:.0%}): claim 2 stands, reworded: \"{CLAIM_STANDS}\". Split: "
                f"{split['looser']['revisit_after_window']:.1%} of the gap from revisiting after the window, "
                f"{split['looser']['tolerate_repeats']:.1%} from tolerating repeats under the looser setting "
                f"({split['primary']['tolerate_repeats']:.1%} under the primary setting).")
    else:
        text = (f"{head} (< {THRESHOLD:.0%}): strict rotation (the full cycle) is close to optimal everywhere; claim 2 "
                f"returns to \"{CLAIM_RETURNS}\"; the three settings are robustness checks only.")
    return {"threshold": THRESHOLD, "threshold_set": "before the check", "baseline": BASELINE,
            "baseline_reason": reason, "reading_chosen": "after the check (decision C1, 2026-10-06)",
            "readings": readings, "split": split, "claim2_stands": stands, "text": text}


def markdown(results: dict, decision: dict, command: str) -> str:
    lines = ["### Diversity-level check (category-affinity oracle, duration_pct, search partition)", "",
             f"Command: `{command}`", "",
             "| Profile | N, n_Q | Full cycle | Strict | Free | Gap P - NP [95%] |", "|---|---|---|---|---|---|"]
    for p, r in results.items():
        s = r["summaries"]
        cum = lambda k: fmt(s[k]["summary"]["cum_reward"]["mean"])  # noqa: E731
        g = r["gap"]
        lines.append(f"| {p} | {r['env_config']['quit_window']}, {r['env_config']['quit_threshold']} | "
                     f"{cum('oracle_affinity_full_cycle')} | {cum('oracle_affinity_strict')} | {cum('oracle_affinity_free')} | "
                     f"{fmt(g['mean'])} [{fmt(g['ci'][0])}, {fmt(g['ci'][1])}] |")
    lines += ["", "Both readings side by side (free minus the reading; cumulative reward [95%], share of gap [95%]):", "",
              "| Profile | Free - full cycle (baseline) | Free - strict (window reading) |", "|---|---|---|"]
    for p, r in results.items():
        cells = []
        for key in ("free_minus_full_cycle", "free_minus_strict"):
            d = r[key]
            cells.append(f"{fmt(d['mean'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}] ({d['share_of_gap']:.1%} "
                         f"[{d['ci_share_of_gap'][0]:.1%}, {d['ci_share_of_gap'][1]:.1%}])")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    lines += ["", "Split of free minus full cycle:", "",
              "| Profile | Revisiting after the window: strict - full cycle | Tolerating repeats: free - strict |", "|---|---|---|"]
    for p, r in results.items():
        c, d = r["strict_minus_full_cycle"], r["free_minus_strict"]
        lines.append(f"| {p} | {fmt(c['mean'])} [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}] ({c['share_of_gap']:.1%}) | "
                     f"{fmt(d['mean'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}] ({d['share_of_gap']:.1%}) |")
    lines += ["", "| Reading | Share under looser | Claim 2 under the 10% rule |", "|---|---|---|"]
    for name, rd in decision["readings"].items():
        tag = " (baseline)" if name == decision["baseline"] else ""
        lines.append(f"| {rd['description']}{tag} | {rd['share_of_gap']['looser']:.1%} | "
                     f"{'stands' if rd['claim2_stands_under_rule'] else 'returns'} |")
    lines += ["", f"**Decision:** {decision['text']}", "",
              f"The {decision['threshold']:.0%} rule was set {decision['threshold_set']}; the choice of reading was made "
              f"{decision['reading_chosen']}.", ""]
    d1 = results[D1["run_under"]]["d1_tolerance_mismatch"]
    s = results[D1["run_under"]]["summaries"]
    lines += [f"#### D1: the free oracle built for {d1['built_for']}, run under {d1['run_under']}", "",
              "| Oracle (run under tighter) | Cum reward / session | Session length | Boredom-quit rate |", "|---|---|---|---|"]
    for key, label in (("oracle_affinity_free", "free, built for tighter"),
                       ("oracle_affinity_free_built_for_primary", "free, built for primary"),
                       ("oracle_affinity_full_cycle", "full cycle (baseline)")):
        sm = s[key]["summary"]
        lines.append(f"| {label} | {fmt(sm['cum_reward']['mean'])} | {fmt(sm['session_length']['mean'], 2)} | "
                     f"{fmt(sm['boredom_quit_rate']['mean'], 3)} |")
    lo, mb = d1["loss"], d1["mismatched_minus_full_cycle"]
    lines += ["", f"Loss per session (built for tighter minus built for primary): {fmt(lo['mean'])} "
              f"[{fmt(lo['ci'][0])}, {fmt(lo['ci'][1])}], {lo['share_of_gap']:.1%} [{lo['ci_share_of_gap'][0]:.1%}, "
              f"{lo['ci_share_of_gap'][1]:.1%}] of the tighter gap. Built for primary minus full cycle: {fmt(mb['mean'])} "
              f"[{fmt(mb['ci'][0])}, {fmt(mb['ci'][1])}] ({mb['share_of_gap']:.1%}).", ""]
    for p, r in results.items():
        lines += [f"#### Ladder under `{p}` (N={r['env_config']['quit_window']}, n_Q={r['env_config']['quit_threshold']})", "",
                  "| Rung | Cum reward / session {95% users} [95% seeds] | Mean reward / rec | Session length | Boredom-quit rate |",
                  "|---|---|---|---|---|"]
        for k, v in r["summaries"].items():
            sm = v["summary"]
            cr = sm["cum_reward"]
            cell = f"{fmt(cr['mean'])} {{{fmt(cr['ci_users'][0])}, {fmt(cr['ci_users'][1])}}}"
            if cr["ci_seeds"] is not None:
                cell += f" [{fmt(cr['ci_seeds'][0])}, {fmt(cr['ci_seeds'][1])}]"
            lines.append(f"| {r['labels'][k]} | {cell} | {fmt(sm['mean_reward']['mean'], 4)} | "
                         f"{fmt(sm['session_length']['mean'], 2)} | {fmt(sm['boredom_quit_rate']['mean'], 3)} |")
        lines.append("")
    return "\n".join(lines)


def main():
    command = ".venv/bin/python scripts/audit_diversity_level.py " + shlex.join(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    search = partitions.load("search")
    results = {p: run_profile(p, search) for p in PROFILES}
    decision = decide(results)
    payload = {"command": command, "decision": decision, "profiles": results}
    (OUT / "diversity_level.json").write_text(json.dumps(payload, indent=2))
    md = markdown(results, decision, command)
    (OUT / "diversity_level.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
