"""Phase 1 environment runs (brief items 5, 6, 8 and the gate), primary quit setting.

  ladder  --partition search   regenerate the reference ladder (no number copied from the audit)
  scorers                      the six scorers on search; comparisons with paired bootstraps
  contrasts                    no environment run: the ranker variant contrasts (duration, text)
                               as paired bootstraps over the stored per-user values
  shadow                       the best scorer on search, and the ladder, once on shadow; the gate

Every run must be declared in runs/phase1/env_grid.json (scripts/phase1_declare.py) and is
written to the run ledger (runs/phase1/ledger.jsonl) by phase1_common.run().

Usage: .venv/bin/python scripts/phase1_env.py {ladder --partition search|scorers|contrasts|shadow}
Writes runs/phase1/ladder_<partition>.json, scorers_search.json, contrasts_search.json, final_shadow.json
"""
import lightgbm  # noqa: F401  (before anything that could import torch)
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import phase1_common as C  # noqa: E402
from kuairec_eval import harness, partitions  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from kuairec_eval.rewards import reward_matrix  # noqa: E402
from policies.phase1 import serving  # noqa: E402

SUMMARY_KEYS = ("cum_reward", "mean_reward", "session_length", "catalog_coverage", "top_category_share",
                "boredom_quit_rate")


def row_record(name: str, label: str, res: dict, log: list, catalog) -> dict:
    return {"name": name, "label": label, "stochastic": res["stochastic"], "seeds": res["seeds"],
            "determinism_check_seed": res.get("determinism_check_seed"),
            "summary": {k: res["summary"][k] for k in SUMMARY_KEYS},
            "duration_profile": C.duration_profile(log, res["seeds"], catalog),
            "per_user": res["per_user"], "policy": name, "manifest_digest": res["manifest_digest"]}


def run_ladder(env, partition: str) -> dict:
    users = partitions.load(partition)
    mat = reward_matrix(env.config.reward_config(), np.asarray(env.users), np.asarray(env.catalog))
    digest16 = json.loads(C.GRID.read_text())["manifest_digest16"]
    rows = {}
    for name, label, factory, _ in C.ladder_specs(env, mat, users):
        res, log = C.run(env, name, factory, partition, C.ladder_hash(name, digest16))
        rows[name] = row_record(name, label, res, log, np.asarray(env.catalog))
        print(f"{partition:7s} {name:24s} cum {rows[name]['summary']['cum_reward']['mean']:.3f}", flush=True)
    del mat
    return rows


def compare(a: dict, b: dict) -> dict:
    return harness.compare(a, b, "cum_reward")


def stronger_probe(ladder: dict) -> str:
    return max(C.PROBES, key=lambda n: ladder[n]["summary"]["cum_reward"]["mean"])


def comparisons(rows: dict, ladder: dict, probe: str, base_scorer: str | None) -> dict:
    P, NP = ladder["oracle_personalized"], ladder["oracle_nonpersonalized"]
    gap = compare(P, NP)
    out = {"gap": gap, "stronger_probe": probe, "per_scorer": {}}
    for name, r in rows.items():
        d = {"vs_rotation_random": compare(r, ladder["rotation_random"]),
             "vs_stronger_probe": compare(r, ladder[probe]),
             "vs_nonpersonalized_oracle": compare(r, NP),
             "gap_captured": C.ratio_bootstrap(C.per_user(r), C.per_user(NP), C.per_user(P), C.per_user(NP))}
        if base_scorer and name != base_scorer and base_scorer in rows:
            d["vs_scorer_a"] = compare(r, rows[base_scorer])
        if name == base_scorer:
            for mode in ("strict", "free"):
                o = ladder[f"oracle_affinity_{mode}"]
                d[f"share_of_affinity_{mode}_lift"] = C.ratio_bootstrap(
                    C.per_user(r), C.per_user(ladder["rotation_random"]), C.per_user(o), C.per_user(ladder["rotation_random"]))
        out["per_scorer"][name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ladder", "scorers", "contrasts", "shadow"])
    ap.add_argument("--partition", default="search", choices=["search"])
    args = ap.parse_args()
    if args.stage == "contrasts":
        rows = json.loads((C.OUT / "scorers_search.json").read_text())["rows"]
        out = {f"{a} - {b}": {**compare(rows[a], rows[b]), "mean_reward": harness.compare(rows[a], rows[b], "mean_reward")}
               for a, b in (("ranker_duration", "ranker_no_duration"), ("ranker_text", "ranker_duration"),
                            ("ranker_text", "ranker_no_duration"), ("ranker_no_duration", "ease"))}
        (C.OUT / "contrasts_search.json").write_text(json.dumps(out, indent=2))
        for k, v in out.items():
            print(f"{k:42s} cum {v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]   per rec "
                  f"{v['mean_reward']['mean']:+.4f} [{v['mean_reward']['ci'][0]:+.4f}, {v['mean_reward']['ci'][1]:+.4f}]")
        return
    env = KuaiEnv()
    catalog = np.asarray(env.catalog)
    if args.stage == "ladder":
        rows = run_ladder(env, args.partition)
        probe = stronger_probe(rows)
        (C.OUT / f"ladder_{args.partition}.json").write_text(json.dumps({"rows": rows, "stronger_probe": probe}, indent=2))
        print("stronger duration probe:", probe)
        return
    ladder = json.loads((C.OUT / "ladder_search.json").read_text())
    if args.stage == "scorers":
        rows = {}
        for name in serving.SCORERS:
            res, log = C.run(env, name, serving.factory(name), "search", C.scorer_hash(name))
            rows[name] = row_record(name, name, res, log, catalog)
            print(f"search  {name:24s} cum {rows[name]['summary']['cum_reward']['mean']:.3f}", flush=True)
        out = {"rows": rows, "comparisons": comparisons(rows, ladder["rows"], ladder["stronger_probe"], "affinity")}
        (C.OUT / "scorers_search.json").write_text(json.dumps(out, indent=2))
        return
    scorers = json.loads((C.OUT / "scorers_search.json").read_text())["rows"]
    best = max(scorers, key=lambda n: scorers[n]["summary"]["cum_reward"]["mean"])
    shadow_ladder = run_ladder(env, "shadow")
    res, log = C.run(env, best, serving.factory(best), "shadow", C.scorer_hash(best))
    final = {best: row_record(best, best, res, log, catalog)}
    probe = ladder["stronger_probe"]
    comp = comparisons(final, shadow_ladder, probe, None)["per_scorer"][best]
    gap = compare(shadow_ladder["oracle_personalized"], shadow_ladder["oracle_nonpersonalized"])
    beats = lambda d: d["ci"][0] > 0  # noqa: E731
    gate = {"best_scorer": best, "chosen_on": "search cumulative reward", "stronger_probe": probe,
            "beats_rotation_random": beats(comp["vs_rotation_random"]),
            "beats_stronger_probe": beats(comp["vs_stronger_probe"]),
            "claim1_passes": beats(comp["vs_rotation_random"]) and beats(comp["vs_stronger_probe"]),
            "beats_nonpersonalized_oracle": beats(comp["vs_nonpersonalized_oracle"])}
    out = {"ladder": shadow_ladder, "best": final[best], "comparisons": comp, "gap": gap, "gate": gate}
    (C.OUT / "final_shadow.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
