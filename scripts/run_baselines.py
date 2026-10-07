"""Run the Phase 0 baselines through the frozen harness and write runs/<partition>_baselines.{json,md}.

Usage: .venv/bin/python scripts/run_baselines.py --partition search
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kuairec_eval import harness  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from policies.baselines import REGISTRY  # noqa: E402

ROWS = [("mean_watch_ratio", "Mean watch_ratio of recommended videos", 4),
        ("cum_reward", "Cumulative reward per session", 3),
        ("session_length", "Session length", 3),
        ("top_category_share", "Share in most-recommended category", 4),
        ("catalog_coverage", "Catalog coverage", 4),
        ("missing_pair_rate", "(diag) unrecorded-pair rate", 5),
        ("boredom_quit_rate", "(diag) sessions ended by boredom", 4),
        ("max_turn_rate", "(diag) sessions ended by max_turn", 4)]


def fmt(x, d):
    return "n/a" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="search", choices=["search", "shadow"])
    ap.add_argument("--policies", nargs="+", default=list(REGISTRY))
    args = ap.parse_args()
    command = ".venv/bin/python scripts/run_baselines.py " + shlex.join(sys.argv[1:])
    env = KuaiEnv()
    results = [harness.run(REGISTRY[p], args.partition, env=env, policy_name=p) for p in args.policies]
    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "uncommitted"
    payload = {"command": command, "git_commit": commit, "results": results}
    (out / f"{args.partition}_baselines.json").write_text(json.dumps(payload, indent=2))

    r0 = results[0]
    lines = [f"Command: `{command}`  ",
             f"Partition: {args.partition} ({r0['n_users']} users), seeds {r0['seeds']}, "
             f"manifest digest `{r0['manifest_digest'][:16]}`, env {r0['env_config']}", "",
             "Each cell: mean over seeds, [95% t-interval over seeds], {95% user-bootstrap interval where defined}.", "",
             "| Metric | " + " | ".join(r["policy"] for r in results) + " |",
             "|---|" + "---|" * len(results)]
    for key, label, d in ROWS:
        cells = []
        for r in results:
            s = r["summary"][key]
            cell = f"{fmt(s['mean'], d)} [{fmt(s['ci_seeds'][0], d)}, {fmt(s['ci_seeds'][1], d)}]"
            if "ci_users" in s:
                cell += f" {{{fmt(s['ci_users'][0], d)}, {fmt(s['ci_users'][1], d)}}}"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    (out / f"{args.partition}_baselines.md").write_text(md)
    print(md)
    for r in results:
        print(f"{r['policy']}: {r['wall_seconds']}s")


if __name__ == "__main__":
    main()
