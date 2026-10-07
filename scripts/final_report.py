"""The only program allowed to read the holdout partition (kuairec_eval.partitions.load).

Phase 0 ships it as a guarded stub: it refuses to run without --confirm-final, so the
holdout is not touched by accident while models are still being selected.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kuairec_eval import harness  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from policies.baselines import REGISTRY  # noqa: E402
from policies.reference import PERMANENT_ROWS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-final", action="store_true")
    ap.add_argument("--policy", choices=sorted(REGISTRY), required=True)
    args = ap.parse_args()
    if not args.confirm_final:
        sys.exit("refusing to read the holdout without --confirm-final")
    env = KuaiEnv()
    rows = [(args.policy, REGISTRY[args.policy])] + [(k, f) for k, _, f in PERMANENT_ROWS]  # ADR-0029
    results = {k: harness.run(f, "holdout", env=env, policy_name=k)["summary"] for k, f in rows}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
