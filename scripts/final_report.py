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
from policies.baselines import REGISTRY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-final", action="store_true")
    ap.add_argument("--policy", choices=sorted(REGISTRY), required=True)
    args = ap.parse_args()
    if not args.confirm_final:
        sys.exit("refusing to read the holdout without --confirm-final")
    result = harness.run(REGISTRY[args.policy], "holdout", policy_name=args.policy)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
