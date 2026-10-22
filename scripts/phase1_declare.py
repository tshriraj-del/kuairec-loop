"""Declare the Phase 1 environment-run grid BEFORE the first environment run (brief item 4).

Refuses to run if the run ledger already has an entry. The grid is small and closed:
  search  the 10 ladder rows and the 6 scorers, each at the configuration chosen offline
  shadow  the 10 ladder rows and ONE scorer: the best of the 6 on search cumulative reward
Rules written into the grid: the stronger duration probe is the one of the four permanent
probes with the higher search cumulative reward; each shadow row runs at most once.

Usage: .venv/bin/python scripts/phase1_declare.py
Writes runs/phase1/env_grid.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kuairec_eval import _integrity  # noqa: E402
from phase1_common import GRID, LADDER_NAMES, LEDGER, ladder_hash, ledger_entries, scorer_hash  # noqa: E402
from policies.phase1.serving import SCORERS  # noqa: E402


def main():
    if ledger_entries():
        sys.exit(f"{LEDGER} already has entries: the grid must be declared before the first environment run")
    digest16 = _integrity.verify()[:16]
    runs = []
    for partition in ("search", "shadow"):
        runs += [{"name": n, "partition": partition, "config_hash": ladder_hash(n, digest16)} for n in LADDER_NAMES]
    for name in SCORERS:
        h = scorer_hash(name)
        runs.append({"name": name, "partition": "search", "config_hash": h})
        runs.append({"name": name, "partition": "shadow", "config_hash": h,
                     "condition": "only if it is the best scorer on search; one scorer on shadow in total"})
    grid = {"declared_at": datetime.now(timezone.utc).isoformat(), "manifest_digest16": digest16,
            "rules": {"stronger_duration_probe": "the higher search cumulative reward among shortest_first, longest_first, "
                                                 "rotation_shortest, rotation_longest",
                      "best_scorer": "the highest search cumulative reward among the six scorers",
                      "shadow": "each shadow row runs once; exactly one scorer runs on shadow",
                      "seeds": "harness rule: stochastic rows seeds 0-9, deterministic rows seed 0 plus the check on seed 1"},
            "runs": runs}
    GRID.write_text(json.dumps(grid, indent=2))
    print(f"declared {len(runs)} environment runs ({sum(r['partition'] == 'search' for r in runs)} on search, "
          f"{sum(r['partition'] == 'shadow' for r in runs)} shadow candidates) at {grid['declared_at']}")


if __name__ == "__main__":
    main()
