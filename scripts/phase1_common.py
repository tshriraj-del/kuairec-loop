"""Phase 1 run plumbing shared by the environment scripts (not policy code).

  * every environment run must be declared in runs/phase1/env_grid.json, written by
    scripts/phase1_declare.py before the first run; a run that is not declared, or a second
    shadow run of the same row, raises;
  * every run appends one ledger line per seed it executed (runs/phase1/ledger.jsonl):
    policy, config hash, seed, partition, role, grid and manifest hashes, git commit, command;
  * a recording wrapper keeps every recommendation for the duration profile;
  * ratio_bootstrap() gives paired intervals for ratios such as the gap captured.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kuairec_eval import harness  # noqa: E402
from policies.phase1.config import config_hash, file_hash, load  # noqa: E402

OUT = ROOT / "runs" / "phase1"
GRID = OUT / "env_grid.json"
LEDGER = OUT / "ledger.jsonl"


class _Recorded:
    def __init__(self, inner, seed: int, log: list):
        self._inner, self._seed, self._log = inner, seed, log
        self.stochastic = inner.stochastic
        self._user = None

    def begin_session(self, user, excluded):
        self._user = user
        self._inner.begin_session(user, excluded)

    def act(self):
        v = self._inner.act()
        self._log.append((self._seed, self._user, int(v)))
        return v

    def observe(self, video, result):
        self._inner.observe(video, result)


def recording(factory, log: list):
    return lambda context, seed: _Recorded(factory(context, seed), seed, log)


def ledger_entries() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def declared(name: str, partition: str) -> dict:
    grid = json.loads(GRID.read_text())
    for row in grid["runs"]:
        if row["name"] == name and row["partition"] == partition:
            return row
    raise PermissionError(f"environment run {name!r} on {partition!r} is not in the declared grid {GRID.name}")


def run(env, name: str, factory, partition: str, config_hash_value: str) -> tuple[dict, list]:
    row = declared(name, partition)
    if row["config_hash"] != config_hash_value:
        raise PermissionError(f"{name}: config hash {config_hash_value} differs from the declared {row['config_hash']}")
    if partition == "shadow" and any(e["policy"] == name and e["partition"] == "shadow" for e in ledger_entries()):
        raise PermissionError(f"{name} already ran on shadow; the shadow partition is used once")
    if partition == "shadow" and row.get("condition"):
        scorers = {r["name"] for r in json.loads(GRID.read_text())["runs"] if r.get("condition")}
        if any(e["partition"] == "shadow" and e["policy"] in scorers for e in ledger_entries()):
            raise PermissionError("a scorer already ran on shadow; exactly one scorer runs there")
    log: list = []
    res = harness.run(recording(factory, log), partition, env=env, policy_name=name)
    seeds = [(s, "reported") for s in res["seeds"]]
    if res.get("determinism_check_seed") is not None:
        seeds.append((res["determinism_check_seed"], "determinism_check"))
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    with open(LEDGER, "a") as fh:
        for seed, role in seeds:
            fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "policy": name,
                                 "config_hash": config_hash_value, "seed": seed, "partition": partition, "role": role,
                                 "grid_sha16": file_hash(GRID), "manifest_digest16": res["manifest_digest"][:16],
                                 "git_head": head, "command": " ".join(shlex.quote(a) for a in sys.argv)}) + "\n")
    return res, log


def duration_profile(log: list, seeds: list[int], catalog: np.ndarray) -> dict:
    """Mean catalog duration percentile (mid-rank among the catalog's canonical durations) and
    median duration of the recommended videos, over the reported seeds."""
    from scipy.stats import rankdata
    from policies.phase1.features import durations
    dur_ms = durations()[0][catalog]
    pct = (rankdata(dur_ms, method="average") - 0.5) / len(dur_ms)
    pos = {int(v): i for i, v in enumerate(catalog)}
    vids = [v for s, _, v in log if s in set(seeds)]
    idx = np.array([pos[v] for v in vids])
    return {"recommendations": int(len(idx)), "mean_duration_percentile": float(pct[idx].mean()),
            "catalog_mean_duration_percentile": float(pct.mean()),
            "median_duration_s": float(np.median(dur_ms[idx]) / 1000),
            "catalog_median_duration_s": float(np.median(dur_ms) / 1000)}


def per_user(res: dict, metric: str = "cum_reward") -> np.ndarray:
    return np.asarray(res["per_user"][metric], dtype=float)


def ratio_bootstrap(num_a, num_b, den_a, den_b) -> dict:
    """Paired bootstrap over users of mean(num_a - num_b) / mean(den_a - den_b)."""
    cfg = load()["eval"]
    num = np.asarray(num_a) - np.asarray(num_b)
    den = np.asarray(den_a) - np.asarray(den_b)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n, out = len(num), []
    for start in range(0, cfg["paired_bootstrap_resamples"], 1000):
        idx = rng.integers(0, n, size=(min(1000, cfg["paired_bootstrap_resamples"] - start), n))
        out.append(num[idx].mean(1) / den[idx].mean(1))
    boot = np.concatenate(out)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"point": float(num.mean() / den.mean()), "ci": [float(lo), float(hi)],
            "resamples": int(len(boot)), "users": int(n)}


def spec_hash(spec: dict) -> str:
    return config_hash(spec)


def ladder_specs(env, matrix: np.ndarray, selection_users: list[int]) -> list[tuple[str, str, object, dict]]:
    """The reference ladder (brief: Goal): random, the PERMANENT_ROWS (rotation-random and the four
    duration probes), the category-affinity oracle with window rotation and free within the quit
    rule, the non-personalized and the personalized oracle. Returns (name, label, factory, spec)."""
    from kuairec_eval.diagnostics import oracles
    from policies.baselines import UniformRandom
    from policies.reference import PERMANENT_ROWS
    users, videos = np.asarray(env.users), np.asarray(env.catalog)
    specs = [("random", "random", UniformRandom, {"module": "policies.baselines.UniformRandom"})]
    specs += [(k, label, f, {"module": "policies.reference", "row": k}) for k, label, f in PERMANENT_ROWS]
    for mode, label in (("strict", "category-affinity oracle, window rotation"),
                        ("free", "category-affinity oracle, free within the quit rule")):
        specs.append((f"oracle_affinity_{mode}", label,
                      oracles.category_affinity_factory(matrix, users, videos, env.categories, env.config, mode),
                      {"module": "kuairec_eval.diagnostics.oracles.category_affinity_factory", "mode": mode}))
    specs.append(("oracle_nonpersonalized", "non-personalized oracle",
                  oracles.nonpersonalized_factory(matrix, users, videos, env.config, selection_users),
                  {"module": "kuairec_eval.diagnostics.oracles.nonpersonalized_factory", "selection": "partition users"}))
    specs.append(("oracle_personalized", "personalized oracle",
                  oracles.personalized_factory(matrix, users, videos, env.config),
                  {"module": "kuairec_eval.diagnostics.oracles.personalized_factory"}))
    return specs


LADDER_NAMES = ["random", "rotation_random", "shortest_first", "longest_first", "rotation_shortest", "rotation_longest",
                "oracle_affinity_strict", "oracle_affinity_free", "oracle_nonpersonalized", "oracle_personalized"]
PROBES = ["shortest_first", "longest_first", "rotation_shortest", "rotation_longest"]


def ladder_hash(name: str, manifest16: str) -> str:
    return spec_hash({"row": name, "manifest16": manifest16})


def scorer_hash(name: str) -> str:
    """Hash of a scorer's selected configuration, the slate rule knowledge and its artifacts."""
    from policies.phase1.config import CACHE
    meta = json.loads((CACHE / "final" / "meta.json").read_text())
    arts = meta["artifacts_sha16"]
    base = {k: arts[f"{k}.pkl"] for k in ("affinity", "item_knn", "ease", "prior")}
    spec = {"name": name, "slate": load()["slate"], "label": meta["label"], "pairs": meta["pairs_file_sha16"]}
    if name in ("affinity", "item_knn", "ease"):
        spec.update({"params": meta["selected"][name]["params"], "artifact": arts[f"{name}.pkl"]})
    else:
        spec.update({"ranker": file_hash(CACHE / "final" / f"{name}.pkl"), "base_artifacts": base,
                     "text": file_hash(CACHE / "caption_emb.npz") if name == "ranker_text" else None})
    return spec_hash(spec)
