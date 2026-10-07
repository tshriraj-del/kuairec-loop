"""Differential check: identical recommendation sequences fed to our KuaiEnv and to
EasyRL4Rec's KuaiEnv (commit pinned in third_party/EasyRL4Rec), every difference in
reward or quit step reported with its cause.

Reference loading, with no edits to EasyRL4Rec's logic files:
  * its own preprocessing (preprocessing_kuairec.py) and KuaiData.load_mat/load_category
    run unmodified in a scratch workspace with symlinks to the raw CSVs;
  * `src.core` is registered as a bare package so its __init__ (which imports torch,
    tianshou and the policy zoo) does not run; submodules load from their real files;
  * `torch` is stubbed with manual_seed only (BaseEnv.seed is never called here);
  * `numba.njit` is an identity decorator (the decorated functions are not on this path).

Sequence sources, per search-partition user (no repeats, length max_turn):
  random      uniform permutations, 3 per user
  popularity  the GlobalPopularity ranking
  diverse     random items that avoid every category in the previous quit_window items,
              to drive sessions to max_turn
  stress      items that share one category, to fire the quit rule at every t
  capped      starts with that user's videos whose watch_ratio > 5, then random

Causes:
  cap_at_5               EasyRL4Rec clips watch_ratio at 5 (KuaiData.load_mat); we return raw
  negative_slice_window  EasyRL4Rec's window seq[t-N:t] wraps for 0 < t < N; the intended
                         window disagrees on whether the user quits at this step
  UNEXPLAINED            anything else (the test fails on any of these)

Usage: .venv/bin/python scripts/differential.py [--users N]
Writes runs/differential_differences.csv and runs/differential_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REF_ROOT = ROOT / "third_party" / "EasyRL4Rec"
WORKSPACE = ROOT / "data" / "cache" / "easyrl4rec_ws"

from kuairec_eval import partitions  # noqa: E402
from kuairec_eval.data import data_dir, load_small_matrix  # noqa: E402
from kuairec_eval.env import EnvConfig, KuaiEnv, boredom_quit  # noqa: E402
from policies.baselines import GlobalPopularity  # noqa: E402


def load_reference(cfg: EnvConfig):
    raw = WORKSPACE / "data" / "KuaiRec" / "data_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("big_matrix.csv", "small_matrix.csv", "item_categories.csv"):
        link = raw / name
        if not link.exists():
            link.symlink_to(data_dir() / name)
    import scipy.stats  # noqa: F401  (import before the stub: scipy probes sys.modules["torch"])
    import sklearn.preprocessing  # noqa: F401

    torch_stub = types.ModuleType("torch")
    torch_stub.manual_seed = lambda *_a, **_k: None
    torch_stub.Tensor = type("Tensor", (), {})
    numba_stub = types.ModuleType("numba")
    numba_stub.njit = lambda f=None, **_k: f if f is not None else (lambda g: g)
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("numba", numba_stub)
    sys.path.insert(0, str(REF_ROOT))
    for pkg, path in (("src", REF_ROOT / "src"), ("src.core", REF_ROOT / "src" / "core")):
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(path)]
        sys.modules[pkg] = mod
    cwd = os.getcwd()
    os.chdir(WORKSPACE)
    saved_argv, sys.argv = sys.argv, [sys.argv[0]]  # their argparse reads sys.argv
    try:
        from src.core.envs.KuaiRec.KuaiData import KuaiData
        from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv as RefKuaiEnv
        mat, lbe_user, lbe_item = KuaiData.load_mat()
        list_feat, _ = KuaiData.load_category()
    finally:
        os.chdir(cwd)
        sys.argv = saved_argv
    env = RefKuaiEnv(mat=mat, lbe_user=lbe_user, lbe_item=lbe_item, list_feat=list_feat,
                     df_dist_small=None, num_leave_compute=cfg.quit_window,
                     leave_threshold=cfg.quit_threshold, max_turn=cfg.max_turn)
    return env, lbe_user, lbe_item


def run_ours(env: KuaiEnv, user: int, seq: list[int]):
    env.reset(user)
    rewards, quit_t = [], None
    for t, v in enumerate(seq):
        r = env.step(v)
        rewards.append(r.reward)
        if r.done:
            quit_t = t
            break
    return rewards, quit_t


def run_ref(ref, uidx: int, seq_idx: list[int]):
    ref.reset()
    ref.cur_user = uidx
    rewards, quit_t = [], None
    for t, a in enumerate(seq_idx):
        _, r, terminated, _, _ = ref.step(a)
        rewards.append(float(r))
        if terminated:
            quit_t = t
            break
    return rewards, quit_t


def build_sequences(users, catalog, cats, small_lookup, cfg, rng):
    pop = list(GlobalPopularity(types.SimpleNamespace(catalog=catalog), 0).ranking[: cfg.max_turn])
    by_cat: dict[int, list[int]] = {}
    for v in catalog:
        for c in cats[v]:
            by_cat.setdefault(c, []).append(v)
    big_cats = [c for c, vs in by_cat.items() if len(vs) >= cfg.max_turn]
    arr = np.asarray(catalog)
    for u in users:
        for k in range(3):
            yield "random", u, rng.permutation(arr)[: cfg.max_turn].tolist()
        yield "popularity", u, pop
        seq, banned_hist = [], []
        for v in rng.permutation(arr).tolist():
            recent = set().union(*banned_hist[-cfg.quit_window:]) if banned_hist else set()
            if not recent & set(cats[v]):
                seq.append(v)
                banned_hist.append(set(cats[v]))
            if len(seq) == cfg.max_turn:
                break
        yield "diverse", u, seq
        c = big_cats[int(rng.integers(len(big_cats)))]
        yield "stress", u, rng.permutation(np.asarray(by_cat[c]))[: cfg.max_turn].tolist()
        high = [v for v in catalog if small_lookup.get((u, v), 0) > 5]
        rest = [v for v in rng.permutation(arr).tolist() if v not in set(high)]
        yield "capped", u, (high + rest)[: cfg.max_turn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=0, help="limit to the first N search users (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs", help="directory for the CSV and summary")
    args = ap.parse_args()
    command = ".venv/bin/python scripts/differential.py " + shlex.join(sys.argv[1:])

    base = EnvConfig.from_toml()
    ours = {m: KuaiEnv(EnvConfig(**{**base.__dict__, "window_mode": m})) for m in ("intended", "easyrl4rec_slice")}
    ref, lbe_user, lbe_item = load_reference(base)
    env0 = ours["intended"]
    catalog, cats = env0.catalog, env0.categories
    assert list(lbe_item.classes_) == list(catalog) and list(lbe_user.classes_) == list(env0.users)
    item_idx = {int(v): i for i, v in enumerate(lbe_item.classes_)}
    user_idx = {int(u): i for i, u in enumerate(lbe_user.classes_)}
    small = load_small_matrix()
    raw = dict(zip(zip(small.user_id.tolist(), small.video_id.tolist()), small.watch_ratio.tolist()))
    del small

    users = partitions.load("search")
    if args.users:
        users = users[: args.users]
    rng = np.random.default_rng(args.seed)
    rows, n_seq, n_steps, per_source, ref_end = [], 0, 0, Counter(), Counter()
    for source, u, seq in build_sequences(users, catalog, cats, raw, base, rng):
        n_seq += 1
        per_source[source] += 1
        ref_r, ref_q = run_ref(ref, user_idx[u], [item_idx[v] for v in seq])
        n_steps += len(ref_r)
        ref_end[(source, "max_turn" if ref_q == base.max_turn - 1 else "boredom")] += 1
        for mode, env in ours.items():
            our_r, our_q = run_ours(env, u, seq)
            for t in range(min(len(our_r), len(ref_r))):
                if our_r[t] != ref_r[t]:
                    rv = raw.get((u, seq[t]))
                    cause = "cap_at_5" if rv is not None and rv > 5 and ref_r[t] == 5.0 and our_r[t] == rv else "UNEXPLAINED"
                    rows.append([source, u, mode, "reward", t, seq[t], our_r[t], ref_r[t], cause])
            if our_q != ref_q:
                t = min(x for x in (our_q, ref_q) if x is not None)
                at_cap = t >= base.max_turn - 1
                i = boredom_quit(seq[:t], seq[t], cats, base.quit_window, base.quit_threshold, "intended")
                s = boredom_quit(seq[:t], seq[t], cats, base.quit_window, base.quit_threshold, "easyrl4rec_slice")
                cause = "negative_slice_window" if (i != s and not at_cap and 0 < t < base.quit_window) else "UNEXPLAINED"
                rows.append([source, u, mode, "quit_step", t, seq[t], our_q, ref_q, cause])

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "differential_differences.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "user", "our_window_mode", "kind", "step", "video", "ours", "reference", "cause"])
        w.writerows(rows)
    table = Counter((r[2], r[3], r[8]) for r in rows)
    summary = {
        "command": command,
        "reference_commit": (REF_ROOT / ".git").exists() and os.popen(f"git -C '{REF_ROOT}' rev-parse HEAD").read().strip(),
        "env_config": base.__dict__,
        "users": len(users),
        "sequences": n_seq,
        "sequences_per_source": dict(per_source),
        "reference_steps": n_steps,
        "reference_session_end": {f"{k[0]}/{k[1]}": n for k, n in sorted(ref_end.items())},
        "differences": [{"our_window_mode": m, "kind": k, "cause": c, "count": n} for (m, k, c), n in sorted(table.items())],
        "unexplained": sum(1 for r in rows if r[8] == "UNEXPLAINED"),
    }
    (out / "differential_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = main()
    sys.exit(1 if s["unexplained"] else 0)
