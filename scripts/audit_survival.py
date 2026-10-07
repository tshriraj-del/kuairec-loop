"""Item 8: random-policy survival curves under the quit rule, and the looser/tighter proposal.

The boredom rule depends only on the categories of the recommended sequence, never on the
user or the reward, so the random policy's survival curve is a property of the catalog. It is
computed two ways:
  * exactly as the harness ran it: session lengths of UniformRandom on the search partition,
    seeds 0-9, frozen environment (N=9, n_Q=1, max_turn=30, intended window);
  * by Monte Carlo over the catalog: 100,000 uniform no-repeat sequences of 30 videos
    (numpy default_rng(0)), session length evaluated for every (N, n_Q) on a grid with the
    intended window, vectorised over sequences (checked against env.boredom_quit below).

Selection rule (fixed before looking at the grid, ADR-0019): keep n_Q = 1 and vary N, the
axis CIRS Fig. 8 varies for KuaiEnv. looser = the N whose Monte Carlo median session length
is closest to 2x the primary's; tighter = closest to 0.5x. Ties: the N whose mean length is
closer to the same multiple.

Usage: .venv/bin/python scripts/audit_survival.py [--json runs/audit/survival.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from kuairec_eval import harness  # noqa: E402
from kuairec_eval.env import KuaiEnv, boredom_quit  # noqa: E402
from policies.baselines import UniformRandom  # noqa: E402

N_SEQ, MAX_TURN, SEED = 100_000, 30, 0
GRID_N = range(1, 16)
GRID_T = (0, 1, 2)


def lengths_for(onehot: np.ndarray, window: int, threshold: int) -> np.ndarray:
    """onehot: (n_seq, MAX_TURN, n_cat) bool. Returns session lengths (1..MAX_TURN)."""
    n_seq = onehot.shape[0]
    cum = np.zeros((n_seq, MAX_TURN + 1, onehot.shape[2]), dtype=np.int16)
    cum[:, 1:] = np.cumsum(onehot, axis=1, dtype=np.int16)
    length = np.full(n_seq, MAX_TURN)
    alive = np.ones(n_seq, dtype=bool)
    for t in range(1, MAX_TURN):
        counts = cum[:, t] - cum[:, max(0, t - window)]
        quit_now = alive & np.any(onehot[:, t] & (counts > threshold), axis=1)
        length[quit_now] = t + 1
        alive &= ~quit_now
    return length


def survival(lengths: np.ndarray) -> list[float]:
    """S(t) = P(session length >= t) for t = 1..MAX_TURN."""
    return [float((lengths >= t).mean()) for t in range(1, MAX_TURN + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    env = KuaiEnv()
    catalog = np.asarray(env.catalog)
    cat_ids = sorted({c for v in env.catalog for c in env.categories[v]})
    col = {c: i for i, c in enumerate(cat_ids)}
    hot = np.zeros((len(catalog), len(cat_ids)), dtype=bool)
    for j, v in enumerate(env.catalog):
        for c in env.categories[v]:
            hot[j, col[c]] = True

    rng = np.random.default_rng(SEED)
    idx = np.stack([rng.choice(len(catalog), size=MAX_TURN, replace=False) for _ in range(N_SEQ)])
    onehot = hot[idx]

    # Cross-check the vectorised rule against env.boredom_quit on 2,000 sequences.
    cfg = env.config
    vec = lengths_for(onehot[:2000], cfg.quit_window, cfg.quit_threshold)
    for s in range(2000):
        seq = [int(catalog[i]) for i in idx[s]]
        n = MAX_TURN
        for t in range(1, MAX_TURN):
            if boredom_quit(seq[:t], seq[t], env.categories, cfg.quit_window, cfg.quit_threshold, "intended"):
                n = t + 1
                break
        assert n == vec[s], (s, n, vec[s])
    print("vectorised rule == env.boredom_quit on 2,000 sequences")

    seeds = list(harness.harness_config()["seeds"])
    from kuairec_eval import partitions
    users = partitions.load("search")
    sess = np.concatenate([harness.run_seed(env, UniformRandom, users, s).sess_len for s in seeds])
    out = {"harness_primary": {"n_sessions": int(len(sess)), "mean": float(sess.mean()),
                               "median": float(np.median(sess)), "survival": survival(sess),
                               "reach_max_turn": float((sess == MAX_TURN).mean())}}
    print(f"harness, random, search x 10 seeds (N=9, n_Q=1): {len(sess):,} sessions, mean {sess.mean():.3f},"
          f" median {np.median(sess):.1f}, P(reach 30) {(sess == MAX_TURN).mean():.4f}")

    grid = {}
    print(f"\nMonte Carlo, {N_SEQ:,} random sequences: median / mean session length / P(reach 30)")
    print("   N " + "".join(f"{'n_Q=' + str(t):>26s}" for t in GRID_T))
    for n in GRID_N:
        cells = []
        for t in GRID_T:
            ln = lengths_for(onehot, n, t)
            grid[(n, t)] = ln
            cells.append(f"{np.median(ln):>8.1f} / {ln.mean():>6.2f} / {(ln == MAX_TURN).mean():>6.4f}")
        print(f"  {n:>2} " + "".join(f"{c:>26s}" for c in cells))
    prim = grid[(9, 1)]
    print(f"\nprimary (9,1) Monte Carlo mean {prim.mean():.3f} vs harness {sess.mean():.3f}")

    target_med = np.median(prim)
    def pick(mult):
        cand = [(abs(np.median(grid[(n, 1)]) - mult * target_med), abs(grid[(n, 1)].mean() - mult * prim.mean()), n)
                for n in GRID_N if n != 9]
        return min(cand)[2]
    looser, tighter = pick(2.0), pick(0.5)
    print(f"proposal (n_Q = 1): looser N = {looser} (median {np.median(grid[(looser, 1)]):.1f}, mean {grid[(looser, 1)].mean():.2f},"
          f" P(reach 30) {(grid[(looser, 1)] == MAX_TURN).mean():.4f}); tighter N = {tighter} (median {np.median(grid[(tighter, 1)]):.1f},"
          f" mean {grid[(tighter, 1)].mean():.2f})")
    print("\nsurvival S(t) = P(length >= t), Monte Carlo:")
    print("   t " + "".join(f"{'N=' + str(n):>9s}" for n in (tighter, 9, looser)) + f"{'harness N=9':>13s}")
    for t in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
        print(f"  {t:>2} " + "".join(f"{survival(grid[(n, 1)])[t - 1]:>9.4f}" for n in (tighter, 9, looser))
              + f"{out['harness_primary']['survival'][t - 1]:>13.4f}")
    # The N-axis rule cannot tighten: for n_Q = 1, N >= 6 all have median 8 and the tie-break
    # lands on N = 15, whose curve equals the primary's through t = 10. The threshold axis at
    # the primary window (N = 9) moves the curve both ways (ADR-0019).
    p_med, p_mean = np.median(prim), prim.mean()
    print("\nthreshold axis at N = 9 (median, mean, ratio of mean to primary, P(reach 30)):")
    for t in GRID_T:
        ln = grid[(9, t)]
        print(f"  n_Q={t}: median {np.median(ln):>5.1f} ({np.median(ln) / p_med:.2f}x)  mean {ln.mean():>6.2f} ({ln.mean() / p_mean:.2f}x)"
              f"  P(reach 30) {(ln == MAX_TURN).mean():.4f}")
    t15 = survival(grid[(tighter, 1)])
    same_until = next((t for t in range(1, MAX_TURN + 1) if abs(t15[t - 1] - survival(prim)[t - 1]) > 1e-9), MAX_TURN + 1) - 1
    print(f"rule's tighter N={tighter}: S(t) identical to primary for t <= {same_until}; mean {grid[(tighter, 1)].mean():.2f} vs {p_mean:.2f}")
    print("\nsurvival S(t), threshold axis at N = 9:")
    print("   t " + "".join(f"{'n_Q=' + str(t):>9s}" for t in GRID_T))
    for t in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
        print(f"  {t:>2} " + "".join(f"{survival(grid[(9, q)])[t - 1]:>9.4f}" for q in GRID_T))
    out["n_axis_rule_result"] = {"looser": {"quit_window": looser, "quit_threshold": 1},
                                 "tighter": {"quit_window": tighter, "quit_threshold": 1},
                                 "tighter_identical_to_primary_through_t": same_until}
    out["monte_carlo"] = {f"N={n},n_Q={t}": {"median": float(np.median(ln)), "mean": float(ln.mean()),
                                             "reach_max_turn": float((ln == MAX_TURN).mean())}
                          for (n, t), ln in grid.items()}
    out["survival_curves"] = {**{f"N={n},n_Q=1": survival(grid[(n, 1)]) for n in (tighter, 9, looser)},
                              **{f"N=9,n_Q={q}": survival(grid[(9, q)]) for q in GRID_T}}
    out["proposal"] = {"primary": {"quit_window": 9, "quit_threshold": 1},
                       "looser": {"quit_window": 9, "quit_threshold": 2},
                       "tighter": {"quit_window": 9, "quit_threshold": 0},
                       "alternative_looser_on_N_axis": {"quit_window": looser, "quit_threshold": 1}}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
