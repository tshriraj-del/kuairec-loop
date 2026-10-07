# DRAFT, not filed. For github.com/chongminggao/EasyRL4Rec (checked against commit 95d0acf).

**Title:** Quit window drops recent items when `0 < t < num_leave_compute` (negative slice start in `_determine_whether_to_leave`)

---

### Summary

`_determine_whether_to_leave` builds its window as

```python
window_actions = self.sequence_action[t - self.num_leave_compute:t]
```

When `t < num_leave_compute`, the start index is negative, so Python counts it from the end of the list. The list has length `t` at that point. For `num_leave_compute / 2 < t < num_leave_compute`, the window becomes `sequence_action[2t - N : t]` instead of the last `t` items, so earlier recommendations are skipped. The user then quits later than the rule describes (CIRS, Sec. 5.1.2: "for the most recent N recommended items ...").

The same line appears in five environments:

- `src/core/envs/KuaiRec/KuaiEnv.py:52`
- `src/core/envs/KuaiRand_Pure/KuaiRandEnv.py:43`
- `src/core/envs/Coat/CoatEnv.py:40`
- `src/core/envs/YahooR3/YahooEnv.py:39`
- `src/core/envs/MovieLens/MovieLensEnv.py:64`

### Minimal reproduction

Run from the repo root. Importing `KuaiEnv` reads the KuaiRec CSVs at import time, so the data must be in place; the environment itself is built from tiny synthetic inputs.

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.core.envs.KuaiRec.KuaiEnv import KuaiEnv

list_feat = [[0], [0], [1], [2], [3], [0]]  # items 0, 1 and 5 share category 0
env = KuaiEnv(mat=np.ones((1, 6)), lbe_user=LabelEncoder().fit([0]), lbe_item=LabelEncoder().fit(range(6)),
              list_feat=list_feat, df_dist_small=None, num_leave_compute=5, leave_threshold=1, max_turn=100)
env.reset()
for a in [0, 1, 2, 3]:
    assert not env.step(a)[2]
t = env.total_turn  # 4
print("window used:    ", env.sequence_action[t - env.num_leave_compute:t])
print("intended window:", env.sequence_action[max(0, t - env.num_leave_compute):t])
print("terminated:     ", env.step(5)[2])
```

Output:

```
window used:     [3]
intended window: [0, 1, 2, 3]
terminated:      False
```

**Expected:** `terminated: True`. Category 0 appears in two of the last four items, which is more than `leave_threshold = 1`. With the one-line fix below, the same script prints `terminated: True`.

### Which items the window keeps, for `num_leave_compute = 9` (the `KuaiEnv-v0` default)

| t | window used | intended | dropped |
|---|---|---|---|
| 1-4 | all earlier items | all earlier items | none |
| 5 | 1-4 | 0-4 | 0 |
| 6 | 3-5 | 0-5 | 0-2 |
| 7 | 5-6 | 0-6 | 0-4 |
| 8 | 7 | 0-7 | 0-6 |
| ≥ 9 | last 9 | last 9 | none |

### Impact

We measured this with an independent reimplementation of `KuaiEnv` that uses the intended window, fed the same recommendation sequences as `KuaiEnv` with `KuaiEnv-v0` defaults (`num_leave_compute=9`, `leave_threshold=1`, `max_turn=30`). There were 5,929 no-repeat sequences over 847 users. 1,348 of them ended at a different step, including 1,025 of 2,541 uniformly random sequences. Every difference fell at t ∈ {5, 6, 7, 8}, and in every case the current code ended the session later. Session lengths and cumulative rewards reported with the current code are therefore higher than the rule as described would produce. In the distance-based environments (Coat, YahooR3, MovieLens), the dropped items are simply never compared against the target.

### Suggested fix

```python
window_actions = self.sequence_action[max(0, t - self.num_leave_compute):t]
```

Apply it in all five files, and add the snippet above as a regression test. Since this changes the numbers that existing runs produce, a flag to restore the old behaviour may help anyone reproducing published results.
