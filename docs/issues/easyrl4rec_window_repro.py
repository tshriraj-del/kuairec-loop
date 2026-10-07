# Minimal reproduction for the EasyRL4Rec issue drafted in easyrl4rec-negative-window.md.
# Run from the EasyRL4Rec repo root. Importing KuaiEnv reads data/KuaiRec/data_raw/*.csv at
# import time (preprocessing_kuairec.py), so the KuaiRec data must be in place; the
# environment itself is built from the tiny synthetic inputs below.
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
