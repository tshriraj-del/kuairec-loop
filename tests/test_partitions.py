import pytest

from kuairec_eval import partitions
from kuairec_eval.data import load_small_matrix


def test_holdout_is_refused_outside_final_report():
    with pytest.raises(partitions.HoldoutAccessError):
        partitions.load("holdout")


def test_harness_cannot_run_on_holdout(env):
    from kuairec_eval import harness
    from policies.baselines import UniformRandom
    with pytest.raises(partitions.HoldoutAccessError):
        harness.run(UniformRandom, "holdout", seeds=[0], env=env)


def test_partitions_are_disjoint_complete_and_reproducible():
    search, shadow = partitions.load("search"), partitions.load("shadow")
    users = load_small_matrix()["user_id"].unique()
    cfg = partitions.config()
    regen = partitions.split(users, cfg["seed"], {k: cfg[k] for k in partitions.NAMES})
    assert regen["search"] == search and regen["shadow"] == shadow
    assert (len(search), len(shadow), len(regen["holdout"])) == (847, 282, 282)
    assert not set(search) & set(shadow)
    assert set(search) | set(shadow) | set(regen["holdout"]) == set(int(u) for u in users)


def test_partitions_are_never_rewritten(tmp_path):
    parts = {"search": [1], "shadow": [2], "holdout": [3]}
    partitions.write(parts, 0, tmp_path)
    with pytest.raises(FileExistsError):
        partitions.write(parts, 0, tmp_path)
