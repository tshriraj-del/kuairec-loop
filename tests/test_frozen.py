"""The hash check passes on the frozen tree and fails when any frozen file changes."""
import json
import shutil

import pytest

from kuairec_eval import _integrity
from kuairec_eval.data import REPO_ROOT, data_dir


def test_frozen_tree_verifies():
    assert len(_integrity.verify()) == 64


@pytest.mark.parametrize("victim", _integrity.FROZEN_CODE)
def test_any_changed_file_fails(tmp_path, victim):
    for rel in _integrity.FROZEN_CODE:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(json.loads(_integrity.MANIFEST_PATH.read_text())))
    _integrity.verify(tmp_path, data_dir(), manifest)          # pristine copy passes
    with open(tmp_path / victim, "a") as f:
        f.write("\n")
    with pytest.raises(_integrity.FrozenEvaluatorError, match=victim):
        _integrity.verify(tmp_path, data_dir(), manifest)


def test_changed_data_fails(tmp_path):
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    for rel in _integrity.FROZEN_DATA:
        shutil.copy2(data_dir() / rel, fake_data / rel)
    with open(fake_data / "item_categories.csv", "a") as f:
        f.write("99999,[1]\n")
    with pytest.raises(_integrity.FrozenEvaluatorError, match="item_categories.csv"):
        _integrity.verify(REPO_ROOT, fake_data, _integrity.MANIFEST_PATH)
