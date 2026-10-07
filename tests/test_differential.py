"""Item 7d: identical sequences through our env and EasyRL4Rec's KuaiEnv.

Runs scripts/differential.py on the first 40 search users (the full 847-user run is
`.venv/bin/python scripts/differential.py`) and fails on any UNEXPLAINED difference.
Skips, visibly, if EasyRL4Rec or its reference dependencies are not installed.
"""
import importlib.util
import subprocess
import sys

import pytest

from kuairec_eval.data import REPO_ROOT


def test_no_unexplained_differences_vs_easyrl4rec(tmp_path):
    if not (REPO_ROOT / "third_party" / "EasyRL4Rec" / "src").exists():
        pytest.skip("third_party/EasyRL4Rec not cloned")
    for mod in ("sklearn", "gymnasium", "logzero", "tqdm"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"reference dependency {mod} missing (uv sync --group reference)")
    proc = subprocess.run([sys.executable, "scripts/differential.py", "--users", "40", "--out", str(tmp_path)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    print(proc.stdout[-4000:])
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
