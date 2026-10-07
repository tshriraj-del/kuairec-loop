"""Policies cannot import the small matrix: static scan plus the runtime guard."""
import ast
from pathlib import Path

import pytest

from kuairec_eval import data as D

POLICY_FILES = sorted(D.POLICIES_DIR.glob("**/*.py"))
# Evaluator-side modules (and every submodule) that hold or reach small-matrix rewards.
BANNED_MODULES = {"kuairec_eval.env", "kuairec_eval.harness", "kuairec_eval._integrity",
                  "kuairec_eval.partitions", "kuairec_eval.rewards", "kuairec_eval.diagnostics",
                  "audit"}
BANNED_NAMES = {"load_small_matrix", "_build_oracle", "_lookup", "_read_matrix", "reward_matrix",
                "personalized_factory", "nonpersonalized_factory"}


def _is_banned(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in BANNED_MODULES)


def test_policy_files_exist():
    assert POLICY_FILES, "no policy files found"


@pytest.mark.parametrize("path", POLICY_FILES, ids=lambda p: p.name)
def test_policy_source_never_references_small_matrix(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not _is_banned(a.name), f"{path.name} imports {a.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not _is_banned(module), f"{path.name} imports from {module}"
            for a in node.names:
                assert not _is_banned(f"{module}.{a.name}"), f"{path.name} imports {module}.{a.name}"
                assert a.name not in BANNED_NAMES and a.name != "*", f"{path.name} imports {a.name}"
        elif isinstance(node, ast.Name):
            assert node.id not in BANNED_NAMES, f"{path.name} uses {node.id}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in BANNED_NAMES, f"{path.name} uses .{node.attr}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "small_matrix" not in node.value, f"{path.name} mentions small_matrix"


def test_runtime_guard_blocks_calls_from_policy_code():
    fake = str(D.POLICIES_DIR / "evil_policy.py")
    code = compile("from kuairec_eval.data import load_small_matrix\nload_small_matrix()", fake, "exec")
    with pytest.raises(D.PolicyDataAccessError):
        exec(code, {})


def test_runtime_guard_allows_evaluator_code():
    # the environment itself must still be able to load it (exercised by the env fixture)
    D._assert_not_called_from_policy()


@pytest.mark.parametrize("src", [
    "from kuairec_eval import env",
    "from kuairec_eval import rewards",
    "from kuairec_eval.diagnostics.oracles import greedy_sequence",
    "import audit.reward_env",
    "from audit import runner",
])
def test_static_scan_catches_evaluator_imports(tmp_path, src):
    bad = tmp_path / "bad.py"
    bad.write_text(src + "\n")
    with pytest.raises(AssertionError):
        test_policy_source_never_references_small_matrix(bad)


def test_runtime_guard_blocks_oracles_from_policy_code():
    from kuairec_eval.diagnostics import DiagnosticsAccessError
    fake = str(D.POLICIES_DIR / "evil_oracle_user.py")
    src = ("import numpy as np\n"
           "from kuairec_eval.diagnostics.oracles import personalized_factory\n"
           "personalized_factory(np.zeros((1, 1)), np.array([0]), np.array([0]), None)\n")
    with pytest.raises(DiagnosticsAccessError):
        exec(compile(src, fake, "exec"), {})
