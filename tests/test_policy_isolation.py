"""Policies cannot import the small matrix: static scan plus the runtime guard."""
import ast
from pathlib import Path

import pytest

from kuairec_eval import data as D

POLICY_FILES = sorted(D.POLICIES_DIR.glob("**/*.py"))
BANNED_MODULES = {"kuairec_eval.env", "kuairec_eval.harness", "kuairec_eval._integrity",
                  "kuairec_eval.partitions"}
BANNED_NAMES = {"load_small_matrix", "_build_oracle", "_lookup", "_read_matrix"}


def test_policy_files_exist():
    assert POLICY_FILES, "no policy files found"


@pytest.mark.parametrize("path", POLICY_FILES, ids=lambda p: p.name)
def test_policy_source_never_references_small_matrix(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name not in BANNED_MODULES, f"{path.name} imports {a.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in BANNED_MODULES, f"{path.name} imports from {node.module}"
            for a in node.names:
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
