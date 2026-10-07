"""DIAGNOSTICS ONLY: oracle ceilings that read small-matrix rewards.

No policy may import or call anything in this package. Enforced three ways:
  * importing it from a file under policies/ raises ImportError (first import only,
    because Python caches modules, hence the call-time guard below);
  * every factory and constructor re-checks the call stack (guard_against_policies);
  * tests/test_policy_isolation.py fails on any policies/ file that references it.
Not in the v0 manifest; evaluator v1 (proposed) adds it.
"""
from __future__ import annotations

import inspect
import os

from kuairec_eval.data import POLICIES_DIR


class DiagnosticsAccessError(ImportError):
    pass


def guard_against_policies() -> None:
    policies = str(POLICIES_DIR) + os.sep
    for frame in inspect.stack(context=0):
        if os.path.abspath(frame.filename).startswith(policies):
            raise DiagnosticsAccessError(
                f"kuairec_eval.diagnostics used from policy code ({frame.filename}:{frame.lineno})")


guard_against_policies()
