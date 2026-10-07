"""Hash check for the frozen evaluator (environment, partitions, harness, config, data).

`verify()` is called by Environment.__init__, partitions.load and harness.run, so
any run fails if a frozen file changed since `scripts/freeze.py` last wrote the
manifest. Re-freezing is a deliberate act that shows up as a changed manifest
digest in every later results file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "kuairec_eval" / "FROZEN_MANIFEST.json"

FROZEN_CODE = [
    "kuairec_eval/__init__.py",
    "kuairec_eval/_integrity.py",
    "kuairec_eval/data.py",
    "kuairec_eval/env.py",
    "kuairec_eval/partitions.py",
    "kuairec_eval/harness.py",
    "config/evaluator.toml",
    "data/partitions/search.json",
    "data/partitions/shadow.json",
    "data/partitions/holdout.json",
]
# Relative to the KuaiRec data dir: the tables the environment answers from.
FROZEN_DATA = ["small_matrix.csv", "item_categories.csv"]


class FrozenEvaluatorError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def compute(repo_root: Path, data_dir: Path) -> dict:
    return {
        "code": {p: sha256_file(repo_root / p) for p in FROZEN_CODE},
        "data": {p: sha256_file(data_dir / p) for p in FROZEN_DATA},
    }


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


_verified: dict[tuple[str, str, str], str] = {}


def verify(repo_root: Path | None = None, data_dir: Path | None = None,
           manifest_path: Path | None = None) -> str:
    """Raise FrozenEvaluatorError on any mismatch; return the manifest digest."""
    from kuairec_eval.data import data_dir as default_data_dir

    repo_root = repo_root or REPO_ROOT
    data_dir = data_dir or default_data_dir()
    manifest_path = manifest_path or MANIFEST_PATH
    if not manifest_path.exists():
        raise FrozenEvaluatorError(f"{manifest_path} missing: run scripts/freeze.py")
    expected = json.loads(manifest_path.read_text())
    key = (str(repo_root), str(data_dir), str(manifest_path))
    # Code and config are rehashed on every call; the data CSVs (~0.5 GB) once per process.
    actual_code = {p: sha256_file(repo_root / p) for p in FROZEN_CODE}
    if key not in _verified:
        actual_data = {p: sha256_file(data_dir / p) for p in FROZEN_DATA}
    else:
        actual_data = expected["data"]
    problems = []
    for section, actual in (("code", actual_code), ("data", actual_data)):
        want = expected.get(section, {})
        for p in sorted(set(want) | set(actual)):
            if want.get(p) != actual.get(p):
                problems.append(f"{section}:{p} expected {str(want.get(p))[:12]} got {str(actual.get(p))[:12]}")
    if problems:
        raise FrozenEvaluatorError("frozen evaluator changed:\n  " + "\n  ".join(problems))
    digest = manifest_digest(expected)
    _verified[key] = digest
    return digest
