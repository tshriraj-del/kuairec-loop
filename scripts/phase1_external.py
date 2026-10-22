"""List every external function the Phase 1 code calls, with its package, version and licence.

Walks the AST of policies/phase1/*.py and scripts/phase1_*.py, resolves calls through import
aliases (import numpy as np; from scipy.stats import rankdata; ...) to fully qualified names, and
groups them by package. Method calls on library objects (df.groupby, booster.predict, ...) cannot
be resolved statically; they are listed with file:line so they can be attributed by hand.

Usage: .venv/bin/python scripts/phase1_external.py
Writes runs/phase1/external_calls.json
"""
import ast
import json
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted((ROOT / "policies" / "phase1").glob("*.py")) + sorted((ROOT / "scripts").glob("phase1_*.py"))
DIST = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "lightgbm": "lightgbm",
        "sentence_transformers": "sentence-transformers"}
LICENCE = {"numpy": "BSD-3-Clause", "scipy": "BSD-3-Clause", "pandas": "BSD-3-Clause", "lightgbm": "MIT",
           "sentence_transformers": "Apache-2.0"}
LOCAL = ("kuairec_eval", "policies", "phase1_common")
# Method calls on objects cannot be resolved from the AST alone; each name is attributed by hand
# to the library (or libraries) whose objects receive it in this code. Unlisted names are flagged.
ATTRIBUTION = {
    "pandas (DataFrame / Series / GroupBy)": ["assign", "duplicated", "fillna", "groupby", "isna", "notna", "itertuples",
                                             "join", "last", "reindex", "set_index", "sort_index", "sort_values",
                                             "to_dict", "to_numpy", "size"],
    "pandas or numpy (receiver-dependent)": ["astype", "max", "mean", "min", "sum", "replace", "tolist", "any",
                                             "percentile"],
    "numpy (ndarray / Generator)": ["ravel", "integers", "random"],
    "scipy.sparse (csr_matrix)": ["toarray"],
    "lightgbm (Booster)": ["predict", "feature_importance"],
    "sentence_transformers (SentenceTransformer)": ["encode", "get_sentence_embedding_dimension"],
    "Python standard library": ["add", "append", "extend", "pop", "update", "get", "items", "values", "index", "split",
                                "strip", "startswith", "removeprefix", "splitlines", "as_posix", "exists", "glob",
                                "mkdir", "read_text", "write_text", "relative_to", "resolve", "hexdigest", "isoformat",
                                "strftime", "read", "write", "add_argument", "parse_args"],
    "this repository (not external)": ["act", "begin_session", "observe", "score", "fit", "subset", "user_affinity",
                                       "user_cosines", "user_rows", "video_block", "bucket_of", "reward_config", "score_fn"],
}


def resolve(node, aliases):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in aliases:
        return ".".join([aliases[node.id]] + parts[::-1])
    return None


def main():
    found = defaultdict(set)
    methods = defaultdict(set)
    for path in FILES:
        tree = ast.parse(path.read_text())
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = resolve(node.func, aliases) if isinstance(node.func, ast.Attribute) else \
                    aliases.get(node.func.id) if isinstance(node.func, ast.Name) else None
                if name and not name.startswith(LOCAL):
                    found[name].add(rel)
                elif name is None and isinstance(node.func, ast.Attribute):
                    methods[node.func.attr].add(f"{rel}:{node.lineno}")
    by_pkg = defaultdict(dict)
    for name, files in sorted(found.items()):
        top = name.split(".")[0]
        by_pkg[top][name] = sorted(files)
    out = {"files_scanned": [p.relative_to(ROOT).as_posix() for p in FILES], "packages": {}}
    for top, calls in sorted(by_pkg.items()):
        dist = DIST.get(top)
        out["packages"][top] = {"version": metadata.version(dist) if dist else f"Python {sys.version.split()[0]} stdlib",
                                "licence": LICENCE.get(top, "PSF-2.0 (Python standard library)"), "calls": calls}
    out["unresolved_method_calls"] = {k: sorted(v) for k, v in sorted(methods.items())}
    owner = {m: lib for lib, names in ATTRIBUTION.items() for m in names}
    out["unattributed_methods"] = sorted(m for m in methods if m not in owner)
    (ROOT / "runs" / "phase1" / "external_calls.json").write_text(json.dumps(out, indent=2))
    md = ["# Reused external functions (Phase 1)", "",
          "_Generated by `.venv/bin/python scripts/phase1_external.py` from the AST of "
          f"{len(FILES)} files: `policies/phase1/*.py` and `scripts/phase1_*.py`. Do not edit by hand._", "",
          "Item-kNN, EASE, the affinity scorer, NDCG, the ratio bootstrap, the slate rule and PCA are written by hand "
          "from the primitives below.", "",
          "Model weights: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, from huggingface.co.", "",
          "## Module-level functions and classes", "", "| Package | Version | Licence | Functions |", "|---|---|---|---|"]
    for top, d in out["packages"].items():
        md.append(f"| {top} | {d['version']} | {d['licence']} | " + ", ".join(f"`{n}`" for n in d["calls"]) + " |")
    md += ["", "## Methods called on library objects (attributed by hand)", "", "| Owner | Methods |", "|---|---|"]
    for lib, names in ATTRIBUTION.items():
        used = [n for n in names if n in methods]
        if used:
            md.append(f"| {lib} | " + ", ".join(f"`{n}`" for n in used) + " |")
    if out["unattributed_methods"]:
        md += ["", "**Unattributed:** " + ", ".join(out["unattributed_methods"])]
    (ROOT / "docs" / "phase1").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "phase1" / "EXTERNAL_FUNCTIONS.md").write_text("\n".join(md) + "\n")
    for top, d in out["packages"].items():
        print(f"{top} {d['version']} ({d['licence']}): {len(d['calls'])} functions")
    print(f"method names: {len(out['unresolved_method_calls'])}; unattributed: {out['unattributed_methods'] or 'none'}")


if __name__ == "__main__":
    main()
