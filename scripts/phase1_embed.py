"""Phase 1: caption embeddings for the text variant of the ranker (ADR-0034).

The ONLY Phase 1 code that imports torch, run as its own process: importing torch before
LightGBM in one process segfaults on this machine (exit 139, two OpenMP runtimes). Training and
policies read the cached array and never import torch.

Text per video = caption + manual cover text (unless "UNKNOWN") + topic tags, from
kuairec_caption_category.csv. Category names are left out, so the text variant adds free text,
not a second copy of the category features. Videos with no text get a zero vector and has_text=0.

Usage: HF_HOME=data/cache/hf .venv/bin/python scripts/phase1_embed.py
Writes data/cache/phase1/caption_emb.npz and runs/phase1/embed_report.json
"""
import ast
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "hf"))
import numpy as np  # noqa: E402

from kuairec_eval.data import load_caption_category  # noqa: E402
from policies.phase1.config import CACHE, load  # noqa: E402


def video_text(row) -> str:
    parts = []
    if isinstance(row.caption, str) and row.caption.strip():
        parts.append(row.caption.strip())
    if isinstance(row.manual_cover_text, str) and row.manual_cover_text not in ("", "UNKNOWN"):
        parts.append(row.manual_cover_text.strip())
    try:
        tags = ast.literal_eval(row.topic_tag) if isinstance(row.topic_tag, str) else []
    except (ValueError, SyntaxError):
        tags = []
    parts.extend(str(t) for t in tags if str(t).strip())
    return " ".join(parts)


def main():
    from sentence_transformers import SentenceTransformer
    model_name = load()["text"]["model"]
    cc = load_caption_category().sort_values("video_id")
    assert cc.video_id.tolist() == list(range(len(cc)))
    texts = [video_text(r) for r in cc.itertuples()]
    has_text = np.array([bool(t) for t in texts])
    t0 = time.time()
    model = SentenceTransformer(model_name, device="cpu")
    emb = np.zeros((len(texts), model.get_sentence_embedding_dimension()), np.float32)
    idx = np.flatnonzero(has_text)
    emb[idx] = model.encode([texts[i] for i in idx], batch_size=128, normalize_embeddings=True,
                            show_progress_bar=False).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "caption_emb.npz", emb=emb, has_text=has_text)
    report = {"model": model_name, "videos": len(texts), "videos_with_text": int(has_text.sum()),
              "dims": int(emb.shape[1]), "seconds": round(time.time() - t0, 1),
              "median_text_chars": float(np.median([len(texts[i]) for i in idx]))}
    (ROOT / "runs" / "phase1" / "embed_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
