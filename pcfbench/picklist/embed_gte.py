"""Build ``embeddings_gte.npy`` for the picklist using GTE-Large.

GTE-Large (``thenlper/gte-large``, MIT-licensed) is the embedder used by
the Parakeet 3-stage mapping baseline (Balaji et al. 2025): paraphrase →
GTE-Large semantic retrieval → LLM rerank. Encoding the same
``reference_product_name`` list every other pcfbench baseline
sees as the picklist keeps Parakeet's candidate set aligned with the
single-shot and agentic mapping evals.

Outputs:
    pcfbench/picklist/embeddings_gte.npy
    pcfbench/picklist/embedding_model_gte.txt
    pcfbench/picklist/embeddings_gte_uuids.json   # row-order sidecar

Usage:
    uv run python -m pcfbench.picklist.embed_gte
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

GTE_MODEL = "thenlper/gte-large"

_PICKLIST_DIR = Path(__file__).resolve().parent
_PICKLIST_JSONL = _PICKLIST_DIR / "ecoinvent_picklist.jsonl"
_EMBEDDINGS_NPY = _PICKLIST_DIR / "embeddings_gte.npy"
_MODEL_TXT = _PICKLIST_DIR / "embedding_model_gte.txt"
# Row-order ``activity_uuid_product_uuid`` sidecar — Parakeet retrieval
# joins back to ecoinvent_picklist.jsonl by row index, so we fail loudly
# if the embeddings .npy and the picklist fall out of sync.
_EMBEDDINGS_UUIDS_JSON = _PICKLIST_DIR / "embeddings_gte_uuids.json"


def main() -> None:
    print(f"Loading picklist from {_PICKLIST_JSONL} ...", flush=True)
    with _PICKLIST_JSONL.open() as f:
        materials = [json.loads(line) for line in f if line.strip()]
    # Parakeet retrieves over reference_product_name only — the same
    # strings the LLM later picks from. activity_name (e.g. "market for
    # X") would inflate similarity for every market activity uniformly.
    texts = [m["reference_product_name"] for m in materials]
    uuids = [m["activity_uuid_product_uuid"] for m in materials]
    print(f"Encoding {len(texts)} reference products with {GTE_MODEL} ...", flush=True)

    model = SentenceTransformer(GTE_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(_EMBEDDINGS_NPY, embeddings)
    _MODEL_TXT.write_text(GTE_MODEL + "\n")
    _EMBEDDINGS_UUIDS_JSON.write_text(json.dumps(uuids))
    print(
        f"Wrote {embeddings.shape} -> {_EMBEDDINGS_NPY}; "
        f"recorded model at {_MODEL_TXT}; "
        f"row-order UUIDs at {_EMBEDDINGS_UUIDS_JSON}",
        flush=True,
    )


if __name__ == "__main__":
    main()
