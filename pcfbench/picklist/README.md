# PCFBench picklist

The picklist (one row per single-geography ``kg`` market activity in
ecoinvent v3.11 cut-off, plus a small set of energy-carrier markets
admitted by exception via ``ENERGY_ALLOWLIST_UUIDS`` in
``build_picklist_json.py``) is **committed** as
``ecoinvent_picklist.jsonl``. Its precomputed embeddings are **not
committed** — they're regenerable from the picklist via the embed
scripts. Layout:

```
ecoinvent_picklist.jsonl         # ecoinvent activity uuid + activity name + reference product + product information
embeddings_gemini.npy            # gemini-embedding-001 (768-dim, L2-normalised)
embeddings_gemini_uuids.json     # row-order activity_uuid_product_uuid sidecar for embeddings_gemini.npy
embedding_model_gemini.txt       # records "gemini-embedding-001"
embeddings_gte.npy               # GTE-Large (1024-dim, L2-normalised) — Parakeet-only
embeddings_gte_uuids.json        # row-order activity_uuid_product_uuid sidecar for embeddings_gte.npy
embedding_model_gte.txt          # records "thenlper/gte-large"
```

``ecoinvent_picklist.jsonl`` is one JSON object per line, each with
exactly four fields taken straight from the public ``Cut-Off AO``
sheet:

```
activity_uuid_product_uuid   # "Activity UUID & Product UUID"
activity_name                # "Activity Name"
reference_product_name       # "Reference Product Name"
product_information          # "Product Information"
```

Rows are sorted by ``activity_uuid_product_uuid`` (a stable, permanent
key across ecoinvent re-releases) so the file is byte-stable when the
upstream xlsx is unchanged. Each ``embeddings*.npy`` ships with a
``embeddings*_uuids.json`` sidecar capturing the row-order UUID list
at embed time; ``MaterialLibrary.load_default()`` cross-checks this
against the current ``ecoinvent_picklist.jsonl`` order and refuses to
load if a stale ``.npy`` no longer aligns with the picklist.

## Bootstrap

### (Re)generate the picklist (~5 seconds; no GCP access required)

```bash
uv run python -m pcfbench.picklist.build_picklist_json
```

Pulls every ``kg`` market activity (one per reference product,
picking ``GLO``→``RoW``→``RER`` for the geography) directly from the
public ecoinvent v3.11 [Database
Overview](https://support.ecoinvent.org/hubfs/Knowledge%20Base/Database/Releases/3.11/Database-Overview-for-ecoinvent-v3.11%20(6).xlsx)
workbook (sheet ``Cut-Off AO``) and writes
``ecoinvent_picklist.jsonl`` here. The script reads only the public
xlsx (cached after first download).

This produces a strict super-set of the prior ~1,663-row picklist
(now 2,574 rows): the four post-``filter_to_single_geo_markets``
filters used internally (``is_waste_from_ecoinvent``, ``is_zero_ef``,
``is_service``, ``is_removed_by_activity``) are deliberately skipped,
and three non-kg energy-carrier markets (electricity, natural gas,
industrial heat) are admitted by exception so Task 2 triage items
whose subject is energy have a valid mapping target.

### Embeddings (Gemini, used by ``MaterialLibrary``)

```bash
uv run python -m pcfbench.picklist.embed_gemini
```

Calls Vertex AI ``gemini-embedding-001`` at
``output_dimensionality=768``, L2-normalised, ~3 minutes at
concurrency=8. Writes ``embeddings_gemini.npy``,
``embeddings_gemini_uuids.json``, and ``embedding_model_gemini.txt``.

### Embeddings (GTE-Large, used by Parakeet)

```bash
uv run python -m pcfbench.picklist.embed_gte
```

Encodes the ``reference_product_name`` column with
``thenlper/gte-large`` (1024-dim, L2-normalised) for the Parakeet
3-stage mapping baseline. Writes ``embeddings_gte.npy``,
``embeddings_gte_uuids.json``, and ``embedding_model_gte.txt``.

## Why embeddings aren't committed

They're regenerable from ``ecoinvent_picklist.jsonl``, large-ish
binary blobs, and embedding-model-version sensitive — keeping them
out of git keeps the diff weight low and avoids stale assets drifting
from the picklist.
