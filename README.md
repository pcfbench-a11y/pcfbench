# PCFBench

Process-based Product Carbon Footprint benchmark for evaluating LLMs and agents on
the operational steps of life-cycle assessment (LCA): bill-of-materials decomposition,
mapping triage, ecoinvent process matching, literature extraction of physical input
rates, and total kgCO₂e prediction against expert-grounded EPDs.

> **Anonymized for NeurIPS 2026 review.** Author and affiliation details are withheld
> until the camera-ready period.

## Tasks

| ID | Task | Items | GT claims | Headline metric |
| -- | ---- | ----: | --------: | --------------- |
| 1 | Product decomposition (BOM) | 94 | 94 | Judge-aligned F₁ on compositional match groups |
| 2 | Mapping triage | 200 | 200 | Accuracy / F₁ on `should_map` binary |
| 3 | Background-database mapping | 109 | 109 | Exact-match top-1 against expert reference products |
| 4 | Material input-rate extraction | 22 | 55 | Claim F₁ on greedy (value, unit) match |
| 5 | Energy input-rate extraction | 14 | 34 | Claim F₁ on greedy (value, unit) match |
| 7 | Total kgCO₂e prediction (EPD)       | 175 | 175 | Median \|RE\|, within-2× / within-5× rate |

(Step 6 is deterministic arithmetic and not separately evaluated.)

Tasks 2 and 3 share an **ecoinvent v3.11 picklist** of 2,571 reference products
(strict superset of the prior v3.10 1,663-row list); the picklist is published
alongside the task data.

## Quick start

```bash
# 1. Clone
git clone https://github.com/anonymous/pcfbench
cd pcfbench

# 2. Install
pip install -e .   # or: uv sync

# 3. Get the data from Kaggle (dataset path is in the paper). Download
#    the JSONL files into ./pcfbench_data/.

# 4. Set credentials for whichever model you'll evaluate
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, or Vertex creds for Gemini/etc.

# 5. Run an eval (example: decomposition on 10 items with Haiku 4.5)
python -m pcfbench.evals.runner pcfbench_decomposition \
    --model "claude-haiku-4-5@20251001" \
    --data-dir pcfbench_data \
    --limit 10
```

Per-item results land in `pcfbench/runs/<eval>__<model>.jsonl`; the summary prints to stdout.

## Available evals

```
pcfbench_decomposition
pcfbench_triage                       pcfbench_triage_agentic
pcfbench_mapping_with_context         pcfbench_mapping_agentic_with_context
pcfbench_extraction                   pcfbench_extraction_query_only_estimate
pcfbench_epd_with_composition
```

The `_agentic` variants give the model tool access to retrieve from the ecoinvent
picklist on demand instead of putting it all in the prompt.

## Repository layout

```
pcfbench/
├── agents/         # Per-task agents (Pydantic AI), one module per task
├── evals/
│   └── runner.py   # CLI entry point: load JSONL, dispatch, score, write JSONL
├── models/
│   ├── factory.py  # build_agent(model_id) -> typed Agent
│   └── registry.py # Frozen sets of supported model ids per provider
├── picklist/
│   ├── build_picklist_json.py  # Rebuild the picklist from the public ecoinvent xlsx
│   ├── embed_gemini.py         # Gemini embeddings (used by MaterialLibrary)
│   └── embed_gte.py            # GTE-Large embeddings (used by Parakeet)
├── scoring/        # Per-task per-item scorers
├── tools/          # Ecoinvent search/inspect tools for the agentic variants
├── sweep_all.py    # Run every (eval × model) combination and produce a summary table
├── README.md
└── LICENSE
```

## Adding a model

1. Add the model id to the appropriate frozen set in `pcfbench/models/registry.py`
   (`ANTHROPIC_MODELS`, `OPENAI_MODELS`, `GEMINI_MODELS`, `DEEPSEEK_VERTEX_MODELS`).
2. Verify the auth path in `pcfbench/models/factory.py` covers your model's provider.
3. Run a small smoke: `python -m pcfbench.evals.runner pcfbench_decomposition --model <new-id> --limit 2`.

## Reproducing the paper

For headline-table parity, run the full sweep — every model in `MODELS_8`
crossed with every eval in `EVALS_5`:

```bash
python -m pcfbench.sweep_all --data-dir pcfbench_data
```

Numbers will differ from the paper if you (a) rebuild the picklist from a newer
ecoinvent release, or (b) run with reasoning settings that drift from the pinned
configuration. The paper's reasoning-on rows use:
- Claude Opus 4.6: extended thinking, 8192-token budget
- Gemini 3.1 Pro: thinking, 8192-token budget
- GPT-5.5: `reasoning_effort=high`

## Datasets

Data is published on Kaggle (path in the paper) and consists of seven JSONL
files plus a Croissant metadata descriptor. Schema details and provenance are in
the dataset's `DATASHEET.md`.

Each task JSONL row has the same envelope:

```json
{
  "id": "...",
  "input": { /* task-specific */ },
  "expected_output": { /* task-specific ground truth */ },
  "metadata": { "product_category": "...", /* task-specific extras */ }
}
```

## Citation

```bibtex
@inproceedings{pcfbench2026,
  title     = {{PCFBench}: Evaluating LLMs and Agents on Process-Based
               Product Carbon Footprint Estimation},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2026},
  note      = {Under review}
}
```

## License

[Apache License 2.0](LICENSE).

The shared ecoinvent v3.11 picklist used by Tasks 2 and 3 ships with this
repository at `pcfbench/picklist/ecoinvent_picklist.jsonl` (2,571 rows;
row order is load-bearing because the precomputed embedding `.npy` files
index into it positionally). It is derived from the publicly available
ecoinvent v3.11 *Database Overview* workbook and can be regenerated with
`python -m pcfbench.picklist.build_picklist_json`.
