"""Sweep 8 models × 5 evals (40 runs). Writes JSONL per (eval, model).

Concurrency is per-eval (8 items in parallel within a run); runs are
sequential to keep API rate limits manageable. Total wall-clock ~1-2hr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from pcfbench.evals.runner import (
    _DEFAULT_DATA_DIR,
    run_eval,
)

MODELS_8 = [
    "claude-haiku-4-5@20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "deepseek-ai/deepseek-v3.2-maas",
]

EVALS_5 = [
    "pcfbench_decomposition",
    "pcfbench_triage",
    "pcfbench_mapping_with_context",
    "pcfbench_extraction",
    "pcfbench_epd_with_composition",
]

_RUNS_DIR = Path(__file__).resolve().parent / "runs"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS_8)
    parser.add_argument("--evals", nargs="*", default=EVALS_5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Directory with pcfbench task*.jsonl files.",
    )
    args = parser.parse_args()

    summaries = []
    total = len(args.evals) * len(args.models)
    done = 0
    for ev in args.evals:
        for model in args.models:
            done += 1
            slug = model.replace("/", "_").replace("@", "_")
            output = _RUNS_DIR / f"{ev}__{slug}.jsonl"
            print(f"\n>>> [{done}/{total}] {ev} | {model}", flush=True)
            t0 = time.perf_counter()
            try:
                summary = await run_eval(
                    eval_name=ev,
                    model_id=model,
                    output_path=output,
                    data_dir=args.data_dir,
                    limit=args.limit,
                    concurrency=args.concurrency,
                )
                summary["ok"] = True
            except Exception as e:
                summary = {
                    "eval_name": ev,
                    "model_id": model,
                    "ok": False,
                    "error": str(e),
                }
                print(f"!!! FAILED {ev}|{model}: {e}", flush=True)
            summary["wall_s"] = time.perf_counter() - t0
            summaries.append(summary)
            print(json.dumps(summary, default=str), flush=True)

    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = _RUNS_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nWrote summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
