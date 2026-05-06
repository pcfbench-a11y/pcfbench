"""Extraction scoring: greedy unit-and-value matching of predicted
claims against ground-truth, then per-item P90|RE| / unit-correctness /
best-RE diagnostics, and run-level claim-F1 from aggregated counts.
"""

from __future__ import annotations

import re
import statistics
from typing import Iterable

from pcfbench.scoring.unit_normalization import (
    normalize_unit,
    normalize_value_and_unit,
)


def _unit_candidates(unit: str) -> list[str]:
    """Defensive: return progressively-stripped variants of a unit string.

    Reasoning models (Opus, Sonnet under thinking) sometimes pack
    descriptive prose into the unit field of structured output —
    e.g. ``'h, precipitation heat treatment time for 7075-T6'`` or
    ``'% of power supplied to extruder'``. The strict matcher then
    can't pair them with clean ground-truth units. Trying a few
    cleanup heuristics for each prediction recovers ~12-15pp of
    extraction F1 for those models without affecting Haiku/Gemini
    (which keep units clean already).
    """
    if not unit:
        return [unit]
    seen: set[str] = set()
    out: list[str] = []
    for candidate in (
        unit,
        unit.split(",", 1)[0].strip(),
        re.split(r"\s*\(", unit, 1)[0].strip(),
        unit.split()[0] if unit.split() else unit,
    ):
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _match_extraction_claims(
    pred_claims: list[tuple[float, str]],
    gt_claims: list[tuple[float, str]],
) -> tuple[list[tuple[int, int, float]], int, int]:
    """Greedy bipartite match of predicted claims to ground-truth claims.

    Pairs are eligible only if their normalised units match. Among
    eligible pairs we greedily pick the lowest relative error first,
    ensuring each prediction and each ground-truth claim is used at most
    once.

    For each predicted claim we try several unit cleanups (raw, comma-
    truncated, paren-truncated, first-token); the first variant that
    normalises to a valid GT unit is used for matching. Predictions
    with already-clean units land on the raw form first and never see
    the fallbacks.

    Returns ``(matches, n_unmatched_gt, n_unmatched_pred)`` where
    ``matches`` is a list of ``(pred_idx, gt_idx, relative_error)``.
    """
    gt_normalized = [normalize_value_and_unit(v, u) for v, u in gt_claims]
    gt_units = {g_unit for _, g_unit in gt_normalized}

    # For each prediction, find the first unit-candidate that normalises
    # to a GT-known unit. Falls back to the strict normalisation if
    # nothing matches (preserves prior behaviour for unknown-unit cases).
    pred_normalized: list[tuple[float, str]] = []
    for v, u in pred_claims:
        chosen: tuple[float, str] | None = None
        for candidate in _unit_candidates(u):
            cv, cu = normalize_value_and_unit(v, candidate)
            if cu in gt_units:
                chosen = (cv, cu)
                break
        if chosen is None:
            chosen = normalize_value_and_unit(v, u)
        pred_normalized.append(chosen)

    candidates: list[tuple[float, int, int]] = []
    for p_idx, (p_val, p_unit) in enumerate(pred_normalized):
        for g_idx, (g_val, g_unit) in enumerate(gt_normalized):
            if g_unit != p_unit:
                continue
            if g_val == 0:
                continue
            rel_err = abs(p_val - g_val) / abs(g_val)
            candidates.append((rel_err, p_idx, g_idx))
    candidates.sort(key=lambda x: x[0])
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for rel_err, p_idx, g_idx in candidates:
        if p_idx in used_p or g_idx in used_g:
            continue
        used_p.add(p_idx)
        used_g.add(g_idx)
        matches.append((p_idx, g_idx, rel_err))
    return matches, len(gt_claims) - len(used_g), len(pred_claims) - len(used_p)


def score_claim_matched_count(
    predicted: list[tuple[float, str]] | None,
    expected: list[tuple[float, str]],
) -> int | None:
    """Per-item count of greedy unit-and-value matches."""
    if predicted is None:
        return None
    matches, _, _ = _match_extraction_claims(predicted, expected)
    return len(matches)


def score_claim_pred_count(
    predicted: list[tuple[float, str]] | None,
) -> int | None:
    if predicted is None:
        return None
    return len(predicted)


def score_claim_gt_count(expected: list[tuple[float, str]]) -> int:
    return len(expected)


def score_p90_re_on_matched(
    predicted: list[tuple[float, str]] | None,
    expected: list[tuple[float, str]],
) -> float | None:
    """Per-item P90 |RE| across matched (pred, gt) pairs.

    P90 (not median) because greedy matching pairs the smallest-RE
    candidates first, which makes the median collapse to ~0 on common
    percentage claims. P90 preserves a meaningful long-tail signal.
    Returns ``None`` if no matches.
    """
    if predicted is None or not predicted:
        return None
    matches, _, _ = _match_extraction_claims(predicted, expected)
    if not matches:
        return None
    rel_errors = sorted(rel_err for _, _, rel_err in matches)
    if len(rel_errors) == 1:
        return rel_errors[0]
    rank = 0.9 * (len(rel_errors) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(rel_errors) - 1)
    frac = rank - lo
    return rel_errors[lo] + frac * (rel_errors[hi] - rel_errors[lo])


def score_unit_correctness(
    predicted: list[tuple[float, str]] | None,
    expected: list[tuple[float, str]],
) -> bool | None:
    """Diagnostic: any predicted unit overlaps any ground-truth unit
    (after normalisation)."""
    if predicted is None or not predicted:
        return None
    pred_units = {normalize_unit(u) for _, u in predicted}
    gt_units = {normalize_unit(u) for _, u in expected}
    return bool(pred_units & gt_units)


def score_best_relative_error(
    predicted: list[tuple[float, str]] | None,
    expected: list[tuple[float, str]],
) -> float | None:
    """Diagnostic: smallest relative error across any unit-matched pair."""
    if predicted is None or not predicted:
        return None
    best: float | None = None
    for p_val, p_unit in predicted:
        np_val, np_unit = normalize_value_and_unit(p_val, p_unit)
        for g_val, g_unit in expected:
            ng_val, ng_unit = normalize_value_and_unit(g_val, g_unit)
            if ng_unit != np_unit:
                continue
            if ng_val == 0:
                continue
            rel_err = abs(np_val - ng_val) / abs(ng_val)
            if best is None or rel_err < best:
                best = rel_err
    return best


# --- Run-level aggregates --------------------------------------------------


def run_claim_f1(
    matched_counts: Iterable[int | None],
    pred_counts: Iterable[int | None],
    gt_counts: Iterable[int],
) -> float:
    """Aggregate claim-F1 = 2PR/(P+R) where P and R use total-matched /
    total-pred / total-gt summed across all dataset items.

    This is the headline metric the existing harness's
    ``run_score_f1`` over filtered ``claim_matched`` / ``claim_pred`` /
    ``claim_gt_count`` per-item scores produces."""
    matched_total = sum(int(m) for m in matched_counts if m is not None)
    pred_total = sum(int(p) for p in pred_counts if p is not None)
    gt_total = sum(int(g) for g in gt_counts)
    p = matched_total / pred_total if pred_total else 0.0
    r = matched_total / gt_total if gt_total else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def run_p90_re_on_matched(values: Iterable[float | None]) -> float:
    """Run-level P90 of per-item P90 |RE| values (same statistic the
    existing harness reports)."""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = 0.9 * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] + frac * (vals[hi] - vals[lo])


def run_median_best_relative_error(values: Iterable[float | None]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return statistics.median(vals)


def run_mean_unit_correctness(values: Iterable[bool | None]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(1.0 if v else 0.0 for v in vals) / len(vals)
