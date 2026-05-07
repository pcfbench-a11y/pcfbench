"""Extraction scoring: exact-tuple match of predicted claims against
ground-truth, then per-item P90|RE| / unit-correctness / best-RE
diagnostics, and run-level claim-F1 from aggregated counts.

A predicted claim ``(value, unit)`` matches a ground-truth claim only
when their ``(round(value, 6), canonical_unit)`` tuples are identical
under the same unit normalization the dataset build uses (so e.g.
``mass%`` ↔ ``%``, ``g/kg`` ↔ ``kg/kg`` with appropriate scaling).
Each pred and each GT can be used at most once (multiset semantics).
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
    """Exact-tuple match of predicted claims to ground-truth claims.

    A predicted claim ``(value, unit)`` matches a GT claim iff their
    ``(round(value, 6), canonical_unit)`` tuples are equal after the
    same unit normalization the dataset build uses (which scales values
    when collapsing equivalent unit families, e.g. ``g/kg`` × 0.001 →
    ``kg/kg``). Each pred and each GT can be used at most once
    (multiset semantics): if a model emits the same tuple ``k`` times
    against a GT that has it ``j`` times, ``min(k, j)`` matches count.

    For each predicted claim we still try several unit cleanups (raw,
    comma-truncated, paren-truncated, first-token); the first variant
    that normalizes to a unit appearing in GT is used. This recovers
    matches when reasoning models pack prose into the unit field
    (e.g. ``"h, precipitation heat treatment time"``).

    Relative error on matched pairs is 0 by construction (only float
    cleanup at the 1e-6 level survives), so the third element of each
    match tuple is reported as 0.0 and ``score_p90_re_on_matched``
    is vestigial under this matcher.

    Returns ``(matches, n_unmatched_gt, n_unmatched_pred)`` where
    ``matches`` is a list of ``(pred_idx, gt_idx, 0.0)``.
    """
    gt_normalized = [normalize_value_and_unit(v, u) for v, u in gt_claims]
    gt_units = {g_unit for _, g_unit in gt_normalized}

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

    # Bucket GT indices by (rounded value, canonical unit). Iteration
    # order over a list of indices in each bucket is insertion order,
    # so ties are resolved deterministically by GT position.
    gt_by_key: dict[tuple[float, str], list[int]] = {}
    for g_idx, (g_val, g_unit) in enumerate(gt_normalized):
        key = (round(g_val, 6), g_unit)
        gt_by_key.setdefault(key, []).append(g_idx)

    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for p_idx, (p_val, p_unit) in enumerate(pred_normalized):
        key = (round(p_val, 6), p_unit)
        for g_idx in gt_by_key.get(key, ()):
            if g_idx in used_g:
                continue
            used_g.add(g_idx)
            matches.append((p_idx, g_idx, 0.0))
            break

    used_p = {p_idx for p_idx, _, _ in matches}
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

    Vestigial under exact-tuple matching: matched pairs have RE == 0
    by construction, so this returns ``0.0`` whenever there is at
    least one match and ``None`` otherwise. Kept for downstream API
    compatibility (``run_p90_re_on_matched``, summary tables) and
    because ``score_best_relative_error`` -- which scans *all*
    unit-matched pairs ignoring assignment -- still carries the
    relative-error signal the old greedy matcher expressed here.
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
