"""Triage scoring: per-item correctness + run-level accuracy / precision /
recall / F1.
"""

from __future__ import annotations


def score_correct(
    pred_should_map: bool | None,
    expected_should_map: bool | None,
) -> bool | None:
    """Per-item: True iff prediction matches expected."""
    if pred_should_map is None or expected_should_map is None:
        return None
    return pred_should_map == expected_should_map


def run_accuracy(items: list[tuple[bool | None, bool | None]]) -> float:
    """items: (pred, expected). Returns mean correctness over scored items."""
    paired = [(p, e) for p, e in items if p is not None and e is not None]
    if not paired:
        return 0.0
    return sum(1.0 for p, e in paired if p == e) / len(paired)


def run_precision(items: list[tuple[bool | None, bool | None]]) -> float:
    """Treat ``should_map=True`` as the positive class."""
    paired = [(p, e) for p, e in items if p is not None and e is not None]
    tp = sum(1 for p, e in paired if p and e)
    fp = sum(1 for p, e in paired if p and not e)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def run_recall(items: list[tuple[bool | None, bool | None]]) -> float:
    paired = [(p, e) for p, e in items if p is not None and e is not None]
    tp = sum(1 for p, e in paired if p and e)
    fn = sum(1 for p, e in paired if not p and e)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def run_f1(items: list[tuple[bool | None, bool | None]]) -> float:
    p = run_precision(items)
    r = run_recall(items)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
