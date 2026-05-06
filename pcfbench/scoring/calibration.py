"""Calibration metrics for self-reported model confidence.

Pairs ``(confidence, correct)`` are evaluated with:
  - **ECE** (expected calibration error, 10 equal-width bins): weighted
    mean of |bin-confidence - bin-accuracy| over non-empty bins.
  - **Brier score**: mean of (confidence - 1{correct})^2.
  - **Mean confidence**: mean of confidence over scored items.

All three are aggregate (run-level); per-item we just store the raw
confidence and the correctness bool — the aggregator does the rest.
"""

from __future__ import annotations


def _paired(
    items: list[tuple[float | None, bool | None]],
) -> list[tuple[float, float]]:
    """Filter to (conf, correct_as_float) pairs. Drops Nones."""
    out: list[tuple[float, float]] = []
    for conf, correct in items:
        if conf is None or correct is None:
            continue
        out.append((float(conf), 1.0 if correct else 0.0))
    return out


def run_mean_confidence(items: list[tuple[float | None, bool | None]]) -> float:
    pairs = _paired(items)
    if not pairs:
        return 0.0
    return sum(c for c, _ in pairs) / len(pairs)


def run_brier(items: list[tuple[float | None, bool | None]]) -> float:
    """Mean (confidence - correct)^2."""
    pairs = _paired(items)
    if not pairs:
        return 0.0
    return sum((c - y) ** 2 for c, y in pairs) / len(pairs)


def run_ece(
    items: list[tuple[float | None, bool | None]],
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error, equal-width bins on [0, 1].

    Bin k = [k/n_bins, (k+1)/n_bins). Confidence == 1.0 lands in the
    last bin. Empty bins are skipped (zero weight, no contribution).
    """
    pairs = _paired(items)
    if not pairs:
        return 0.0
    n = len(pairs)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for conf, y in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, y))
    total = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bin_conf = sum(c for c, _ in bucket) / len(bucket)
        bin_acc = sum(y for _, y in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(bin_conf - bin_acc)
    return total
