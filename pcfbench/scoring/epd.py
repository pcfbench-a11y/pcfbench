"""EPD (Task 7) scoring: signed relative error per item, then
run-level median |RE|, within-factor-2 / within-factor-5.

``within_factor_F`` is symmetric in the prediction-to-truth ratio:
``abs(log2(pred / truth)) < log2(F)``, equivalently
``1/F <= pred/truth <= F``. The legacy harness used
``abs(rel_err) < F-1``, which is asymmetric — it credited any
under-prediction (e.g. ``pred = 0.1 * truth`` registered as "within
2×" because |rel_err|=0.9 < 1.0) while bounding over-predictions
correctly. The symmetric form is what reviewers expect when they read
"within 2×" and is what the paper now reports.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def score_relative_error(
    predicted_kgco2e: float | None,
    expected_kgco2e: float | None,
) -> float | None:
    """Signed relative error: ``(pred - truth) / truth``. ``None`` if
    either value is missing or truth is zero."""
    if predicted_kgco2e is None or expected_kgco2e is None or expected_kgco2e == 0:
        return None
    return (predicted_kgco2e - expected_kgco2e) / expected_kgco2e


def run_median_abs_relative_error(values: Iterable[float | None]) -> float:
    vals = [abs(float(v)) for v in values if v is not None]
    if not vals:
        return 0.0
    return statistics.median(vals)


def _within_factor(values: Iterable[float | None], factor: float) -> float:
    """Fraction with ``abs(log2(pred / truth)) < log2(factor)``.

    Inputs are signed ``rel_err = (pred - truth) / truth``, so
    ``ratio = pred / truth = rel_err + 1``. Predictions of zero or
    negative kgCO₂e are never within any finite factor.
    """
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    log_F = math.log2(factor)

    def _ok(rel_err: float) -> bool:
        ratio = rel_err + 1.0
        if ratio <= 0:
            return False
        return abs(math.log2(ratio)) < log_F

    return sum(1 for v in vals if _ok(v)) / len(vals)


def run_within_factor_2(values: Iterable[float | None]) -> float:
    """Fraction of items with prediction within a 2× ratio of truth
    (symmetric: ``0.5 <= pred/truth <= 2.0``,
    equivalently ``abs(log2(pred/truth)) < 1.0``)."""
    return _within_factor(values, 2.0)


def run_within_factor_5(values: Iterable[float | None]) -> float:
    """Fraction of items with prediction within a 5× ratio of truth
    (symmetric: ``0.2 <= pred/truth <= 5.0``,
    equivalently ``abs(log2(pred/truth)) < log2(5)``)."""
    return _within_factor(values, 5.0)


def run_mean_relative_error(values: Iterable[float | None]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
