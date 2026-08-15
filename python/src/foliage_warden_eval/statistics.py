"""Dependency-free statistical helpers with explicit conventions."""

from __future__ import annotations

import math
from collections.abc import Iterable


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def percentiles(values: Iterable[int | float], requested: Iterable[float]) -> dict[str, float] | None:
    """Return linearly interpolated (R-7/NumPy-default) percentiles."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    result: dict[str, float] = {}
    for percentile in requested:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentiles must be between 0 and 100")
        position = (len(ordered) - 1) * percentile / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            value = ordered[lower]
        else:
            fraction = position - lower
            value = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
        result[f"p{percentile:g}_ms"] = value
    return result


def _log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right > left:
        left, right = right, left
    return left + math.log1p(math.exp(right - left))


def _log_poisson_cdf(count: int, mean: float) -> float:
    if mean == 0.0:
        return 0.0
    log_mean = math.log(mean)
    total = -math.inf
    for value in range(count + 1):
        term = -mean + value * log_mean - math.lgamma(value + 1)
        total = _log_add(total, term)
    return total


def poisson_upper_mean(count: int, confidence: float = 0.95) -> float:
    """Exact one-sided Garwood upper bound for a Poisson mean.

    For zero observed events this is ``-log(1-confidence)``, the familiar
    rule-of-three value of 2.996 at 95% confidence.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1.0 - confidence
    if count == 0:
        return -math.log(alpha)

    target = math.log(alpha)
    low = float(count)
    high = max(2.0, count + 1.0)
    while _log_poisson_cdf(count, high) > target:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if _log_poisson_cdf(count, middle) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def poisson_rate_upper_bound(
    count: int,
    exposure_hours: float,
    confidence: float = 0.95,
) -> float | None:
    if exposure_hours < 0.0:
        raise ValueError("exposure_hours must be >= 0")
    if exposure_hours == 0.0:
        return None
    return poisson_upper_mean(count, confidence) / exposure_hours
