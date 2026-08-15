from __future__ import annotations

import math

import pytest

from foliage_warden_eval.statistics import (
    percentiles,
    poisson_rate_upper_bound,
    poisson_upper_mean,
)


def test_rule_of_three_zero_event_bound() -> None:
    assert poisson_upper_mean(0, 0.95) == pytest.approx(-math.log(0.05))
    assert poisson_rate_upper_bound(0, 50.0, 0.95) == pytest.approx(0.05991464547)


def test_exact_nonzero_poisson_bound() -> None:
    # Standard one-sided 95% Garwood limit for one observed event.
    assert poisson_upper_mean(1, 0.95) == pytest.approx(4.743864518, rel=1e-8)
    assert poisson_rate_upper_bound(1, 0.0, 0.95) is None


def test_percentiles_use_linear_interpolation() -> None:
    assert percentiles([100, 200], [50, 95]) == {"p50_ms": 150.0, "p95_ms": 195.0}
    assert percentiles([], [50]) is None


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1])
def test_invalid_confidence_rejected(confidence: float) -> None:
    with pytest.raises(ValueError):
        poisson_upper_mean(0, confidence)

