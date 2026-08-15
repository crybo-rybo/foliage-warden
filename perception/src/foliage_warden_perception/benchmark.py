"""Stage-level latency summaries kept separate from deterministic observations."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from .types import _wire_float


@dataclass(slots=True)
class BenchmarkAccumulator:
    _samples: dict[str, list[float]] = field(default_factory=dict)

    def add(self, stage: str, milliseconds: float) -> None:
        if not math.isfinite(milliseconds) or milliseconds < 0.0:
            raise ValueError("benchmark samples must be finite and non-negative")
        self._samples.setdefault(stage, []).append(milliseconds)

    @property
    def frame_count(self) -> int:
        return len(self._samples.get("total", []))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def to_dict(self) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for stage in sorted(self._samples):
            values = self._samples[stage]
            if not values:
                continue
            stages[stage] = {
                "count": len(values),
                "max_ms": _wire_float(max(values), 3),
                "mean_ms": _wire_float(statistics.fmean(values), 3),
                "min_ms": _wire_float(min(values), 3),
                "p50_ms": _wire_float(self._percentile(values, 0.50), 3),
                "p95_ms": _wire_float(self._percentile(values, 0.95), 3),
            }
        total = self._samples.get("total", [])
        effective_fps = 1000.0 / statistics.fmean(total) if total and sum(total) > 0.0 else 0.0
        return {
            "effective_fps": _wire_float(effective_fps, 3),
            "frame_count": self.frame_count,
            "stages": stages,
        }
