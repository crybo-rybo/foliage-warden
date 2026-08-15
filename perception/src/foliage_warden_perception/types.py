"""Typed, normalized perception primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ObjectClass(str, Enum):
    """The only object classes exposed by the perception boundary."""

    PERSON = "PERSON"
    CAT = "CAT"


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """An axis-aligned ``x, y, width, height`` box in normalized image space."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized box coordinates must be finite")
        tolerance = 1e-9
        if self.x < 0.0 or self.y < 0.0 or self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("normalized box must have a non-negative origin and positive size")
        if self.x + self.width > 1.0 + tolerance or self.y + self.height > 1.0 + tolerance:
            raise ValueError("normalized box must fit within [0, 1] image space")

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: NormalizedBox) -> float:
        intersection_width = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        intersection_height = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        intersection = intersection_width * intersection_height
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "height": _wire_float(self.height),
            "width": _wire_float(self.width),
            "x": _wire_float(self.x),
            "y": _wire_float(self.y),
        }


@dataclass(frozen=True, slots=True)
class Detection:
    """One post-NMS cat or person detection."""

    object_class: ObjectClass
    class_id: int
    confidence: float
    bbox: NormalizedBox
    prediction_index: int = 0

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and within [0, 1]")
        if self.prediction_index < 0:
            raise ValueError("prediction_index must be non-negative")


def _wire_float(value: float, digits: int = 6) -> float:
    """Bound platform noise while retaining far more precision than a camera pixel."""

    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


JsonObject = dict[str, Any]
