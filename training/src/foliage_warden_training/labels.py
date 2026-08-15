from __future__ import annotations

import hashlib
import json
from enum import Enum


class BehaviorLabel(str, Enum):
    PASSING = "PASSING"
    SNIFFING = "SNIFFING"
    EATING = "EATING"
    DIGGING = "DIGGING"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


BEHAVIOR_LABELS = tuple(label.value for label in BehaviorLabel)
LABEL_TO_INDEX = {label: index for index, label in enumerate(BEHAVIOR_LABELS)}
HARMFUL_LABELS = frozenset({BehaviorLabel.EATING.value, BehaviorLabel.DIGGING.value})
LABEL_SCHEMA_VERSION = 1


def label_schema_id() -> str:
    payload = json.dumps(
        {"version": LABEL_SCHEMA_VERSION, "labels": BEHAVIOR_LABELS},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_label(value: object) -> str:
    if not isinstance(value, str) or value not in LABEL_TO_INDEX:
        expected = ", ".join(BEHAVIOR_LABELS)
        raise ValueError(f"label must be one of [{expected}], got {value!r}")
    return value
