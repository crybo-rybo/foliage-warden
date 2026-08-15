"""Small immutable values shared by the simulator modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScheduledInput:
    at_ms: int
    sequence: int
    event_id: str
    payload: JsonObject = field(compare=False)


@dataclass(order=True, frozen=True, slots=True)
class InternalWork:
    due_ms: int
    insertion_order: int
    kind: str = field(compare=False)
    payload: JsonObject = field(compare=False)


@dataclass(slots=True)
class Persistence:
    supporting_ms: list[int] = field(default_factory=list)
    last_seen_ms: int | None = None


@dataclass(slots=True)
class Incident:
    incident_id: str
    track_id: str
    zone_id: str
    behavior: str
    started_ms: int
    last_harmful_ms: int
    latest_observation: JsonObject
    action_latched: bool = False
    clear_since_ms: int | None = None
    ready_ms: int | None = None
    ended_ms: int | None = None


@dataclass(slots=True)
class PendingCommand:
    command: JsonObject
    issued_at_ms: int
    state_before: str
    state_after: str
    trigger_event_id: str
    incident_id: str | None
    observation: JsonObject | None
    evidence_snapshot: JsonObject
    safety_snapshot: JsonObject
    response: str
    epoch: int
    finished: bool = False


@dataclass(frozen=True, slots=True)
class Assertion:
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"name": self.name, "passed": self.passed}
        if not self.passed:
            result["actual"] = self.actual
            result["expected"] = self.expected
        return result
