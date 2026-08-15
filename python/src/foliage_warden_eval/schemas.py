"""Typed JSON wire schemas used by replay and evaluation.

The parser is deliberately strict. A typo in a safety flag must fail evaluation
instead of silently falling back to a permissive default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, TypeAlias

SCHEMA_VERSION = 1
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class SchemaError(ValueError):
    """Raised when a JSON object does not conform to a replay schema."""


class Behavior(str, Enum):
    PASSING = "PASSING"
    SNIFFING = "SNIFFING"
    EATING = "EATING"
    DIGGING = "DIGGING"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    @property
    def is_harmful(self) -> bool:
        return self in (Behavior.EATING, Behavior.DIGGING)


class PolicyState(str, Enum):
    DISARMED = "DISARMED"
    MONITORING = "MONITORING"
    TRACKING = "TRACKING"
    CONFIRMING = "CONFIRMING"
    AIMING = "AIMING"
    READY = "READY"
    BURST = "BURST"
    COOLDOWN = "COOLDOWN"
    HOLD = "HOLD"
    FAULT = "FAULT"


class ActionType(str, Enum):
    ARM = "ARM"
    DISARM = "DISARM"
    HOME = "HOME"
    GOTO_PRESET = "GOTO_PRESET"
    PAN_LEFT = "PAN_LEFT"
    PAN_RIGHT = "PAN_RIGHT"
    TILT_UP = "TILT_UP"
    TILT_DOWN = "TILT_DOWN"
    HOLD = "HOLD"
    BURST = "BURST"
    ESTOP = "ESTOP"
    STATUS = "STATUS"


class AckStatus(str, Enum):
    ACK = "ACK"
    DENIED = "DENIED"
    TIMEOUT = "TIMEOUT"
    MISSING = "MISSING"


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be a JSON object")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SchemaError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _required(data: Mapping[str, Any], name: str, context: str) -> Any:
    if name not in data:
        raise SchemaError(f"{context}.{name} is required")
    return data[name]


def _string(value: Any, context: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context} must be a number")
    return float(value)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{context} must be a boolean")
    return value


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if not isinstance(value, str):
        raise SchemaError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(member.value for member in enum_type)
        raise SchemaError(f"{context} must be one of: {choices}") from error


def _metadata(value: Any, context: str) -> dict[str, JsonValue]:
    mapping = _require_mapping(value, context)
    result = dict(mapping)
    # JSON round trips are also a practical recursive type check, but avoid an
    # import and retain the original useful field-level error here.
    def validate_json(item: Any, path: str) -> None:
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate_json(child, f"{path}[{index}]")
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for key, child in item.items():
                validate_json(child, f"{path}.{key}")
            return
        raise SchemaError(f"{path} is not a JSON value")

    validate_json(result, context)
    return result


def _base_fields(data: Mapping[str, Any], record_type: str, context: str) -> None:
    if data.get("record_type") != record_type:
        raise SchemaError(f"{context}.record_type must be {record_type!r}")
    version = _integer(data.get("schema_version", SCHEMA_VERSION), f"{context}.schema_version", minimum=1)
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"{context}.schema_version {version} is unsupported; expected {SCHEMA_VERSION}"
        )


def _wire_dict(instance: Any) -> dict[str, JsonValue]:
    def convert(value: Any) -> JsonValue:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(key): convert(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(child) for child in value]
        return value

    return convert(asdict(instance))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GroundTruthEvent:
    record_type: ClassVar[str] = "ground_truth_event"

    event_id: str
    session_id: str
    behavior: Behavior
    start_ms: int
    end_ms: int
    zone_id: str | None = None
    staged_safe: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GroundTruthEvent:
        data = _require_mapping(raw, "ground_truth_event")
        allowed = {
            "record_type", "schema_version", "event_id", "session_id", "behavior",
            "start_ms", "end_ms", "zone_id", "staged_safe", "metadata",
        }
        _reject_unknown(data, allowed, "ground_truth_event")
        _base_fields(data, cls.record_type, "ground_truth_event")
        start_ms = _integer(_required(data, "start_ms", "ground_truth_event"), "ground_truth_event.start_ms")
        end_ms = _integer(_required(data, "end_ms", "ground_truth_event"), "ground_truth_event.end_ms")
        assert start_ms is not None and end_ms is not None
        if end_ms <= start_ms:
            raise SchemaError("ground_truth_event.end_ms must be greater than start_ms")
        return cls(
            event_id=_string(_required(data, "event_id", "ground_truth_event"), "ground_truth_event.event_id"),  # type: ignore[arg-type]
            session_id=_string(_required(data, "session_id", "ground_truth_event"), "ground_truth_event.session_id"),  # type: ignore[arg-type]
            behavior=_enum(Behavior, _required(data, "behavior", "ground_truth_event"), "ground_truth_event.behavior"),
            start_ms=start_ms,
            end_ms=end_ms,
            zone_id=_string(data.get("zone_id"), "ground_truth_event.zone_id", optional=True),
            staged_safe=_boolean(data.get("staged_safe", False), "ground_truth_event.staged_safe"),
            metadata=_metadata(data.get("metadata", {}), "ground_truth_event.metadata"),
            schema_version=SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        result = _wire_dict(self)
        result["record_type"] = self.record_type
        return result


@dataclass(frozen=True, slots=True)
class PredictionEvent:
    record_type: ClassVar[str] = "prediction_event"

    event_id: str
    session_id: str
    behavior: Behavior
    start_ms: int
    end_ms: int
    score: float
    would_action: bool = False
    ready_ms: int | None = None
    zone_id: str | None = None
    incident_id: str | None = None
    track_id: str | None = None
    person_present: bool = False
    cat_count: int = 1
    stale_input: bool = False
    track_lost: bool = False
    hardware_ready: bool = True
    no_fire_intersection: bool = False
    model_id: str | None = None
    config_hash: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PredictionEvent:
        data = _require_mapping(raw, "prediction_event")
        allowed = {
            "record_type", "schema_version", "event_id", "session_id", "behavior",
            "start_ms", "end_ms", "score", "would_action", "ready_ms", "zone_id",
            "incident_id", "track_id", "person_present", "cat_count", "stale_input",
            "track_lost", "hardware_ready", "no_fire_intersection", "model_id",
            "config_hash", "metadata",
        }
        _reject_unknown(data, allowed, "prediction_event")
        _base_fields(data, cls.record_type, "prediction_event")
        start_ms = _integer(_required(data, "start_ms", "prediction_event"), "prediction_event.start_ms")
        end_ms = _integer(_required(data, "end_ms", "prediction_event"), "prediction_event.end_ms")
        ready_ms = _integer(data.get("ready_ms"), "prediction_event.ready_ms", optional=True)
        assert start_ms is not None and end_ms is not None
        if end_ms <= start_ms:
            raise SchemaError("prediction_event.end_ms must be greater than start_ms")
        if ready_ms is not None and ready_ms < start_ms:
            raise SchemaError("prediction_event.ready_ms must be >= start_ms")
        score = _number(_required(data, "score", "prediction_event"), "prediction_event.score")
        if not 0.0 <= score <= 1.0:
            raise SchemaError("prediction_event.score must be between 0 and 1")
        return cls(
            event_id=_string(_required(data, "event_id", "prediction_event"), "prediction_event.event_id"),  # type: ignore[arg-type]
            session_id=_string(_required(data, "session_id", "prediction_event"), "prediction_event.session_id"),  # type: ignore[arg-type]
            behavior=_enum(Behavior, _required(data, "behavior", "prediction_event"), "prediction_event.behavior"),
            start_ms=start_ms,
            end_ms=end_ms,
            score=score,
            would_action=_boolean(data.get("would_action", False), "prediction_event.would_action"),
            ready_ms=ready_ms,
            zone_id=_string(data.get("zone_id"), "prediction_event.zone_id", optional=True),
            incident_id=_string(data.get("incident_id"), "prediction_event.incident_id", optional=True),
            track_id=_string(data.get("track_id"), "prediction_event.track_id", optional=True),
            person_present=_boolean(data.get("person_present", False), "prediction_event.person_present"),
            cat_count=_integer(data.get("cat_count", 1), "prediction_event.cat_count"),  # type: ignore[arg-type]
            stale_input=_boolean(data.get("stale_input", False), "prediction_event.stale_input"),
            track_lost=_boolean(data.get("track_lost", False), "prediction_event.track_lost"),
            hardware_ready=_boolean(data.get("hardware_ready", True), "prediction_event.hardware_ready"),
            no_fire_intersection=_boolean(data.get("no_fire_intersection", False), "prediction_event.no_fire_intersection"),
            model_id=_string(data.get("model_id"), "prediction_event.model_id", optional=True),
            config_hash=_string(data.get("config_hash"), "prediction_event.config_hash", optional=True),
            metadata=_metadata(data.get("metadata", {}), "prediction_event.metadata"),
            schema_version=SCHEMA_VERSION,
        )

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def reached_ready(self) -> bool:
        return self.ready_ms is not None

    def to_dict(self) -> dict[str, JsonValue]:
        result = _wire_dict(self)
        result["record_type"] = self.record_type
        return result


@dataclass(frozen=True, slots=True)
class SessionRecord:
    record_type: ClassVar[str] = "session"

    session_id: str
    monitored_duration_ms: int
    startup_state: PolicyState = PolicyState.DISARMED
    final_state: PolicyState | None = None
    failure_reason: str | None = None
    track_observation_ms: int | None = None
    track_lost_ms: int | None = None
    behavior_observation_ms: int | None = None
    unknown_behavior_ms: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SessionRecord:
        data = _require_mapping(raw, "session")
        allowed = {
            "record_type", "schema_version", "session_id", "monitored_duration_ms",
            "startup_state", "final_state", "failure_reason", "track_observation_ms",
            "track_lost_ms", "behavior_observation_ms", "unknown_behavior_ms", "metadata",
        }
        _reject_unknown(data, allowed, "session")
        _base_fields(data, cls.record_type, "session")
        track_observation_ms = _integer(data.get("track_observation_ms"), "session.track_observation_ms", optional=True)
        track_lost_ms = _integer(data.get("track_lost_ms"), "session.track_lost_ms", optional=True)
        behavior_observation_ms = _integer(data.get("behavior_observation_ms"), "session.behavior_observation_ms", optional=True)
        unknown_behavior_ms = _integer(data.get("unknown_behavior_ms"), "session.unknown_behavior_ms", optional=True)
        if (track_observation_ms is None) != (track_lost_ms is None):
            raise SchemaError("session track_observation_ms and track_lost_ms must be provided together")
        if (behavior_observation_ms is None) != (unknown_behavior_ms is None):
            raise SchemaError("session behavior_observation_ms and unknown_behavior_ms must be provided together")
        if track_observation_ms is not None and track_lost_ms is not None and track_lost_ms > track_observation_ms:
            raise SchemaError("session.track_lost_ms cannot exceed track_observation_ms")
        if behavior_observation_ms is not None and unknown_behavior_ms is not None and unknown_behavior_ms > behavior_observation_ms:
            raise SchemaError("session.unknown_behavior_ms cannot exceed behavior_observation_ms")
        final_raw = data.get("final_state")
        return cls(
            session_id=_string(_required(data, "session_id", "session"), "session.session_id"),  # type: ignore[arg-type]
            monitored_duration_ms=_integer(_required(data, "monitored_duration_ms", "session"), "session.monitored_duration_ms"),  # type: ignore[arg-type]
            startup_state=_enum(PolicyState, data.get("startup_state", "DISARMED"), "session.startup_state"),
            final_state=None if final_raw is None else _enum(PolicyState, final_raw, "session.final_state"),
            failure_reason=_string(data.get("failure_reason"), "session.failure_reason", optional=True),
            track_observation_ms=track_observation_ms,
            track_lost_ms=track_lost_ms,
            behavior_observation_ms=behavior_observation_ms,
            unknown_behavior_ms=unknown_behavior_ms,
            metadata=_metadata(data.get("metadata", {}), "session.metadata"),
            schema_version=SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        result = _wire_dict(self)
        result["record_type"] = self.record_type
        return result


@dataclass(frozen=True, slots=True)
class ActionRecord:
    record_type: ClassVar[str] = "action"

    action_id: str
    session_id: str
    timestamp_ms: int
    command_id: str
    action: ActionType
    state_before: PolicyState
    state_after: PolicyState
    incident_id: str | None = None
    ack_status: AckStatus | None = None
    is_retry: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ActionRecord:
        data = _require_mapping(raw, "action")
        allowed = {
            "record_type", "schema_version", "action_id", "session_id", "timestamp_ms",
            "command_id", "action", "state_before", "state_after", "incident_id",
            "ack_status", "is_retry", "metadata",
        }
        _reject_unknown(data, allowed, "action")
        _base_fields(data, cls.record_type, "action")
        ack_raw = data.get("ack_status")
        return cls(
            action_id=_string(_required(data, "action_id", "action"), "action.action_id"),  # type: ignore[arg-type]
            session_id=_string(_required(data, "session_id", "action"), "action.session_id"),  # type: ignore[arg-type]
            timestamp_ms=_integer(_required(data, "timestamp_ms", "action"), "action.timestamp_ms"),  # type: ignore[arg-type]
            command_id=_string(_required(data, "command_id", "action"), "action.command_id"),  # type: ignore[arg-type]
            action=_enum(ActionType, _required(data, "action", "action"), "action.action"),
            state_before=_enum(PolicyState, _required(data, "state_before", "action"), "action.state_before"),
            state_after=_enum(PolicyState, _required(data, "state_after", "action"), "action.state_after"),
            incident_id=_string(data.get("incident_id"), "action.incident_id", optional=True),
            ack_status=None if ack_raw is None else _enum(AckStatus, ack_raw, "action.ack_status"),
            is_retry=_boolean(data.get("is_retry", False), "action.is_retry"),
            metadata=_metadata(data.get("metadata", {}), "action.metadata"),
            schema_version=SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        result = _wire_dict(self)
        result["record_type"] = self.record_type
        return result


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """A manifest row; all rows in a session/group are assigned together."""

    item_id: str
    session_id: str
    group_id: str | None = None
    path: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DatasetItem:
        data = _require_mapping(raw, "dataset_item")
        allowed = {"item_id", "session_id", "group_id", "path", "metadata"}
        _reject_unknown(data, allowed, "dataset_item")
        return cls(
            item_id=_string(_required(data, "item_id", "dataset_item"), "dataset_item.item_id"),  # type: ignore[arg-type]
            session_id=_string(_required(data, "session_id", "dataset_item"), "dataset_item.session_id"),  # type: ignore[arg-type]
            group_id=_string(data.get("group_id"), "dataset_item.group_id", optional=True),
            path=_string(data.get("path"), "dataset_item.path", optional=True),
            metadata=_metadata(data.get("metadata", {}), "dataset_item.metadata"),
        )

    @property
    def split_group(self) -> str:
        return self.group_id or self.session_id

    def to_dict(self) -> dict[str, JsonValue]:
        return _wire_dict(self)


ReplayRecord: TypeAlias = SessionRecord | PredictionEvent | ActionRecord


def parse_ground_truth(raw: Mapping[str, Any]) -> GroundTruthEvent:
    return GroundTruthEvent.from_dict(raw)


def parse_replay_record(raw: Mapping[str, Any]) -> ReplayRecord:
    record_type = raw.get("record_type")
    parsers = {
        SessionRecord.record_type: SessionRecord.from_dict,
        PredictionEvent.record_type: PredictionEvent.from_dict,
        ActionRecord.record_type: ActionRecord.from_dict,
    }
    try:
        parser = parsers[record_type]
    except (KeyError, TypeError) as error:
        choices = ", ".join(sorted(parsers))
        raise SchemaError(f"record_type must be one of: {choices}") from error
    return parser(raw)
