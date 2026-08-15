"""Strict validation and trigger extraction for observe-only perception JSON."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .errors import ObservationError
from .types import RecorderFrame

RECORD_CANONICALIZATION = "JSON_SORTED_KEYS_COMPACT_UTF8_V1"
BINDING_STREAM_CANONICALIZATION = "JSONL_FRAME_BINDINGS_SORTED_KEYS_COMPACT_UTF8_V1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_OBSERVATION_JSON_BYTES = 1024 * 1024
MAX_OBSERVATION_JSON_DEPTH = 32
MAX_OBSERVATION_JSON_NODES = 20_000
MAX_OBSERVATION_TRACKS = 512


@dataclass(frozen=True, slots=True)
class ObservationOrder:
    sequence: int
    captured_at_ms: int
    camera_id: str
    source_kind: str
    source_name: str
    observation_id: str
    frame_id: str


@dataclass(frozen=True, slots=True)
class PerceptionBinding:
    sequence: int
    captured_at_ms: int
    observation_id: str
    frame_id: str
    perception_record_sha256: str

    def to_dict(self, encoded_frame_index: int) -> dict[str, Any]:
        return {
            "captured_at_ms": self.captured_at_ms,
            "encoded_frame_index": encoded_frame_index,
            "frame_id": self.frame_id,
            "observation_id": self.observation_id,
            "perception_record_sha256": self.perception_record_sha256,
            "sequence": self.sequence,
        }


@dataclass(frozen=True, slots=True)
class TriggerEvidence:
    active: bool
    track_ids: tuple[str, ...]
    maximum_approach_overlap: float


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SAFE_INTEGER:
        raise ObservationError(f"{field} must be a non-negative integer")
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservationError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ObservationError(f"{field} must be a non-empty string")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ObservationError(f"{field} must be a canonical identifier")
    return value


def validate_json_value(value: Any, field: str = "observation record") -> None:
    active_containers: set[int] = set()
    nodes = 0
    text_bytes = 0

    def account_text(text: str) -> None:
        nonlocal text_bytes
        if len(text) > MAX_OBSERVATION_JSON_BYTES:
            raise ObservationError(
                f"observation record exceeds the {MAX_OBSERVATION_JSON_BYTES}-byte canonical limit"
            )
        try:
            text_bytes += len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ObservationError(
                "observation record is not canonical JSON: text is not valid UTF-8"
            ) from error
        if text_bytes > MAX_OBSERVATION_JSON_BYTES:
            raise ObservationError(
                f"observation record exceeds the {MAX_OBSERVATION_JSON_BYTES}-byte canonical limit"
            )

    stack: list[tuple[Any, str, int, bool]] = [(value, field, 0, False)]
    while stack:
        item, context, depth, exiting = stack.pop()
        if exiting:
            active_containers.remove(id(item))
            continue
        nodes += 1
        if nodes > MAX_OBSERVATION_JSON_NODES:
            raise ObservationError(
                f"observation record exceeds the {MAX_OBSERVATION_JSON_NODES}-node limit"
            )
        if depth > MAX_OBSERVATION_JSON_DEPTH:
            raise ObservationError(
                f"observation record exceeds the {MAX_OBSERVATION_JSON_DEPTH}-level depth limit"
            )
        if item is None or type(item) is bool:
            continue
        if type(item) is str:
            account_text(item)
            continue
        if type(item) is int:
            if not -_MAX_SAFE_INTEGER <= item <= _MAX_SAFE_INTEGER:
                raise ObservationError(f"{context} integer is outside the interoperable safe range")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ObservationError(f"{context} contains a non-finite number")
            continue
        if type(item) not in {list, dict}:
            raise ObservationError(f"{context} contains a non-JSON value")

        identity = id(item)
        if identity in active_containers:
            raise ObservationError(f"{context} contains a cyclic JSON value")
        if nodes + len(item) > MAX_OBSERVATION_JSON_NODES:
            raise ObservationError(
                f"observation record exceeds the {MAX_OBSERVATION_JSON_NODES}-node limit"
            )
        active_containers.add(identity)
        stack.append((item, context, depth, True))
        if type(item) is list:
            stack.extend(
                (child, f"{context}[{index}]", depth + 1, False)
                for index, child in reversed(list(enumerate(item)))
            )
            continue
        for key in item:
            if type(key) is not str:
                raise ObservationError(f"{context} object keys must be strings")
            account_text(key)
        stack.extend(
            (child, f"{context}.{key}", depth + 1, False)
            for key, child in reversed(list(item.items()))
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the recorder's strict, byte-stable JSON representation."""

    validate_json_value(value)
    try:
        result = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (RecursionError, TypeError, UnicodeEncodeError, ValueError) as error:
        raise ObservationError(f"observation record is not canonical JSON: {error}") from error
    if len(result) > MAX_OBSERVATION_JSON_BYTES:
        raise ObservationError(
            f"observation record exceeds the {MAX_OBSERVATION_JSON_BYTES}-byte canonical limit"
        )
    return result


def binding_stream_sha256(bindings: Sequence[PerceptionBinding]) -> str:
    """Hash ordered canonical frame bindings using unambiguous JSONL framing."""

    digest = hashlib.sha256()
    for encoded_frame_index, binding in enumerate(bindings):
        digest.update(canonical_json_bytes(binding.to_dict(encoded_frame_index)))
        digest.update(b"\n")
    return digest.hexdigest()


def _overlap(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservationError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ObservationError(f"{field} must be a finite number in [0, 1]")
    return result


def validate_and_extract_trigger(
    record: Any,
    frame: RecorderFrame,
    *,
    previous: ObservationOrder | None,
    minimum_approach_overlap: float,
) -> tuple[ObservationOrder, TriggerEvidence, PerceptionBinding]:
    """Reject ambiguous pairing before any frame enters a privacy-sensitive buffer."""

    root = _mapping(record, "observation record")
    canonical_record = canonical_json_bytes(root)
    if root.get("record_type") != "perception_observation":
        raise ObservationError("record_type must be perception_observation")
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ObservationError("schema_version must be 1")
    if root.get("mode") != "OBSERVE_ONLY":
        raise ObservationError("recorder accepts only OBSERVE_ONLY observations")
    if root.get("would_action") is not False:
        raise ObservationError("would_action must be exactly false")

    sequence = _integer(root.get("sequence"), "sequence")
    frame_record = _mapping(root.get("frame"), "frame")
    frame_index = _integer(frame_record.get("index"), "frame.index")
    observation = _mapping(root.get("observation"), "observation")
    captured_at_ms = _integer(observation.get("captured_at_ms"), "observation.captured_at_ms")
    camera_id = _identifier(observation.get("camera_id"), "observation.camera_id")
    observation_id = _identifier(observation.get("observation_id"), "observation.observation_id")
    frame_id = _identifier(observation.get("frame_id"), "observation.frame_id")
    source = _mapping(root.get("source"), "source")
    source_kind = _text(source.get("kind"), "source.kind")
    source_name = _text(source.get("name"), "source.name")

    if sequence != frame.sequence or frame_index != frame.sequence:
        raise ObservationError("observation sequence does not match its paired frame")
    if captured_at_ms != frame.captured_at_ms:
        raise ObservationError("observation timestamp does not match its paired frame")
    if camera_id != frame.camera_id:
        raise ObservationError("observation camera_id does not match its paired frame")
    if source_kind != frame.source_kind or source_name != frame.source_name:
        raise ObservationError("observation source does not match its paired frame")

    current = ObservationOrder(
        sequence,
        captured_at_ms,
        camera_id,
        source_kind,
        source_name,
        observation_id,
        frame_id,
    )
    if previous is not None:
        if sequence <= previous.sequence:
            raise ObservationError("observation sequences must be strictly increasing")
        if captured_at_ms <= previous.captured_at_ms:
            raise ObservationError("observation timestamps must be strictly increasing")
        if (camera_id, source_kind, source_name) != (
            previous.camera_id,
            previous.source_kind,
            previous.source_name,
        ):
            raise ObservationError("camera and source identity cannot change within one recorder")
        if observation_id == previous.observation_id or frame_id == previous.frame_id:
            raise ObservationError("observation_id and frame_id must change with every frame")

    tracks = observation.get("tracks")
    if not isinstance(tracks, list):
        raise ObservationError("observation.tracks must be an array")
    if len(tracks) > MAX_OBSERVATION_TRACKS:
        raise ObservationError(
            f"observation.tracks exceeds the {MAX_OBSERVATION_TRACKS}-track limit"
        )
    cat_count = _integer(root.get("cat_count"), "cat_count")
    cat_tracks = 0
    trigger_tracks: list[tuple[str, float]] = []
    seen_track_ids: set[str] = set()
    for index, value in enumerate(tracks):
        track = _mapping(value, f"observation.tracks[{index}]")
        track_id = _identifier(track.get("track_id"), f"observation.tracks[{index}].track_id")
        if track_id in seen_track_ids:
            raise ObservationError("track_id values must be unique within an observation")
        seen_track_ids.add(track_id)
        object_class = _text(track.get("class"), f"observation.tracks[{index}].class")
        if object_class != "CAT":
            continue
        cat_tracks += 1
        region = _mapping(
            track.get("region_evidence"),
            f"observation.tracks[{index}].region_evidence",
        )
        overlap = _overlap(
            region.get("approach_overlap"),
            f"observation.tracks[{index}].region_evidence.approach_overlap",
        )
        if overlap >= minimum_approach_overlap:
            trigger_tracks.append((track_id, overlap))
    if cat_count != cat_tracks:
        raise ObservationError("cat_count does not match CAT tracks")

    trigger_tracks.sort(key=lambda item: item[0])
    record_sha256 = hashlib.sha256(canonical_record).hexdigest()
    return (
        current,
        TriggerEvidence(
            active=bool(trigger_tracks),
            track_ids=tuple(item[0] for item in trigger_tracks),
            maximum_approach_overlap=max((item[1] for item in trigger_tracks), default=0.0),
        ),
        PerceptionBinding(
            sequence=sequence,
            captured_at_ms=captured_at_ms,
            observation_id=observation_id,
            frame_id=frame_id,
            perception_record_sha256=record_sha256,
        ),
    )
