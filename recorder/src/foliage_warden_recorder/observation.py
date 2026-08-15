"""Strict validation and trigger extraction for observe-only perception JSON."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ObservationError
from .types import RecorderFrame


@dataclass(frozen=True, slots=True)
class ObservationOrder:
    sequence: int
    captured_at_ms: int
    camera_id: str
    source_kind: str
    source_name: str


@dataclass(frozen=True, slots=True)
class TriggerEvidence:
    active: bool
    track_ids: tuple[str, ...]
    maximum_approach_overlap: float


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
) -> tuple[ObservationOrder, TriggerEvidence]:
    """Reject ambiguous pairing before any frame enters a privacy-sensitive buffer."""

    root = _mapping(record, "observation record")
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
    camera_id = _text(observation.get("camera_id"), "observation.camera_id")
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

    current = ObservationOrder(sequence, captured_at_ms, camera_id, source_kind, source_name)
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

    tracks = observation.get("tracks")
    if not isinstance(tracks, list):
        raise ObservationError("observation.tracks must be an array")
    cat_count = _integer(root.get("cat_count"), "cat_count")
    cat_tracks = 0
    trigger_tracks: list[tuple[str, float]] = []
    seen_track_ids: set[str] = set()
    for index, value in enumerate(tracks):
        track = _mapping(value, f"observation.tracks[{index}]")
        track_id = _text(track.get("track_id"), f"observation.tracks[{index}].track_id")
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
    return current, TriggerEvidence(
        active=bool(trigger_tracks),
        track_ids=tuple(item[0] for item in trigger_tracks),
        maximum_approach_overlap=max((item[1] for item in trigger_tracks), default=0.0),
    )
