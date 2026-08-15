"""Small deterministic IoU tracker for observe-only prototyping."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Detection, ObjectClass


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    detection: Detection
    track_id: str
    age_frames: int
    age_ms: int
    quality: float
    ambiguous: bool


@dataclass(slots=True)
class _TrackState:
    numeric_id: int
    object_class: ObjectClass
    detection: Detection
    first_frame_index: int
    first_timestamp_ms: int
    last_frame_index: int
    hits: int = 1
    missed_updates: int = 0

    @property
    def track_id(self) -> str:
        return f"{self.object_class.value.lower()}-{self.numeric_id:06d}"


class IouTracker:
    """Greedy class-specific IoU association with stable tie breaking.

    This is deliberately modest rather than pretending to be a production
    motion tracker. Quality combines three-frame maturity with continuity;
    tracks absent from the current frame are retained for reacquisition but
    are never emitted as current observations.
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_missed_frames: int = 5) -> None:
        if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("tracker IoU threshold must be finite and within [0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._states: dict[int, _TrackState] = {}
        self._next_id = 1
        self._last_frame_index: int | None = None
        self._last_timestamp_ms: int | None = None

    def reset(self) -> None:
        self._states.clear()
        self._next_id = 1
        self._last_frame_index = None
        self._last_timestamp_ms = None

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[TrackedDetection]:
        if frame_index < 0 or timestamp_ms < 0:
            raise ValueError("frame index and timestamp must be non-negative")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("tracker frame indices must be strictly increasing")
        if self._last_timestamp_ms is not None and timestamp_ms < self._last_timestamp_ms:
            raise ValueError("tracker timestamps must be non-decreasing")
        self._last_frame_index = frame_index
        self._last_timestamp_ms = timestamp_ms

        ordered_detections = sorted(
            enumerate(detections),
            key=lambda item: (
                item[1].object_class.value,
                -item[1].confidence,
                item[1].prediction_index,
                item[0],
            ),
        )
        candidate_pairs: list[tuple[float, int, int]] = []
        candidate_tracks_by_detection: dict[int, int] = {}
        candidate_detections_by_track: dict[int, int] = {}
        for detection_index, detection in ordered_detections:
            for numeric_id, state in sorted(self._states.items()):
                if state.object_class is not detection.object_class:
                    continue
                overlap = state.detection.bbox.iou(detection.bbox)
                if overlap < self.iou_threshold:
                    continue
                candidate_pairs.append((overlap, numeric_id, detection_index))
                candidate_tracks_by_detection[detection_index] = (
                    candidate_tracks_by_detection.get(detection_index, 0) + 1
                )
                candidate_detections_by_track[numeric_id] = (
                    candidate_detections_by_track.get(numeric_id, 0) + 1
                )
        candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        assignments: dict[int, int] = {}
        assigned_tracks: set[int] = set()
        for _, numeric_id, detection_index in candidate_pairs:
            if numeric_id in assigned_tracks or detection_index in assignments:
                continue
            assignments[detection_index] = numeric_id
            assigned_tracks.add(numeric_id)

        for numeric_id, state in list(self._states.items()):
            if numeric_id not in assigned_tracks:
                state.missed_updates += 1
                if state.missed_updates > self.max_missed_frames:
                    del self._states[numeric_id]

        results: list[TrackedDetection] = []
        for detection_index, detection in ordered_detections:
            numeric_id = assignments.get(detection_index)
            if numeric_id is None:
                numeric_id = self._next_id
                self._next_id += 1
                state = _TrackState(
                    numeric_id=numeric_id,
                    object_class=detection.object_class,
                    detection=detection,
                    first_frame_index=frame_index,
                    first_timestamp_ms=timestamp_ms,
                    last_frame_index=frame_index,
                )
                self._states[numeric_id] = state
            else:
                state = self._states[numeric_id]
                state.detection = detection
                state.last_frame_index = frame_index
                state.hits += 1
                state.missed_updates = 0

            age_frames = frame_index - state.first_frame_index + 1
            continuity = state.hits / age_frames
            maturity = min(1.0, state.hits / 3.0)
            quality = max(0.0, min(1.0, continuity * maturity))
            ambiguous = (
                candidate_tracks_by_detection.get(detection_index, 0) > 1
                or candidate_detections_by_track.get(numeric_id, 0) > 1
            )
            results.append(
                TrackedDetection(
                    detection=detection,
                    track_id=state.track_id,
                    age_frames=age_frames,
                    age_ms=timestamp_ms - state.first_timestamp_ms,
                    quality=quality,
                    ambiguous=ambiguous,
                )
            )
        results.sort(key=lambda item: item.track_id)
        return results
