"""Bounded rolling buffer and one-clip-per-continuous-incident state machine."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .encoding import ClipEncoder
from .errors import ObservationError, RecorderStateError
from .observation import (
    BINDING_STREAM_CANONICALIZATION,
    RECORD_CANONICALIZATION,
    ObservationOrder,
    PerceptionBinding,
    TriggerEvidence,
    binding_stream_sha256,
    validate_and_extract_trigger,
)
from .storage import IncidentStore, PublishedIncident
from .types import RecorderConfig, RecorderFrame


@dataclass(frozen=True, slots=True)
class _TriggerSample:
    sequence: int
    captured_at_ms: int
    track_ids: tuple[str, ...]
    maximum_approach_overlap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at_ms": self.captured_at_ms,
            "maximum_approach_overlap": self.maximum_approach_overlap,
            "sequence": self.sequence,
            "track_ids": list(self.track_ids),
        }


@dataclass(frozen=True, slots=True)
class _PairedFrame:
    frame: RecorderFrame
    binding: PerceptionBinding

    @property
    def sequence(self) -> int:
        return self.frame.sequence

    @property
    def captured_at_ms(self) -> int:
        return self.frame.captured_at_ms

    @property
    def pixel_bytes(self) -> int:
        return self.frame.pixel_bytes


@dataclass(slots=True)
class _Incident:
    frames: list[_PairedFrame]
    pixel_bytes: int
    first_trigger: _TriggerSample
    last_trigger_ms: int
    last_observation_triggered: bool
    trigger_samples: list[_TriggerSample] = field(default_factory=list)


class IncidentRecorder:
    """Observe-only recorder with no action, policy, arming, or network surface."""

    def __init__(
        self,
        store: IncidentStore,
        encoder: ClipEncoder,
        config: RecorderConfig,
    ) -> None:
        self._store = store
        self._encoder = encoder
        self._config = config
        self._prebuffer: deque[_PairedFrame] = deque()
        self._prebuffer_bytes = 0
        self._active: _Incident | None = None
        self._suppressed_until_clear = False
        self._previous: ObservationOrder | None = None
        self._seen_observation_ids: set[str] = set()
        self._seen_frame_ids: set[str] = set()
        self._closed = False

    @property
    def buffered_frame_count(self) -> int:
        return len(self._prebuffer)

    @property
    def buffered_byte_count(self) -> int:
        return self._prebuffer_bytes

    @property
    def incident_active(self) -> bool:
        return self._active is not None

    @property
    def active_frame_count(self) -> int:
        return len(self._active.frames) if self._active is not None else 0

    @property
    def active_byte_count(self) -> int:
        return self._active.pixel_bytes if self._active is not None else 0

    @property
    def suppressed_until_clear(self) -> bool:
        return self._suppressed_until_clear

    @property
    def accepted_observation_count(self) -> int:
        return len(self._seen_observation_ids)

    def _drop_oldest_prebuffer(self) -> None:
        dropped = self._prebuffer.popleft()
        self._prebuffer_bytes -= dropped.pixel_bytes

    def _clear_prebuffer(self) -> None:
        self._prebuffer.clear()
        self._prebuffer_bytes = 0

    def _append_prebuffer(self, paired: _PairedFrame) -> None:
        self._prebuffer.append(paired)
        self._prebuffer_bytes += paired.pixel_bytes
        cutoff = paired.captured_at_ms - self._config.pre_event_ms
        while self._prebuffer and self._prebuffer[0].captured_at_ms < cutoff:
            self._drop_oldest_prebuffer()
        while self._prebuffer and (
            len(self._prebuffer) > self._config.max_buffer_frames
            or self._prebuffer_bytes > self._config.max_buffer_bytes
        ):
            self._drop_oldest_prebuffer()

    @staticmethod
    def _sample(frame: RecorderFrame, trigger: TriggerEvidence) -> _TriggerSample:
        return _TriggerSample(
            frame.sequence,
            frame.captured_at_ms,
            trigger.track_ids,
            trigger.maximum_approach_overlap,
        )

    def _start(self, paired: _PairedFrame, trigger: TriggerEvidence) -> None:
        sample = self._sample(paired.frame, trigger)
        frames = list(self._prebuffer)
        pixel_bytes = self._prebuffer_bytes
        if not frames or frames[-1].sequence != paired.sequence:
            frames = [paired]
            pixel_bytes = paired.pixel_bytes
        while len(frames) > 1 and (
            len(frames) > self._config.max_active_frames
            or pixel_bytes > self._config.max_active_bytes
        ):
            pixel_bytes -= frames.pop(0).pixel_bytes
        self._clear_prebuffer()
        if (
            len(frames) > self._config.max_active_frames
            or pixel_bytes > self._config.max_active_bytes
        ):
            self._suppressed_until_clear = True
            return
        self._active = _Incident(
            frames=frames,
            pixel_bytes=pixel_bytes,
            first_trigger=sample,
            last_trigger_ms=paired.captured_at_ms,
            last_observation_triggered=True,
            trigger_samples=[sample],
        )

    def _metadata(self, incident: _Incident, termination: str) -> dict[str, Any]:
        first = incident.frames[0].frame
        last = incident.frames[-1].frame
        trigger = incident.first_trigger
        incident_id = f"incident-{trigger.captured_at_ms:013d}-{trigger.sequence:010d}"
        bindings = [paired.binding for paired in incident.frames]
        return {
            "incident_id": incident_id,
            "mode": "OBSERVE_ONLY",
            "privacy": {"audio": False, "display": False, "network": False},
            "perception_provenance": {
                "binding_stream_canonicalization": BINDING_STREAM_CANONICALIZATION,
                "frame_bindings": [
                    binding.to_dict(encoded_frame_index)
                    for encoded_frame_index, binding in enumerate(bindings)
                ],
                "record_canonicalization": RECORD_CANONICALIZATION,
                "record_count": len(bindings),
                "stream_sha256": binding_stream_sha256(bindings),
            },
            "record_type": "observation_clip",
            "resource_limits": {
                "max_active_bytes": self._config.max_active_bytes,
                "max_active_frames": self._config.max_active_frames,
                "max_buffer_bytes": self._config.max_buffer_bytes,
                "max_buffer_frames": self._config.max_buffer_frames,
            },
            "schema_version": 1,
            "source": {
                "camera_id": first.camera_id,
                "kind": first.source_kind,
                "name": first.source_name,
            },
            "termination": termination,
            "timeline": {
                "duration_ms": last.captured_at_ms - first.captured_at_ms,
                "end_captured_at_ms": last.captured_at_ms,
                "end_sequence": last.sequence,
                "first_trigger_at_ms": trigger.captured_at_ms,
                "last_trigger_at_ms": incident.last_trigger_ms,
                "start_captured_at_ms": first.captured_at_ms,
                "start_sequence": first.sequence,
            },
            "trigger": {
                "minimum_approach_overlap": self._config.minimum_approach_overlap,
                "rule": "CAT_IN_APPROACH_ZONE",
                "samples": [sample.to_dict() for sample in incident.trigger_samples],
            },
        }

    def _finalize(self, termination: str) -> PublishedIncident:
        incident = self._active
        if incident is None or not incident.frames:
            raise RecorderStateError("cannot finalize without an active incident")
        self._active = None
        metadata = self._metadata(incident, termination)
        return self._store.publish(
            incident_id=metadata["incident_id"],
            frames=[paired.frame for paired in incident.frames],
            metadata=metadata,
            encoder=self._encoder,
            fps=self._config.nominal_fps,
        )

    def process(
        self,
        frame: RecorderFrame,
        observation: dict[str, Any],
    ) -> PublishedIncident | None:
        if self._closed:
            raise RecorderStateError("recorder is closed")
        if self.accepted_observation_count >= self._config.max_accepted_observations:
            raise RecorderStateError(
                "recorder reached max_accepted_observations; rotate the bounded session"
            )
        order, trigger, binding = validate_and_extract_trigger(
            observation,
            frame,
            previous=self._previous,
            minimum_approach_overlap=self._config.minimum_approach_overlap,
        )
        if binding.observation_id in self._seen_observation_ids:
            raise ObservationError("observation_id must be unique across the recorder stream")
        if binding.frame_id in self._seen_frame_ids:
            raise ObservationError("frame_id must be unique across the recorder stream")
        self._seen_observation_ids.add(binding.observation_id)
        self._seen_frame_ids.add(binding.frame_id)
        self._previous = order

        if self._suppressed_until_clear:
            if not trigger.active:
                self._suppressed_until_clear = False
                if frame.pixel_bytes <= self._config.max_buffer_bytes:
                    self._append_prebuffer(_PairedFrame(frame.owned_copy(), binding))
            return None

        if self._active is None:
            if frame.pixel_bytes <= max(
                self._config.max_buffer_bytes,
                self._config.max_active_bytes if trigger.active else 0,
            ):
                paired = _PairedFrame(frame.owned_copy(), binding)
                self._append_prebuffer(paired)
            else:
                paired = _PairedFrame(frame, binding)
            if trigger.active:
                self._start(paired, trigger)
            return None

        incident = self._active
        if frame.captured_at_ms - incident.frames[0].captured_at_ms > self._config.max_clip_ms:
            published = self._finalize("max_clip_duration")
            if trigger.active:
                self._suppressed_until_clear = True
            else:
                if frame.pixel_bytes <= self._config.max_buffer_bytes:
                    self._append_prebuffer(_PairedFrame(frame.owned_copy(), binding))
            return published

        if (
            trigger.active
            and not incident.last_observation_triggered
            and frame.captured_at_ms - incident.last_trigger_ms > self._config.post_event_ms
        ):
            published = self._finalize("post_event_elapsed")
            paired = _PairedFrame(frame.owned_copy(), binding)
            self._append_prebuffer(paired)
            self._start(paired, trigger)
            return published

        paired = _PairedFrame(frame.owned_copy(), binding)
        frame_bytes = paired.pixel_bytes
        if len(incident.frames) >= self._config.max_active_frames:
            published = self._finalize("max_active_frames")
            if trigger.active:
                self._suppressed_until_clear = True
            elif frame_bytes <= self._config.max_buffer_bytes:
                self._append_prebuffer(paired)
            return published
        if incident.pixel_bytes + frame_bytes > self._config.max_active_bytes:
            published = self._finalize("max_active_bytes")
            if trigger.active:
                self._suppressed_until_clear = True
            elif frame_bytes <= self._config.max_buffer_bytes:
                self._append_prebuffer(paired)
            return published

        incident.frames.append(paired)
        incident.pixel_bytes += frame_bytes
        incident.last_observation_triggered = trigger.active
        if trigger.active:
            sample = self._sample(paired.frame, trigger)
            incident.trigger_samples.append(sample)
            incident.last_trigger_ms = frame.captured_at_ms

        if frame.captured_at_ms - incident.frames[0].captured_at_ms >= self._config.max_clip_ms:
            published = self._finalize("max_clip_duration")
            self._suppressed_until_clear = trigger.active
            return published
        if (
            not trigger.active
            and frame.captured_at_ms - incident.last_trigger_ms >= self._config.post_event_ms
        ):
            return self._finalize("post_event_elapsed")
        return None

    def close(self) -> PublishedIncident | None:
        if self._closed:
            return None
        self._closed = True
        if self._active is not None:
            return self._finalize("source_end")
        self._clear_prebuffer()
        return None

    def abort(self) -> None:
        """Discard all in-memory frames without publishing a partial clip."""

        self._active = None
        self._clear_prebuffer()
        self._closed = True

    def process_many(
        self,
        pairs: Sequence[tuple[RecorderFrame, dict[str, Any]]],
    ) -> list[PublishedIncident]:
        published: list[PublishedIncident] = []
        for frame, observation in pairs:
            incident = self.process(frame, observation)
            if incident is not None:
                published.append(incident)
        final = self.close()
        if final is not None:
            published.append(final)
        return published
