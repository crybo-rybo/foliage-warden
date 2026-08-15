"""Observe-only perception runner and deterministic JSONL wire contract."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Protocol, TextIO

import numpy as np
import numpy.typing as npt

from .benchmark import BenchmarkAccumulator
from .geometry import RegionEvidence, Zone, evidence_for_box
from .sources import Frame, FrameSource
from .tracking import IouTracker, TrackedDetection
from .types import Detection, JsonObject, ObjectClass, _wire_float
from .yolox import DetectorTimings

SCHEMA_VERSION = 1
OBSERVE_ONLY_MODE = "OBSERVE_ONLY"


class TimedDetector(Protocol):
    def detect_timed(
        self, frame: npt.NDArray[np.uint8]
    ) -> tuple[list[Detection], DetectorTimings]: ...


def stable_json(value: JsonObject) -> str:
    """Serialize one record deterministically and reject NaN/Infinity."""

    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _unknown_behavior() -> JsonObject:
    return {
        "label": "UNKNOWN",
        "raw_label": "OTHER_UNKNOWN",
        "scores": {
            "CLEAR": 0.0,
            "DIGGING": 0.0,
            "EATING": 0.0,
            "UNKNOWN": 1.0,
        },
    }


def _base_track(track: TrackedDetection) -> JsonObject:
    return {
        "ambiguous": track.ambiguous,
        "bbox": track.detection.bbox.to_dict(),
        "class": track.detection.object_class.value,
        "detection_confidence": _wire_float(track.detection.confidence),
        "track_age_ms": track.age_ms,
        "track_id": track.track_id,
        "track_quality": _wire_float(track.quality),
    }


def _track_and_evidence(
    track: TrackedDetection,
    zones: tuple[Zone, ...],
) -> tuple[JsonObject, JsonObject]:
    evidence = evidence_for_box(track.detection.bbox, zones)
    policy_track = _base_track(track)
    if track.detection.object_class is ObjectClass.CAT:
        policy_track.update(
            {
                "aim_preset_id": None,
                "behavior": _unknown_behavior(),
                "no_fire_intersection": evidence.no_fire_intersection,
                "region_evidence": evidence.policy_dict(),
                "zone_id": evidence.zone_id,
            }
        )
    return policy_track, _evidence_record(track, evidence)


def _evidence_record(track: TrackedDetection, evidence: RegionEvidence) -> JsonObject:
    return {
        "no_fire_overlap": _wire_float(evidence.no_fire_overlap),
        "overlaps": [overlap.to_dict() for overlap in evidence.overlaps],
        "track_age_frames": track.age_frames,
        "track_id": track.track_id,
    }


def build_observation_record(
    frame: Frame,
    tracks: Iterable[TrackedDetection],
    zones: tuple[Zone, ...] = (),
    *,
    model_id: str = "synthetic",
    model_sha256: str | None = None,
) -> JsonObject:
    """Build an output that embeds the canonical policy-observation shape."""

    policy_tracks: list[JsonObject] = []
    zone_evidence: list[JsonObject] = []
    sorted_tracks = sorted(tracks, key=lambda item: item.track_id)
    for track in sorted_tracks:
        policy_track, evidence = _track_and_evidence(track, zones)
        policy_tracks.append(policy_track)
        zone_evidence.append(evidence)
    cat_count = sum(
        track.detection.object_class is ObjectClass.CAT for track in sorted_tracks
    )
    person_present = any(
        track.detection.object_class is ObjectClass.PERSON for track in sorted_tracks
    )
    frame_id = f"{frame.camera_id}:frame:{frame.index:08d}"
    observation_id = f"{frame.camera_id}:observation:{frame.index:08d}"
    model: JsonObject = {"id": model_id}
    if model_sha256 is not None:
        model["sha256"] = model_sha256
    return {
        "behavior": "UNKNOWN",
        "cat_count": cat_count,
        "frame": {
            "height": int(frame.bgr.shape[0]),
            "index": frame.index,
            "width": int(frame.bgr.shape[1]),
        },
        "mode": OBSERVE_ONLY_MODE,
        "model": model,
        "observation": {
            "camera_id": frame.camera_id,
            "captured_at_ms": frame.captured_at_ms,
            "frame_id": frame_id,
            "observation_id": observation_id,
            "tracks": policy_tracks,
        },
        "person_present": person_present,
        "record_type": "perception_observation",
        "schema_version": SCHEMA_VERSION,
        "sequence": frame.index,
        "source": {"kind": frame.source_kind, "name": frame.source_name},
        "would_action": False,
        "zone_evidence": zone_evidence,
    }


def run_pipeline(
    source: FrameSource,
    detector: TimedDetector,
    tracker: IouTracker,
    output: TextIO,
    *,
    zones: tuple[Zone, ...] = (),
    max_frames: int | None = None,
    model_id: str = "synthetic",
    model_sha256: str | None = None,
) -> BenchmarkAccumulator:
    """Process frames, emitting observations only; this function has no action API."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")
    benchmark = BenchmarkAccumulator()
    iterator = iter(source)
    emitted = 0
    try:
        while max_frames is None or emitted < max_frames:
            total_start = time.perf_counter_ns()
            capture_start = total_start
            try:
                frame = next(iterator)
            except StopIteration:
                break
            after_capture = time.perf_counter_ns()

            detections, detector_timings = detector.detect_timed(frame.bgr)
            after_detection = time.perf_counter_ns()
            tracked = tracker.update(
                detections,
                frame_index=frame.index,
                timestamp_ms=frame.captured_at_ms,
            )
            after_tracking = time.perf_counter_ns()
            record = build_observation_record(
                frame,
                tracked,
                zones,
                model_id=model_id,
                model_sha256=model_sha256,
            )
            after_regions = time.perf_counter_ns()
            output.write(stable_json(record))
            output.write("\n")
            output.flush()
            after_output = time.perf_counter_ns()

            benchmark.add("capture", (after_capture - capture_start) / 1_000_000.0)
            benchmark.add("preprocess", detector_timings.preprocess_ms)
            benchmark.add("inference", detector_timings.inference_ms)
            benchmark.add("postprocess", detector_timings.postprocess_ms)
            benchmark.add("tracking", (after_tracking - after_detection) / 1_000_000.0)
            benchmark.add("regions", (after_regions - after_tracking) / 1_000_000.0)
            benchmark.add("jsonl", (after_output - after_regions) / 1_000_000.0)
            benchmark.add("total", (after_output - total_start) / 1_000_000.0)
            emitted += 1
    finally:
        source.close()
    return benchmark
