from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from foliage_warden_recorder import (
    IncidentRecorder,
    IncidentStore,
    OpenCvAviEncoder,
    RecorderConfig,
    RecorderFrame,
)
from foliage_warden_shadow.contracts import stable_json


def unknown_behavior() -> dict[str, Any]:
    return {
        "label": "UNKNOWN",
        "raw_label": "OTHER_UNKNOWN",
        "scores": {"CLEAR": 0.0, "DIGGING": 0.0, "EATING": 0.0, "UNKNOWN": 1.0},
    }


def cat_track(
    track_id: str = "cat-a",
    *,
    approach_overlap: float = 0.8,
) -> dict[str, Any]:
    return {
        "aim_preset_id": None,
        "ambiguous": False,
        "bbox": {"height": 0.4, "width": 0.35, "x": 0.35, "y": 0.28},
        "behavior": unknown_behavior(),
        "class": "CAT",
        "detection_confidence": 0.98,
        "no_fire_intersection": False,
        "region_evidence": {
            "approach_overlap": approach_overlap,
            "foliage_overlap": 0.4,
            "motion_score": 0.7,
            "soil_overlap": 0.3,
        },
        "track_age_ms": 900,
        "track_id": track_id,
        "track_quality": 0.95,
        "zone_id": "pot-1-approach" if approach_overlap > 0.0 else None,
    }


def person_track(track_id: str = "person-a") -> dict[str, Any]:
    return {
        "ambiguous": False,
        "bbox": {"height": 0.8, "width": 0.3, "x": 0.05, "y": 0.1},
        "class": "PERSON",
        "detection_confidence": 0.99,
        "track_age_ms": 1000,
        "track_id": track_id,
        "track_quality": 0.99,
    }


def _zone_evidence(track: dict[str, Any]) -> dict[str, Any]:
    if track["class"] == "PERSON":
        return {
            "no_fire_overlap": 0.0,
            "overlaps": [],
            "track_age_frames": 30,
            "track_id": track["track_id"],
        }
    region = track["region_evidence"]
    return {
        "no_fire_overlap": 0.0,
        "overlaps": [
            {
                "overlap": region["approach_overlap"],
                "zone_id": "pot-1-approach",
                "zone_type": "approach",
            },
            {
                "overlap": region["foliage_overlap"],
                "zone_id": "pot-1-foliage",
                "zone_type": "foliage",
            },
            {
                "overlap": region["soil_overlap"],
                "zone_id": "pot-1-soil",
                "zone_type": "soil",
            },
        ],
        "track_age_frames": 27,
        "track_id": track["track_id"],
    }


def perception_record(
    sequence: int,
    captured_at_ms: int,
    *,
    tracks: list[dict[str, Any]] | None = None,
    width: int = 32,
    height: int = 24,
) -> dict[str, Any]:
    selected = deepcopy(tracks or [])
    return {
        "behavior": "UNKNOWN",
        "cat_count": sum(track["class"] == "CAT" for track in selected),
        "frame": {"height": height, "index": sequence, "width": width},
        "mode": "OBSERVE_ONLY",
        "model": {"id": "synthetic-detector", "sha256": "a" * 64},
        "observation": {
            "camera_id": "camera-1",
            "captured_at_ms": captured_at_ms,
            "frame_id": f"camera-1:frame:{sequence:08d}",
            "observation_id": f"camera-1:observation:{sequence:08d}",
            "tracks": selected,
        },
        "person_present": any(track["class"] == "PERSON" for track in selected),
        "record_type": "perception_observation",
        "schema_version": 1,
        "sequence": sequence,
        "source": {"kind": "video", "name": "input.avi"},
        "would_action": False,
        "zone_evidence": [_zone_evidence(track) for track in selected],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(stable_json(record) + "\n" for record in records), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_bindings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for ordinal, record in enumerate(records):
        observation = record["observation"]
        result.append(
            {
                "captured_at_ms": observation["captured_at_ms"],
                "encoded_frame_index": ordinal,
                "frame_id": observation["frame_id"],
                "observation_id": observation["observation_id"],
                "perception_record_sha256": hashlib.sha256(
                    stable_json(record).encode("utf-8")
                ).hexdigest(),
                "sequence": record["sequence"],
            }
        )
    return result


def write_incident(
    root: Path,
    records: list[dict[str, Any]],
    *,
    frames: list[np.ndarray] | None = None,
    provenance: bool = True,
    fps: float = 10.0,
) -> tuple[Path, Path, dict[str, Any]]:
    if frames is None:
        frames = []
        for ordinal in range(len(records)):
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[:, :, 0] = 20 + ordinal * 20
            frame[:, :, 1] = 80 + ordinal * 10
            frame[:, :, 2] = 180 - ordinal * 20
            frames.append(frame)
    assert len(frames) == len(records)
    first_trigger_record = next(
        record
        for record in records
        if any(
            track["class"] == "CAT" and track["region_evidence"]["approach_overlap"] >= 0.25
            for track in record["observation"]["tracks"]
        )
    )
    first_observation = first_trigger_record["observation"]
    incident_id = (
        f"incident-{first_observation['captured_at_ms']:013d}-"
        f"{first_trigger_record['sequence']:010d}"
    )
    incident = root / incident_id
    incident.mkdir(mode=0o700)
    clip_path = incident / "clip.avi"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    os.chmod(clip_path, 0o600)

    trigger_samples = []
    for record in records:
        qualifying = sorted(
            (
                track["track_id"],
                track["region_evidence"]["approach_overlap"],
            )
            for track in record["observation"]["tracks"]
            if track["class"] == "CAT" and track["region_evidence"]["approach_overlap"] >= 0.25
        )
        if qualifying:
            trigger_samples.append(
                {
                    "captured_at_ms": record["observation"]["captured_at_ms"],
                    "maximum_approach_overlap": max(value for _, value in qualifying),
                    "sequence": record["sequence"],
                    "track_ids": [track_id for track_id, _ in qualifying],
                }
            )
    bindings = frame_bindings(records)
    metadata: dict[str, Any] = {
        "clip": {
            "audio": False,
            "byte_size": clip_path.stat().st_size,
            "codec": "MJPG",
            "container": "avi",
            "filename": "clip.avi",
            "fps": fps,
            "frame_count": len(records),
            "height": height,
            "sha256": file_sha256(clip_path),
            "width": width,
        },
        "incident_id": incident_id,
        "mode": "OBSERVE_ONLY",
        "privacy": {"audio": False, "display": False, "network": False},
        "record_type": "observation_clip",
        "resource_limits": {
            "max_active_bytes": 1024 * 1024,
            "max_active_frames": 100,
            "max_buffer_bytes": 1024 * 1024,
            "max_buffer_frames": 100,
        },
        "schema_version": 1,
        "source": {"camera_id": "camera-1", "kind": "video", "name": "input.avi"},
        "termination": "source_end",
        "timeline": {
            "duration_ms": (
                records[-1]["observation"]["captured_at_ms"]
                - records[0]["observation"]["captured_at_ms"]
            ),
            "end_captured_at_ms": records[-1]["observation"]["captured_at_ms"],
            "end_sequence": records[-1]["sequence"],
            "first_trigger_at_ms": trigger_samples[0]["captured_at_ms"],
            "last_trigger_at_ms": trigger_samples[-1]["captured_at_ms"],
            "start_captured_at_ms": records[0]["observation"]["captured_at_ms"],
            "start_sequence": records[0]["sequence"],
        },
        "trigger": {
            "minimum_approach_overlap": 0.25,
            "rule": "CAT_IN_APPROACH_ZONE",
            "samples": trigger_samples,
        },
    }
    if provenance:
        metadata["perception_provenance"] = {
            "binding_stream_canonicalization": ("JSONL_FRAME_BINDINGS_SORTED_KEYS_COMPACT_UTF8_V1"),
            "frame_bindings": bindings,
            "record_canonicalization": "JSON_SORTED_KEYS_COMPACT_UTF8_V1",
            "record_count": len(records),
            "stream_sha256": hashlib.sha256(
                "".join(stable_json(binding) + "\n" for binding in bindings).encode("utf-8")
            ).hexdigest(),
        }
    metadata_path = incident / "metadata.json"
    metadata_path.write_text(stable_json(metadata) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)
    perceptions = root / "perception.jsonl"
    write_jsonl(perceptions, records)
    return incident, perceptions, metadata


def write_recorder_incident(
    root: Path,
    records: list[dict[str, Any]],
    *,
    frames: list[np.ndarray] | None = None,
    fps: float = 10.0,
) -> tuple[Path, Path, dict[str, Any]]:
    """Publish an incident through the real recorder/OpenCV producer boundary."""

    if frames is None:
        frames = []
        for ordinal in range(len(records)):
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[:, :, 0] = 20 + ordinal * 20
            frame[:, :, 1] = 80 + ordinal * 10
            frame[:, :, 2] = 180 - ordinal * 20
            frames.append(frame)
    assert len(frames) == len(records)
    config = RecorderConfig(
        pre_event_ms=1_000,
        post_event_ms=1_000,
        max_clip_ms=10_000,
        max_buffer_frames=max(10, len(records) + 1),
        nominal_fps=fps,
        minimum_approach_overlap=0.25,
        max_incidents=10,
        max_disk_bytes=1024 * 1024 * 1024,
        max_buffer_bytes=64 * 1024 * 1024,
        max_active_frames=max(10, len(records) + 1),
        max_active_bytes=64 * 1024 * 1024,
    )
    store = IncidentStore(root / "recorder-output", config)
    recorder = IncidentRecorder(store, OpenCvAviEncoder(cv2_module=cv2), config)
    published = None
    for record, pixels in zip(records, frames, strict=True):
        observation = record["observation"]
        source = record["source"]
        result = recorder.process(
            RecorderFrame(
                sequence=record["sequence"],
                captured_at_ms=observation["captured_at_ms"],
                camera_id=observation["camera_id"],
                source_kind=source["kind"],
                source_name=source["name"],
                pixels=pixels,
            ),
            deepcopy(record),
        )
        assert result is None
    published = recorder.close()
    assert published is not None

    perceptions = root / "recorder-perceptions.jsonl"
    write_jsonl(perceptions, records)
    os.chmod(perceptions, 0o600)
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    return published.directory, perceptions, metadata


def default_records() -> list[dict[str, Any]]:
    return [
        perception_record(10, 100),
        perception_record(20, 200, tracks=[cat_track()]),
        perception_record(40, 300, tracks=[cat_track()]),
    ]
