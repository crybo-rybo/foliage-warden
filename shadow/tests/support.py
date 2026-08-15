from __future__ import annotations

from copy import deepcopy
from typing import Any

MODEL_SHA = "1" * 64
CONFIG_SHA = "2" * 64


def unknown_behavior() -> dict[str, Any]:
    return {
        "label": "UNKNOWN",
        "raw_label": "OTHER_UNKNOWN",
        "scores": {"CLEAR": 0.0, "DIGGING": 0.0, "EATING": 0.0, "UNKNOWN": 1.0},
    }


def cat_track(
    track_id: str = "cat-a",
    *,
    age_ms: int = 900,
    quality: float = 0.95,
    no_fire: bool = False,
    foliage_overlap: float = 0.58,
    soil_overlap: float = 0.52,
    motion_score: float = 0.83,
) -> dict[str, Any]:
    return {
        "aim_preset_id": None,
        "ambiguous": False,
        "bbox": {"height": 0.4, "width": 0.35, "x": 0.35, "y": 0.28},
        "behavior": unknown_behavior(),
        "class": "CAT",
        "detection_confidence": 0.98,
        "no_fire_intersection": no_fire,
        "region_evidence": {
            "approach_overlap": 0.91,
            "foliage_overlap": foliage_overlap,
            "motion_score": motion_score,
            "soil_overlap": soil_overlap,
        },
        "track_age_ms": age_ms,
        "track_id": track_id,
        "track_quality": quality,
        "zone_id": "pot-1-approach",
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


def perception_record(
    index: int,
    *,
    captured_at_ms: int,
    tracks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = deepcopy(tracks if tracks is not None else [cat_track(age_ms=900 + index * 400)])
    frame_id = f"camera-1:frame:{index:08d}"
    observation_id = f"camera-1:observation:{index:08d}"
    return {
        "behavior": "UNKNOWN",
        "cat_count": sum(track["class"] == "CAT" for track in selected),
        "frame": {"height": 720, "index": index, "width": 1280},
        "mode": "OBSERVE_ONLY",
        "model": {"id": "synthetic-detector", "sha256": "a" * 64},
        "observation": {
            "camera_id": "camera-1",
            "captured_at_ms": captured_at_ms,
            "frame_id": frame_id,
            "observation_id": observation_id,
            "tracks": selected,
        },
        "person_present": any(track["class"] == "PERSON" for track in selected),
        "record_type": "perception_observation",
        "schema_version": 1,
        "sequence": index,
        "source": {"kind": "synthetic", "name": "injected-shadow-fixture"},
        "would_action": False,
        "zone_evidence": [_zone_evidence(track) for track in selected],
    }


def _zone_evidence(track: dict[str, Any]) -> dict[str, Any]:
    if track["class"] == "CAT":
        region = track["region_evidence"]
        no_fire_overlap = 0.2 if track["no_fire_intersection"] else 0.0
        overlaps = [
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
            {
                "overlap": no_fire_overlap,
                "zone_id": "example-human-walkway",
                "zone_type": "no_fire",
            },
        ]
    else:
        no_fire_overlap = 0.0
        overlaps = []
    return {
        "no_fire_overlap": no_fire_overlap,
        "overlaps": overlaps,
        "track_age_frames": max(1, track["track_age_ms"] // 33),
        "track_id": track["track_id"],
    }


def probabilities(label: str) -> dict[str, float]:
    values = {
        "PASSING": 0.01,
        "SNIFFING": 0.01,
        "EATING": 0.01,
        "DIGGING": 0.01,
        "OTHER": 0.01,
        "UNKNOWN": 0.01,
    }
    values[label] = 0.95
    return values


def behavior_prediction(
    perception: dict[str, Any],
    track_id: str,
    label: str,
    *,
    sequence: int,
    latency_ms: int = 10,
    frame_id: str | None = None,
    captured_at_ms: int | None = None,
) -> dict[str, Any]:
    observation = perception["observation"]
    captured = observation["captured_at_ms"] if captured_at_ms is None else captured_at_ms
    return {
        "captured_at_ms": captured,
        "config": {"id": "synthetic-behavior-config", "sha256": CONFIG_SHA},
        "frame_id": frame_id or observation["frame_id"],
        "mode": "OBSERVE_ONLY",
        "model": {"id": "synthetic-behavior", "sha256": MODEL_SHA},
        "observation_id": observation["observation_id"],
        "predicted_at_ms": observation["captured_at_ms"] + latency_ms,
        "predicted_label": label,
        "probabilities": probabilities(label),
        "record_type": "behavior_prediction",
        "schema_version": 1,
        "sequence": sequence,
        "track_id": track_id,
        "would_action": False,
    }


def series(
    count: int,
    *,
    interval_ms: int = 400,
    tracks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        selected = deepcopy(tracks) if tracks is not None else [cat_track(age_ms=900 + index * 400)]
        for track in selected:
            if track["class"] == "CAT":
                track["track_age_ms"] += index * 400 if tracks is not None else 0
        result.append(
            perception_record(index, captured_at_ms=100 + index * interval_ms, tracks=selected)
        )
    return result


def predictions_for(
    perceptions: list[dict[str, Any]],
    label: str,
    *,
    latency_ms: int = 10,
) -> list[dict[str, Any]]:
    result = []
    sequence = 0
    for perception in perceptions:
        for track in perception["observation"]["tracks"]:
            if track["class"] != "CAT":
                continue
            result.append(
                behavior_prediction(
                    perception,
                    track["track_id"],
                    label,
                    sequence=sequence,
                    latency_ms=latency_ms,
                )
            )
            sequence += 1
    return result
