from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from foliage_warden_perception.geometry import Point, Zone, ZoneType
from foliage_warden_perception.pipeline import build_observation_record, run_pipeline, stable_json
from foliage_warden_perception.sources import Frame
from foliage_warden_perception.tracking import IouTracker
from foliage_warden_perception.types import Detection, NormalizedBox, ObjectClass
from foliage_warden_perception.yolox import DetectorTimings


@dataclass
class SyntheticSource:
    frames: list[Frame]
    closed: bool = False

    def __iter__(self):
        return iter(self.frames)

    def close(self) -> None:
        self.closed = True


class SyntheticDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect_timed(self, frame: np.ndarray) -> tuple[list[Detection], DetectorTimings]:
        return list(self.detections), DetectorTimings(1.0, 2.0, 3.0)


def _frame(index: int = 0) -> Frame:
    return Frame(
        index=index,
        captured_at_ms=index * 40,
        camera_id="camera-1",
        source_kind="video",
        source_name="fixed.mp4",
        bgr=np.zeros((10, 20, 3), dtype=np.uint8),
    )


def _cat() -> Detection:
    return Detection(ObjectClass.CAT, 15, 0.9, NormalizedBox(0.25, 0.25, 0.5, 0.5), 7)


def test_observation_is_explicitly_unknown_observe_only_and_normalized() -> None:
    tracker = IouTracker()
    [track] = tracker.update([_cat()], frame_index=0, timestamp_ms=0)
    zone = Zone(
        "plant",
        ZoneType.APPROACH,
        (Point(0, 0), Point(0.5, 0), Point(0.5, 1), Point(0, 1)),
    )

    record = build_observation_record(_frame(), [track], (zone,))

    assert record["mode"] == "OBSERVE_ONLY"
    assert record["behavior"] == "UNKNOWN"
    assert record["would_action"] is False
    assert record["cat_count"] == 1
    assert record["person_present"] is False
    [policy_track] = record["observation"]["tracks"]
    assert policy_track["class"] == "CAT"
    assert policy_track["behavior"] == {
        "label": "UNKNOWN",
        "raw_label": "OTHER_UNKNOWN",
        "scores": {"CLEAR": 0.0, "DIGGING": 0.0, "EATING": 0.0, "UNKNOWN": 1.0},
    }
    assert policy_track["region_evidence"]["approach_overlap"] == 0.5
    assert policy_track["bbox"] == {"height": 0.5, "width": 0.5, "x": 0.25, "y": 0.25}
    assert record["zone_evidence"][0]["track_age_frames"] == 1

    common_schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "common.schema.json").read_text(
            encoding="utf-8"
        )
    )
    policy_observation_schema = {**common_schema, "$ref": "#/$defs/policy_observation"}
    Draft202012Validator(policy_observation_schema).validate(record["observation"])


def test_fixed_frames_and_detections_produce_byte_identical_jsonl() -> None:
    def execute() -> tuple[str, SyntheticSource]:
        source = SyntheticSource([_frame(0), _frame(1)])
        output = io.StringIO()
        benchmark = run_pipeline(
            source,
            SyntheticDetector([_cat()]),
            IouTracker(),
            output,
            max_frames=2,
        )
        assert benchmark.frame_count == 2
        return output.getvalue(), source

    first, first_source = execute()
    second, second_source = execute()

    assert first == second
    assert first_source.closed and second_source.closed
    records = [json.loads(line) for line in first.splitlines()]
    assert [record["sequence"] for record in records] == [0, 1]
    assert all(record["would_action"] is False for record in records)


def test_max_frames_stops_before_source_disconnect_and_closes() -> None:
    source = SyntheticSource([_frame(0), _frame(1)])
    output = io.StringIO()

    benchmark = run_pipeline(
        source,
        SyntheticDetector([]),
        IouTracker(),
        output,
        max_frames=1,
    )

    assert benchmark.frame_count == 1
    assert len(output.getvalue().splitlines()) == 1
    assert source.closed


def test_stable_json_sorts_keys_and_rejects_non_finite_values() -> None:
    assert stable_json({"z": 1, "a": 2}) == '{"a":2,"z":1}'
    with pytest.raises(ValueError, match="Out of range"):
        stable_json({"value": float("nan")})
