from __future__ import annotations

import json

import pytest
from foliage_warden_eval.jsonl import read_jsonl, stable_json, write_jsonl
from foliage_warden_eval.schemas import (
    Behavior,
    GroundTruthEvent,
    PredictionEvent,
    SchemaError,
    parse_ground_truth,
)


def test_ground_truth_round_trip() -> None:
    event = GroundTruthEvent(
        event_id="event-1",
        session_id="session-1",
        behavior=Behavior.EATING,
        start_ms=100,
        end_ms=200,
        zone_id="pot-1",
        metadata={"lighting": "day"},
    )
    assert GroundTruthEvent.from_dict(event.to_dict()) == event
    assert event.to_dict()["record_type"] == "ground_truth_event"


def test_prediction_schema_rejects_permissive_or_malformed_values() -> None:
    base = {
        "record_type": "prediction_event",
        "event_id": "prediction-1",
        "session_id": "session-1",
        "behavior": "EATING",
        "start_ms": 100,
        "end_ms": 200,
        "score": 0.9,
    }
    with pytest.raises(SchemaError, match="unknown field"):
        PredictionEvent.from_dict({**base, "person_presnt": True})
    with pytest.raises(SchemaError, match="score must be between"):
        PredictionEvent.from_dict({**base, "score": 1.1})
    with pytest.raises(SchemaError, match="end_ms must be greater"):
        PredictionEvent.from_dict({**base, "end_ms": 100})
    with pytest.raises(SchemaError, match="would_action must be a boolean"):
        PredictionEvent.from_dict({**base, "would_action": 1})


def test_jsonl_round_trip_is_stable_and_ignores_blank_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    events = [
        GroundTruthEvent("b", "session", Behavior.DIGGING, 10, 20),
        GroundTruthEvent("a", "session", Behavior.EATING, 30, 40),
    ]
    write_jsonl(path, events)
    original = path.read_text(encoding="utf-8")
    path.write_text(f"\n{original}\n", encoding="utf-8")
    assert read_jsonl(path, parse_ground_truth) == events
    write_jsonl(path, events)
    assert path.read_text(encoding="utf-8") == original
    for line in original.splitlines():
        assert line == stable_json(json.loads(line))


def test_jsonl_error_includes_file_and_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n{}\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"bad\.jsonl:2:.*record_type"):
        read_jsonl(path, parse_ground_truth)

