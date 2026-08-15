from __future__ import annotations

import json
from pathlib import Path

import pytest

from foliage_warden_eval.cli import main
from foliage_warden_eval.evaluator import EvaluationInputError, evaluate
from foliage_warden_eval.jsonl import read_jsonl
from foliage_warden_eval.schemas import (
    ActionRecord,
    ActionType,
    Behavior,
    GroundTruthEvent,
    PolicyState,
    PredictionEvent,
    SessionRecord,
    parse_ground_truth,
    parse_replay_record,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_records():
    ground_truth = read_jsonl(FIXTURES / "ground_truth.jsonl", parse_ground_truth)
    replay = read_jsonl(FIXTURES / "replay.jsonl", parse_replay_record)
    sessions = [record for record in replay if isinstance(record, SessionRecord)]
    predictions = [record for record in replay if isinstance(record, PredictionEvent)]
    actions = [record for record in replay if isinstance(record, ActionRecord)]
    return ground_truth, sessions, predictions, actions


def test_fixture_report_covers_event_exposure_latency_quality_and_safety() -> None:
    ground_truth, sessions, predictions, actions = fixture_records()
    report = evaluate(ground_truth, predictions, sessions, actions)
    assert report["event_metrics"]["overall"] == {
        "f1": pytest.approx(2 / 3),
        "false_negatives": 1,
        "false_positives": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "true_positives": 2,
    }
    exposure = report["exposure_and_false_actions"]
    assert exposure["monitored_hours"] == 10.0
    assert exposure["false_would_actions"] == 1
    assert exposure["observed_false_would_actions_per_hour"] == 0.1
    assert exposure["one_sided_upper_bound_per_hour"] == pytest.approx(0.4743864518)
    assert report["latency"]["behavior_onset_to_ready"]["mean_ms"] == 1500
    assert report["latency"]["ready_to_burst"]["percentiles"]["p50_ms"] == 150
    assert report["quality"]["track_loss"]["rate"] == pytest.approx(2_000 / 30_000)
    assert report["quality"]["unknown_behavior"]["rate"] == 0.1
    assert report["safety"]["passed"] is True
    assert report["data_quality"]["passed"] is True
    assert [item["ground_truth_event_id"] for item in report["matching"]["matches"]] == [
        "gt-eating-1",
        "gt-digging-1",
    ]


def test_evaluation_is_deterministic_under_input_reordering() -> None:
    ground_truth, sessions, predictions, actions = fixture_records()
    first = evaluate(ground_truth, predictions, sessions, actions)
    second = evaluate(reversed(ground_truth), reversed(predictions), reversed(sessions), reversed(actions))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_missing_exposure_is_not_silently_invented() -> None:
    ground_truth, _sessions, predictions, actions = fixture_records()
    report = evaluate(ground_truth, predictions, [], actions)
    exposure = report["exposure_and_false_actions"]
    assert exposure["monitored_hours"] == 0.0
    assert exposure["observed_false_would_actions_per_hour"] is None
    assert exposure["one_sided_upper_bound_per_hour"] is None
    assert report["data_quality"]["passed"] is False


def test_duplicate_event_ids_fail_instead_of_matching_ambiguously() -> None:
    ground_truth, sessions, predictions, _actions = fixture_records()
    duplicate = GroundTruthEvent.from_dict(ground_truth[0].to_dict())
    with pytest.raises(EvaluationInputError, match="duplicate ground-truth"):
        evaluate([*ground_truth, duplicate], predictions, sessions)


def test_cli_writes_byte_identical_reports(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = [
        "evaluate",
        "--ground-truth", str(FIXTURES / "ground_truth.jsonl"),
        "--replay", str(FIXTURES / "replay.jsonl"),
    ]
    assert main([*args, "--output", str(first)]) == 0
    assert main([*args, "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    parsed = json.loads(first.read_text(encoding="utf-8"))
    assert parsed["event_metrics"]["overall"]["true_positives"] == 2


def test_cli_can_fail_ci_on_safety_violation(tmp_path) -> None:
    truth_path = tmp_path / "truth.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    truth_path.write_text("", encoding="utf-8")
    unsafe = PredictionEvent(
        "unsafe",
        "session",
        Behavior.UNKNOWN,
        0,
        100,
        0.1,
        would_action=True,
        ready_ms=50,
        incident_id="incident",
        person_present=True,
    )
    session = SessionRecord("session", 3_600_000)
    action = ActionRecord(
        "action", "session", 60, "command", ActionType.BURST,
        PolicyState.READY, PolicyState.COOLDOWN, incident_id="incident",
    )
    from foliage_warden_eval.jsonl import write_jsonl

    write_jsonl(replay_path, [session, unsafe, action])
    assert main(
        [
            "evaluate",
            "--ground-truth", str(truth_path),
            "--replay", str(replay_path),
            "--fail-on-safety-violation",
        ]
    ) == 3
