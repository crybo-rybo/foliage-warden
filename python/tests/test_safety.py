from __future__ import annotations

from foliage_warden_eval.safety import check_safety
from foliage_warden_eval.schemas import (
    ActionRecord,
    ActionType,
    Behavior,
    PolicyState,
    PredictionEvent,
    SessionRecord,
)


def test_safe_replay_passes() -> None:
    session = SessionRecord("session", 10_000, final_state=PolicyState.MONITORING)
    prediction = PredictionEvent(
        "prediction",
        "session",
        Behavior.EATING,
        100,
        300,
        0.99,
        would_action=True,
        ready_ms=200,
        incident_id="incident",
    )
    action = ActionRecord(
        "action",
        "session",
        250,
        "command",
        ActionType.BURST,
        PolicyState.READY,
        PolicyState.COOLDOWN,
        incident_id="incident",
    )
    report = check_safety([session], [prediction], [action])
    assert report.passed
    assert report.violations == ()
    assert report.checks_run > 0


def test_unsafe_replay_reports_each_invariant_deterministically() -> None:
    session = SessionRecord(
        "session",
        10_000,
        startup_state=PolicyState.MONITORING,
        final_state=PolicyState.COOLDOWN,
        failure_reason="camera_loss",
    )
    prediction = PredictionEvent(
        "prediction",
        "session",
        Behavior.UNKNOWN,
        100,
        300,
        0.5,
        would_action=True,
        ready_ms=200,
        incident_id="incident",
        person_present=True,
        cat_count=2,
        stale_input=True,
        track_lost=True,
        hardware_ready=False,
        no_fire_intersection=True,
    )
    actions = [
        ActionRecord(
            "action-1", "session", 150, "duplicate-command", ActionType.BURST,
            PolicyState.MONITORING, PolicyState.BURST, incident_id="incident", is_retry=True,
        ),
        ActionRecord(
            "action-2", "session", 250, "duplicate-command", ActionType.BURST,
            PolicyState.READY, PolicyState.COOLDOWN, incident_id="incident",
        ),
    ]
    report = check_safety([session], [prediction], actions)
    codes = {violation.code for violation in report.violations}
    assert not report.passed
    assert {
        "STARTUP_NOT_DISARMED",
        "FAILURE_NOT_FAIL_CLOSED",
        "PERSON_PRESENT_AUTHORIZED",
        "AMBIGUOUS_CAT_COUNT_AUTHORIZED",
        "STALE_INPUT_AUTHORIZED",
        "LOST_TRACK_AUTHORIZED",
        "HARDWARE_NOT_READY_AUTHORIZED",
        "NO_FIRE_INTERSECTION_AUTHORIZED",
        "NON_HARMFUL_BEHAVIOR_AUTHORIZED",
        "BURST_NOT_FROM_READY",
        "BURST_RETRY",
        "BURST_BEFORE_READY",
        "PERSON_PRESENT_BURST",
        "DUPLICATE_COMMAND_ID",
        "MULTIPLE_BURSTS_PER_INCIDENT",
    } <= codes
    assert list(report.violations) == sorted(
        report.violations,
        key=lambda item: (item.session_id, item.record_id, item.code, item.message),
    )

