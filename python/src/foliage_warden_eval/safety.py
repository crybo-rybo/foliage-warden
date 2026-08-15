"""Deterministic checks for the fail-closed policy contract."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .schemas import (
    ActionRecord,
    ActionType,
    PolicyState,
    PredictionEvent,
    SessionRecord,
)


@dataclass(frozen=True, slots=True)
class SafetyViolation:
    code: str
    session_id: str
    record_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "record_id": self.record_id,
            "session_id": self.session_id,
        }


@dataclass(frozen=True, slots=True)
class SafetyReport:
    passed: bool
    checks_run: int
    violations: tuple[SafetyViolation, ...]

    def to_dict(self) -> dict[str, object]:
        counts = Counter(violation.code for violation in self.violations)
        return {
            "checks_run": self.checks_run,
            "passed": self.passed,
            "violation_counts": dict(sorted(counts.items())),
            "violations": [violation.to_dict() for violation in self.violations],
        }


def check_safety(
    sessions: Iterable[SessionRecord],
    predictions: Iterable[PredictionEvent],
    actions: Iterable[ActionRecord],
) -> SafetyReport:
    """Check replay records against the README's non-negotiable invariants."""

    session_items = sorted(sessions, key=lambda item: item.session_id)
    prediction_items = sorted(predictions, key=lambda item: (item.session_id, item.start_ms, item.event_id))
    action_items = sorted(actions, key=lambda item: (item.session_id, item.timestamp_ms, item.action_id))
    violations: list[SafetyViolation] = []
    checks_run = 0

    def check(condition: bool, code: str, session_id: str, record_id: str, message: str) -> None:
        nonlocal checks_run
        checks_run += 1
        if not condition:
            violations.append(SafetyViolation(code, session_id, record_id, message))

    for session in session_items:
        check(
            session.startup_state == PolicyState.DISARMED,
            "STARTUP_NOT_DISARMED",
            session.session_id,
            session.session_id,
            f"startup state was {session.startup_state.value}, expected DISARMED",
        )
        if session.failure_reason is not None:
            check(
                session.final_state in (PolicyState.DISARMED, PolicyState.FAULT),
                "FAILURE_NOT_FAIL_CLOSED",
                session.session_id,
                session.session_id,
                "a failed session must finish in DISARMED or FAULT",
            )

    predictions_by_incident: dict[tuple[str, str], list[PredictionEvent]] = defaultdict(list)
    for prediction in prediction_items:
        if prediction.incident_id is not None:
            predictions_by_incident[(prediction.session_id, prediction.incident_id)].append(prediction)
        authorized = prediction.reached_ready or prediction.would_action
        if authorized:
            check(
                not prediction.person_present,
                "PERSON_PRESENT_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                "person presence must block READY and would-action",
            )
            check(
                prediction.cat_count == 1,
                "AMBIGUOUS_CAT_COUNT_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                f"cat_count={prediction.cat_count}; exactly one cat is required",
            )
            check(
                not prediction.stale_input,
                "STALE_INPUT_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                "stale input must block READY and would-action",
            )
            check(
                not prediction.track_lost,
                "LOST_TRACK_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                "track loss must block READY and would-action",
            )
            check(
                prediction.hardware_ready,
                "HARDWARE_NOT_READY_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                "hardware-not-ready must block READY and would-action",
            )
            check(
                not prediction.no_fire_intersection,
                "NO_FIRE_INTERSECTION_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                "no-fire intersection must block READY and would-action",
            )
            check(
                prediction.behavior.is_harmful,
                "NON_HARMFUL_BEHAVIOR_AUTHORIZED",
                prediction.session_id,
                prediction.event_id,
                f"{prediction.behavior.value} must not reach READY or would-action",
            )
        if prediction.would_action:
            check(
                prediction.ready_ms is not None,
                "WOULD_ACTION_WITHOUT_READY",
                prediction.session_id,
                prediction.event_id,
                "a would-action must have a READY timestamp",
            )
            check(
                prediction.incident_id is not None,
                "WOULD_ACTION_WITHOUT_INCIDENT",
                prediction.session_id,
                prediction.event_id,
                "a would-action needs an incident ID for one-shot enforcement",
            )

    command_owners: dict[tuple[str, str], ActionRecord] = {}
    bursts_by_incident: dict[tuple[str, str], list[ActionRecord]] = defaultdict(list)
    for action in action_items:
        command_key = (action.session_id, action.command_id)
        previous = command_owners.get(command_key)
        checks_run += 1
        if previous is not None:
            violations.append(
                SafetyViolation(
                    "DUPLICATE_COMMAND_ID",
                    action.session_id,
                    action.action_id,
                    f"command_id {action.command_id!r} was already used by {previous.action_id!r}",
                )
            )
        else:
            command_owners[command_key] = action

        if action.action != ActionType.BURST:
            continue
        check(
            action.state_before == PolicyState.READY,
            "BURST_NOT_FROM_READY",
            action.session_id,
            action.action_id,
            f"BURST was issued from {action.state_before.value}, expected READY",
        )
        check(
            not action.is_retry,
            "BURST_RETRY",
            action.session_id,
            action.action_id,
            "BURST must never be retried automatically",
        )
        check(
            action.incident_id is not None,
            "BURST_WITHOUT_INCIDENT",
            action.session_id,
            action.action_id,
            "BURST needs an incident ID for one-shot enforcement",
        )
        if action.incident_id is None:
            continue
        incident_key = (action.session_id, action.incident_id)
        bursts_by_incident[incident_key].append(action)
        decisions = predictions_by_incident.get(incident_key, [])
        check(
            bool(decisions),
            "BURST_WITHOUT_DECISION",
            action.session_id,
            action.action_id,
            "BURST has no replayable prediction decision for its incident",
        )
        ready_times = [decision.ready_ms for decision in decisions if decision.ready_ms is not None]
        if ready_times:
            check(
                action.timestamp_ms >= min(ready_times),
                "BURST_BEFORE_READY",
                action.session_id,
                action.action_id,
                "BURST timestamp precedes the incident's READY timestamp",
            )
        if any(decision.person_present for decision in decisions):
            check(
                False,
                "PERSON_PRESENT_BURST",
                action.session_id,
                action.action_id,
                "BURST was issued for an incident with a person present",
            )

    for (session_id, incident_id), burst_items in sorted(bursts_by_incident.items()):
        checks_run += 1
        if len(burst_items) > 1:
            for duplicate in burst_items[1:]:
                violations.append(
                    SafetyViolation(
                        "MULTIPLE_BURSTS_PER_INCIDENT",
                        session_id,
                        duplicate.action_id,
                        f"incident {incident_id!r} produced {len(burst_items)} BURST commands",
                    )
                )

    violations.sort(key=lambda item: (item.session_id, item.record_id, item.code, item.message))
    return SafetyReport(not violations, checks_run, tuple(violations))

