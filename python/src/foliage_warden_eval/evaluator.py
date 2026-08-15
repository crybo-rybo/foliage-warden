"""Compose matching, metrics, exposure statistics, and safety checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .matching import MatchConfig, MatchResult, match_events
from .safety import check_safety
from .schemas import (
    SCHEMA_VERSION,
    ActionRecord,
    ActionType,
    Behavior,
    GroundTruthEvent,
    JsonValue,
    PredictionEvent,
    SessionRecord,
)
from .statistics import percentiles, poisson_rate_upper_bound, safe_ratio


class EvaluationInputError(ValueError):
    """Raised when records are individually valid but ambiguous as a set."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    match: MatchConfig = field(default_factory=MatchConfig)
    target_behaviors: tuple[Behavior, ...] = (Behavior.EATING, Behavior.DIGGING)
    would_action_only: bool = True
    poisson_confidence: float = 0.95
    latency_percentiles: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)

    def __post_init__(self) -> None:
        if not self.target_behaviors:
            raise ValueError("target_behaviors must not be empty")
        if len(set(self.target_behaviors)) != len(self.target_behaviors):
            raise ValueError("target_behaviors must be unique")
        if not 0.0 < self.poisson_confidence < 1.0:
            raise ValueError("poisson_confidence must be between 0 and 1")
        for percentile in self.latency_percentiles:
            if not 0.0 <= percentile <= 100.0:
                raise ValueError("latency percentiles must be between 0 and 100")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "latency_percentiles": list(self.latency_percentiles),
            "matching": {
                "max_onset_delta_ms": self.match.max_onset_delta_ms,
                "min_temporal_iou": self.match.min_temporal_iou,
                "require_behavior_match": self.match.require_behavior_match,
                "require_zone_match": self.match.require_zone_match,
            },
            "poisson_confidence": self.poisson_confidence,
            "prediction_filter": "would_action" if self.would_action_only else "all_target_behavior_events",
            "target_behaviors": sorted(behavior.value for behavior in self.target_behaviors),
        }


def _validate_unique(records: Iterable[object], kind: str, field_name: str) -> None:
    seen: dict[object, object] = {}
    for record in records:
        value = getattr(record, field_name)
        if value in seen:
            raise EvaluationInputError(f"duplicate {kind} {field_name} {value!r}")
        seen[value] = record


def _metric_counts(result: MatchResult) -> dict[str, int | float]:
    true_positives = len(result.matches)
    false_positives = len(result.unmatched_predictions)
    false_negatives = len(result.unmatched_truth)
    precision = safe_ratio(true_positives, true_positives + false_positives)
    recall = safe_ratio(true_positives, true_positives + false_negatives)
    # Undefined ratios use the explicit zero convention. Counts remain in the
    # report so consumers can distinguish "no data" from measured zero.
    precision_value = 0.0 if precision is None else precision
    recall_value = 0.0 if recall is None else recall
    f1 = (
        0.0
        if precision_value + recall_value == 0.0
        else 2.0 * precision_value * recall_value / (precision_value + recall_value)
    )
    return {
        "f1": f1,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "precision": precision_value,
        "recall": recall_value,
        "true_positives": true_positives,
    }


def _distribution(values: list[int], requested: tuple[float, ...]) -> dict[str, JsonValue]:
    if not values:
        return {
            "count": 0,
            "maximum_ms": None,
            "mean_ms": None,
            "minimum_ms": None,
            "percentiles": None,
        }
    return {
        "count": len(values),
        "maximum_ms": max(values),
        "mean_ms": sum(values) / len(values),
        "minimum_ms": min(values),
        "percentiles": percentiles(values, requested),
    }


def _duration_rate(
    sessions: list[SessionRecord],
    predictions: list[PredictionEvent],
    numerator_name: str,
) -> dict[str, JsonValue]:
    if numerator_name == "track_lost_ms":
        denominator_name = "track_observation_ms"
        predicate = lambda prediction: prediction.track_lost
    elif numerator_name == "unknown_behavior_ms":
        denominator_name = "behavior_observation_ms"
        predicate = lambda prediction: prediction.behavior == Behavior.UNKNOWN
    else:
        raise ValueError(f"unsupported duration numerator: {numerator_name}")

    covered = [
        session
        for session in sessions
        if getattr(session, numerator_name) is not None
        and getattr(session, denominator_name) is not None
    ]
    if covered:
        numerator = sum(int(getattr(session, numerator_name)) for session in covered)
        denominator = sum(int(getattr(session, denominator_name)) for session in covered)
        source = "session_counters"
        coverage = len(covered)
    else:
        numerator = sum(prediction.duration_ms for prediction in predictions if predicate(prediction))
        denominator = sum(prediction.duration_ms for prediction in predictions)
        source = "prediction_event_durations"
        coverage = 0
    return {
        "denominator_ms": denominator,
        "numerator_ms": numerator,
        "rate": safe_ratio(numerator, denominator),
        "session_counter_coverage": {
            "sessions_with_counters": coverage,
            "total_sessions": len(sessions),
        },
        "source": source,
    }


def evaluate(
    ground_truth: Iterable[GroundTruthEvent],
    predictions: Iterable[PredictionEvent],
    sessions: Iterable[SessionRecord],
    actions: Iterable[ActionRecord] = (),
    config: EvaluationConfig | None = None,
) -> dict[str, JsonValue]:
    """Evaluate event decisions and return a deterministic JSON-ready report."""

    if config is None:
        config = EvaluationConfig()
    truth_items = list(ground_truth)
    prediction_items = list(predictions)
    session_items = list(sessions)
    action_items = list(actions)
    _validate_unique(truth_items, "ground-truth event", "event_id")
    _validate_unique(prediction_items, "prediction event", "event_id")
    _validate_unique(session_items, "session", "session_id")
    _validate_unique(action_items, "action", "action_id")

    target_set = set(config.target_behaviors)
    target_truth = [event for event in truth_items if event.behavior in target_set]
    if config.would_action_only:
        # Every would-action enters the precision denominator. An UNKNOWN or
        # otherwise non-harmful authorization is a false positive, not ignored.
        target_predictions = [event for event in prediction_items if event.would_action]
    else:
        target_predictions = [event for event in prediction_items if event.behavior in target_set]
    overall = match_events(target_truth, target_predictions, config.match)

    by_behavior: dict[str, JsonValue] = {}
    for behavior in sorted(config.target_behaviors, key=lambda item: item.value):
        behavior_result = match_events(
            (event for event in target_truth if event.behavior == behavior),
            (event for event in target_predictions if event.behavior == behavior),
            config.match,
        )
        by_behavior[behavior.value] = _metric_counts(behavior_result)

    monitored_ms = sum(session.monitored_duration_ms for session in session_items)
    monitored_hours = monitored_ms / 3_600_000.0
    false_would_actions = len(overall.unmatched_predictions)
    observed_rate = safe_ratio(false_would_actions, monitored_hours)
    upper_rate = poisson_rate_upper_bound(
        false_would_actions,
        monitored_hours,
        config.poisson_confidence,
    )

    onset_to_ready: list[int] = []
    ready_to_burst: list[int] = []
    bursts_by_incident: dict[tuple[str, str], list[ActionRecord]] = defaultdict(list)
    for action in action_items:
        if action.action == ActionType.BURST and action.incident_id is not None:
            bursts_by_incident[(action.session_id, action.incident_id)].append(action)
    for burst_items in bursts_by_incident.values():
        burst_items.sort(key=lambda item: (item.timestamp_ms, item.action_id))
    for matched in overall.matches:
        prediction = matched.prediction
        if prediction.ready_ms is None:
            continue
        onset_to_ready.append(prediction.ready_ms - matched.truth.start_ms)
        if prediction.incident_id is None:
            continue
        burst_items = bursts_by_incident.get((prediction.session_id, prediction.incident_id), [])
        if burst_items:
            ready_to_burst.append(burst_items[0].timestamp_ms - prediction.ready_ms)

    session_ids = {session.session_id for session in session_items}
    used_session_ids = {
        *(event.session_id for event in truth_items),
        *(event.session_id for event in prediction_items),
        *(action.session_id for action in action_items),
    }
    warnings = [
        f"session {session_id!r} has events but no session exposure record"
        for session_id in sorted(used_session_ids - session_ids)
    ]
    if monitored_ms == 0:
        warnings.append("monitored exposure is zero; false-action rates are undefined")
    track_counter_sessions = sum(session.track_observation_ms is not None for session in session_items)
    behavior_counter_sessions = sum(session.behavior_observation_ms is not None for session in session_items)
    if 0 < track_counter_sessions < len(session_items):
        warnings.append("track-loss session counters have partial coverage")
    if 0 < behavior_counter_sessions < len(session_items):
        warnings.append("UNKNOWN-behavior session counters have partial coverage")

    safety = check_safety(session_items, prediction_items, action_items)
    return {
        "config": config.to_dict(),
        "data_quality": {"passed": not warnings, "warnings": sorted(warnings)},
        "dataset": {
            "action_records": len(action_items),
            "ground_truth_events": len(truth_items),
            "prediction_events": len(prediction_items),
            "sessions": len(session_items),
            "target_ground_truth_events": len(target_truth),
            "target_prediction_events": len(target_predictions),
        },
        "event_metrics": {
            "by_behavior": by_behavior,
            "overall": _metric_counts(overall),
            "zero_division_convention": 0.0,
        },
        "exposure_and_false_actions": {
            "false_would_actions": false_would_actions,
            "monitored_duration_ms": monitored_ms,
            "monitored_hours": monitored_hours,
            "observed_false_would_actions_per_hour": observed_rate,
            "one_sided_upper_bound_per_hour": upper_rate,
            "upper_bound_confidence": config.poisson_confidence,
            "upper_bound_method": "exact_poisson_garwood (rule_of_three_when_zero)",
        },
        "latency": {
            "behavior_onset_to_ready": _distribution(onset_to_ready, config.latency_percentiles),
            "ready_to_burst": _distribution(ready_to_burst, config.latency_percentiles),
        },
        "matching": {
            "matches": [match.to_dict() for match in overall.matches],
            "unmatched_ground_truth_event_ids": [event.event_id for event in overall.unmatched_truth],
            "unmatched_prediction_event_ids": [event.event_id for event in overall.unmatched_predictions],
        },
        "quality": {
            "track_loss": _duration_rate(session_items, prediction_items, "track_lost_ms"),
            "unknown_behavior": _duration_rate(session_items, prediction_items, "unknown_behavior_ms"),
        },
        "safety": safety.to_dict(),
        "schema_version": SCHEMA_VERSION,
    }
