from __future__ import annotations

from foliage_warden_eval.matching import MatchConfig, match_events, temporal_iou
from foliage_warden_eval.schemas import Behavior, GroundTruthEvent, PredictionEvent


def truth(event_id: str, start: int, end: int, *, behavior: Behavior = Behavior.EATING, zone: str | None = "pot") -> GroundTruthEvent:
    return GroundTruthEvent(event_id, "session", behavior, start, end, zone_id=zone)


def prediction(event_id: str, start: int, end: int, *, behavior: Behavior = Behavior.EATING, zone: str | None = "pot") -> PredictionEvent:
    return PredictionEvent(event_id, "session", behavior, start, end, 0.9, zone_id=zone)


def test_temporal_iou_uses_incident_intervals() -> None:
    assert temporal_iou(truth("t", 0, 100), prediction("p", 50, 150)) == 1 / 3
    assert temporal_iou(truth("t", 0, 100), prediction("p", 100, 200)) == 0.0


def test_matching_is_one_to_one_and_maximum_cardinality() -> None:
    # t1 can use p1 or p2, while t2 can only use p1. A naive best-first
    # greedy matcher can consume p1 for t1 and report only one true positive.
    result = match_events(
        [truth("t1", 0, 100), truth("t2", 80, 180)],
        [prediction("p1", 75, 105), prediction("p2", 0, 60)],
        MatchConfig(min_temporal_iou=0.1),
    )
    assert len(result.matches) == 2
    assert not result.unmatched_truth
    assert not result.unmatched_predictions
    assert {(match.truth.event_id, match.prediction.event_id) for match in result.matches} == {
        ("t1", "p2"),
        ("t2", "p1"),
    }


def test_behavior_zone_onset_and_non_overlap_gate_candidates() -> None:
    truths = [truth("t", 100, 300, behavior=Behavior.DIGGING, zone="pot-1")]
    assert len(match_events(truths, [prediction("p", 110, 290)]).matches) == 0
    assert len(
        match_events(
            truths,
            [prediction("p", 110, 290, behavior=Behavior.DIGGING, zone="pot-2")],
        ).matches
    ) == 0
    assert len(
        match_events(
            truths,
            [prediction("p", 250, 350, behavior=Behavior.DIGGING, zone="pot-1")],
            MatchConfig(min_temporal_iou=0.1, max_onset_delta_ms=50),
        ).matches
    ) == 0
    assert len(
        match_events(
            truths,
            [prediction("p", 300, 400, behavior=Behavior.DIGGING, zone="pot-1")],
            MatchConfig(min_temporal_iou=0.0),
        ).matches
    ) == 0

