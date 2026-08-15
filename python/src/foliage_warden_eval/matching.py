"""One-to-one temporal matching of complete incidents."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .schemas import GroundTruthEvent, PredictionEvent


@dataclass(frozen=True, slots=True)
class MatchConfig:
    min_temporal_iou: float = 0.1
    max_onset_delta_ms: int | None = None
    require_behavior_match: bool = True
    require_zone_match: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_temporal_iou <= 1.0:
            raise ValueError("min_temporal_iou must be between 0 and 1")
        if self.max_onset_delta_ms is not None and self.max_onset_delta_ms < 0:
            raise ValueError("max_onset_delta_ms must be >= 0")


@dataclass(frozen=True, slots=True)
class EventMatch:
    truth: GroundTruthEvent
    prediction: PredictionEvent
    temporal_iou: float
    onset_delta_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "ground_truth_event_id": self.truth.event_id,
            "onset_delta_ms": self.onset_delta_ms,
            "prediction_event_id": self.prediction.event_id,
            "session_id": self.truth.session_id,
            "temporal_iou": self.temporal_iou,
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[EventMatch, ...]
    unmatched_truth: tuple[GroundTruthEvent, ...]
    unmatched_predictions: tuple[PredictionEvent, ...]


def temporal_iou(truth: GroundTruthEvent, prediction: PredictionEvent) -> float:
    intersection = max(0, min(truth.end_ms, prediction.end_ms) - max(truth.start_ms, prediction.start_ms))
    if intersection == 0:
        return 0.0
    union = max(truth.end_ms, prediction.end_ms) - min(truth.start_ms, prediction.start_ms)
    return intersection / union


def _candidate(
    truth: GroundTruthEvent,
    prediction: PredictionEvent,
    config: MatchConfig,
) -> tuple[float, int] | None:
    if truth.session_id != prediction.session_id:
        return None
    if config.require_behavior_match and truth.behavior != prediction.behavior:
        return None
    if (
        config.require_zone_match
        and truth.zone_id is not None
        and prediction.zone_id is not None
        and truth.zone_id != prediction.zone_id
    ):
        return None
    overlap = temporal_iou(truth, prediction)
    # Even a threshold of zero must not match disjoint incidents.
    if overlap <= 0.0 or overlap < config.min_temporal_iou:
        return None
    onset_delta = prediction.start_ms - truth.start_ms
    if config.max_onset_delta_ms is not None and abs(onset_delta) > config.max_onset_delta_ms:
        return None
    return overlap, onset_delta


def match_events(
    ground_truth: Iterable[GroundTruthEvent],
    predictions: Iterable[PredictionEvent],
    config: MatchConfig | None = None,
) -> MatchResult:
    """Return a deterministic maximum-cardinality bipartite event matching.

    Candidate edges are ranked by higher temporal IoU, then smaller onset
    error, then stable event IDs. The augmenting-path algorithm guarantees the
    maximum possible number of one-to-one matches rather than greedily taking
    a locally attractive pair that suppresses another true positive.
    """

    if config is None:
        config = MatchConfig()
    truths = sorted(ground_truth, key=lambda item: (item.session_id, item.start_ms, item.event_id))
    predicted = sorted(predictions, key=lambda item: (item.session_id, item.start_ms, item.event_id))
    candidates: dict[int, list[tuple[int, float, int]]] = {}
    candidate_values: dict[tuple[int, int], tuple[float, int]] = {}
    for truth_index, truth in enumerate(truths):
        edges: list[tuple[int, float, int]] = []
        for prediction_index, prediction in enumerate(predicted):
            value = _candidate(truth, prediction, config)
            if value is None:
                continue
            overlap, onset_delta = value
            edges.append((prediction_index, overlap, onset_delta))
            candidate_values[(truth_index, prediction_index)] = value
        edges.sort(
            key=lambda edge: (
                -edge[1],
                abs(edge[2]),
                predicted[edge[0]].event_id,
            )
        )
        candidates[truth_index] = edges

    # Scarce truths first tends to preserve high-quality pairings while Kuhn's
    # augmenting paths retain the maximum-cardinality guarantee.
    truth_order = sorted(
        range(len(truths)),
        key=lambda index: (len(candidates[index]), truths[index].session_id, truths[index].start_ms, truths[index].event_id),
    )
    prediction_to_truth: dict[int, int] = {}

    def augment(truth_index: int, seen_predictions: set[int]) -> bool:
        for prediction_index, _overlap, _delta in candidates[truth_index]:
            if prediction_index in seen_predictions:
                continue
            seen_predictions.add(prediction_index)
            previous_truth = prediction_to_truth.get(prediction_index)
            if previous_truth is None or augment(previous_truth, seen_predictions):
                prediction_to_truth[prediction_index] = truth_index
                return True
        return False

    for truth_index in truth_order:
        augment(truth_index, set())

    matched_truth_indices = set(prediction_to_truth.values())
    match_items: list[EventMatch] = []
    for prediction_index, truth_index in prediction_to_truth.items():
        overlap, onset_delta = candidate_values[(truth_index, prediction_index)]
        match_items.append(
            EventMatch(
                truth=truths[truth_index],
                prediction=predicted[prediction_index],
                temporal_iou=overlap,
                onset_delta_ms=onset_delta,
            )
        )
    match_items.sort(key=lambda item: (item.truth.session_id, item.truth.start_ms, item.truth.event_id))

    return MatchResult(
        matches=tuple(match_items),
        unmatched_truth=tuple(
            truth for index, truth in enumerate(truths) if index not in matched_truth_indices
        ),
        unmatched_predictions=tuple(
            prediction for index, prediction in enumerate(predicted) if index not in prediction_to_truth
        ),
    )
