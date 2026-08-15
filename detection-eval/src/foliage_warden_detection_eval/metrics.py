"""Deterministic AP50 with one-to-one detection/ground-truth matching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class GroundTruth:
    annotation_id: int
    image_id: int
    category_id: int
    bbox: BBox
    ignore: bool = False


@dataclass(frozen=True, slots=True)
class Prediction:
    image_id: int
    category_id: int
    bbox: BBox
    score: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    iou_threshold: float
    classes: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"classes": self.classes, "iou_threshold": _wire_float(self.iou_threshold)}


def _validate_bbox(bbox: BBox, context: str) -> None:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{context} bbox must contain four finite values")
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ValueError(f"{context} bbox width and height must be positive")


def bbox_iou(left: BBox, right: BBox) -> float:
    """Axis-aligned IoU for COCO ``x, y, width, height`` boxes."""

    _validate_bbox(left, "left")
    _validate_bbox(right, "right")
    left_x2 = left[0] + left[2]
    left_y2 = left[1] + left[3]
    right_x2 = right[0] + right[2]
    right_y2 = right[1] + right[3]
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0.0 else 0.0


def _wire_float(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded


def _box_list(box: BBox) -> list[float]:
    return [_wire_float(value) for value in box]


def _sort_predictions(predictions: list[Prediction]) -> list[Prediction]:
    for index, prediction in enumerate(predictions):
        _validate_bbox(prediction.bbox, f"prediction {index}")
        if not math.isfinite(prediction.score) or not 0.0 <= prediction.score <= 1.0:
            raise ValueError(f"prediction {index} score must be finite and within [0, 1]")
    return sorted(
        predictions,
        key=lambda item: (-item.score, item.image_id, item.bbox, item.category_id),
    )


def _average_precision_101(decisions: list[bool], ground_truth_count: int) -> float | None:
    if ground_truth_count == 0:
        return None
    true_positives = 0
    false_positives = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for is_true_positive in decisions:
        true_positives += int(is_true_positive)
        false_positives += int(not is_true_positive)
        recalls.append(true_positives / ground_truth_count)
        precisions.append(true_positives / (true_positives + false_positives))
    interpolated: list[float] = []
    for recall_step in range(101):
        threshold = recall_step / 100.0
        interpolated.append(
            max(
                (
                    precision
                    for recall, precision in zip(recalls, precisions, strict=True)
                    if recall >= threshold
                ),
                default=0.0,
            )
        )
    return sum(interpolated) / len(interpolated)


def _evaluate_class(
    ground_truths: list[GroundTruth],
    predictions: list[Prediction],
    *,
    category_id: int,
    iou_threshold: float,
    example_limit: int,
    score_threshold: float,
) -> dict[str, Any]:
    class_ground_truths = [item for item in ground_truths if item.category_id == category_id]
    class_predictions = _sort_predictions(
        [item for item in predictions if item.category_id == category_id]
    )
    regular_by_image: dict[int, list[GroundTruth]] = {}
    ignored_by_image: dict[int, list[GroundTruth]] = {}
    for truth in class_ground_truths:
        _validate_bbox(truth.bbox, f"ground truth {truth.annotation_id}")
        destination = ignored_by_image if truth.ignore else regular_by_image
        destination.setdefault(truth.image_id, []).append(truth)
    for values in regular_by_image.values():
        values.sort(key=lambda item: item.annotation_id)
    for values in ignored_by_image.values():
        values.sort(key=lambda item: item.annotation_id)

    matched_ids: set[int] = set()
    records: list[tuple[Prediction, bool | None, int | None]] = []
    for prediction in class_predictions:
        candidates = [
            truth
            for truth in regular_by_image.get(prediction.image_id, [])
            if truth.annotation_id not in matched_ids
        ]
        ranked = sorted(
            ((bbox_iou(prediction.bbox, truth.bbox), truth) for truth in candidates),
            key=lambda item: (-item[0], item[1].annotation_id),
        )
        if ranked and ranked[0][0] >= iou_threshold:
            matched_ids.add(ranked[0][1].annotation_id)
            records.append((prediction, True, ranked[0][1].annotation_id))
            continue

        ignored_overlap = max(
            (
                bbox_iou(prediction.bbox, truth.bbox)
                for truth in ignored_by_image.get(prediction.image_id, [])
            ),
            default=0.0,
        )
        if ignored_overlap >= iou_threshold:
            records.append((prediction, None, None))
            continue

        records.append((prediction, False, None))

    ap_decisions = [outcome for _, outcome, _ in records if outcome is not None]
    operating_records = [record for record in records if record[0].score >= score_threshold]
    operating_decisions = [outcome for _, outcome, _ in operating_records if outcome is not None]
    operating_matched_ids = {
        annotation_id
        for _, outcome, annotation_id in operating_records
        if outcome is True and annotation_id is not None
    }
    false_positive_examples: list[dict[str, Any]] = []
    for prediction, outcome, _ in operating_records:
        if outcome is False and len(false_positive_examples) < example_limit:
            all_regular = regular_by_image.get(prediction.image_id, [])
            best_iou = max(
                (bbox_iou(prediction.bbox, truth.bbox) for truth in all_regular),
                default=0.0,
            )
            false_positive_examples.append(
                {
                    "bbox": _box_list(prediction.bbox),
                    "best_ground_truth_iou": _wire_float(best_iou),
                    "image_id": prediction.image_id,
                    "score": _wire_float(prediction.score),
                }
            )

    regular_ground_truths = sorted(
        (truth for truth in class_ground_truths if not truth.ignore),
        key=lambda item: (item.image_id, item.annotation_id),
    )
    misses: list[dict[str, Any]] = []
    for truth in regular_ground_truths:
        if truth.annotation_id in operating_matched_ids:
            continue
        image_predictions = [
            prediction
            for prediction, _, _ in operating_records
            if prediction.image_id == truth.image_id
        ]
        best_iou = max(
            (bbox_iou(truth.bbox, prediction.bbox) for prediction in image_predictions),
            default=0.0,
        )
        if len(misses) < example_limit:
            misses.append(
                {
                    "annotation_id": truth.annotation_id,
                    "bbox": _box_list(truth.bbox),
                    "best_prediction_iou": _wire_float(best_iou),
                    "image_id": truth.image_id,
                }
            )

    true_positives = sum(operating_decisions)
    false_positives = len(operating_decisions) - true_positives
    ground_truth_count = len(regular_ground_truths)
    false_negatives = ground_truth_count - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )
    recall = true_positives / ground_truth_count if ground_truth_count else None
    average_precision = _average_precision_101(ap_decisions, ground_truth_count)
    return {
        "ap50": None if average_precision is None else _wire_float(average_precision),
        "ap_evaluated_prediction_count": len(ap_decisions),
        "category_id": category_id,
        "evaluated_prediction_count": len(operating_decisions),
        "false_negative_count": false_negatives,
        "false_positive_count": false_positives,
        "false_positive_examples": false_positive_examples,
        "ground_truth_count": ground_truth_count,
        "ignored_ground_truth_count": len(class_ground_truths) - ground_truth_count,
        "ignored_prediction_count": sum(outcome is None for _, outcome, _ in operating_records),
        "miss_examples": misses,
        "precision": None if precision is None else _wire_float(precision),
        "prediction_count": len(class_predictions),
        "recall": None if recall is None else _wire_float(recall),
        "score_threshold": _wire_float(score_threshold),
        "true_positive_count": true_positives,
    }


def evaluate_detections(
    ground_truths: list[GroundTruth],
    predictions: list[Prediction],
    categories: dict[str, int],
    *,
    iou_threshold: float = 0.5,
    example_limit: int = 10,
    score_thresholds: dict[str, float] | None = None,
) -> EvaluationResult:
    """Evaluate each requested class independently at the AP50 IoU threshold.

    Predictions are greedily score-ordered and each non-ignored ground-truth
    box can match at most one prediction. Crowd/ignored boxes do not count as
    positives and absorb otherwise-unmatched overlapping predictions.
    """

    if not math.isfinite(iou_threshold) or iou_threshold != 0.5:
        raise ValueError("iou_threshold must be 0.5 for this AP50 evaluator")
    if example_limit < 0:
        raise ValueError("example_limit must be non-negative")
    thresholds = score_thresholds if score_thresholds is not None else {}
    for name, threshold in thresholds.items():
        if name not in categories:
            raise ValueError(f"score threshold references unknown category {name!r}")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"score threshold for {name} must be within [0, 1]")
    classes = {
        name: _evaluate_class(
            ground_truths,
            predictions,
            category_id=category_id,
            iou_threshold=iou_threshold,
            example_limit=example_limit,
            score_threshold=thresholds.get(name, 0.0),
        )
        for name, category_id in sorted(categories.items())
    }
    return EvaluationResult(iou_threshold=iou_threshold, classes=classes)
