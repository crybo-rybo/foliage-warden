from __future__ import annotations

import pytest

from foliage_warden_detection_eval.metrics import (
    GroundTruth,
    Prediction,
    bbox_iou,
    evaluate_detections,
)


def test_bbox_iou_uses_xywh_geometry() -> None:
    assert bbox_iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 10.0, 10.0)) == pytest.approx(25.0 / 175.0)


def test_duplicate_prediction_is_false_positive_not_second_match() -> None:
    truth = [GroundTruth(1, 10, 17, (0.0, 0.0, 10.0, 10.0))]
    predictions = [
        Prediction(10, 17, (0.0, 0.0, 10.0, 10.0), 0.9),
        Prediction(10, 17, (0.0, 0.0, 10.0, 10.0), 0.8),
    ]

    cat = evaluate_detections(truth, predictions, {"cat": 17}).classes["cat"]

    assert cat["true_positive_count"] == 1
    assert cat["false_positive_count"] == 1
    assert cat["false_negative_count"] == 0
    assert cat["precision"] == 0.5
    assert cat["recall"] == 1.0
    assert cat["ap50"] == 1.0


def test_high_scoring_false_positive_reduces_ap50() -> None:
    truth = [GroundTruth(1, 10, 17, (0.0, 0.0, 10.0, 10.0))]
    predictions = [
        Prediction(10, 17, (20.0, 20.0, 5.0, 5.0), 0.9),
        Prediction(10, 17, (0.0, 0.0, 10.0, 10.0), 0.8),
    ]

    cat = evaluate_detections(truth, predictions, {"cat": 17}).classes["cat"]

    assert cat["ap50"] == 0.5
    assert cat["false_positive_examples"] == [
        {
            "bbox": [20.0, 20.0, 5.0, 5.0],
            "best_ground_truth_iou": 0.0,
            "image_id": 10,
            "score": 0.9,
        }
    ]


def test_crowd_overlap_absorbs_unmatched_prediction() -> None:
    truth = [
        GroundTruth(1, 10, 17, (0.0, 0.0, 10.0, 10.0)),
        GroundTruth(2, 11, 17, (0.0, 0.0, 20.0, 20.0), ignore=True),
    ]
    predictions = [
        Prediction(10, 17, (0.0, 0.0, 10.0, 10.0), 0.9),
        Prediction(11, 17, (0.0, 0.0, 20.0, 20.0), 0.8),
    ]

    cat = evaluate_detections(truth, predictions, {"cat": 17}).classes["cat"]

    assert cat["prediction_count"] == 2
    assert cat["evaluated_prediction_count"] == 1
    assert cat["ignored_prediction_count"] == 1
    assert cat["ignored_ground_truth_count"] == 1
    assert cat["precision"] == 1.0


def test_class_metrics_and_examples_are_stably_ordered() -> None:
    truth = [
        GroundTruth(3, 30, 1, (0.0, 0.0, 5.0, 5.0)),
        GroundTruth(2, 20, 17, (1.0, 1.0, 5.0, 5.0)),
        GroundTruth(1, 10, 17, (1.0, 1.0, 5.0, 5.0)),
    ]

    result = evaluate_detections(
        truth,
        [],
        {"person": 1, "cat": 17},
        example_limit=10,
    ).to_dict()

    assert list(result["classes"]) == ["cat", "person"]
    assert [item["image_id"] for item in result["classes"]["cat"]["miss_examples"]] == [
        10,
        20,
    ]
    assert result["classes"]["person"]["precision"] is None
    assert result["classes"]["person"]["recall"] == 0.0


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="AP50"):
        evaluate_detections([], [], {"cat": 17}, iou_threshold=0.0)


def test_precision_recall_use_operating_threshold_while_ap_uses_score_curve() -> None:
    truth = [GroundTruth(1, 10, 17, (0.0, 0.0, 10.0, 10.0))]
    predictions = [
        Prediction(10, 17, (0.0, 0.0, 10.0, 10.0), 0.4),
        Prediction(10, 17, (20.0, 20.0, 5.0, 5.0), 0.3),
    ]

    cat = evaluate_detections(
        truth,
        predictions,
        {"cat": 17},
        score_thresholds={"cat": 0.5},
    ).classes["cat"]

    assert cat["ap50"] == 1.0
    assert cat["ap_evaluated_prediction_count"] == 2
    assert cat["evaluated_prediction_count"] == 0
    assert cat["score_threshold"] == 0.5
    assert cat["precision"] is None
    assert cat["recall"] == 0.0
