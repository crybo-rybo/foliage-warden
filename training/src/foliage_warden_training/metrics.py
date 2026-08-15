from __future__ import annotations

from typing import Any

import numpy as np

from .labels import BEHAVIOR_LABELS, HARMFUL_LABELS, LABEL_TO_INDEX


def _probability_metrics(
    targets: list[int],
    probability_rows: list[list[float]],
) -> dict[str, Any]:
    probabilities = np.asarray(probability_rows, dtype=np.float64)
    expected_shape = (len(targets), len(BEHAVIOR_LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"probabilities must have shape {expected_shape}, got {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("each probability row must sum to one")

    target_array = np.asarray(targets, dtype=np.int64)
    one_hot = np.eye(len(BEHAVIOR_LABELS), dtype=np.float64)[target_array]
    target_probabilities = probabilities[np.arange(len(targets)), target_array]
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correctness = predicted == target_array

    calibration_error = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        in_bin = (confidence >= lower) & (
            confidence <= upper if bin_index == 9 else confidence < upper
        )
        if np.any(in_bin):
            calibration_error += float(
                np.mean(in_bin)
                * abs(float(np.mean(correctness[in_bin])) - float(np.mean(confidence[in_bin])))
            )

    harmful_indices = [LABEL_TO_INDEX[label] for label in sorted(HARMFUL_LABELS)]
    harmful_scores = probabilities[:, harmful_indices].sum(axis=1)
    harmful_actual = np.isin(target_array, harmful_indices)
    operating_points: dict[str, dict[str, float | int]] = {}
    for threshold in (0.5, 0.7, 0.9):
        harmful_predicted = harmful_scores >= threshold
        true_positive = int(np.sum(harmful_actual & harmful_predicted))
        false_positive = int(np.sum(~harmful_actual & harmful_predicted))
        false_negative = int(np.sum(harmful_actual & ~harmful_predicted))
        true_negative = int(np.sum(~harmful_actual & ~harmful_predicted))
        operating_points[f"{threshold:.2f}"] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            ),
            "false_positive_rate": (
                false_positive / (false_positive + true_negative)
                if false_positive + true_negative
                else 0.0
            ),
        }

    top_two = np.argpartition(probabilities, -2, axis=1)[:, -2:]
    return {
        "negative_log_likelihood": float(
            -np.mean(np.log(np.clip(target_probabilities, 1e-12, 1.0)))
        ),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "top_2_accuracy": float(np.mean(np.any(top_two == target_array[:, None], axis=1))),
        "expected_calibration_error_10_bin": calibration_error,
        "harmful_probability_operating_points": operating_points,
    }


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]] | None = None,
) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    if not targets:
        raise ValueError("cannot calculate metrics for an empty dataset")

    class_count = len(BEHAVIOR_LABELS)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        if not (0 <= target < class_count and 0 <= prediction < class_count):
            raise ValueError("target and prediction indices must be valid behavior labels")
        confusion[target, prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    supported_f1: list[float] = []
    for index, label in enumerate(BEHAVIOR_LABELS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            supported_f1.append(f1)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    harmful_indices = {LABEL_TO_INDEX[label] for label in HARMFUL_LABELS}
    harmful_tp = sum(
        target in harmful_indices and prediction in harmful_indices
        for target, prediction in zip(targets, predictions, strict=True)
    )
    harmful_predicted = sum(prediction in harmful_indices for prediction in predictions)
    harmful_actual = sum(target in harmful_indices for target in targets)
    harmful_precision = harmful_tp / harmful_predicted if harmful_predicted else 0.0
    harmful_recall = harmful_tp / harmful_actual if harmful_actual else 0.0

    result = {
        "clips": len(targets),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_f1_supported_classes": float(np.mean(supported_f1)),
        "harmful_binary": {
            "labels": sorted(HARMFUL_LABELS),
            "actual": harmful_actual,
            "predicted": harmful_predicted,
            "true_positive": harmful_tp,
            "precision": harmful_precision,
            "recall": harmful_recall,
        },
        "unknown_prediction_rate": (
            sum(prediction == LABEL_TO_INDEX["UNKNOWN"] for prediction in predictions)
            / len(predictions)
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "rows": "actual",
            "columns": "predicted",
            "labels": list(BEHAVIOR_LABELS),
            "values": confusion.tolist(),
        },
    }
    if probabilities is not None:
        result["probability_quality"] = _probability_metrics(targets, probabilities)
    return result
