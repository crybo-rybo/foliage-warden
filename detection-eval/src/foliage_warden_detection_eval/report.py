"""Stable prediction and evaluation report construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .cache import sha256_file
from .constants import PUBLIC_DATA_WARNING
from .dataset import PreparedSubset
from .errors import DetectionEvalError
from .metrics import GroundTruth, Prediction, evaluate_detections


def _wire_float(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded


def predictions_to_coco(predictions: list[Prediction]) -> list[dict[str, Any]]:
    values = [
        {
            "bbox": [_wire_float(value) for value in prediction.bbox],
            "category_id": prediction.category_id,
            "image_id": prediction.image_id,
            "score": _wire_float(prediction.score),
        }
        for prediction in predictions
    ]
    return sorted(
        values,
        key=lambda item: (
            -item["score"],
            item["image_id"],
            item["category_id"],
            item["bbox"],
        ),
    )


def _predictions_from_canonical(values: list[dict[str, Any]]) -> list[Prediction]:
    """Recover the exact metric input represented by the prediction artifact."""

    return [
        Prediction(
            image_id=int(item["image_id"]),
            category_id=int(item["category_id"]),
            bbox=tuple(float(value) for value in item["bbox"]),
            score=float(item["score"]),
        )
        for item in values
    ]


def write_stable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))


def stable_json_bytes(value: Any) -> bytes:
    """Serialize an artifact canonically for byte equality and content binding."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def require_exact_report(path: Path, report: dict[str, Any]) -> None:
    """Fail closed unless a regenerated report is byte-identical to a locked report."""

    try:
        expected = path.read_bytes()
    except OSError as error:
        raise DetectionEvalError(f"cannot read expected report {path}: {error}") from error
    actual = stable_json_bytes(report)
    if actual != expected:
        raise DetectionEvalError(
            "regenerated report differs from locked expected report: "
            f"SHA-256 {hashlib.sha256(actual).hexdigest()} != "
            f"{hashlib.sha256(expected).hexdigest()}"
        )


def build_report(
    prepared: PreparedSubset,
    predictions: list[Prediction],
    model: dict[str, Any],
    *,
    iou_threshold: float,
    example_limit: int,
    score_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    selected_ids = {int(item["id"]) for item in prepared.manifest["images"]}
    categories = {
        str(name): int(category_id) for name, category_id in prepared.manifest["categories"].items()
    }
    target_ids = set(categories.values())
    ground_truths = [
        GroundTruth(
            annotation_id=annotation.annotation_id,
            image_id=annotation.image_id,
            category_id=annotation.category_id,
            bbox=annotation.bbox,
            ignore=annotation.iscrowd,
        )
        for annotation in prepared.index.annotations
        if annotation.image_id in selected_ids and annotation.category_id in target_ids
    ]
    prediction_artifact = predictions_to_coco(predictions)
    result = evaluate_detections(
        ground_truths,
        _predictions_from_canonical(prediction_artifact),
        categories,
        iou_threshold=iou_threshold,
        example_limit=example_limit,
        score_thresholds=score_thresholds,
    )
    return {
        "dataset": {
            "image_count": len(selected_ids),
            "manifest_sha256": sha256_file(prepared.manifest_path),
            "name": prepared.manifest["dataset"],
            "selection": prepared.manifest["selection"],
        },
        "definitions": {
            "ap50": (
                "101-point interpolated average precision at one-to-one IoU >= 0.50; "
                "this is not the full pycocotools area/maxDet metric suite"
            ),
            "crowd": (
                "iscrowd ground truths are excluded from recall; otherwise-unmatched "
                "predictions with standard IoU >= threshold against them are ignored"
            ),
            "precision_recall": (
                "computed at the per-class operating score thresholds; AP50 uses all "
                "predictions retained by the detector confidence floors"
            ),
        },
        "metrics": result.to_dict(),
        "model": model,
        "prediction_artifact": {
            "format": "COCO detection results JSON array",
            "sha256": stable_json_sha256(prediction_artifact),
        },
        "prediction_count": len(predictions),
        "schema_version": 2,
        "scope_warning": PUBLIC_DATA_WARNING,
    }
