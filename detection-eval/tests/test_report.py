from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from foliage_warden_detection_eval.coco import CocoAnnotation, CocoImage, CocoIndex
from foliage_warden_detection_eval.dataset import PreparedSubset
from foliage_warden_detection_eval.errors import DetectionEvalError
from foliage_warden_detection_eval.metrics import Prediction
from foliage_warden_detection_eval.report import (
    build_report,
    predictions_to_coco,
    require_exact_report,
    write_stable_json,
)


def test_report_bytes_are_stable_and_have_scope_warning(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n")
    manifest = {
        "categories": {"cat": 17, "person": 1},
        "dataset": "tiny",
        "images": [
            {
                "file_name": "one.jpg",
                "height": 100,
                "id": 1,
                "stratum": "cat_positive",
                "width": 100,
            }
        ],
        "selection": {"algorithm": "test"},
    }
    index = CocoIndex(
        images={1: CocoImage(1, "one.jpg", 100, 100, 1)},
        annotations=(CocoAnnotation(10, 1, 17, (10, 10, 20, 20), False),),
        category_ids_by_name={"person": 1, "cat": 17},
        category_names_by_id={1: "person", 17: "cat"},
        licenses=(),
    )
    prepared = PreparedSubset(
        annotation_path=tmp_path / "annotations.json",
        dataset_root=tmp_path,
        image_dir=tmp_path,
        index=index,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    prediction = Prediction(1, 17, (10, 10, 20, 20), 0.75)
    report = build_report(
        prepared,
        [prediction],
        {"id": "test", "sha256": "abc"},
        iou_threshold=0.5,
        example_limit=5,
        score_thresholds={"cat": 0.5, "person": 0.5},
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    prediction_path = tmp_path / "predictions.json"

    write_stable_json(first, report)
    write_stable_json(second, report)
    write_stable_json(prediction_path, predictions_to_coco([prediction]))

    assert first.read_bytes() == second.read_bytes()
    assert report["schema_version"] == 2
    assert (
        report["prediction_artifact"]["sha256"]
        == hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    )
    assert "installed camera" in report["scope_warning"]
    assert report["metrics"]["classes"]["cat"]["ap50"] == 1.0


def test_locked_report_guard_rejects_any_metric_change(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    report = {
        "metrics": {"cat": {"ap50": 0.5}},
        "prediction_artifact": {"sha256": "a" * 64},
    }
    write_stable_json(expected, report)

    require_exact_report(expected, report)
    changed = copy.deepcopy(report)
    changed["metrics"]["cat"]["ap50"] = 0.6

    with pytest.raises(DetectionEvalError, match="regenerated report differs"):
        require_exact_report(expected, changed)


def test_metrics_use_the_exact_rounded_prediction_artifact(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "categories": {"cat": 17},
        "dataset": "tiny",
        "images": [{"id": 1}],
        "selection": {"algorithm": "test"},
    }
    index = CocoIndex(
        images={1: CocoImage(1, "one.jpg", 100, 100, 1)},
        annotations=(CocoAnnotation(10, 1, 17, (10, 10, 20, 20), False),),
        category_ids_by_name={"cat": 17},
        category_names_by_id={17: "cat"},
        licenses=(),
    )
    prepared = PreparedSubset(
        annotation_path=tmp_path / "annotations.json",
        dataset_root=tmp_path,
        image_dir=tmp_path,
        index=index,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    prediction = Prediction(1, 17, (10, 10, 20, 20), 0.4999996)

    report = build_report(
        prepared,
        [prediction],
        {"id": "test", "sha256": "abc"},
        iou_threshold=0.5,
        example_limit=5,
        score_thresholds={"cat": 0.5},
    )

    assert predictions_to_coco([prediction])[0]["score"] == 0.5
    assert report["metrics"]["classes"]["cat"]["true_positive_count"] == 1
