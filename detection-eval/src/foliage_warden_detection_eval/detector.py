"""Adapter from the pinned perception package to COCO prediction records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import PreparedSubset
from .errors import DetectionEvalError
from .metrics import Prediction


def run_pinned_detector(
    prepared: PreparedSubset,
    *,
    registry_path: Path,
    model_id: str,
    model_path: Path | None,
    cat_confidence: float,
    person_confidence: float,
    nms_iou: float,
    backend_target: str,
) -> tuple[list[Prediction], dict[str, Any]]:
    """Run the repository-pinned YOLOX implementation on each selected image."""

    try:
        import numpy as np
        from foliage_warden_perception.dependencies import require_cv2
        from foliage_warden_perception.errors import PerceptionError
        from foliage_warden_perception.registry import (
            load_model_spec,
            resolve_and_verify_model,
        )
        from foliage_warden_perception.types import ObjectClass
        from foliage_warden_perception.yolox import YOLOXDetector
    except (ImportError, OSError) as error:
        raise DetectionEvalError(
            "the perception package and desktop OpenCV are required; run "
            "`uv sync --project detection-eval --extra desktop --group dev`"
        ) from error

    try:
        spec = load_model_spec(registry_path, model_id)
        verified_model = resolve_and_verify_model(registry_path, spec, model_path)
        cv2 = require_cv2()
        detector = YOLOXDetector(
            verified_model,
            spec,
            cat_confidence=cat_confidence,
            person_confidence=person_confidence,
            nms_iou_threshold=nms_iou,
            backend_target=backend_target,
            cv2_module=cv2,
        )
        categories = prepared.manifest["categories"]
        category_by_object_class = {
            ObjectClass.CAT: int(categories["cat"]),
            ObjectClass.PERSON: int(categories["person"]),
        }
        predictions: list[Prediction] = []
        raw_images = sorted(prepared.manifest["images"], key=lambda item: item["id"])
        for raw_image in raw_images:
            image_id = int(raw_image["id"])
            expected_width = int(raw_image["width"])
            expected_height = int(raw_image["height"])
            image_path = prepared.image_dir / str(raw_image["file_name"])
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise DetectionEvalError(f"OpenCV could not decode selected image {image_path}")
            height, width = frame.shape[:2]
            if width != expected_width or height != expected_height:
                raise DetectionEvalError(
                    f"decoded shape {width}x{height} differs from COCO metadata "
                    f"{expected_width}x{expected_height} for image {image_id}"
                )
            for detection in detector.detect(frame):
                box = detection.bbox
                predictions.append(
                    Prediction(
                        image_id=image_id,
                        category_id=category_by_object_class[detection.object_class],
                        bbox=(
                            box.x * width,
                            box.y * height,
                            box.width * width,
                            box.height * height,
                        ),
                        score=detection.confidence,
                    )
                )
    except (PerceptionError, OSError, ValueError, KeyError, TypeError) as error:
        raise DetectionEvalError(f"detector evaluation failed: {error}") from error

    predictions.sort(key=lambda item: (-item.score, item.image_id, item.category_id, item.bbox))
    model = {
        "backend_target": backend_target,
        "cat_confidence_floor": cat_confidence,
        "description": spec.description,
        "id": spec.model_id,
        "nms_iou_threshold": nms_iou,
        "person_confidence_floor": person_confidence,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "sha256": spec.sha256,
        "source_revision": spec.source_revision,
    }
    return predictions, model
