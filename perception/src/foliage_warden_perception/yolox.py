"""OpenCV Zoo YOLOX-S preprocessing, decode, and inference adapter.

The letterbox and grid decode follow OpenCV Zoo's Apache-2.0 licensed YOLOX
demo at revision ``47534e27c9851bb1128ccc0102f1145e27f23f98``. The code here
was rewritten to provide normalized typed output and deterministic, per-class
NumPy NMS that works with OpenCV 4.8 (which lacks ``NMSBoxesBatched``).
See ``THIRD_PARTY_NOTICES.md`` for attribution.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .dependencies import require_cv2
from .errors import ModelError
from .registry import ModelSpec
from .types import Detection, NormalizedBox, ObjectClass

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

YOLOX_STRIDES = (8, 16, 32)


@dataclass(frozen=True, slots=True)
class LetterboxMetadata:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    scale: float
    pad_x: int = 0
    pad_y: int = 0


@dataclass(frozen=True, slots=True)
class DetectorTimings:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


def letterbox_bgr(
    frame: npt.NDArray[np.uint8],
    *,
    input_width: int,
    input_height: int,
    cv2_module: Any | None = None,
) -> tuple[FloatArray, LetterboxMetadata]:
    """Top-left letterbox a BGR frame and return the model's RGB NCHW float blob.

    The pinned OpenCV Zoo export expects values in ``[0, 255]``. It does not
    apply ``/ 255``, mean subtraction, or standard-deviation normalization.
    """

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame must be a non-empty HxWx3 BGR array")
    if input_width <= 0 or input_height <= 0:
        raise ValueError("model input dimensions must be positive")
    cv2 = cv2_module if cv2_module is not None else require_cv2()

    original_height, original_width = frame.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(original_width * scale))
    resized_height = max(1, int(original_height * scale))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32, copy=False)
    padded = np.full((input_height, input_width, 3), 114.0, dtype=np.float32)
    padded[:resized_height, :resized_width] = resized
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    metadata = LetterboxMetadata(
        original_width=original_width,
        original_height=original_height,
        input_width=input_width,
        input_height=input_height,
        resized_width=resized_width,
        resized_height=resized_height,
        scale=scale,
    )
    return blob, metadata


def generate_yolox_grid(
    input_width: int,
    input_height: int,
    strides: tuple[int, ...] = YOLOX_STRIDES,
) -> tuple[FloatArray, FloatArray]:
    """Generate row-major YOLOX grid cells and their expanded strides."""

    if input_width <= 0 or input_height <= 0:
        raise ValueError("model input dimensions must be positive")
    grids: list[FloatArray] = []
    expanded: list[FloatArray] = []
    for stride in strides:
        if stride <= 0:
            raise ValueError("YOLOX strides must be positive")
        grid_height = input_height // stride
        grid_width = input_width // stride
        grid_x, grid_y = np.meshgrid(
            np.arange(grid_width, dtype=np.float32),
            np.arange(grid_height, dtype=np.float32),
            indexing="xy",
        )
        grid = np.stack((grid_x, grid_y), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), float(stride), dtype=np.float32))
    return np.concatenate(grids, axis=0), np.concatenate(expanded, axis=0)


def _xyxy_iou(box: FloatArray, others: FloatArray) -> FloatArray:
    intersection_x1 = np.maximum(box[0], others[:, 0])
    intersection_y1 = np.maximum(box[1], others[:, 1])
    intersection_x2 = np.minimum(box[2], others[:, 2])
    intersection_y2 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
        0.0, intersection_y2 - intersection_y1
    )
    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    other_areas = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = box_area + other_areas - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0.0,
    )


def class_aware_nms(
    boxes_xyxy: FloatArray,
    scores: FloatArray,
    class_ids: IntArray,
    *,
    iou_threshold: float,
    prediction_indices: IntArray | None = None,
) -> IntArray:
    """Return deterministic keep indices after independent per-class NMS.

    Score ties are broken using the original YOLOX row index, not input list
    order or an OpenCV-version-specific implementation detail.
    """

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape [N, 4]")
    if len(boxes) != len(score_values) or len(boxes) != len(classes):
        raise ValueError("boxes, scores, and class_ids must have equal length")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("NMS IoU threshold must be within [0, 1]")
    if prediction_indices is None:
        original_indices = np.arange(len(boxes), dtype=np.int64)
    else:
        original_indices = np.asarray(prediction_indices, dtype=np.int64).reshape(-1)
        if len(original_indices) != len(boxes):
            raise ValueError("prediction_indices must have the same length as boxes")

    kept: list[int] = []
    for class_id in sorted(int(value) for value in np.unique(classes)):
        candidates = [index for index in range(len(boxes)) if int(classes[index]) == class_id]
        candidates.sort(
            key=lambda index: (-float(score_values[index]), int(original_indices[index]))
        )
        while candidates:
            current = candidates.pop(0)
            kept.append(current)
            if not candidates:
                break
            remaining = np.asarray(candidates, dtype=np.int64)
            overlaps = _xyxy_iou(boxes[current], boxes[remaining])
            candidates = [
                index
                for index, overlap in zip(candidates, overlaps, strict=True)
                if float(overlap) <= iou_threshold
            ]

    kept.sort(
        key=lambda index: (
            -float(score_values[index]),
            int(classes[index]),
            int(original_indices[index]),
        )
    )
    return np.asarray(kept, dtype=np.int64)


def decode_yolox(
    outputs: npt.ArrayLike,
    metadata: LetterboxMetadata,
    *,
    person_class_id: int = 0,
    cat_class_id: int = 15,
    person_confidence: float = 0.5,
    cat_confidence: float = 0.5,
    nms_iou_threshold: float = 0.5,
    strides: tuple[int, ...] = YOLOX_STRIDES,
) -> list[Detection]:
    """Decode raw ``[1, N, 5+C]`` YOLOX output into normalized detections."""

    for name, threshold in (
        ("person confidence", person_confidence),
        ("cat confidence", cat_confidence),
        ("NMS IoU", nms_iou_threshold),
    ):
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} threshold must be finite and within [0, 1]")

    predictions = np.asarray(outputs, dtype=np.float32)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise ModelError(
            f"unexpected YOLOX output shape {predictions.shape}; expected [1, N, 5+C]"
        )
    class_count = predictions.shape[1] - 5
    if max(person_class_id, cat_class_id) >= class_count:
        raise ModelError(
            f"YOLOX output has {class_count} classes but registry references class "
            f"{max(person_class_id, cat_class_id)}"
        )

    grid, expanded_strides = generate_yolox_grid(
        metadata.input_width,
        metadata.input_height,
        strides,
    )
    if len(predictions) != len(grid):
        raise ModelError(
            f"YOLOX output has {len(predictions)} rows; input "
            f"{metadata.input_width}x{metadata.input_height} and strides {strides} require "
            f"{len(grid)}"
        )

    decoded = predictions.copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * expanded_strides
    with np.errstate(over="ignore", invalid="ignore"):
        decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * expanded_strides

    all_scores = decoded[:, 4:5] * decoded[:, 5:]
    finite_scores = np.where(np.isfinite(all_scores), all_scores, -np.inf)
    best_class_ids = np.argmax(finite_scores, axis=1).astype(np.int64)
    best_scores = finite_scores[np.arange(len(decoded)), best_class_ids]
    thresholds = np.where(
        best_class_ids == person_class_id,
        person_confidence,
        np.where(best_class_ids == cat_class_id, cat_confidence, np.inf),
    )
    selected = np.flatnonzero(best_scores >= thresholds).astype(np.int64)
    if len(selected) == 0:
        return []

    centers = decoded[selected, :2]
    sizes = decoded[selected, 2:4]
    boxes = np.empty((len(selected), 4), dtype=np.float32)
    boxes[:, :2] = centers - sizes / 2.0
    boxes[:, 2:] = centers + sizes / 2.0
    selected_scores = np.clip(best_scores[selected], 0.0, 1.0).astype(np.float32)
    selected_classes = best_class_ids[selected]

    finite_geometry = np.all(np.isfinite(boxes), axis=1)
    positive_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    valid = finite_geometry & positive_geometry
    boxes = boxes[valid]
    selected_scores = selected_scores[valid]
    selected_classes = selected_classes[valid]
    selected = selected[valid]
    if len(selected) == 0:
        return []

    keep = class_aware_nms(
        boxes,
        selected_scores,
        selected_classes,
        iou_threshold=nms_iou_threshold,
        prediction_indices=selected,
    )

    detections: list[Detection] = []
    for kept_index in keep:
        prediction_index = int(selected[kept_index])
        class_id = int(selected_classes[kept_index])
        box = boxes[kept_index]
        x1 = float(np.clip((box[0] - metadata.pad_x) / metadata.scale, 0, metadata.original_width))
        y1 = float(np.clip((box[1] - metadata.pad_y) / metadata.scale, 0, metadata.original_height))
        x2 = float(np.clip((box[2] - metadata.pad_x) / metadata.scale, 0, metadata.original_width))
        y2 = float(np.clip((box[3] - metadata.pad_y) / metadata.scale, 0, metadata.original_height))
        if x2 <= x1 or y2 <= y1:
            continue
        object_class = ObjectClass.PERSON if class_id == person_class_id else ObjectClass.CAT
        detections.append(
            Detection(
                object_class=object_class,
                class_id=class_id,
                confidence=float(selected_scores[kept_index]),
                bbox=NormalizedBox(
                    x=x1 / metadata.original_width,
                    y=y1 / metadata.original_height,
                    width=(x2 - x1) / metadata.original_width,
                    height=(y2 - y1) / metadata.original_height,
                ),
                prediction_index=prediction_index,
            )
        )
    detections.sort(
        key=lambda detection: (
            -detection.confidence,
            detection.object_class.value,
            detection.prediction_index,
        )
    )
    return detections


class YOLOXDetector:
    """Thin OpenCV DNN adapter for the pinned OpenCV Zoo model."""

    def __init__(
        self,
        model_path: str | Path,
        spec: ModelSpec,
        *,
        person_confidence: float = 0.5,
        cat_confidence: float = 0.5,
        nms_iou_threshold: float = 0.5,
        backend_target: str = "opencv",
        cv2_module: Any | None = None,
        net: Any | None = None,
    ) -> None:
        if spec.input_color.upper() != "RGB":
            raise ModelError(f"unsupported model input color {spec.input_color!r}; expected RGB")
        self._cv2 = cv2_module if cv2_module is not None else require_cv2()
        self.spec = spec
        self.person_confidence = person_confidence
        self.cat_confidence = cat_confidence
        self.nms_iou_threshold = nms_iou_threshold
        try:
            self._net = net if net is not None else self._cv2.dnn.readNet(str(model_path))
            self._configure_backend(backend_target)
        except Exception as error:
            if isinstance(error, (ModelError, ValueError)):
                raise
            raise ModelError(
                f"OpenCV could not load/configure YOLOX model {model_path} for "
                f"backend {backend_target!r}: {error}"
            ) from error

    def _configure_backend(self, backend_target: str) -> None:
        cv2 = self._cv2
        pairs = {
            "opencv": (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),
            "cuda": (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA),
            "cuda-fp16": (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16),
        }
        if backend_target not in pairs:
            raise ValueError(f"unknown backend target {backend_target!r}")
        backend, target = pairs[backend_target]
        self._net.setPreferableBackend(backend)
        self._net.setPreferableTarget(target)

    def detect_timed(
        self, frame: npt.NDArray[np.uint8]
    ) -> tuple[list[Detection], DetectorTimings]:
        start = time.perf_counter_ns()
        blob, metadata = letterbox_bgr(
            frame,
            input_width=self.spec.input_width,
            input_height=self.spec.input_height,
            cv2_module=self._cv2,
        )
        after_preprocess = time.perf_counter_ns()
        try:
            self._net.setInput(blob)
            output_names = self._net.getUnconnectedOutLayersNames()
            raw_outputs = self._net.forward(output_names)
        except Exception as error:
            raise ModelError(f"OpenCV YOLOX inference failed: {error}") from error
        after_inference = time.perf_counter_ns()

        if isinstance(raw_outputs, (tuple, list)):
            if len(raw_outputs) != 1:
                raise ModelError(f"expected one YOLOX output tensor, received {len(raw_outputs)}")
            raw_output = raw_outputs[0]
        else:
            raw_output = raw_outputs
        detections = decode_yolox(
            raw_output,
            metadata,
            person_class_id=self.spec.person_class_id,
            cat_class_id=self.spec.cat_class_id,
            person_confidence=self.person_confidence,
            cat_confidence=self.cat_confidence,
            nms_iou_threshold=self.nms_iou_threshold,
        )
        after_postprocess = time.perf_counter_ns()
        timings = DetectorTimings(
            preprocess_ms=(after_preprocess - start) / 1_000_000.0,
            inference_ms=(after_inference - after_preprocess) / 1_000_000.0,
            postprocess_ms=(after_postprocess - after_inference) / 1_000_000.0,
        )
        return detections, timings

    def detect(self, frame: npt.NDArray[np.uint8]) -> list[Detection]:
        detections, _ = self.detect_timed(frame)
        return detections
