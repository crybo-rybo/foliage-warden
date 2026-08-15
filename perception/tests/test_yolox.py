from __future__ import annotations

import math

import numpy as np
import pytest

from foliage_warden_perception.errors import ModelError
from foliage_warden_perception.registry import ModelSpec
from foliage_warden_perception.types import ObjectClass
from foliage_warden_perception.yolox import (
    LetterboxMetadata,
    YOLOXDetector,
    class_aware_nms,
    decode_yolox,
    generate_yolox_grid,
    letterbox_bgr,
)


def test_letterbox_uses_rgb_top_left_padding_without_normalization() -> None:
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)

    blob, metadata = letterbox_bgr(
        frame,
        input_width=4,
        input_height=4,
        cv2_module=cv2,
    )

    assert blob.shape == (1, 3, 4, 4)
    assert blob.dtype == np.float32
    assert blob.flags.c_contiguous
    assert blob[0, :, 0, 0].tolist() == [30.0, 20.0, 10.0]
    assert np.all(blob[0, :, 2:, :] == 114.0)
    assert metadata.scale == 1.0
    assert metadata.resized_width == 4
    assert metadata.resized_height == 2
    assert metadata.pad_x == metadata.pad_y == 0


def test_generate_grid_is_row_major_and_supports_non_square_input() -> None:
    grid, strides = generate_yolox_grid(32, 16)

    assert grid.shape == (10, 2)
    assert strides.shape == (10, 1)
    assert grid[:5].tolist() == [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1]]
    assert strides[:8].reshape(-1).tolist() == [8.0] * 8
    assert grid[-1].tolist() == [1, 0]
    assert strides[-1, 0] == 16.0


def test_class_aware_nms_suppresses_only_same_class_and_breaks_ties_by_source_row() -> None:
    boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.8, 0.8, 0.7, 0.6], dtype=np.float32)
    classes = np.asarray([15, 15, 0, 15], dtype=np.int64)
    source_rows = np.asarray([9, 3, 8, 1], dtype=np.int64)

    keep = class_aware_nms(
        boxes,
        scores,
        classes,
        iou_threshold=0.5,
        prediction_indices=source_rows,
    )

    assert keep.tolist() == [1, 2, 3]


def _metadata() -> LetterboxMetadata:
    return LetterboxMetadata(
        original_width=32,
        original_height=32,
        input_width=32,
        input_height=32,
        resized_width=32,
        resized_height=32,
        scale=1.0,
    )


def _encode_box(
    output: np.ndarray,
    row: int,
    *,
    center: tuple[float, float],
    size: tuple[float, float],
    objectness: float,
    class_id: int,
    class_probability: float,
) -> None:
    grid, strides = generate_yolox_grid(32, 32)
    stride = float(strides[row, 0])
    output[0, row, :2] = np.asarray(center) / stride - grid[row]
    output[0, row, 2:4] = np.log(np.asarray(size) / stride)
    output[0, row, 4] = objectness
    output[0, row, 5 + class_id] = class_probability


def test_decode_filters_to_full_argmax_cat_person_and_keeps_cross_class_overlap() -> None:
    output = np.zeros((1, 21, 85), dtype=np.float32)
    _encode_box(
        output,
        0,
        center=(12, 12),
        size=(8, 8),
        objectness=1.0,
        class_id=15,
        class_probability=0.9,
    )
    _encode_box(
        output,
        1,
        center=(12, 12),
        size=(8, 8),
        objectness=1.0,
        class_id=15,
        class_probability=0.8,
    )
    _encode_box(
        output,
        2,
        center=(12, 12),
        size=(8, 8),
        objectness=1.0,
        class_id=0,
        class_probability=0.7,
    )
    _encode_box(
        output,
        3,
        center=(24, 24),
        size=(4, 4),
        objectness=1.0,
        class_id=16,
        class_probability=0.95,
    )
    output[0, 3, 5 + 15] = 0.8

    detections = decode_yolox(output, _metadata(), nms_iou_threshold=0.5)

    assert [detection.object_class for detection in detections] == [
        ObjectClass.CAT,
        ObjectClass.PERSON,
    ]
    assert [detection.prediction_index for detection in detections] == [0, 2]
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].bbox.x == pytest.approx(0.25)
    assert detections[0].bbox.y == pytest.approx(0.25)
    assert detections[0].bbox.width == pytest.approx(0.25)
    assert detections[0].bbox.height == pytest.approx(0.25)


def test_decode_multiplies_objectness_and_class_probability() -> None:
    output = np.zeros((1, 21, 85), dtype=np.float32)
    _encode_box(
        output,
        0,
        center=(12, 12),
        size=(8, 8),
        objectness=0.8,
        class_id=15,
        class_probability=0.75,
    )

    assert decode_yolox(output, _metadata(), cat_confidence=0.61) == []
    [detection] = decode_yolox(output, _metadata(), cat_confidence=0.6)
    assert detection.confidence == pytest.approx(0.6)


def test_decode_clips_unletterboxed_boxes_to_original_image() -> None:
    output = np.zeros((1, 21, 85), dtype=np.float32)
    _encode_box(
        output,
        0,
        center=(3, 3),
        size=(8, 8),
        objectness=1.0,
        class_id=0,
        class_probability=1.0,
    )
    metadata = LetterboxMetadata(16, 8, 32, 32, 32, 16, 2.0)

    [detection] = decode_yolox(output, metadata)

    assert detection.bbox.x == 0.0
    assert detection.bbox.y == 0.0
    assert detection.bbox.width == pytest.approx(7 / 32)
    assert detection.bbox.height == pytest.approx(7 / 16)


@pytest.mark.parametrize(
    "output",
    [np.zeros((21, 5), dtype=np.float32), np.zeros((1, 20, 85), dtype=np.float32)],
)
def test_decode_rejects_malformed_output(output: np.ndarray) -> None:
    with pytest.raises(ModelError, match="YOLOX output"):
        decode_yolox(output, _metadata())


def test_nms_validates_shapes_and_threshold() -> None:
    with pytest.raises(ValueError, match="shape"):
        class_aware_nms(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            iou_threshold=0.5,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        class_aware_nms(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.int64),
            iou_threshold=math.nan,
        )


class _FakeNet:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.backend: int | None = None
        self.target: int | None = None
        self.input: np.ndarray | None = None

    def setPreferableBackend(self, backend: int) -> None:
        self.backend = backend

    def setPreferableTarget(self, target: int) -> None:
        self.target = target

    def setInput(self, blob: np.ndarray) -> None:
        self.input = blob

    def getUnconnectedOutLayersNames(self) -> tuple[str]:
        return ("output",)

    def forward(self, names: tuple[str]) -> tuple[np.ndarray]:
        assert names == ("output",)
        return (self.output,)


def test_detector_adapter_configures_opencv_and_runs_synthetic_network() -> None:
    cv2 = pytest.importorskip("cv2")
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    output[0, 0, :2] = 0.5
    output[0, 0, 4] = 1.0
    output[0, 0, 5] = 1.0
    net = _FakeNet(output)
    spec = ModelSpec(
        model_id="synthetic",
        description="synthetic",
        filename="unused.onnx",
        input_width=640,
        input_height=640,
        input_color="RGB",
        person_class_id=0,
        cat_class_id=15,
        sha256="0" * 64,
        source_revision="synthetic",
        url="https://example.invalid",
    )
    detector = YOLOXDetector("unused.onnx", spec, cv2_module=cv2, net=net)

    detections, timings = detector.detect_timed(np.zeros((32, 32, 3), dtype=np.uint8))

    assert net.backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert net.target == cv2.dnn.DNN_TARGET_CPU
    assert net.input is not None and net.input.shape == (1, 3, 640, 640)
    assert len(detections) == 1
    assert detections[0].object_class is ObjectClass.PERSON
    assert timings.preprocess_ms >= 0.0
    assert timings.inference_ms >= 0.0
    assert timings.postprocess_ms >= 0.0
