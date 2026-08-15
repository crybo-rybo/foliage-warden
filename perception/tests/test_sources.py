from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foliage_warden_perception.errors import SourceError
from foliage_warden_perception.sources import CameraSource, ImageSource, VideoSource


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, opened: bool = True, fps: float = 25.0) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.fps = fps
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def get(self, property_id: int) -> float:
        return self.fps

    def set(self, property_id: int, value: float) -> bool:
        self.settings.append((property_id, value))
        return True

    def release(self) -> None:
        self.released = True


class FakeCv2:
    IMREAD_COLOR = 1
    CAP_GSTREAMER = 1800
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5

    def __init__(self, capture: FakeCapture | None = None, image: np.ndarray | None = None) -> None:
        self.capture = capture
        self.image = image
        self.capture_args: tuple[object, ...] | None = None

    def VideoCapture(self, *args: object) -> FakeCapture:
        self.capture_args = args
        assert self.capture is not None
        return self.capture

    def imread(self, path: str, mode: int) -> np.ndarray | None:
        return self.image


def _frame() -> np.ndarray:
    return np.zeros((2, 3, 3), dtype=np.uint8)


def test_image_source_emits_one_zero_time_frame(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"synthetic placeholder")
    source = ImageSource(path, cv2_module=FakeCv2(image=_frame()))

    [frame] = list(source)

    assert frame.index == 0
    assert frame.captured_at_ms == 0
    assert frame.source_kind == "image"
    assert frame.source_name == "frame.jpg"


def test_video_source_uses_nominal_media_timeline_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"synthetic placeholder")
    capture = FakeCapture([_frame(), _frame()], fps=25.0)
    source = VideoSource(path, cv2_module=FakeCv2(capture=capture))

    frames = list(source)
    source.close()

    assert [frame.captured_at_ms for frame in frames] == [0, 40]
    assert capture.released


def test_camera_applies_requested_properties_and_disconnect_is_an_error() -> None:
    capture = FakeCapture([_frame()])
    cv2 = FakeCv2(capture=capture)
    source = CameraSource(
        "/dev/video-test",
        width=640,
        height=480,
        fps=20.0,
        gstreamer=True,
        cv2_module=cv2,
    )
    iterator = iter(source)

    first = next(iterator)
    assert first.captured_at_ms == 0
    with pytest.raises(SourceError, match="did not provide frame 1"):
        next(iterator)
    assert cv2.capture_args == ("/dev/video-test", cv2.CAP_GSTREAMER)
    assert capture.settings == [
        (cv2.CAP_PROP_FRAME_WIDTH, 640.0),
        (cv2.CAP_PROP_FRAME_HEIGHT, 480.0),
        (cv2.CAP_PROP_FPS, 20.0),
    ]
    source.close()


def test_capture_open_failure_is_clear() -> None:
    capture = FakeCapture([], opened=False)
    with pytest.raises(SourceError, match="could not open camera"):
        CameraSource(0, cv2_module=FakeCv2(capture=capture))
    assert capture.released
