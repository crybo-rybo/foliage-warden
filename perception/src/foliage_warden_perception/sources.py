"""Image, video, and camera input adapters with no implicit display or recording."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .dependencies import require_cv2
from .errors import SourceError


@dataclass(frozen=True, slots=True)
class Frame:
    index: int
    captured_at_ms: int
    camera_id: str
    source_kind: str
    source_name: str
    bgr: npt.NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.index < 0 or self.captured_at_ms < 0:
            raise ValueError("frame index and captured_at_ms must be non-negative")
        if not self.camera_id:
            raise ValueError("camera_id must not be empty")
        if self.bgr.ndim != 3 or self.bgr.shape[2] != 3:
            raise ValueError("frame pixels must be an HxWx3 BGR array")


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class ImageSource:
    def __init__(
        self,
        path: str | Path,
        *,
        camera_id: str = "camera-1",
        cv2_module: Any | None = None,
    ) -> None:
        self.path = Path(path)
        self.camera_id = camera_id
        self._cv2 = cv2_module if cv2_module is not None else require_cv2()

    def __iter__(self) -> Iterator[Frame]:
        if not self.path.is_file():
            raise SourceError(f"image file not found at {self.path}")
        image = self._cv2.imread(str(self.path), self._cv2.IMREAD_COLOR)
        if image is None:
            raise SourceError(f"OpenCV could not decode image {self.path}")
        yield Frame(0, 0, self.camera_id, "image", self.path.name, image)

    def close(self) -> None:
        return None


class _CaptureSource:
    def __init__(
        self,
        capture_input: int | str,
        *,
        camera_id: str,
        source_kind: str,
        source_name: str,
        nominal_fps: float | None,
        width: int | None = None,
        height: int | None = None,
        gstreamer: bool = False,
        cv2_module: Any | None = None,
    ) -> None:
        self._cv2 = cv2_module if cv2_module is not None else require_cv2()
        self.camera_id = camera_id
        self.source_kind = source_kind
        self.source_name = source_name
        try:
            if gstreamer:
                self._capture = self._cv2.VideoCapture(capture_input, self._cv2.CAP_GSTREAMER)
            else:
                self._capture = self._cv2.VideoCapture(capture_input)
        except Exception as error:
            raise SourceError(f"OpenCV could not create {source_kind} capture: {error}") from error
        if not self._capture.isOpened():
            self._capture.release()
            raise SourceError(f"OpenCV could not open {source_kind} source {source_name!r}")

        if source_kind == "camera":
            if width is not None:
                self._capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height is not None:
                self._capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            if nominal_fps is not None:
                self._capture.set(self._cv2.CAP_PROP_FPS, float(nominal_fps))

        reported_fps = float(self._capture.get(self._cv2.CAP_PROP_FPS))
        if nominal_fps is not None:
            self._fps = nominal_fps
        elif math.isfinite(reported_fps) and reported_fps > 0.0:
            self._fps = reported_fps
        else:
            self._fps = 30.0
        if not math.isfinite(self._fps) or self._fps <= 0.0:
            self.close()
            raise SourceError("nominal capture FPS must be finite and positive")

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if index == 0 or self.source_kind == "camera":
                    raise SourceError(
                        f"{self.source_kind} source {self.source_name!r} did not provide "
                        f"frame {index}"
                    )
                break
            if not isinstance(frame, np.ndarray):
                raise SourceError("OpenCV capture returned a non-array frame")
            timestamp_ms = round(index * 1000.0 / self._fps)
            yield Frame(
                index=index,
                captured_at_ms=timestamp_ms,
                camera_id=self.camera_id,
                source_kind=self.source_kind,
                source_name=self.source_name,
                bgr=frame,
            )
            index += 1

    def close(self) -> None:
        self._capture.release()


class VideoSource(_CaptureSource):
    def __init__(
        self,
        path: str | Path,
        *,
        camera_id: str = "camera-1",
        cv2_module: Any | None = None,
    ) -> None:
        video_path = Path(path)
        if not video_path.is_file():
            raise SourceError(f"video file not found at {video_path}")
        super().__init__(
            str(video_path),
            camera_id=camera_id,
            source_kind="video",
            source_name=video_path.name,
            nominal_fps=None,
            cv2_module=cv2_module,
        )


class CameraSource(_CaptureSource):
    def __init__(
        self,
        device: int | str = 0,
        *,
        camera_id: str = "camera-1",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        gstreamer: bool = False,
        cv2_module: Any | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise SourceError("camera width and height must be positive")
        super().__init__(
            device,
            camera_id=camera_id,
            source_kind="camera",
            source_name=str(device),
            nominal_fps=fps,
            width=width,
            height=height,
            gstreamer=gstreamer,
            cv2_module=cv2_module,
        )
