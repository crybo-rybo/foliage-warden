"""Local-video and JSONL adapters; neither adapter accesses camera, audio, display, or network."""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from .errors import ObservationError, RecorderError
from .types import RecorderFrame


class LocalVideoSource:
    """Decode an explicit local video path into camera-like recorder frames."""

    def __init__(
        self,
        path: str | Path,
        *,
        camera_id: str = "camera-1",
        fallback_fps: float = 30.0,
        cv2_module: Any | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise RecorderError(f"local video file not found at {self.path}")
        if self.path.is_symlink():
            raise RecorderError("local video path must not be a symbolic link")
        if not math.isfinite(fallback_fps) or fallback_fps <= 0:
            raise ValueError("fallback_fps must be finite and positive")
        if cv2_module is None:
            try:
                cv2_module = importlib.import_module("cv2")
            except ImportError as error:
                raise RecorderError(
                    "OpenCV is unavailable; install the desktop extra or use JetPack's system cv2"
                ) from error
        self._cv2 = cv2_module
        self._capture = cv2_module.VideoCapture(str(self.path.resolve(strict=True)))
        if not self._capture.isOpened():
            self._capture.release()
            raise RecorderError(f"OpenCV could not open local video {self.path}")
        reported = float(self._capture.get(cv2_module.CAP_PROP_FPS))
        self.fps = reported if math.isfinite(reported) and reported > 0 else fallback_fps
        self.camera_id = camera_id
        self.closed = False

    def __iter__(self) -> Iterator[RecorderFrame]:
        index = 0
        while True:
            ok, pixels = self._capture.read()
            if not ok or pixels is None:
                break
            yield RecorderFrame(
                sequence=index,
                captured_at_ms=round(index * 1000.0 / self.fps),
                camera_id=self.camera_id,
                source_kind="video",
                source_name=self.path.name,
                pixels=pixels,
            )
            index += 1

    def close(self) -> None:
        if not self.closed:
            self._capture.release()
            self.closed = True


class JsonlObservations:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for line_number, line in enumerate(self._stream, start=1):
            if not line.strip():
                raise ObservationError(f"blank JSONL record at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ObservationError(
                    f"invalid JSON at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ObservationError(f"JSONL line {line_number} must contain an object")
            yield value
