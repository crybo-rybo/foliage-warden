"""Local-video and JSONL adapters; neither adapter accesses camera, audio, display, or network."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from .errors import ObservationError, RecorderError
from .jsonio import StrictJsonError, strict_json_loads
from .observation import MAX_OBSERVATION_JSON_BYTES, validate_json_value
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
    def __init__(
        self,
        stream: TextIO,
        *,
        max_record_bytes: int = MAX_OBSERVATION_JSON_BYTES,
    ) -> None:
        if type(max_record_bytes) is not int or max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be a positive integer")
        self._stream = stream
        self._max_record_bytes = max_record_bytes

    def __iter__(self) -> Iterator[dict[str, Any]]:
        line_number = 0
        # readline's bound is measured in decoded characters, not UTF-8 bytes.
        # Reading at most byte_limit + 3 characters still bounds temporary memory
        # while leaving room for CRLF and an over-limit sentinel character.
        character_limit = self._max_record_bytes + 3
        while line := self._stream.readline(character_limit):
            line_number += 1
            if not line.endswith("\n") and len(line) == character_limit:
                raise ObservationError(
                    f"JSONL record at line {line_number} exceeds "
                    f"{self._max_record_bytes} UTF-8 bytes"
                )
            payload = line.removesuffix("\n").removesuffix("\r")
            try:
                payload_bytes = payload.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ObservationError(
                    f"invalid JSON at line {line_number}: text is not valid UTF-8"
                ) from error
            if len(payload_bytes) > self._max_record_bytes:
                raise ObservationError(
                    f"JSONL record at line {line_number} exceeds "
                    f"{self._max_record_bytes} UTF-8 bytes"
                )
            if not payload.strip():
                raise ObservationError(f"blank JSONL record at line {line_number}")
            try:
                value = strict_json_loads(payload)
            except StrictJsonError as error:
                raise ObservationError(f"invalid JSON at line {line_number}: {error}") from error
            except ValueError as error:
                raise ObservationError(f"invalid JSON at line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ObservationError(f"JSONL line {line_number} must contain an object")
            validate_json_value(value)
            yield value
