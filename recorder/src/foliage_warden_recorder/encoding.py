"""Clip encoders. OpenCV is imported lazily and never opens audio or a display."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import StorageError
from .types import RecorderFrame


@dataclass(frozen=True, slots=True)
class ClipEncoding:
    container: str
    codec: str
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if not self.container or not self.codec:
            raise ValueError("clip container and codec must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("clip dimensions must be positive")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("clip FPS must be finite and positive")


class ClipEncoder(Protocol):
    """Injected encoding boundary used by both OpenCV and synthetic tests."""

    @property
    def suffix(self) -> str: ...

    def encode(
        self,
        frames: Sequence[RecorderFrame],
        destination: Path,
        *,
        fps: float,
    ) -> ClipEncoding: ...


class OpenCvAviEncoder:
    """Encode silent MJPEG AVI without invoking GUI, network, or camera APIs."""

    suffix = ".avi"

    def __init__(self, *, codec: str = "MJPG", cv2_module: Any | None = None) -> None:
        if len(codec) != 4 or not codec.isascii():
            raise ValueError("OpenCV codec must be exactly four ASCII characters")
        self.codec = codec
        self._cv2 = cv2_module

    def _opencv(self) -> Any:
        if self._cv2 is None:
            try:
                self._cv2 = importlib.import_module("cv2")
            except ImportError as error:
                raise StorageError(
                    "OpenCV is unavailable; install the desktop extra or use JetPack's system cv2"
                ) from error
        return self._cv2

    def encode(
        self,
        frames: Sequence[RecorderFrame],
        destination: Path,
        *,
        fps: float,
    ) -> ClipEncoding:
        if not frames:
            raise StorageError("refusing to encode an empty clip")
        first_shape = getattr(frames[0].pixels, "shape", None)
        if not isinstance(first_shape, tuple) or len(first_shape) != 3 or first_shape[2] != 3:
            raise StorageError("OpenCV clip frames must be HxWx3 arrays")
        height, width = int(first_shape[0]), int(first_shape[1])
        if width <= 0 or height <= 0:
            raise StorageError("OpenCV clip frame dimensions must be positive")
        for frame in frames:
            if getattr(frame.pixels, "shape", None) != first_shape:
                raise StorageError("all frames in a clip must have identical dimensions")

        cv2 = self._opencv()
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*self.codec),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise StorageError(f"OpenCV could not open silent clip writer at {destination.name}")
        try:
            for frame in frames:
                writer.write(frame.pixels)
        except Exception as error:
            raise StorageError(f"OpenCV failed while encoding clip: {error}") from error
        finally:
            writer.release()
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise StorageError("OpenCV produced no clip bytes")
        return ClipEncoding("avi", self.codec, width, height, float(fps))
