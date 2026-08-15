"""Small dependency-free types shared by the recorder adapters and core."""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


def pixel_byte_size(pixels: Any) -> int:
    """Return the decoded pixel payload size used by the recorder's memory limits."""

    declared = getattr(pixels, "nbytes", None)
    if declared is not None:
        if type(declared) is not int or declared < 0:
            raise ValueError("frame pixels nbytes must be a non-negative integer")
        return declared
    if isinstance(pixels, str):
        return len(pixels.encode("utf-8"))
    if isinstance(pixels, (bytes, bytearray)):
        return len(pixels)
    try:
        return memoryview(pixels).nbytes
    except TypeError:
        # Synthetic encoders use small scalar sentinels. They still need a finite,
        # conservative accounting value even though they are not image arrays.
        return sys.getsizeof(pixels)


@dataclass(frozen=True, slots=True)
class RecorderFrame:
    """One decoded BGR frame paired with a perception observation."""

    sequence: int
    captured_at_ms: int
    camera_id: str
    source_kind: str
    source_name: str
    pixels: Any

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("frame sequence must be a non-negative integer")
        if (
            isinstance(self.captured_at_ms, bool)
            or not isinstance(self.captured_at_ms, int)
            or self.captured_at_ms < 0
        ):
            raise ValueError("frame captured_at_ms must be a non-negative integer")
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("frame camera_id must not be empty")
        if self.source_kind not in {"camera", "video", "synthetic"}:
            raise ValueError("frame source_kind must be camera, video, or synthetic")
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("frame source_name must not be empty")

    @property
    def pixel_bytes(self) -> int:
        return pixel_byte_size(self.pixels)

    def owned_copy(self) -> RecorderFrame:
        """Copy pixel storage so capture backends may safely reuse their buffers."""

        return RecorderFrame(
            sequence=self.sequence,
            captured_at_ms=self.captured_at_ms,
            camera_id=self.camera_id,
            source_kind=self.source_kind,
            source_name=self.source_name,
            pixels=deepcopy(self.pixels),
        )


@dataclass(frozen=True, slots=True)
class RecorderConfig:
    """Privacy and resource bounds; every limit is explicit and finite."""

    pre_event_ms: int = 3_000
    post_event_ms: int = 3_000
    max_clip_ms: int = 15_000
    max_buffer_frames: int = 300
    nominal_fps: float = 30.0
    minimum_approach_overlap: float = 0.01
    max_incidents: int = 100
    max_disk_bytes: int = 5 * 1024 * 1024 * 1024
    max_buffer_bytes: int = 256 * 1024 * 1024
    max_active_frames: int = 600
    max_active_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("pre_event_ms", "post_event_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "max_clip_ms",
            "max_buffer_frames",
            "max_buffer_bytes",
            "max_active_frames",
            "max_active_bytes",
            "max_incidents",
            "max_disk_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.nominal_fps, bool)
            or not isinstance(self.nominal_fps, (int, float))
            or not math.isfinite(self.nominal_fps)
            or self.nominal_fps <= 0
        ):
            raise ValueError("nominal_fps must be finite and positive")
        if (
            isinstance(self.minimum_approach_overlap, bool)
            or not isinstance(self.minimum_approach_overlap, (int, float))
            or not math.isfinite(self.minimum_approach_overlap)
            or not 0.0 < self.minimum_approach_overlap <= 1.0
        ):
            raise ValueError("minimum_approach_overlap must be in (0, 1]")
