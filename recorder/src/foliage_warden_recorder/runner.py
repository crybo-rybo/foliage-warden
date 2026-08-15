"""Pair a frame source and observation stream without silently dropping either side."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import zip_longest
from typing import Any, Protocol

from .core import IncidentRecorder
from .errors import ObservationError
from .storage import PublishedIncident
from .types import RecorderFrame

_MISSING = object()


class ClosableFrameSource(Protocol):
    def __iter__(self) -> Iterable[RecorderFrame]: ...

    def close(self) -> None: ...


def run_paired(
    frames: ClosableFrameSource,
    observations: Iterable[dict[str, Any]],
    recorder: IncidentRecorder,
) -> list[PublishedIncident]:
    """Consume exact one-to-one pairs and abort on mismatch or malformed input."""

    published: list[PublishedIncident] = []
    try:
        for frame, observation in zip_longest(frames, observations, fillvalue=_MISSING):
            if frame is _MISSING:
                raise ObservationError("observation stream is longer than the frame source")
            if observation is _MISSING:
                raise ObservationError("frame source is longer than the observation stream")
            incident = recorder.process(frame, observation)
            if incident is not None:
                published.append(incident)
        final = recorder.close()
        if final is not None:
            published.append(final)
        return published
    except Exception:
        recorder.abort()
        raise
    finally:
        frames.close()
