"""Observe-only, privacy-bounded incident clip recording."""

from .core import IncidentRecorder
from .encoding import ClipEncoder, ClipEncoding, OpenCvAviEncoder
from .errors import ObservationError, RecorderError, RecorderStateError, StorageError
from .runner import run_paired
from .storage import IncidentStore, PublishedIncident
from .types import RecorderConfig, RecorderFrame

__all__ = [
    "ClipEncoder",
    "ClipEncoding",
    "IncidentRecorder",
    "IncidentStore",
    "ObservationError",
    "OpenCvAviEncoder",
    "PublishedIncident",
    "RecorderConfig",
    "RecorderError",
    "RecorderFrame",
    "RecorderStateError",
    "StorageError",
    "run_paired",
]
