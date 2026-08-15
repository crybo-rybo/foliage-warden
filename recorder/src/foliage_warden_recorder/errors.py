"""Recorder-specific failures with safe, user-facing messages."""


class RecorderError(RuntimeError):
    """Base class for recorder failures."""


class ObservationError(RecorderError):
    """A perception observation is malformed, unsafe, or out of order."""


class StorageError(RecorderError):
    """A clip could not be safely staged, published, or retained."""


class RecorderStateError(RecorderError):
    """The recorder was used after it was closed or aborted."""
