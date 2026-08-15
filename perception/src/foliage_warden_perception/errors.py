"""User-facing errors raised by the observe-only perception package."""


class PerceptionError(RuntimeError):
    """Base class for expected configuration and runtime failures."""


class DependencyError(PerceptionError):
    """Raised when a required native or Python dependency is unavailable."""


class ModelError(PerceptionError):
    """Raised when the pinned detector artifact cannot be safely loaded."""


class SourceError(PerceptionError):
    """Raised when an image, video, or camera source cannot provide frames."""


class ZoneError(PerceptionError):
    """Raised when scene calibration geometry is invalid."""
