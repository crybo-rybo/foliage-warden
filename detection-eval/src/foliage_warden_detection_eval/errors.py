"""Domain errors with actionable CLI messages."""


class DetectionEvalError(Exception):
    """Expected data, cache, or evaluation failure."""


class OfflineCacheError(DetectionEvalError):
    """Required verified bytes were unavailable in offline mode."""
