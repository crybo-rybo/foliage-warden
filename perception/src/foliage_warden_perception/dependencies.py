"""Lazy dependency loading keeps missing native libraries understandable."""

from __future__ import annotations

import importlib
from typing import Any

from .errors import DependencyError


def require_cv2() -> Any:
    """Import OpenCV or raise an actionable error without hiding linker failures."""

    try:
        return importlib.import_module("cv2")
    except (ImportError, OSError) as error:
        raise DependencyError(
            "OpenCV (cv2) is required. On Jetson install/use the JetPack system "
            "python3-opencv package; for desktop development run "
            "`uv sync --project perception --extra desktop --group dev`. "
            f"Original import failure: {error}"
        ) from error
