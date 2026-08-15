"""Dependency-light command bootstrap for actionable import failures."""

from __future__ import annotations

import sys


def entrypoint() -> None:
    try:
        from .cli import main
    except (ImportError, OSError) as error:
        missing = getattr(error, "name", None)
        if missing == "numpy" or "numpy" in str(error).lower():
            print(
                "error: NumPy is required. Install the package dependencies, or on Jetson "
                "install python3-numpy before using this source tree via PYTHONPATH.",
                file=sys.stderr,
            )
            raise SystemExit(2) from error
        raise
    raise SystemExit(main())
