"""Command line interface for strict offline incident assembly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foliage_warden_shadow.contracts import stable_json

from .core import MAX_LOGICAL_LATENCY_MS, assemble_incident
from .errors import AssemblyError


def _bounded_ms(value: str, *, maximum: int, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from error
    if not 0 <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{label} must be within [0, {maximum}]")
    return parsed


def _window_ms(value: str) -> int:
    return _bounded_ms(value, maximum=30_000, label="window_ms")


def _latency_ms(value: str) -> int:
    return _bounded_ms(
        value,
        maximum=MAX_LOGICAL_LATENCY_MS,
        label="logical_latency_ms",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-assemble",
        description=(
            "Offline-only assembly of one verified recorder incident and its original strict "
            "perception JSONL into full-frame RGB NumPy clips and shadow inference requests."
        ),
    )
    parser.add_argument("incident_directory", type=Path)
    parser.add_argument("perception_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--window-ms",
        type=_window_ms,
        required=True,
        help="Exact causal window; a real perception frame must exist at target minus this value.",
    )
    parser.add_argument(
        "--logical-latency-ms",
        type=_latency_ms,
        default=10,
        help=(
            "Replay-only prediction latency, capped at shadow's default 50 ms timeout "
            "(default: 10)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = assemble_incident(
            args.incident_directory,
            args.perception_jsonl,
            args.output_dir,
            window_ms=args.window_ms,
            logical_latency_ms=args.logical_latency_ms,
        )
    except (AssemblyError, OSError, ValueError) as error:
        print(f"offline incident assembly failed: {error}", file=sys.stderr)
        return 2
    summary = {
        "mode": "OBSERVE_ONLY",
        "output_directory": str(result.output_directory),
        "request_count": result.request_count,
        "skipped_target_count": result.skipped_target_count,
        "would_action": False,
    }
    print(stable_json(summary))
    return 0


def entrypoint() -> int:
    return main()
