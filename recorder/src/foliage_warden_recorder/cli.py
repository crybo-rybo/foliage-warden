"""Offline-only CLI for pairing an existing local video with perception JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import JsonlObservations, LocalVideoSource
from .core import IncidentRecorder
from .encoding import OpenCvAviEncoder
from .errors import RecorderError
from .runner import run_paired
from .storage import IncidentStore
from .types import RecorderConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-recorder",
        description="Create privacy-bounded observe-only clips from local video and JSONL",
    )
    parser.add_argument("video", type=Path, help="explicit local video path")
    parser.add_argument("--observations", type=Path, required=True, help="matching JSONL path")
    parser.add_argument("--output-dir", type=Path, required=True, help="explicit recorder root")
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--pre-event-ms", type=int, default=3_000)
    parser.add_argument("--post-event-ms", type=int, default=3_000)
    parser.add_argument("--max-clip-ms", type=int, default=15_000)
    parser.add_argument("--max-buffer-frames", type=int, default=300)
    parser.add_argument("--max-buffer-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-active-frames", type=int, default=600)
    parser.add_argument("--max-active-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--minimum-approach-overlap", type=float, default=0.01)
    parser.add_argument("--max-incidents", type=int, default=100)
    parser.add_argument("--max-disk-bytes", type=int, default=5 * 1024 * 1024 * 1024)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source: LocalVideoSource | None = None
    try:
        source = LocalVideoSource(
            args.video,
            camera_id=args.camera_id,
            fallback_fps=args.fallback_fps,
        )
        config = RecorderConfig(
            pre_event_ms=args.pre_event_ms,
            post_event_ms=args.post_event_ms,
            max_clip_ms=args.max_clip_ms,
            max_buffer_frames=args.max_buffer_frames,
            max_buffer_bytes=args.max_buffer_bytes,
            max_active_frames=args.max_active_frames,
            max_active_bytes=args.max_active_bytes,
            nominal_fps=source.fps,
            minimum_approach_overlap=args.minimum_approach_overlap,
            max_incidents=args.max_incidents,
            max_disk_bytes=args.max_disk_bytes,
        )
        store = IncidentStore(args.output_dir, config)
        recorder = IncidentRecorder(store, OpenCvAviEncoder(), config)
        with args.observations.open("r", encoding="utf-8") as stream:
            published = run_paired(source, JsonlObservations(stream), recorder)
    except (OSError, ValueError, RecorderError) as error:
        print(f"recorder error: {error}", file=sys.stderr)
        return 2
    finally:
        if source is not None:
            source.close()

    summary = {
        "incident_count": len(published),
        "mode": "OBSERVE_ONLY",
        "output_dir": str(args.output_dir),
        "record_type": "recorder_summary",
    }
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
