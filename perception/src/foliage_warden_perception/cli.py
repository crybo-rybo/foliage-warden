"""Command-line entry point for image, video, and camera observation."""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from .dependencies import require_cv2
from .errors import PerceptionError
from .geometry import load_zones
from .pipeline import run_pipeline, stable_json
from .registry import (
    DEFAULT_MODEL_ID,
    default_registry_path,
    load_model_spec,
    resolve_and_verify_model,
)
from .sources import CameraSource, FrameSource, ImageSource, VideoSource
from .tracking import IouTracker
from .yolox import YOLOXDetector


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _device(value: str) -> int | str:
    return int(value) if re.fullmatch(r"-?\d+", value) else value


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry",
        type=Path,
        help="model registry (default: repository models/registry.json)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="registry model key")
    parser.add_argument(
        "--model",
        type=Path,
        help="override model artifact path (still checksum verified)",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        help="calibration export, scene JSON, or runtime config containing normalized zones",
    )
    parser.add_argument("--camera-id", default="camera-1")
    parser.add_argument("--cat-confidence", type=_probability, default=0.5)
    parser.add_argument("--person-confidence", type=_probability, default=0.5)
    parser.add_argument("--nms-iou", type=_probability, default=0.5)
    parser.add_argument("--tracker-iou", type=_probability, default=0.3)
    parser.add_argument("--max-missed-frames", type=_non_negative_int, default=5)
    parser.add_argument(
        "--backend-target",
        choices=("opencv", "cuda", "cuda-fp16"),
        default="opencv",
        help="OpenCV DNN backend/target pair",
    )
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument(
        "--output",
        default="-",
        help="JSONL destination; '-' writes stdout (default)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="write a stage timing summary to stderr after observations complete",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-perception",
        description=(
            "Observe-only cat/person detection, tracking, and zone evidence. "
            "It cannot classify behavior or issue an action."
        ),
    )
    subparsers = parser.add_subparsers(dest="source_command", required=True)

    image = subparsers.add_parser("image", help="process one still image")
    image.add_argument("input", type=Path)
    _add_common_arguments(image)

    video = subparsers.add_parser("video", help="process a video file")
    video.add_argument("input", type=Path)
    _add_common_arguments(video)

    camera = subparsers.add_parser("camera", help="observe a camera without display or recording")
    camera.add_argument("--device", type=_device, default=0, help="device index/path or pipeline")
    camera.add_argument("--width", type=_positive_int, default=1280)
    camera.add_argument("--height", type=_positive_int, default=720)
    camera.add_argument("--fps", type=float, default=30.0)
    camera.add_argument(
        "--gstreamer",
        action="store_true",
        help="interpret --device as an OpenCV GStreamer pipeline",
    )
    _add_common_arguments(camera)
    return parser


def _make_source(args: argparse.Namespace, cv2: object) -> FrameSource:
    if args.source_command == "image":
        return ImageSource(args.input, camera_id=args.camera_id, cv2_module=cv2)
    if args.source_command == "video":
        return VideoSource(args.input, camera_id=args.camera_id, cv2_module=cv2)
    if args.fps <= 0.0:
        raise PerceptionError("camera FPS must be positive")
    return CameraSource(
        args.device,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        gstreamer=args.gstreamer,
        cv2_module=cv2,
    )


@contextlib.contextmanager
def _output_stream(destination: str) -> Iterator[TextIO]:
    if destination == "-":
        yield sys.stdout
        return
    with Path(destination).open("w", encoding="utf-8") as output:
        yield output


def _run(args: argparse.Namespace) -> int:
    registry_path = args.registry if args.registry is not None else default_registry_path()
    spec = load_model_spec(registry_path, args.model_id)
    model_path = resolve_and_verify_model(registry_path, spec, args.model)
    zones = load_zones(args.zones) if args.zones is not None else ()
    cv2 = require_cv2()
    detector = YOLOXDetector(
        model_path,
        spec,
        person_confidence=args.person_confidence,
        cat_confidence=args.cat_confidence,
        nms_iou_threshold=args.nms_iou,
        backend_target=args.backend_target,
        cv2_module=cv2,
    )
    tracker = IouTracker(
        iou_threshold=args.tracker_iou,
        max_missed_frames=args.max_missed_frames,
    )
    source = _make_source(args, cv2)

    with _output_stream(args.output) as output:
        benchmark = run_pipeline(
            source,
            detector,
            tracker,
            output,
            zones=zones,
            max_frames=args.max_frames,
            model_id=spec.model_id,
            model_sha256=spec.sha256,
        )
    if args.benchmark:
        summary = {
            "backend_target": args.backend_target,
            "model_id": spec.model_id,
            "record_type": "perception_benchmark",
            "schema_version": 1,
            **benchmark.to_dict(),
        }
        sys.stderr.write(stable_json(summary) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (PerceptionError, OSError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2
