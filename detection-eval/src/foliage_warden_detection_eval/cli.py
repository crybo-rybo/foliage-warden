"""CLI for preparing and evaluating the deterministic public-data subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .constants import PUBLIC_DATA_WARNING
from .dataset import load_prepared_subset, prepare_subset
from .detector import run_pinned_detector
from .errors import DetectionEvalError
from .report import build_report, predictions_to_coco, require_exact_report, write_stable_json

DEFAULT_CACHE = Path("artifacts/detection-eval/coco2017")
DEFAULT_REPORT = Path("artifacts/detection-eval/coco-yolox-report.json")
DEFAULT_PREDICTIONS = Path("artifacts/detection-eval/coco-yolox-predictions.json")
DEFAULT_REGISTRY = Path("models/registry.json")
DEFAULT_MODEL_ID = "yolox_s_opencv_zoo"


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _probability(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return result


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--coco-root",
        type=Path,
        help="existing root containing annotations/instances_val2017.json and val2017/",
    )
    parser.add_argument("--max-images", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="require all annotation and selected image bytes to be present and verified",
    )


def _add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cat-confidence", type=_probability, default=0.001)
    parser.add_argument("--person-confidence", type=_probability, default=0.001)
    parser.add_argument("--cat-operating-threshold", type=_probability, default=0.5)
    parser.add_argument("--person-operating-threshold", type=_probability, default=0.5)
    parser.add_argument("--nms-iou", type=_probability, default=0.5)
    parser.add_argument(
        "--backend-target",
        choices=("opencv", "cuda", "cuda-fp16"),
        default="opencv",
    )
    parser.add_argument("--example-limit", type=_non_negative_int, default=10)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--expected-report",
        type=Path,
        help="fail unless the regenerated canonical report exactly matches this locked report",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-detection-eval",
        description=(
            "Reproducible cat/person detection baseline on a bounded COCO 2017 val subset. "
            "It does not evaluate behavior or physical-action safety."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="verify annotations and selected images")
    _add_prepare_arguments(prepare)

    evaluate = subparsers.add_parser("evaluate", help="evaluate an already prepared manifest")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    _add_detector_arguments(evaluate)

    run = subparsers.add_parser("run", help="prepare the subset and run the pinned detector")
    _add_prepare_arguments(run)
    _add_detector_arguments(run)
    return parser


def _prepare(args: argparse.Namespace):
    return prepare_subset(
        cache_dir=args.cache_dir,
        coco_root=args.coco_root,
        max_images=args.max_images,
        seed=args.seed,
        offline=args.offline,
        manifest_path=args.manifest,
    )


def _evaluate(args: argparse.Namespace, prepared) -> None:
    if args.cat_operating_threshold < args.cat_confidence:
        raise DetectionEvalError(
            "cat operating threshold cannot be below detector confidence floor"
        )
    if args.person_operating_threshold < args.person_confidence:
        raise DetectionEvalError(
            "person operating threshold cannot be below detector confidence floor"
        )
    predictions, model = run_pinned_detector(
        prepared,
        registry_path=args.registry,
        model_id=args.model_id,
        model_path=args.model,
        cat_confidence=args.cat_confidence,
        person_confidence=args.person_confidence,
        nms_iou=args.nms_iou,
        backend_target=args.backend_target,
    )
    report = build_report(
        prepared,
        predictions,
        model,
        iou_threshold=0.5,
        example_limit=args.example_limit,
        score_thresholds={
            "cat": args.cat_operating_threshold,
            "person": args.person_operating_threshold,
        },
    )
    if args.expected_report is not None:
        require_exact_report(args.expected_report, report)
    prediction_artifact = predictions_to_coco(predictions)
    write_stable_json(args.predictions, prediction_artifact)
    write_stable_json(args.report, report)
    print(
        json.dumps(
            {
                "manifest": str(prepared.manifest_path),
                "predictions": str(args.predictions),
                "report": str(args.report),
                "scope_warning": PUBLIC_DATA_WARNING,
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepared = _prepare(args)
            print(
                json.dumps(
                    {
                        "dataset_root": str(prepared.dataset_root),
                        "image_count": len(prepared.manifest["images"]),
                        "manifest": str(prepared.manifest_path),
                        "scope_warning": PUBLIC_DATA_WARNING,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "evaluate":
            prepared = load_prepared_subset(args.manifest, args.dataset_root)
        else:
            prepared = _prepare(args)
        _evaluate(args, prepared)
        return 0
    except (DetectionEvalError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())
