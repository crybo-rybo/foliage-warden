"""Command-line interface for event-level replay evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .evaluator import EvaluationConfig, EvaluationInputError, evaluate
from .jsonl import read_jsonl, stable_json, write_json
from .matching import MatchConfig
from .schemas import (
    ActionRecord,
    Behavior,
    PredictionEvent,
    SchemaError,
    SessionRecord,
    parse_ground_truth,
    parse_replay_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-eval",
        description="Evaluate whole behavior incidents and simulated would-actions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a ground-truth/replay pair")
    evaluate_parser.add_argument("--ground-truth", required=True, type=Path)
    evaluate_parser.add_argument("--replay", required=True, type=Path)
    evaluate_parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    evaluate_parser.add_argument("--min-temporal-iou", type=float, default=0.1)
    evaluate_parser.add_argument("--max-onset-delta-ms", type=int)
    evaluate_parser.add_argument("--ignore-zone", action="store_true")
    evaluate_parser.add_argument(
        "--all-target-predictions",
        action="store_true",
        help="score all harmful-label events instead of only would-actions",
    )
    evaluate_parser.add_argument(
        "--target-behavior",
        action="append",
        choices=[behavior.value for behavior in Behavior],
        help="repeat to override the default EATING,DIGGING target set",
    )
    evaluate_parser.add_argument("--confidence", type=float, default=0.95)
    evaluate_parser.add_argument(
        "--fail-on-safety-violation",
        action="store_true",
        help="return exit status 3 when the report contains a safety violation",
    )
    return parser


def _run_evaluate(args: argparse.Namespace) -> int:
    truth = read_jsonl(args.ground_truth, parse_ground_truth)
    replay = read_jsonl(args.replay, parse_replay_record)
    sessions = [record for record in replay if isinstance(record, SessionRecord)]
    predictions = [record for record in replay if isinstance(record, PredictionEvent)]
    actions = [record for record in replay if isinstance(record, ActionRecord)]
    target_values = args.target_behavior or [Behavior.EATING.value, Behavior.DIGGING.value]
    config = EvaluationConfig(
        match=MatchConfig(
            min_temporal_iou=args.min_temporal_iou,
            max_onset_delta_ms=args.max_onset_delta_ms,
            require_zone_match=not args.ignore_zone,
        ),
        target_behaviors=tuple(Behavior(value) for value in target_values),
        would_action_only=not args.all_target_predictions,
        poisson_confidence=args.confidence,
    )
    report = evaluate(truth, predictions, sessions, actions, config)
    if args.output is None:
        sys.stdout.write(stable_json(report, pretty=True))
    else:
        write_json(args.output, report)
    if args.fail_on_safety_violation and not report["safety"]["passed"]:  # type: ignore[index]
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            return _run_evaluate(args)
    except (OSError, SchemaError, EvaluationInputError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

