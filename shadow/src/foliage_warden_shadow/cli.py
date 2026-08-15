"""CLI for strict JSONL fusion and deterministic mock-only replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from foliage_warden_sim.engine import SimulationError
from foliage_warden_sim.validation import ContractError as SimulatorContractError

from .contracts import (
    ContractError,
    parse_behavior_stream,
    parse_perception_stream,
    read_jsonl,
    stable_json,
)
from .fusion import FusionOptions
from .runner import execute_shadow, write_json, write_jsonl


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _bounded_ms(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 30_000:
        raise argparse.ArgumentTypeError("must be within [0, 30000]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-shadow",
        description=(
            "Fuse exact-match behavior predictions with observe-only perception, then run the "
            "existing deterministic simulator with its MOCK actuator."
        ),
    )
    parser.add_argument("perception_jsonl", type=Path)
    parser.add_argument("behavior_jsonl", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=_repository_root() / "config" / "simulation-safe.example.json",
    )
    parser.add_argument("--scenario-id")
    parser.add_argument("--scenario-out", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--evaluator-jsonl", type=Path)
    parser.add_argument("--event-jsonl", type=Path)
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument("--fusion-jsonl", type=Path)
    parser.add_argument("--prediction-timeout-ms", type=_bounded_ms, default=50)
    parser.add_argument("--max-prediction-latency-ms", type=_bounded_ms, default=250)
    parser.add_argument("--pretty", action="store_true")
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path}: config root must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        perceptions = read_jsonl(args.perception_jsonl, parse_perception_stream)
        predictions = read_jsonl(args.behavior_jsonl, parse_behavior_stream)
        config = _read_object(args.config)
        run = execute_shadow(
            perceptions,
            predictions,
            config,
            args.config,
            options=FusionOptions(
                prediction_timeout_ms=args.prediction_timeout_ms,
                max_prediction_latency_ms=args.max_prediction_latency_ms,
            ),
            scenario_id=args.scenario_id,
            scenario_out=args.scenario_out,
        )
        if args.summary is not None:
            write_json(args.summary, run.summary)
        if args.evaluator_jsonl is not None:
            write_jsonl(args.evaluator_jsonl, run.evaluator_records)
        if args.event_jsonl is not None:
            write_jsonl(args.event_jsonl, run.simulation.event_records)
        if args.audit_jsonl is not None:
            write_jsonl(args.audit_jsonl, run.simulation.audit_records)
        if args.fusion_jsonl is not None:
            write_jsonl(
                args.fusion_jsonl,
                (diagnostic.to_dict() for diagnostic in run.fusion.diagnostics),
            )
    except (
        ContractError,
        SimulatorContractError,
        SimulationError,
        OSError,
        ValueError,
    ) as error:
        print(f"shadow replay failed: {error}", file=sys.stderr)
        return 2

    if args.pretty:
        print(json.dumps(run.summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(stable_json(run.summary))
    return 0
