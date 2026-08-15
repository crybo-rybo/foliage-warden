"""Command-line entry point for deterministic scenario replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import SimulationError, run_scenario, stable_json, write_jsonl
from .resources import default_scenario_dir
from .validation import ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-sim",
        description="Run fail-closed Foliage Warden scenarios on a virtual clock.",
    )
    parser.add_argument("scenarios", nargs="*", type=Path, help="scenario JSON file(s)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every *.json fixture in --scenario-dir (the default when no file is given)",
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=default_scenario_dir(),
        help="fixture directory",
    )
    parser.add_argument(
        "--config", type=Path, help="override each scenario's config_ref"
    )
    parser.add_argument(
        "--schema-dir", type=Path, help="override the repository schema directory"
    )
    parser.add_argument(
        "--summary", type=Path, help="also write the aggregate summary JSON"
    )
    parser.add_argument(
        "--event-jsonl", type=Path, help="write canonical event-record JSONL"
    )
    parser.add_argument("--audit-jsonl", type=Path, help="write action-audit JSONL")
    parser.add_argument(
        "--evaluator-jsonl",
        type=Path,
        help="write replay records accepted directly by foliage-warden-eval",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="write a byte-stable deterministic trace for later conformance comparison",
    )
    parser.add_argument(
        "--compare-trace",
        type=Path,
        help="fail unless generated trace bytes exactly match this file",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="pretty-print stdout summary"
    )
    return parser


def _select_paths(args: argparse.Namespace) -> list[Path]:
    if args.scenarios and args.all:
        raise ContractError("pass scenario paths or --all, not both")
    if args.scenarios:
        return sorted(
            (path.resolve() for path in args.scenarios), key=lambda path: str(path)
        )
    paths = sorted(args.scenario_dir.resolve().glob("*.json"))
    if not paths:
        raise ContractError(f"no scenario JSON files found in {args.scenario_dir}")
    return paths


def _trace(results: list[object]) -> bytes:
    # Assertions/explanatory text are deliberately excluded. This is the policy trace:
    # state, reason, command, action outcome, and canonical records only.
    payload = {
        "schema_version": 1,
        "traces": [
            {
                "scenario_id": result.scenario_id,
                "signature_sha256": result.deterministic_signature(),
                "event_records": result.event_records,
                "audit_records": result.audit_records,
                "evaluator_records": result.evaluator_records,
            }
            for result in results
        ],
    }
    return (stable_json(payload) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = _select_paths(args)
        results = [
            run_scenario(path, config_path=args.config, schema_dir=args.schema_dir)
            for path in paths
        ]
    except (ContractError, SimulationError, OSError, ValueError) as error:
        print(f"simulation failed: {error}", file=sys.stderr)
        return 2

    aggregate = {
        "passed": all(result.passed for result in results),
        "scenario_count": len(results),
        "scenarios": [result.summary() for result in results],
        "schema_version": 1,
        "trace_sha256": stable_trace_digest(results),
    }
    compact = stable_json(aggregate)
    if args.pretty:
        import json

        print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(compact)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(compact + "\n", encoding="utf-8")
    if args.event_jsonl:
        write_jsonl(
            args.event_jsonl,
            (record for result in results for record in result.event_records),
        )
    if args.audit_jsonl:
        write_jsonl(
            args.audit_jsonl,
            (record for result in results for record in result.audit_records),
        )
    if args.evaluator_jsonl:
        write_jsonl(
            args.evaluator_jsonl,
            (record for result in results for record in result.evaluator_records),
        )

    trace = _trace(results)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_bytes(trace)
    trace_matches = True
    if args.compare_trace:
        try:
            trace_matches = args.compare_trace.read_bytes() == trace
        except OSError as error:
            print(f"trace comparison failed: {error}", file=sys.stderr)
            return 2
        if not trace_matches:
            print(
                f"trace mismatch: generated bytes differ from {args.compare_trace}",
                file=sys.stderr,
            )
    return 0 if aggregate["passed"] and trace_matches else 1


def stable_trace_digest(results: list[object]) -> str:
    from hashlib import sha256

    return sha256(_trace(results)).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
