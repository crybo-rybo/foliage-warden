"""CLI for deterministic offline temporal ONNX inference."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

from .contracts import (
    SHA256,
    ContractError,
    JsonObject,
    parse_perception_stream,
    read_jsonl,
    stable_json,
)
from .inference import infer_behavior_predictions, parse_inference_requests


def _bounded_ms(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 <= parsed <= 30_000:
        raise argparse.ArgumentTypeError("must be within [0, 30000]")
    return parsed


def _sha256(value: str) -> str:
    if SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-shadow-infer",
        description=(
            "Run a verified temporal ONNX model on pre-extracted NumPy RGB clips and emit "
            "strict observe-only behavior_prediction JSONL."
        ),
    )
    parser.add_argument("perception_jsonl", type=Path)
    parser.add_argument("request_jsonl", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expected-onnx-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--logical-latency-ms",
        type=_bounded_ms,
        required=True,
        help="Replay-only latency; every predicted_at_ms must equal capture plus this value.",
    )
    parser.add_argument(
        "--window-ms",
        type=_bounded_ms,
        required=True,
        help="Declared causal clip window ending at each perception capture timestamp.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomic mode-0600 JSONL destination. Omit to write only after success to stdout.",
    )
    return parser


def _write_private_jsonl(path: Path, records: Iterable[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(stable_json(record))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve(strict=False) == second.resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output is not None and args.output.suffix != ".jsonl":
            raise ContractError("output path must end in lowercase .jsonl")
        if args.output is not None and any(
            _same_path(args.output, source)
            for source in (
                args.perception_jsonl,
                args.request_jsonl,
                args.model,
                args.metadata,
            )
        ):
            raise ContractError("output must not overwrite an input, model, or metadata file")
        perceptions = read_jsonl(args.perception_jsonl, parse_perception_stream)
        requests = read_jsonl(args.request_jsonl, parse_inference_requests)
        predictions = infer_behavior_predictions(
            perceptions,
            requests,
            manifest_directory=args.request_jsonl.parent,
            model_path=args.model,
            metadata_path=args.metadata,
            expected_onnx_sha256=args.expected_onnx_sha256,
            logical_latency_ms=args.logical_latency_ms,
            window_ms=args.window_ms,
        )
        records = [prediction.to_dict() for prediction in predictions]
        if args.output is None:
            sys.stdout.write("".join(stable_json(record) + "\n" for record in records))
        else:
            _write_private_jsonl(args.output, records)
    except (ContractError, OSError, ValueError) as error:
        print(f"offline behavior inference failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
