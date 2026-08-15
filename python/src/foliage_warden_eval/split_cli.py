"""CLI for leakage-safe dataset manifest splitting."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .jsonl import read_jsonl, stable_json, write_json
from .schemas import DatasetItem, SchemaError
from .splitting import split_by_session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foliage-warden-split",
        description="Split a dataset without leaking sessions or groups.",
    )
    parser.add_argument("manifest", type=Path, help="JSONL of dataset-item objects")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        items = read_jsonl(args.manifest, DatasetItem.from_dict)
        result = split_by_session(
            items,
            ratios={"train": args.train, "validation": args.validation, "test": args.test},
            seed=args.seed,
        ).to_dict()
        if args.output is None:
            sys.stdout.write(stable_json(result, pretty=True))
        else:
            write_json(args.output, result)
        return 0
    except (OSError, SchemaError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
