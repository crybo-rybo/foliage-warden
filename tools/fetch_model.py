#!/usr/bin/env python3
"""Fetch a pinned model artifact and verify it before making it visible."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "registry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(registry_path: Path, model_id: str) -> dict[str, object]:
    with registry_path.open(encoding="utf-8") as stream:
        registry = json.load(stream)
    try:
        model = registry["models"][model_id]
    except KeyError as exc:
        available = ", ".join(sorted(registry.get("models", {}))) or "none"
        raise ValueError(f"unknown model {model_id!r}; available: {available}") from exc
    return model


def fetch(model: dict[str, object], destination_dir: Path, *, force: bool) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / str(model["filename"])
    expected = str(model["sha256"])

    if destination.exists() and not force:
        actual = sha256(destination)
        if actual == expected:
            print(f"already verified: {destination}")
            return destination
        raise RuntimeError(
            f"refusing to overwrite {destination}: digest is {actual}, expected {expected}; "
            "pass --force to replace it"
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_dir, prefix=f".{destination.name}.", suffix=".part", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            request = urllib.request.Request(
                str(model["url"]), headers={"User-Agent": "foliage-warden-model-fetcher/1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    temp.write(chunk)

        actual = sha256(temp_path)
        if actual != expected:
            raise RuntimeError(f"download digest is {actual}, expected {expected}")
        os.replace(temp_path, destination)
        temp_path = None
        print(f"fetched and verified: {destination}")
        return destination
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="model identifier from models/registry.json")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--force", action="store_true", help="replace an existing artifact")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model = load_model(args.registry, args.model_id)
        fetch(model, args.destination, force=args.force)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

