"""Strict JSON Lines I/O with stable, atomic output."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO, TypeVar

from .schemas import JsonValue, SchemaError

T = TypeVar("T")


def _objects(lines: Iterable[str], source: str) -> Iterator[tuple[int, Mapping[str, Any]]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SchemaError(
                f"{source}:{line_number}: invalid JSON: {error.msg} at column {error.colno}"
            ) from error
        if not isinstance(value, dict):
            raise SchemaError(f"{source}:{line_number}: each JSONL record must be an object")
        yield line_number, value


def read_jsonl(path: str | Path, parser: Callable[[Mapping[str, Any]], T]) -> list[T]:
    """Read and parse a JSONL file, attaching filename/line context to errors."""

    resolved = Path(path)
    records: list[T] = []
    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, value in _objects(stream, str(resolved)):
            try:
                records.append(parser(value))
            except SchemaError as error:
                raise SchemaError(f"{resolved}:{line_number}: {error}") from error
    return records


def read_jsonl_stream(stream: TextIO, parser: Callable[[Mapping[str, Any]], T], source: str = "<stream>") -> list[T]:
    records: list[T] = []
    for line_number, value in _objects(stream, source):
        try:
            records.append(parser(value))
        except SchemaError as error:
            raise SchemaError(f"{source}:{line_number}: {error}") from error
    return records


def stable_json(value: JsonValue | Mapping[str, Any], *, pretty: bool = False) -> str:
    """Serialize deterministically and reject NaN/Infinity."""

    if pretty:
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def write_jsonl(
    path: str | Path,
    records: Iterable[Any],
    *,
    serializer: Callable[[Any], Mapping[str, Any]] | None = None,
) -> None:
    """Atomically write records using stable key order.

    Records may be mappings or typed objects exposing ``to_dict``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def serialize(record: Any) -> Mapping[str, Any]:
        if serializer is not None:
            return serializer(record)
        if isinstance(record, Mapping):
            return record
        to_dict = getattr(record, "to_dict", None)
        if callable(to_dict):
            result = to_dict()
            if isinstance(result, Mapping):
                return result
        raise TypeError("record must be a mapping or expose to_dict()")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                stream.write(stable_json(serialize(record)))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: JsonValue | Mapping[str, Any], *, pretty: bool = True) -> None:
    """Atomically write one deterministic JSON document."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(stable_json(value, pretty=pretty))
            if not pretty:
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

