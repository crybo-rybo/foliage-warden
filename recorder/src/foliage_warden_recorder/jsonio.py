"""Shared strict-JSON decoding for observation and stored metadata boundaries."""

from __future__ import annotations

import json
import math
from typing import Any

MAX_SAFE_INTEGER = 9_007_199_254_740_991


class StrictJsonError(ValueError):
    """A JSON token is ambiguous, non-interoperable, or too deeply nested."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise StrictJsonError(f"non-finite number {token}")
    return value


def _safe_integer(token: str) -> int:
    try:
        value = int(token)
    except ValueError as error:
        raise StrictJsonError("integer token exceeds the decoder's safe size") from error
    if not -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
        raise StrictJsonError(f"integer is outside the interoperable safe range: {token}")
    return value


def _reject_non_finite_constant(token: str) -> None:
    raise StrictJsonError(f"non-finite number {token}")


def _validate_utf8_strings(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as error:
                raise StrictJsonError("JSON string is not valid UTF-8") from error
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item)
            stack.extend(item.values())


def strict_json_loads(text: str) -> Any:
    """Decode one strict JSON value while rejecting ambiguous Python extensions."""

    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
            parse_float=_finite_float,
            parse_int=_safe_integer,
        )
    except RecursionError as error:
        raise StrictJsonError("JSON nesting exceeds the decoder limit") from error
    _validate_utf8_strings(value)
    return value
