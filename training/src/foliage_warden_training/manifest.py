from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .labels import BEHAVIOR_LABELS, validate_label

SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = frozenset({"clip_id", "path", "label", "split", "session_id", "day"})


class ManifestError(ValueError):
    """Raised when a manifest would produce an invalid or leaky experiment."""


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    path: Path
    source_path: str
    label: str
    split: str
    session_id: str
    day: str
    staged_safe: bool
    camera_id: str | None
    metadata: dict[str, Any]


def _required_string(row: dict[str, Any], field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location}: {field!r} must be a non-empty string")
    return value.strip()


def _parse_record(row: object, manifest_dir: Path, location: str) -> ClipRecord:
    if not isinstance(row, dict):
        raise ManifestError(f"{location}: expected a JSON object")

    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise ManifestError(f"{location}: missing required fields: {', '.join(sorted(missing))}")

    clip_id = _required_string(row, "clip_id", location)
    source_path = _required_string(row, "path", location)
    label = _required_string(row, "label", location)
    split = _required_string(row, "split", location)
    session_id = _required_string(row, "session_id", location)
    day_value = _required_string(row, "day", location)

    try:
        validate_label(label)
    except ValueError as error:
        raise ManifestError(f"{location}: {error}") from error
    if split not in SPLITS:
        raise ManifestError(f"{location}: split must be one of {SPLITS}, got {split!r}")
    try:
        date.fromisoformat(day_value)
    except ValueError as error:
        raise ManifestError(f"{location}: day must be an ISO date (YYYY-MM-DD)") from error

    staged_safe = row.get("staged_safe", False)
    if not isinstance(staged_safe, bool):
        raise ManifestError(f"{location}: staged_safe must be a boolean when provided")
    camera_id = row.get("camera_id")
    if camera_id is not None and (not isinstance(camera_id, str) or not camera_id.strip()):
        raise ManifestError(f"{location}: camera_id must be a non-empty string when provided")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ManifestError(f"{location}: metadata must be an object when provided")

    clip_path = Path(source_path)
    if not clip_path.is_absolute():
        clip_path = manifest_dir / clip_path
    return ClipRecord(
        clip_id=clip_id,
        path=clip_path.resolve(),
        source_path=source_path,
        label=label,
        split=split,
        session_id=session_id,
        day=day_value,
        staged_safe=staged_safe,
        camera_id=camera_id.strip() if isinstance(camera_id, str) else None,
        metadata=metadata,
    )


def _check_exclusive_group(records: Iterable[ClipRecord], field: str) -> None:
    split_by_value: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_by_value[getattr(record, field)].add(record.split)
    leaked = {value: splits for value, splits in split_by_value.items() if len(splits) > 1}
    if leaked:
        details = "; ".join(
            f"{value!r} appears in {sorted(splits)}" for value, splits in sorted(leaked.items())
        )
        raise ManifestError(f"split leakage through {field}: {details}")


def validate_records(records: list[ClipRecord], *, require_files: bool = True) -> None:
    if not records:
        raise ManifestError("manifest contains no clips")

    for field in ("clip_id", "path"):
        values = [str(getattr(record, field)) for record in records]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ManifestError(f"duplicate {field}: {', '.join(duplicates)}")

    _check_exclusive_group(records, "session_id")
    _check_exclusive_group(records, "day")

    if require_files:
        missing = [str(record.path) for record in records if not record.path.is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise ManifestError(f"missing clip files: {preview}{suffix}")


def load_manifest(path: str | Path, *, require_files: bool = True) -> list[ClipRecord]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"manifest does not exist: {manifest_path}")

    records: list[ClipRecord] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{manifest_path}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestError(f"{location}: invalid JSON: {error.msg}") from error
            records.append(_parse_record(row, manifest_path.parent, location))
    validate_records(records, require_files=require_files)
    return records


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_records(records: Iterable[ClipRecord]) -> dict[str, Any]:
    records = list(records)
    by_split = Counter(record.split for record in records)
    by_label = Counter(record.label for record in records)
    by_split_label: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = Counter(record.label for record in records if record.split == split)
        by_split_label[split] = {label: counts[label] for label in BEHAVIOR_LABELS}
    return {
        "clips": len(records),
        "sessions": len({record.session_id for record in records}),
        "days": len({record.day for record in records}),
        "staged_safe_clips": sum(record.staged_safe for record in records),
        "by_split": {split: by_split[split] for split in SPLITS},
        "by_label": {label: by_label[label] for label in BEHAVIOR_LABELS},
        "by_split_label": by_split_label,
    }
