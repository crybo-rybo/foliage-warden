"""Revisioned, atomic annotation storage and deterministic JSONL export."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from .manifest import Manifest
from .validation import AnnotationError, validate_annotation


class RevisionConflict(RuntimeError):
    """Raised when a browser tries to save against stale annotation state."""


def _stable_json(value: Any, *, pretty: bool = False) -> str:
    options: dict[str, Any] = {"allow_nan": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sorted_annotations(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        values,
        key=lambda item: (
            item["session_id"],
            item["media_id"],
            item["start_ms"],
            item["end_ms"],
            item["event_id"],
        ),
    )


class AnnotationStore:
    def __init__(self, path: str | Path, manifest: Manifest):
        self.path = Path(path).resolve()
        self.manifest = manifest
        self._lock = threading.RLock()
        self._state = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "annotations": [],
            "history": [],
            "revision": 0,
            "schema_version": 1,
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationError(
                f"cannot read annotation store {self.path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise AnnotationError("annotation store must be a JSON object")
        required = {"annotations", "history", "revision", "schema_version"}
        if set(raw) != required:
            raise AnnotationError(
                "annotation store fields do not match schema version 1"
            )
        if raw["schema_version"] != 1:
            raise AnnotationError("annotation store schema_version must be 1")
        revision = raw["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise AnnotationError("annotation store revision must be an integer >= 0")
        if not isinstance(raw["annotations"], list) or not isinstance(
            raw["history"], list
        ):
            raise AnnotationError(
                "annotation store annotations and history must be arrays"
            )

        annotations = [
            validate_annotation(value, self.manifest) for value in raw["annotations"]
        ]
        ids = [value["event_id"] for value in annotations]
        if len(ids) != len(set(ids)):
            raise AnnotationError("annotation store contains duplicate event_id values")
        history: list[dict[str, Any]] = []
        for index, entry in enumerate(raw["history"]):
            if not isinstance(entry, dict) or set(entry) != {
                "annotation",
                "reason",
                "superseded_at_revision",
            }:
                raise AnnotationError(f"annotation store history[{index}] is invalid")
            if entry["reason"] not in {"archived", "updated"}:
                raise AnnotationError(
                    f"annotation store history[{index}].reason is invalid"
                )
            superseded = entry["superseded_at_revision"]
            if (
                isinstance(superseded, bool)
                or not isinstance(superseded, int)
                or superseded < 1
            ):
                raise AnnotationError(
                    f"annotation store history[{index}].superseded_at_revision is invalid"
                )
            history.append(
                {
                    "annotation": validate_annotation(
                        entry["annotation"], self.manifest
                    ),
                    "reason": entry["reason"],
                    "superseded_at_revision": superseded,
                }
            )
        return {
            "annotations": _sorted_annotations(annotations),
            "history": history,
            "revision": revision,
            "schema_version": 1,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(_stable_json(self._state))

    def _check_revision(self, expected_revision: Any) -> None:
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise AnnotationError("expected_revision must be an integer >= 0")
        if expected_revision != self._state["revision"]:
            raise RevisionConflict(
                f"annotation store changed: expected revision {expected_revision}, "
                f"current revision is {self._state['revision']}"
            )

    def _persist(self, state: dict[str, Any]) -> None:
        _atomic_write(self.path, _stable_json(state, pretty=True) + "\n")

    def upsert(self, raw: Any, expected_revision: Any) -> dict[str, Any]:
        annotation = validate_annotation(raw, self.manifest)
        with self._lock:
            self._check_revision(expected_revision)
            next_state = deepcopy(self._state)
            existing_index = next(
                (
                    index
                    for index, value in enumerate(next_state["annotations"])
                    if value["event_id"] == annotation["event_id"]
                ),
                None,
            )
            if (
                existing_index is not None
                and next_state["annotations"][existing_index] == annotation
            ):
                return self.snapshot()

            next_revision = next_state["revision"] + 1
            if existing_index is None:
                next_state["annotations"].append(annotation)
            else:
                previous = next_state["annotations"][existing_index]
                next_state["history"].append(
                    {
                        "annotation": previous,
                        "reason": "updated",
                        "superseded_at_revision": next_revision,
                    }
                )
                next_state["annotations"][existing_index] = annotation
            next_state["annotations"] = _sorted_annotations(next_state["annotations"])
            next_state["revision"] = next_revision
            self._persist(next_state)
            self._state = next_state
            return self.snapshot()

    def archive(self, event_id: Any, expected_revision: Any) -> dict[str, Any]:
        if not isinstance(event_id, str):
            raise AnnotationError("event_id must be a string")
        with self._lock:
            self._check_revision(expected_revision)
            next_state = deepcopy(self._state)
            existing_index = next(
                (
                    index
                    for index, value in enumerate(next_state["annotations"])
                    if value["event_id"] == event_id
                ),
                None,
            )
            if existing_index is None:
                raise AnnotationError(f"unknown event_id: {event_id}")
            next_revision = next_state["revision"] + 1
            previous = next_state["annotations"].pop(existing_index)
            next_state["history"].append(
                {
                    "annotation": previous,
                    "reason": "archived",
                    "superseded_at_revision": next_revision,
                }
            )
            next_state["revision"] = next_revision
            self._persist(next_state)
            self._state = next_state
            return self.snapshot()

    def export_records(self) -> list[dict[str, Any]]:
        with self._lock:
            records: list[dict[str, Any]] = []
            for annotation in _sorted_annotations(self._state["annotations"]):
                record: dict[str, Any] = {
                    "behavior": annotation["behavior"],
                    "end_ms": annotation["end_ms"],
                    "event_id": annotation["event_id"],
                    "metadata": {
                        "group_id": annotation["group_id"],
                        "media_id": annotation["media_id"],
                        "person_present": annotation["person_present"],
                        "privacy_restricted": annotation["privacy_restricted"],
                        "rationale": annotation["rationale"],
                    },
                    "record_type": "ground_truth_event",
                    "schema_version": 1,
                    "session_id": annotation["session_id"],
                    "staged_safe": annotation["staged_safe"],
                    "start_ms": annotation["start_ms"],
                }
                if annotation["zone_id"] is not None:
                    record["zone_id"] = annotation["zone_id"]
                records.append(record)
            return records

    def export_jsonl(self) -> str:
        return "".join(f"{_stable_json(record)}\n" for record in self.export_records())

    def write_export(self, path: str | Path) -> None:
        _atomic_write(Path(path).resolve(), self.export_jsonl())
