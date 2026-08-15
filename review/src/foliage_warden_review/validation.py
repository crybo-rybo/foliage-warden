"""Annotation validation against an already-validated manifest."""

from __future__ import annotations

from typing import Any

from .manifest import Manifest, ManifestError, validate_identifier

BEHAVIORS = {"PASSING", "SNIFFING", "EATING", "DIGGING", "OTHER", "UNKNOWN"}
ANNOTATION_FIELDS = {
    "behavior",
    "end_ms",
    "event_id",
    "group_id",
    "media_id",
    "person_present",
    "privacy_restricted",
    "rationale",
    "session_id",
    "staged_safe",
    "start_ms",
    "zone_id",
}


class AnnotationError(ValueError):
    """Raised when an annotation is incomplete or internally inconsistent."""


def _identifier(value: Any, context: str) -> str:
    try:
        return validate_identifier(value, context)
    except ManifestError as error:
        raise AnnotationError(str(error)) from error


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise AnnotationError(f"{context} must be a boolean")
    return value


def _milliseconds(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnnotationError(f"{context} must be an integer >= 0")
    return value


def validate_annotation(raw: Any, manifest: Manifest) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise AnnotationError("annotation must be a JSON object")
    missing = sorted(ANNOTATION_FIELDS - set(raw))
    if missing:
        raise AnnotationError(
            f"annotation missing required field(s): {', '.join(missing)}"
        )
    unknown = sorted(set(raw) - ANNOTATION_FIELDS)
    if unknown:
        raise AnnotationError(f"annotation has unknown field(s): {', '.join(unknown)}")

    event_id = _identifier(raw["event_id"], "annotation.event_id")
    session_id = _identifier(raw["session_id"], "annotation.session_id")
    group_id = _identifier(raw["group_id"], "annotation.group_id")
    media_id = _identifier(raw["media_id"], "annotation.media_id")
    media = manifest.media_by_key.get((session_id, media_id))
    if media is None:
        raise AnnotationError("annotation references media outside the manifest")
    if group_id != media.group_id:
        raise AnnotationError("annotation.group_id does not match the manifest session")

    behavior = raw["behavior"]
    if behavior not in BEHAVIORS:
        raise AnnotationError(
            f"annotation.behavior must be one of: {', '.join(sorted(BEHAVIORS))}"
        )
    start_ms = _milliseconds(raw["start_ms"], "annotation.start_ms")
    end_ms = _milliseconds(raw["end_ms"], "annotation.end_ms")
    if end_ms <= start_ms:
        raise AnnotationError("annotation.end_ms must be greater than start_ms")
    if end_ms > media.duration_ms:
        raise AnnotationError("annotation.end_ms exceeds the manifest duration")

    rationale = raw["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2000:
        raise AnnotationError(
            "annotation.rationale must be non-empty text of at most 2000 characters"
        )

    zone_id = raw["zone_id"]
    if zone_id is not None:
        zone_id = _identifier(zone_id, "annotation.zone_id")
    staged_safe = _boolean(raw["staged_safe"], "annotation.staged_safe")
    person_present = _boolean(raw["person_present"], "annotation.person_present")
    privacy_restricted = _boolean(
        raw["privacy_restricted"], "annotation.privacy_restricted"
    )
    if person_present and not privacy_restricted:
        raise AnnotationError(
            "annotation.privacy_restricted must be true whenever person_present is true"
        )

    return {
        "behavior": behavior,
        "end_ms": end_ms,
        "event_id": event_id,
        "group_id": group_id,
        "media_id": media_id,
        "person_present": person_present,
        "privacy_restricted": privacy_restricted,
        "rationale": rationale.strip(),
        "session_id": session_id,
        "staged_safe": staged_safe,
        "start_ms": start_ms,
        "zone_id": zone_id,
    }
