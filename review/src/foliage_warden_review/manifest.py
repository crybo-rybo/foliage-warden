"""Strict manifest loading and media containment checks."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".m4v", ".mov", ".mp4", ".webm"}


class ManifestError(ValueError):
    """Raised when a review manifest is invalid or unsafe."""


def validate_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ManifestError(
            f"{context} must start with an alphanumeric character and contain only "
            "alphanumerics, '.', '_' or '-' (maximum 128 characters)"
        )
    return value


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be a JSON object")
    return value


def _strict_fields(
    value: dict[str, Any], required: set[str], optional: set[str], context: str
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ManifestError(
            f"{context} missing required field(s): {', '.join(missing)}"
        )
    unknown = sorted(set(value) - required - optional)
    if unknown:
        raise ManifestError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _optional_text(value: Any, context: str, *, maximum: int = 500) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ManifestError(
            f"{context} must be non-empty text of at most {maximum} characters"
        )
    return value.strip()


def _safe_media_path(root: Path, raw: Any, context: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ManifestError(f"{context} must be a non-empty relative path")
    if "\\" in raw or "\x00" in raw:
        raise ManifestError(f"{context} contains a forbidden path character")
    parts = raw.split("/")
    candidate = Path(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ManifestError(f"{context} must stay beneath the manifest directory")
    try:
        resolved = (root / candidate).resolve(strict=True)
    except OSError as error:
        raise ManifestError(
            f"{context} does not name a readable file: {raw}"
        ) from error
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ManifestError(f"{context} escapes the manifest directory") from error
    if not resolved.is_file():
        raise ManifestError(f"{context} must name a regular file")
    return resolved


@dataclass(frozen=True, slots=True)
class MediaItem:
    session_id: str
    group_id: str
    media_id: str
    kind: str
    duration_ms: int
    path: Path
    zone_id: str | None
    description: str | None
    token: str

    @property
    def mime_type(self) -> str:
        return mimetypes.guess_type(self.path.name)[0] or "application/octet-stream"

    def client_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "display_name": self.path.name,
            "duration_ms": self.duration_ms,
            "kind": self.kind,
            "media_id": self.media_id,
            "media_url": f"/media/{self.token}",
            "zone_id": self.zone_id,
        }


@dataclass(frozen=True, slots=True)
class Session:
    session_id: str
    group_id: str
    zone_id: str | None
    media: tuple[MediaItem, ...]

    def client_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "media": [item.client_dict() for item in self.media],
            "session_id": self.session_id,
            "zone_id": self.zone_id,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    path: Path
    sessions: tuple[Session, ...]
    media_by_key: dict[tuple[str, str], MediaItem]
    media_by_token: dict[str, MediaItem]

    def client_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sessions": [session.client_dict() for session in self.sessions],
        }


def _media_token(session_id: str, media_id: str) -> str:
    digest = hashlib.sha256(f"{session_id}\0{media_id}".encode()).hexdigest()
    return digest[:32]


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {manifest_path}: {error}") from error
    document = _mapping(raw, "manifest")
    _strict_fields(document, {"schema_version", "sessions"}, set(), "manifest")
    if document["schema_version"] != 1:
        raise ManifestError("manifest.schema_version must be 1")
    session_values = document["sessions"]
    if not isinstance(session_values, list) or not session_values:
        raise ManifestError("manifest.sessions must be a non-empty array")

    root = manifest_path.parent
    sessions: list[Session] = []
    media_by_key: dict[tuple[str, str], MediaItem] = {}
    media_by_token: dict[str, MediaItem] = {}
    seen_sessions: set[str] = set()

    for session_index, session_raw in enumerate(session_values):
        context = f"manifest.sessions[{session_index}]"
        session_data = _mapping(session_raw, context)
        _strict_fields(
            session_data,
            {"session_id", "group_id", "media"},
            {"zone_id"},
            context,
        )
        session_id = validate_identifier(
            session_data["session_id"], f"{context}.session_id"
        )
        group_id = validate_identifier(session_data["group_id"], f"{context}.group_id")
        if session_id in seen_sessions:
            raise ManifestError(f"duplicate session_id: {session_id}")
        seen_sessions.add(session_id)
        zone_id = session_data.get("zone_id")
        if zone_id is not None:
            zone_id = validate_identifier(zone_id, f"{context}.zone_id")
        media_values = session_data["media"]
        if not isinstance(media_values, list) or not media_values:
            raise ManifestError(f"{context}.media must be a non-empty array")

        session_media: list[MediaItem] = []
        seen_media: set[str] = set()
        for media_index, media_raw in enumerate(media_values):
            media_context = f"{context}.media[{media_index}]"
            media_data = _mapping(media_raw, media_context)
            _strict_fields(
                media_data,
                {"media_id", "path", "kind", "duration_ms"},
                {"description", "zone_id"},
                media_context,
            )
            media_id = validate_identifier(
                media_data["media_id"], f"{media_context}.media_id"
            )
            if media_id in seen_media:
                raise ManifestError(f"duplicate media_id in {session_id}: {media_id}")
            seen_media.add(media_id)
            kind = media_data["kind"]
            if kind not in {"image", "video"}:
                raise ManifestError(f"{media_context}.kind must be 'image' or 'video'")
            duration_ms = media_data["duration_ms"]
            if (
                isinstance(duration_ms, bool)
                or not isinstance(duration_ms, int)
                or duration_ms <= 0
            ):
                raise ManifestError(
                    f"{media_context}.duration_ms must be a positive integer"
                )
            resolved = _safe_media_path(
                root, media_data["path"], f"{media_context}.path"
            )
            allowed_suffixes = IMAGE_SUFFIXES if kind == "image" else VIDEO_SUFFIXES
            if resolved.suffix.lower() not in allowed_suffixes:
                raise ManifestError(
                    f"{media_context}.path extension does not match kind {kind!r}"
                )
            media_zone = media_data.get("zone_id", zone_id)
            if media_zone is not None:
                media_zone = validate_identifier(media_zone, f"{media_context}.zone_id")
            description = _optional_text(
                media_data.get("description"), f"{media_context}.description"
            )
            token = _media_token(session_id, media_id)
            item = MediaItem(
                session_id=session_id,
                group_id=group_id,
                media_id=media_id,
                kind=kind,
                duration_ms=duration_ms,
                path=resolved,
                zone_id=media_zone,
                description=description,
                token=token,
            )
            session_media.append(item)
            media_by_key[(session_id, media_id)] = item
            media_by_token[token] = item

        sessions.append(Session(session_id, group_id, zone_id, tuple(session_media)))

    return Manifest(
        path=manifest_path,
        sessions=tuple(sessions),
        media_by_key=media_by_key,
        media_by_token=media_by_token,
    )
