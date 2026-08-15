"""Atomic incident publication and path-confined retention."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .encoding import ClipEncoder
from .errors import StorageError
from .jsonio import strict_json_loads
from .types import RecorderConfig, RecorderFrame

_INCIDENT_NAME = re.compile(r"incident-[0-9]{13}-[0-9]{10}\Z")
_STAGING_NAME = re.compile(r"incident-[0-9]{13}-[0-9]{10}\.tmp-[A-Za-z0-9_-]+\Z")
_SAFE_SUFFIX = re.compile(r"\.[a-z0-9]{1,10}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_RECORD_CANONICALIZATION = "JSON_SORTED_KEYS_COMPACT_UTF8_V1"
_BINDING_STREAM_CANONICALIZATION = "JSONL_FRAME_BINDINGS_SORTED_KEYS_COMPACT_UTF8_V1"


@dataclass(frozen=True, slots=True)
class PublishedIncident:
    incident_id: str
    directory: Path
    clip_path: Path
    metadata_path: Path


class IncidentStore:
    """Publish clip/metadata pairs with a single atomic directory rename."""

    def __init__(self, output_dir: str | Path, config: RecorderConfig) -> None:
        raw_root = Path(output_dir)
        if raw_root.is_symlink():
            raise StorageError("output directory must not be a symbolic link")
        raw_root.mkdir(parents=True, exist_ok=True)
        self.root = raw_root.resolve(strict=True)
        if not self.root.is_dir():
            raise StorageError("output directory is not a directory")
        self._force_private_directory(self.root, "output directory")
        self.config = config
        self.incidents_dir = self.root / "incidents"
        self.staging_dir = self.root / ".staging"
        self._ensure_private_directory(self.incidents_dir)
        self._ensure_private_directory(self.staging_dir)
        self._recover_staging()
        self._published()

    def _verify_managed_directory(self, path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise StorageError(f"managed directory {path.name} must remain a real directory")
        resolved = path.resolve(strict=True)
        if resolved.parent != self.root or resolved.name != path.name:
            raise StorageError(f"managed directory {path.name} escaped output root")
        self._verify_private_directory(path, f"managed directory {path.name}")

    def _verify_layout(self) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise StorageError("output directory must remain a real directory")
        self._verify_private_directory(self.root, "output directory")
        self._verify_managed_directory(self.incidents_dir)
        self._verify_managed_directory(self.staging_dir)

    @staticmethod
    def _verify_private_directory(path: Path, label: str) -> None:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if mode & 0o077:
            raise StorageError(f"{label} permissions must not allow group or other access")

    @classmethod
    def _force_private_directory(cls, path: Path, label: str) -> None:
        try:
            os.chmod(path, 0o700, follow_symlinks=False)
        except OSError as error:
            raise StorageError(f"could not make {label} private: {error}") from error
        cls._verify_private_directory(path, label)

    def _ensure_private_directory(self, path: Path) -> None:
        if path.is_symlink():
            raise StorageError(f"managed directory {path.name} must not be a symbolic link")
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
        if not path.is_dir() or path.resolve(strict=True).parent != self.root:
            raise StorageError(f"managed directory {path.name} escaped output root")
        self._force_private_directory(path, f"managed directory {path.name}")

    def _inside(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(parent.resolve(strict=True))
        except ValueError:
            return False
        return True

    def _remove_managed(self, path: Path, *, staging: bool = False) -> None:
        pattern = _STAGING_NAME if staging else _INCIDENT_NAME
        parent = self.staging_dir if staging else self.incidents_dir
        if (
            path.parent != parent
            or not pattern.fullmatch(path.name)
            or not self._inside(path, parent)
        ):
            raise StorageError("refusing to remove a path outside managed recorder storage")
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def _recover_staging(self) -> None:
        self._verify_layout()
        for child in sorted(self.staging_dir.iterdir(), key=lambda path: path.name):
            if _STAGING_NAME.fullmatch(child.name):
                self._remove_managed(child, staging=True)

    def _published(self) -> list[Path]:
        self._verify_layout()
        result: list[Path] = []
        for child in self.incidents_dir.iterdir():
            if not _INCIDENT_NAME.fullmatch(child.name):
                continue
            self._validate_published_directory(child)
            result.append(child)
        return sorted(result, key=lambda path: path.name)

    def _validate_published_directory(self, path: Path) -> None:
        if path.is_symlink() or not path.is_dir() or not self._inside(path, self.incidents_dir):
            raise StorageError(f"managed-looking incident {path.name} is not a safe directory")
        self._verify_private_directory(path, f"managed-looking incident {path.name}")
        metadata_path = path / "metadata.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise StorageError(f"managed-looking incident {path.name} has no safe metadata")
        metadata_bytes = self._read_bounded_regular(
            metadata_path,
            max_bytes=_MAX_METADATA_BYTES,
            label=f"managed-looking incident {path.name} metadata",
        )
        try:
            metadata = strict_json_loads(metadata_bytes.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise StorageError(
                f"managed-looking incident {path.name} has invalid metadata"
            ) from error
        if not isinstance(metadata, dict) or not isinstance(metadata.get("clip"), dict):
            raise StorageError(f"managed-looking incident {path.name} has invalid metadata")
        clip = metadata["clip"]
        clip_name = clip.get("filename")
        expected_byte_size = clip.get("byte_size")
        expected_sha256 = clip.get("sha256")
        frame_count = clip.get("frame_count")
        privacy = metadata.get("privacy")
        if (
            metadata.get("record_type") != "observation_clip"
            or metadata.get("incident_id") != path.name
            or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 1
            or metadata.get("mode") != "OBSERVE_ONLY"
            or not isinstance(privacy, dict)
            or set(privacy) != {"audio", "display", "network"}
            or any(value is not False for value in privacy.values())
            or clip.get("audio") is not False
        ):
            raise StorageError(f"managed-looking incident {path.name} has mismatched metadata")
        if (
            not isinstance(clip_name, str)
            or not clip_name.startswith("clip.")
            or not _SAFE_SUFFIX.fullmatch(clip_name[4:])
        ):
            raise StorageError(f"managed-looking incident {path.name} has an unsafe clip name")
        clip_path = path / clip_name
        if clip_path.is_symlink() or not clip_path.is_file():
            raise StorageError(f"managed-looking incident {path.name} has no safe clip")
        if {item.name for item in path.iterdir()} != {"metadata.json", clip_name}:
            raise StorageError(f"managed-looking incident {path.name} contains unexpected files")
        if type(expected_byte_size) is not int or expected_byte_size <= 0:
            raise StorageError(f"managed-looking incident {path.name} has invalid clip byte_size")
        if not isinstance(expected_sha256, str) or not _SHA256_HEX.fullmatch(expected_sha256):
            raise StorageError(f"managed-looking incident {path.name} has invalid clip SHA-256")
        if type(frame_count) is not int or frame_count <= 0:
            raise StorageError(f"managed-looking incident {path.name} has invalid frame_count")
        self._validate_perception_provenance(
            path.name,
            metadata,
            frame_count,
            required=False,
        )
        actual_byte_size, actual_sha256 = self._file_identity(
            clip_path,
            label=f"managed-looking incident {path.name} clip",
        )
        if actual_byte_size != expected_byte_size:
            raise StorageError(f"managed-looking incident {path.name} clip byte_size mismatch")
        if actual_sha256 != expected_sha256:
            raise StorageError(f"managed-looking incident {path.name} clip SHA-256 mismatch")

    @staticmethod
    def _validate_perception_provenance(
        incident_name: str,
        metadata: dict[str, Any],
        frame_count: int,
        *,
        required: bool,
        frames: Sequence[RecorderFrame] | None = None,
    ) -> None:
        if "perception_provenance" not in metadata:
            if required:
                raise StorageError(
                    f"managed-looking incident {incident_name} has no perception provenance"
                )
            return
        provenance = metadata.get("perception_provenance")
        expected_provenance_keys = {
            "binding_stream_canonicalization",
            "frame_bindings",
            "record_canonicalization",
            "record_count",
            "stream_sha256",
        }
        if not isinstance(provenance, dict) or set(provenance) != expected_provenance_keys:
            raise StorageError(
                f"managed-looking incident {incident_name} has invalid perception provenance"
            )
        if provenance["record_canonicalization"] != _RECORD_CANONICALIZATION or (
            provenance["binding_stream_canonicalization"] != _BINDING_STREAM_CANONICALIZATION
        ):
            raise StorageError(
                f"managed-looking incident {incident_name} has unknown provenance canonicalization"
            )
        record_count = provenance["record_count"]
        bindings = provenance["frame_bindings"]
        if (
            type(record_count) is not int
            or record_count != frame_count
            or not isinstance(bindings, list)
            or len(bindings) != frame_count
        ):
            raise StorageError(
                f"managed-looking incident {incident_name} provenance count mismatch"
            )
        expected_binding_keys = {
            "captured_at_ms",
            "encoded_frame_index",
            "frame_id",
            "observation_id",
            "perception_record_sha256",
            "sequence",
        }
        digest = hashlib.sha256()
        previous_order: tuple[int, int] | None = None
        observation_ids: set[str] = set()
        frame_ids: set[str] = set()
        for encoded_frame_index, binding in enumerate(bindings):
            if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
                raise StorageError(
                    f"managed-looking incident {incident_name} has invalid frame binding"
                )
            sequence = binding["sequence"]
            captured_at_ms = binding["captured_at_ms"]
            if (
                type(binding["encoded_frame_index"]) is not int
                or binding["encoded_frame_index"] != encoded_frame_index
                or type(sequence) is not int
                or not 0 <= sequence <= _MAX_SAFE_INTEGER
                or type(captured_at_ms) is not int
                or not 0 <= captured_at_ms <= _MAX_SAFE_INTEGER
                or not isinstance(binding["observation_id"], str)
                or _IDENTIFIER.fullmatch(binding["observation_id"]) is None
                or not isinstance(binding["frame_id"], str)
                or _IDENTIFIER.fullmatch(binding["frame_id"]) is None
                or not isinstance(binding["perception_record_sha256"], str)
                or _SHA256_HEX.fullmatch(binding["perception_record_sha256"]) is None
            ):
                raise StorageError(
                    f"managed-looking incident {incident_name} has invalid frame binding"
                )
            observation_id = binding["observation_id"]
            frame_id = binding["frame_id"]
            if observation_id in observation_ids or frame_id in frame_ids:
                raise StorageError(
                    f"managed-looking incident {incident_name} has duplicate frame binding IDs"
                )
            observation_ids.add(observation_id)
            frame_ids.add(frame_id)
            if frames is not None:
                frame = frames[encoded_frame_index]
                if sequence != frame.sequence or captured_at_ms != frame.captured_at_ms:
                    raise StorageError(
                        f"managed-looking incident {incident_name} frame binding "
                        "disagrees with its supplied frame"
                    )
            order = (sequence, captured_at_ms)
            if previous_order is not None and (
                sequence <= previous_order[0] or captured_at_ms <= previous_order[1]
            ):
                raise StorageError(
                    f"managed-looking incident {incident_name} has unordered frame bindings"
                )
            previous_order = order
            digest.update(
                json.dumps(
                    binding,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            digest.update(b"\n")
        timeline = metadata.get("timeline")
        if not isinstance(timeline, dict) or (
            bindings[0]["sequence"] != timeline.get("start_sequence")
            or bindings[-1]["sequence"] != timeline.get("end_sequence")
            or bindings[0]["captured_at_ms"] != timeline.get("start_captured_at_ms")
            or bindings[-1]["captured_at_ms"] != timeline.get("end_captured_at_ms")
        ):
            raise StorageError(
                f"managed-looking incident {incident_name} provenance timeline mismatch"
            )
        stream_sha256 = provenance["stream_sha256"]
        if not isinstance(stream_sha256, str) or not _SHA256_HEX.fullmatch(stream_sha256):
            raise StorageError(
                f"managed-looking incident {incident_name} has invalid provenance SHA-256"
            )
        if digest.hexdigest() != stream_sha256:
            raise StorageError(
                f"managed-looking incident {incident_name} perception stream SHA-256 mismatch"
            )

    @staticmethod
    def _verify_private_regular(identity: os.stat_result, *, label: str) -> None:
        if not stat.S_ISREG(identity.st_mode):
            raise StorageError(f"{label} is not a regular file")
        if stat.S_IMODE(identity.st_mode) & 0o077:
            raise StorageError(f"{label} permissions must not allow group or other access")
        get_effective_uid = getattr(os, "geteuid", None)
        if get_effective_uid is not None and identity.st_uid != get_effective_uid():
            raise StorageError(f"{label} must be owned by the current user")

    @staticmethod
    def _read_bounded_regular(path: Path, *, max_bytes: int, label: str) -> bytes:
        descriptor: int | None = None
        result = bytearray()
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            identity = os.fstat(descriptor)
            IncidentStore._verify_private_regular(identity, label=label)
            if identity.st_size > max_bytes:
                raise StorageError(f"{label} exceeds the {max_bytes}-byte limit")
            while chunk := os.read(
                descriptor,
                min(_HASH_CHUNK_BYTES, max_bytes + 1 - len(result)),
            ):
                result.extend(chunk)
                if len(result) > max_bytes:
                    raise StorageError(f"{label} exceeds the {max_bytes}-byte limit")
        except OSError as error:
            raise StorageError(f"could not read {label}: {error}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return bytes(result)

    @staticmethod
    def _file_identity(path: Path, *, label: str) -> tuple[int, str]:
        digest = hashlib.sha256()
        byte_size = 0
        descriptor: int | None = None
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            IncidentStore._verify_private_regular(os.fstat(descriptor), label=label)
            while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
                byte_size += len(chunk)
                digest.update(chunk)
        except OSError as error:
            raise StorageError(f"could not read {label}: {error}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return byte_size, digest.hexdigest()

    @staticmethod
    def _tree_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.iterdir() if item.is_file())

    def _make_space(self, incoming_bytes: int) -> None:
        if incoming_bytes > self.config.max_disk_bytes:
            raise StorageError("encoded incident exceeds max_disk_bytes; nothing was published")
        published = self._published()
        total_bytes = sum(self._tree_size(path) for path in published)
        while published and (
            len(published) >= self.config.max_incidents
            or total_bytes + incoming_bytes > self.config.max_disk_bytes
        ):
            oldest = published.pop(0)
            size = self._tree_size(oldest)
            self._remove_managed(oldest)
            total_bytes -= size
        if len(published) >= self.config.max_incidents:
            raise StorageError("could not satisfy max_incidents retention")
        if total_bytes + incoming_bytes > self.config.max_disk_bytes:
            raise StorageError("could not satisfy max_disk_bytes retention")

    @staticmethod
    def _stable_json(value: dict[str, Any]) -> bytes:
        return (
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

    def publish(
        self,
        *,
        incident_id: str,
        frames: Sequence[RecorderFrame],
        metadata: dict[str, Any],
        encoder: ClipEncoder,
        fps: float,
    ) -> PublishedIncident:
        self._verify_layout()
        if not _INCIDENT_NAME.fullmatch(incident_id):
            raise StorageError("incident_id is not a recorder-generated identifier")
        if not frames:
            raise StorageError("cannot publish an incident without frames")
        privacy = metadata.get("privacy")
        if (
            metadata.get("record_type") != "observation_clip"
            or metadata.get("incident_id") != incident_id
            or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version") != 1
            or metadata.get("mode") != "OBSERVE_ONLY"
            or not isinstance(privacy, dict)
            or set(privacy) != {"audio", "display", "network"}
            or any(value is not False for value in privacy.values())
        ):
            raise StorageError("new incident has mismatched metadata")
        self._validate_perception_provenance(
            incident_id,
            metadata,
            len(frames),
            required=True,
            frames=frames,
        )
        if not _SAFE_SUFFIX.fullmatch(encoder.suffix):
            raise StorageError("encoder suffix must be a short lowercase file extension")
        destination = self.incidents_dir / incident_id
        if destination.exists() or destination.is_symlink():
            raise StorageError(f"incident {incident_id} already exists; refusing overwrite")

        stage = Path(tempfile.mkdtemp(prefix=f"{incident_id}.tmp-", dir=self.staging_dir))
        if not self._inside(stage, self.staging_dir):
            raise StorageError("temporary incident directory escaped output root")
        self._verify_private_directory(stage, "temporary incident directory")
        clip_name = f"clip{encoder.suffix}"
        clip_path = stage / clip_name
        metadata_path = stage / "metadata.json"
        try:
            encoding = encoder.encode(frames, clip_path, fps=fps)
            if not clip_path.is_file() or clip_path.is_symlink():
                raise StorageError("encoder did not produce a regular clip file")
            os.chmod(clip_path, 0o600)
            clip_bytes, clip_sha256 = self._file_identity(clip_path, label="encoded clip")
            if clip_bytes <= 0:
                raise StorageError("encoder produced an empty clip")
            complete_metadata = {
                **metadata,
                "clip": {
                    "audio": False,
                    "byte_size": clip_bytes,
                    "codec": encoding.codec,
                    "container": encoding.container,
                    "filename": clip_name,
                    "fps": encoding.fps,
                    "frame_count": len(frames),
                    "height": encoding.height,
                    "sha256": clip_sha256,
                    "width": encoding.width,
                },
            }
            metadata_bytes = self._stable_json(complete_metadata)
            if len(metadata_bytes) > _MAX_METADATA_BYTES:
                raise StorageError(
                    f"incident metadata exceeds the {_MAX_METADATA_BYTES}-byte limit"
                )
            metadata_path.write_bytes(metadata_bytes)
            os.chmod(metadata_path, 0o600)
            with clip_path.open("rb") as stream:
                os.fsync(stream.fileno())
            with metadata_path.open("rb") as stream:
                os.fsync(stream.fileno())
            incoming_bytes = clip_bytes + metadata_path.stat().st_size
            self._make_space(incoming_bytes)
            stage_fd = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            os.rename(stage, destination)
            self._verify_private_directory(destination, "published incident directory")
            directory_fd = os.open(self.incidents_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception as error:
            if stage.exists() or stage.is_symlink():
                self._remove_managed(stage, staging=True)
            if isinstance(error, StorageError):
                raise
            raise StorageError(f"failed to publish incident {incident_id}: {error}") from error

        return PublishedIncident(
            incident_id,
            destination,
            destination / clip_name,
            destination / "metadata.json",
        )
