"""Strict, offline assembly of recorder incidents into shadow inference inputs.

The adapter reconstructs structural ordinal alignment and, when present, verifies
recorder-bound canonical perception bytes. Neither path authenticates that the
named camera produced a particular exposure or that detections describe its pixels.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from foliage_warden_shadow.contracts import (
    IDENTIFIER,
    MAX_SAFE_INTEGER,
    SHA256,
    ContractError,
    PerceptionObservation,
    parse_perception_stream,
    stable_json,
)
from foliage_warden_shadow.inference import CLIP_FORMAT, parse_inference_requests

from .errors import AssemblyError

SCHEMA_VERSION = 1
ASSEMBLER_VERSION = "0.1.0"
PROVENANCE_RECORD_TYPE = "recorder_shadow_assembly_provenance"
REQUESTS_FILENAME = "behavior-inference-requests.jsonl"
PERCEPTIONS_FILENAME = "selected-perceptions.jsonl"
INCIDENT_PERCEPTIONS_FILENAME = "incident-perceptions.jsonl"
PROVENANCE_FILENAME = "provenance.json"
MAX_METADATA_BYTES = 1024 * 1024
MAX_PERCEPTION_BYTES = 64 * 1024 * 1024
MAX_PERCEPTION_RECORDS = 100_000
MAX_SOURCE_CLIP_BYTES = 1024 * 1024 * 1024
MAX_DECODED_INCIDENT_BYTES = 512 * 1024 * 1024
MAX_INCIDENT_FRAMES = 2_000
MAX_SHADOW_CLIP_BYTES = 256 * 1024 * 1024
MAX_ASSEMBLED_CLIP_BYTES = 1024 * 1024 * 1024
MAX_ASSEMBLED_TARGETS = 1_000
MAX_SELECTED_FRAME_ENTRIES = 1_000_000
MAX_LOGICAL_LATENCY_MS = 50
_HASH_CHUNK_BYTES = 1024 * 1024
_INCIDENT_NAME = re.compile(r"incident-[0-9]{13}-[0-9]{10}\Z")
_CLIP_NAME = re.compile(r"clip\.[a-z0-9]{1,10}\Z")
_TERMINATIONS = {
    "max_active_bytes",
    "max_active_frames",
    "max_clip_duration",
    "post_event_elapsed",
    "source_end",
}
_METADATA_FIELDS = {
    "clip",
    "incident_id",
    "mode",
    "privacy",
    "record_type",
    "resource_limits",
    "schema_version",
    "source",
    "termination",
    "timeline",
    "trigger",
}
_RECORD_CANONICALIZATION = "JSON_SORTED_KEYS_COMPACT_UTF8_V1"
_BINDING_STREAM_CANONICALIZATION = "JSONL_FRAME_BINDINGS_SORTED_KEYS_COMPACT_UTF8_V1"
_STRUCTURAL_WARNING = (
    "Ordinal, identity, timeline, and digest checks structurally re-establish this mapping; "
    "they do not authenticate that the encoded pixels and perception records originated "
    "from the named camera exposure."
)
_FULL_FRAME_WARNING = (
    "Every emitted array is decoded full-frame RGB; no track crop, crop correctness, or "
    "track-conditioned pixel claim is made."
)
_QUALITY_WARNING = (
    "Temporal model quality, calibration, domain transfer, and deployment safety remain "
    "unvalidated."
)
_LOSSY_WARNING = (
    "Recorder MJPEG decoding may be lossy, so emitted pixels need not equal the in-memory "
    "pixels originally presented to perception."
)


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Paths and counts from one atomically published assembly."""

    output_directory: Path
    requests_path: Path
    perceptions_path: Path
    incident_perceptions_path: Path
    provenance_path: Path
    request_count: int
    skipped_target_count: int


@dataclass(frozen=True, slots=True)
class _IncidentContract:
    metadata: dict[str, Any]
    metadata_sha256: str
    clip_name: str
    clip_sha256: str
    clip_byte_size: int
    codec: str
    container: str
    frame_count: int
    height: int
    width: int
    fps: float
    start_sequence: int
    end_sequence: int
    start_captured_at_ms: int
    end_captured_at_ms: int


@dataclass(frozen=True, slots=True)
class _PerceptionInput:
    records: list[PerceptionObservation]
    raw_sha256: str
    byte_size: int
    filename: str


@dataclass(frozen=True, slots=True)
class _Window:
    target: PerceptionObservation
    track_id: str
    first_ordinal: int
    last_ordinal: int
    ordinals: tuple[int, ...]
    sequences: tuple[int, ...]
    timestamps_ms: tuple[int, ...]


class _StrictJsonError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssemblyError(f"{context} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise AssemblyError(f"{context} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise AssemblyError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _integer(value: Any, context: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or not minimum <= value <= MAX_SAFE_INTEGER:
        qualifier = "positive" if positive else "non-negative"
        raise AssemblyError(f"{context} must be a {qualifier} safe integer")
    return value


def _number(
    value: Any,
    context: str,
    *,
    positive: bool = False,
    probability: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise AssemblyError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise AssemblyError(f"{context} must be a finite number")
    if positive and result <= 0.0:
        raise AssemblyError(f"{context} must be positive")
    if probability and not 0.0 <= result <= 1.0:
        raise AssemblyError(f"{context} must be within [0, 1]")
    return result


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise AssemblyError(f"{context} must be a canonical identifier")
    return value


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        raise AssemblyError(f"{context} must be a bounded non-empty string")
    return value


def _verify_private(st: os.stat_result, context: str) -> None:
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise AssemblyError(f"{context} permissions must not allow group or other access")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise AssemblyError(f"{context} must be owned by the current user")


def _verify_regular(st: os.stat_result, context: str, *, private: bool) -> None:
    if not stat.S_ISREG(st.st_mode):
        raise AssemblyError(f"{context} must be a regular file")
    if st.st_nlink != 1:
        raise AssemblyError(f"{context} must have exactly one hard link")
    if private:
        _verify_private(st, context)


def _same_file_state(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _read_descriptor(
    descriptor: int,
    initial: os.stat_result,
    *,
    limit: int,
    context: str,
) -> bytes:
    if initial.st_size > limit:
        raise AssemblyError(f"{context} exceeds the {limit}-byte limit")
    chunks: list[bytes] = []
    total = 0
    while chunk := os.read(descriptor, min(_HASH_CHUNK_BYTES, limit + 1 - total)):
        total += len(chunk)
        if total > limit:
            raise AssemblyError(f"{context} exceeds the {limit}-byte limit")
        chunks.append(chunk)
    final = os.fstat(descriptor)
    if not _same_file_state(initial, final) or total != initial.st_size:
        raise AssemblyError(f"{context} changed while it was being read")
    return b"".join(chunks)


def _open_path_bytes(path: Path, *, limit: int, context: str) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        initial = os.fstat(descriptor)
        _verify_regular(initial, context, private=False)
        return _read_descriptor(descriptor, initial, limit=limit, context=context), initial
    except OSError as error:
        raise AssemblyError(f"could not read {context}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_incident_file(
    directory_descriptor: int,
    filename: str,
    *,
    limit: int,
    context: str,
) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        initial = os.fstat(descriptor)
        _verify_regular(initial, context, private=True)
        return _read_descriptor(descriptor, initial, limit=limit, context=context), initial
    except OSError as error:
        raise AssemblyError(f"could not read {context}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_json_constant(value: str) -> None:
    raise AssemblyError(f"JSON contains forbidden constant {value}")


def _parse_json_object(data: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise AssemblyError(f"{context} is not valid UTF-8") from error
    except json.JSONDecodeError as error:
        raise AssemblyError(f"{context} is invalid JSON: {error.msg}") from error
    except _StrictJsonError as error:
        raise AssemblyError(f"{context} is invalid JSON: {error}") from error
    return dict(_object(value, context))


def _validate_trigger(
    value: Any,
    *,
    start_sequence: int,
    end_sequence: int,
    start_ms: int,
    end_ms: int,
) -> None:
    trigger = _object(value, "metadata.trigger")
    _exact_fields(
        trigger,
        {"minimum_approach_overlap", "rule", "samples"},
        "metadata.trigger",
    )
    minimum = _number(
        trigger["minimum_approach_overlap"],
        "metadata.trigger.minimum_approach_overlap",
        positive=True,
        probability=True,
    )
    if trigger["rule"] != "CAT_IN_APPROACH_ZONE":
        raise AssemblyError("metadata.trigger.rule must be CAT_IN_APPROACH_ZONE")
    samples = trigger["samples"]
    if not isinstance(samples, list) or not samples:
        raise AssemblyError("metadata.trigger.samples must be a non-empty list")
    previous: tuple[int, int] | None = None
    for index, raw_sample in enumerate(samples):
        context = f"metadata.trigger.samples[{index}]"
        sample = _object(raw_sample, context)
        _exact_fields(
            sample,
            {"captured_at_ms", "maximum_approach_overlap", "sequence", "track_ids"},
            context,
        )
        sequence = _integer(sample["sequence"], f"{context}.sequence")
        captured = _integer(sample["captured_at_ms"], f"{context}.captured_at_ms")
        maximum = _number(
            sample["maximum_approach_overlap"],
            f"{context}.maximum_approach_overlap",
            probability=True,
        )
        if not start_sequence <= sequence <= end_sequence or not start_ms <= captured <= end_ms:
            raise AssemblyError(f"{context} lies outside the incident timeline")
        if maximum < minimum:
            raise AssemblyError(f"{context} maximum overlap is below the recorder threshold")
        track_ids = sample["track_ids"]
        if not isinstance(track_ids, list) or not track_ids:
            raise AssemblyError(f"{context}.track_ids must be a non-empty list")
        checked_ids = [_identifier(item, f"{context}.track_ids") for item in track_ids]
        if checked_ids != sorted(set(checked_ids)):
            raise AssemblyError(f"{context}.track_ids must be sorted and unique")
        order = (captured, sequence)
        if previous is not None and (captured <= previous[0] or sequence <= previous[1]):
            raise AssemblyError("metadata.trigger.samples must be strictly ordered")
        previous = order


def _validate_perception_provenance(
    value: Any,
    *,
    frame_count: int,
    start_sequence: int,
    end_sequence: int,
) -> None:
    provenance = _object(value, "metadata.perception_provenance")
    _exact_fields(
        provenance,
        {
            "binding_stream_canonicalization",
            "frame_bindings",
            "record_canonicalization",
            "record_count",
            "stream_sha256",
        },
        "metadata.perception_provenance",
    )
    if provenance["record_canonicalization"] != _RECORD_CANONICALIZATION:
        raise AssemblyError("metadata perception record canonicalization is unsupported")
    if provenance["binding_stream_canonicalization"] != _BINDING_STREAM_CANONICALIZATION:
        raise AssemblyError("metadata perception binding canonicalization is unsupported")
    if (
        _integer(
            provenance["record_count"],
            "metadata.perception_provenance.record_count",
            positive=True,
        )
        != frame_count
    ):
        raise AssemblyError("metadata perception record_count disagrees with clip.frame_count")
    stream_sha = provenance["stream_sha256"]
    if not isinstance(stream_sha, str) or SHA256.fullmatch(stream_sha) is None:
        raise AssemblyError("metadata perception stream_sha256 is invalid")
    bindings = provenance["frame_bindings"]
    if not isinstance(bindings, list) or len(bindings) != frame_count:
        raise AssemblyError("metadata frame_bindings must contain one item per encoded frame")
    previous: tuple[int, int] | None = None
    observation_ids: set[str] = set()
    frame_ids: set[str] = set()
    for ordinal, raw_binding in enumerate(bindings):
        context = f"metadata.perception_provenance.frame_bindings[{ordinal}]"
        binding = _object(raw_binding, context)
        _exact_fields(
            binding,
            {
                "captured_at_ms",
                "encoded_frame_index",
                "frame_id",
                "observation_id",
                "perception_record_sha256",
                "sequence",
            },
            context,
        )
        if _integer(binding["encoded_frame_index"], f"{context}.encoded_frame_index") != ordinal:
            raise AssemblyError("metadata encoded frame indices must be contiguous from zero")
        sequence = _integer(binding["sequence"], f"{context}.sequence")
        captured_at_ms = _integer(binding["captured_at_ms"], f"{context}.captured_at_ms")
        if not start_sequence <= sequence <= end_sequence:
            raise AssemblyError(f"{context}.sequence lies outside the incident")
        observation_id = _identifier(binding["observation_id"], f"{context}.observation_id")
        frame_id = _identifier(binding["frame_id"], f"{context}.frame_id")
        digest = binding["perception_record_sha256"]
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise AssemblyError(f"{context}.perception_record_sha256 is invalid")
        order = (captured_at_ms, sequence)
        if previous is not None and (captured_at_ms <= previous[0] or sequence <= previous[1]):
            raise AssemblyError("metadata frame_bindings must be strictly ordered")
        if observation_id in observation_ids or frame_id in frame_ids:
            raise AssemblyError("metadata frame_bindings contain duplicate identities")
        previous = order
        observation_ids.add(observation_id)
        frame_ids.add(frame_id)
    binding_bytes = _jsonl_bytes(bindings)
    if hashlib.sha256(binding_bytes).hexdigest() != stream_sha:
        raise AssemblyError("metadata perception binding stream SHA-256 is invalid")


def _validate_metadata(metadata: dict[str, Any], incident_name: str) -> _IncidentContract:
    allowed_fields = set(_METADATA_FIELDS)
    if "perception_provenance" in metadata:
        allowed_fields.add("perception_provenance")
    _exact_fields(metadata, allowed_fields, "metadata")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise AssemblyError("metadata.schema_version must be the integer 1")
    if metadata["record_type"] != "observation_clip":
        raise AssemblyError("metadata.record_type must be observation_clip")
    if metadata["mode"] != "OBSERVE_ONLY":
        raise AssemblyError("metadata.mode must be OBSERVE_ONLY")
    if metadata["incident_id"] != incident_name:
        raise AssemblyError("metadata.incident_id does not match the incident directory")

    privacy = _object(metadata["privacy"], "metadata.privacy")
    _exact_fields(privacy, {"audio", "display", "network"}, "metadata.privacy")
    if any(privacy[field] is not False for field in ("audio", "display", "network")):
        raise AssemblyError("metadata privacy flags must all be false")

    limits = _object(metadata["resource_limits"], "metadata.resource_limits")
    _exact_fields(
        limits,
        {"max_active_bytes", "max_active_frames", "max_buffer_bytes", "max_buffer_frames"},
        "metadata.resource_limits",
    )
    for field in limits:
        _integer(limits[field], f"metadata.resource_limits.{field}", positive=True)

    source = _object(metadata["source"], "metadata.source")
    _exact_fields(source, {"camera_id", "kind", "name"}, "metadata.source")
    _identifier(source["camera_id"], "metadata.source.camera_id")
    source_kind = _nonempty_string(source["kind"], "metadata.source.kind")
    if source_kind not in {"camera", "synthetic", "video"}:
        raise AssemblyError("metadata.source.kind is not a recorder source kind")
    _nonempty_string(source["name"], "metadata.source.name")

    if not isinstance(metadata["termination"], str) or metadata["termination"] not in _TERMINATIONS:
        raise AssemblyError("metadata.termination is not a recorder termination value")

    timeline = _object(metadata["timeline"], "metadata.timeline")
    _exact_fields(
        timeline,
        {
            "duration_ms",
            "end_captured_at_ms",
            "end_sequence",
            "first_trigger_at_ms",
            "last_trigger_at_ms",
            "start_captured_at_ms",
            "start_sequence",
        },
        "metadata.timeline",
    )
    start_sequence = _integer(timeline["start_sequence"], "metadata.timeline.start_sequence")
    end_sequence = _integer(timeline["end_sequence"], "metadata.timeline.end_sequence")
    start_ms = _integer(timeline["start_captured_at_ms"], "metadata.timeline.start_captured_at_ms")
    end_ms = _integer(timeline["end_captured_at_ms"], "metadata.timeline.end_captured_at_ms")
    duration = _integer(timeline["duration_ms"], "metadata.timeline.duration_ms")
    first_trigger = _integer(
        timeline["first_trigger_at_ms"], "metadata.timeline.first_trigger_at_ms"
    )
    last_trigger = _integer(timeline["last_trigger_at_ms"], "metadata.timeline.last_trigger_at_ms")
    if end_sequence < start_sequence:
        raise AssemblyError("metadata timeline sequence endpoints are reversed")
    if end_ms < start_ms or duration != end_ms - start_ms:
        raise AssemblyError("metadata timeline duration disagrees with capture endpoints")
    if not start_ms <= first_trigger <= last_trigger <= end_ms:
        raise AssemblyError("metadata trigger times lie outside the capture timeline")

    clip = _object(metadata["clip"], "metadata.clip")
    _exact_fields(
        clip,
        {
            "audio",
            "byte_size",
            "codec",
            "container",
            "filename",
            "fps",
            "frame_count",
            "height",
            "sha256",
            "width",
        },
        "metadata.clip",
    )
    if clip["audio"] is not False:
        raise AssemblyError("metadata.clip.audio must be false")
    clip_name = clip["filename"]
    if (
        not isinstance(clip_name, str)
        or _CLIP_NAME.fullmatch(clip_name) is None
        or clip_name != "clip.avi"
    ):
        raise AssemblyError("metadata.clip.filename is not a confined recorder clip name")
    clip_sha = clip["sha256"]
    if not isinstance(clip_sha, str) or SHA256.fullmatch(clip_sha) is None:
        raise AssemblyError("metadata.clip.sha256 must be a lowercase SHA-256 digest")
    clip_bytes = _integer(clip["byte_size"], "metadata.clip.byte_size", positive=True)
    if clip_bytes > MAX_SOURCE_CLIP_BYTES:
        raise AssemblyError("metadata.clip.byte_size exceeds the offline source limit")
    frame_count = _integer(clip["frame_count"], "metadata.clip.frame_count", positive=True)
    height = _integer(clip["height"], "metadata.clip.height", positive=True)
    width = _integer(clip["width"], "metadata.clip.width", positive=True)
    fps = _number(clip["fps"], "metadata.clip.fps", positive=True)
    codec = _nonempty_string(clip["codec"], "metadata.clip.codec")
    container = _nonempty_string(clip["container"], "metadata.clip.container")
    if len(codec) != 4 or not codec.isascii():
        raise AssemblyError("metadata.clip.codec must be a four-character ASCII code")
    if container != "avi":
        raise AssemblyError("metadata.clip.container must be avi")
    if frame_count > limits["max_active_frames"]:
        raise AssemblyError("metadata frame_count exceeds declared max_active_frames")
    if frame_count > MAX_INCIDENT_FRAMES:
        raise AssemblyError("metadata frame_count exceeds the offline incident-frame limit")
    decoded_bytes = frame_count * height * width * 3
    if decoded_bytes > MAX_DECODED_INCIDENT_BYTES:
        raise AssemblyError("decoded incident would exceed the offline memory limit")
    if decoded_bytes > limits["max_active_bytes"]:
        raise AssemblyError("decoded incident exceeds metadata.resource_limits.max_active_bytes")

    if "perception_provenance" in metadata:
        _validate_perception_provenance(
            metadata["perception_provenance"],
            frame_count=frame_count,
            start_sequence=start_sequence,
            end_sequence=end_sequence,
        )

    _validate_trigger(
        metadata["trigger"],
        start_sequence=start_sequence,
        end_sequence=end_sequence,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    samples = metadata["trigger"]["samples"]
    if samples[0]["captured_at_ms"] != first_trigger:
        raise AssemblyError("metadata first trigger time disagrees with trigger samples")
    if samples[-1]["captured_at_ms"] != last_trigger:
        raise AssemblyError("metadata last trigger time disagrees with trigger samples")
    expected_id = f"incident-{samples[0]['captured_at_ms']:013d}-{samples[0]['sequence']:010d}"
    if incident_name != expected_id:
        raise AssemblyError("incident_id does not encode the first trigger identity")

    return _IncidentContract(
        metadata=metadata,
        metadata_sha256="",
        clip_name=clip_name,
        clip_sha256=clip_sha,
        clip_byte_size=clip_bytes,
        codec=codec,
        container=container,
        frame_count=frame_count,
        height=height,
        width=width,
        fps=fps,
        start_sequence=start_sequence,
        end_sequence=end_sequence,
        start_captured_at_ms=start_ms,
        end_captured_at_ms=end_ms,
    )


def _open_incident(incident_directory: Path) -> tuple[int, Path, _IncidentContract]:
    if incident_directory.is_symlink():
        raise AssemblyError("incident directory must not be a symbolic link")
    try:
        real = incident_directory.resolve(strict=True)
    except OSError as error:
        raise AssemblyError(f"could not resolve incident directory: {error}") from error
    if _INCIDENT_NAME.fullmatch(real.name) is None:
        raise AssemblyError("incident directory name is not recorder-generated")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            real,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_state = os.fstat(descriptor)
        if not stat.S_ISDIR(directory_state.st_mode):
            raise AssemblyError("incident path is not a real directory")
        _verify_private(directory_state, "incident directory")
        metadata_bytes, _ = _read_incident_file(
            descriptor,
            "metadata.json",
            limit=MAX_METADATA_BYTES,
            context="incident metadata",
        )
        metadata = _parse_json_object(metadata_bytes, "incident metadata")
        contract = _validate_metadata(metadata, real.name)
        entries = set(os.listdir(descriptor))
        if entries != {"metadata.json", contract.clip_name}:
            raise AssemblyError("incident directory must contain only metadata and its clip")
        return (
            descriptor,
            real,
            _IncidentContract(
                metadata=contract.metadata,
                metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                clip_name=contract.clip_name,
                clip_sha256=contract.clip_sha256,
                clip_byte_size=contract.clip_byte_size,
                codec=contract.codec,
                container=contract.container,
                frame_count=contract.frame_count,
                height=contract.height,
                width=contract.width,
                fps=contract.fps,
                start_sequence=contract.start_sequence,
                end_sequence=contract.end_sequence,
                start_captured_at_ms=contract.start_captured_at_ms,
                end_captured_at_ms=contract.end_captured_at_ms,
            ),
        )
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _snapshot_clip(
    directory_descriptor: int,
    contract: _IncidentContract,
    temporary_directory: Path,
) -> Path:
    source: int | None = None
    destination: int | None = None
    snapshot = temporary_directory / contract.clip_name
    try:
        source = os.open(
            contract.clip_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        initial = os.fstat(source)
        _verify_regular(initial, "incident clip", private=True)
        if initial.st_size != contract.clip_byte_size:
            raise AssemblyError("incident clip byte_size disagrees with metadata")
        destination = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(source, _HASH_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_SOURCE_CLIP_BYTES:
                raise AssemblyError("incident clip exceeds the offline source limit")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                view = view[written:]
        final = os.fstat(source)
        if not _same_file_state(initial, final) or total != initial.st_size:
            raise AssemblyError("incident clip changed while it was being snapshotted")
        if digest.hexdigest() != contract.clip_sha256:
            raise AssemblyError("incident clip SHA-256 disagrees with metadata")
        os.fsync(destination)
        os.close(destination)
        destination = None
        os.chmod(snapshot, 0o600, follow_symlinks=False)
        return snapshot
    except OSError as error:
        raise AssemblyError(f"could not snapshot incident clip: {error}") from error
    finally:
        if source is not None:
            os.close(source)
        if destination is not None:
            os.close(destination)
        if not snapshot.is_file():
            snapshot.unlink(missing_ok=True)


def _parse_perception_bytes(data: bytes, context: str) -> list[PerceptionObservation]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AssemblyError(f"{context} is not valid UTF-8") from error
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise AssemblyError(f"{context}:{line_number}: blank JSONL lines are forbidden")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except json.JSONDecodeError as error:
            raise AssemblyError(f"{context}:{line_number}: invalid JSON: {error.msg}") from error
        except _StrictJsonError as error:
            raise AssemblyError(f"{context}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise AssemblyError(f"{context}:{line_number}: record must be an object")
        rows.append(value)
        if len(rows) > MAX_PERCEPTION_RECORDS:
            raise AssemblyError(
                f"{context} exceeds the {MAX_PERCEPTION_RECORDS}-record offline limit"
            )
    if not rows:
        raise AssemblyError(f"{context} is empty")
    try:
        return parse_perception_stream(rows)
    except ContractError as error:
        raise AssemblyError(f"{context}: {error}") from error


def _load_perceptions(path: Path) -> _PerceptionInput:
    if path.is_symlink():
        raise AssemblyError("perception JSONL must not be a symbolic link")
    data, _ = _open_path_bytes(
        path,
        limit=MAX_PERCEPTION_BYTES,
        context="perception JSONL",
    )
    return _PerceptionInput(
        records=_parse_perception_bytes(data, "perception JSONL"),
        raw_sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        filename=path.name,
    )


def _trigger_sample(record: PerceptionObservation, minimum: float) -> dict[str, Any] | None:
    qualifying = sorted(
        (
            track["track_id"],
            float(track["region_evidence"]["approach_overlap"]),
        )
        for track in record.tracks
        if track["class"] == "CAT"
        and float(track["region_evidence"]["approach_overlap"]) >= minimum
    )
    if not qualifying:
        return None
    return {
        "captured_at_ms": record.captured_at_ms,
        "maximum_approach_overlap": max(overlap for _, overlap in qualifying),
        "sequence": record.sequence,
        "track_ids": [track_id for track_id, _ in qualifying],
    }


def _reconcile_perceptions(
    perception_input: _PerceptionInput,
    contract: _IncidentContract,
) -> list[PerceptionObservation]:
    incident_records = [
        record
        for record in perception_input.records
        if contract.start_sequence <= record.sequence <= contract.end_sequence
    ]
    if len(incident_records) != contract.frame_count:
        raise AssemblyError(
            "inclusive perception selection count disagrees with metadata.clip.frame_count"
        )
    if incident_records[0].sequence != contract.start_sequence:
        raise AssemblyError("perception start sequence disagrees with incident metadata")
    if incident_records[-1].sequence != contract.end_sequence:
        raise AssemblyError("perception end sequence disagrees with incident metadata")
    if incident_records[0].captured_at_ms != contract.start_captured_at_ms:
        raise AssemblyError("perception start timestamp disagrees with incident metadata")
    if incident_records[-1].captured_at_ms != contract.end_captured_at_ms:
        raise AssemblyError("perception end timestamp disagrees with incident metadata")
    if any(
        current.captured_at_ms <= previous.captured_at_ms
        for previous, current in pairwise(incident_records)
    ):
        raise AssemblyError("incident perception timestamps must be strictly increasing")

    source = contract.metadata["source"]
    for record in incident_records:
        frame = record.record["frame"]
        record_source = record.record["source"]
        if record.camera_id != source["camera_id"] or record_source != {
            "kind": source["kind"],
            "name": source["name"],
        }:
            raise AssemblyError(
                f"perception sequence {record.sequence} source disagrees with incident metadata"
            )
        if frame["height"] != contract.height or frame["width"] != contract.width:
            raise AssemblyError(
                f"perception sequence {record.sequence} dimensions disagree with the clip"
            )

    minimum = float(contract.metadata["trigger"]["minimum_approach_overlap"])
    recomputed = [
        sample
        for record in incident_records
        if (sample := _trigger_sample(record, minimum)) is not None
    ]
    if recomputed != contract.metadata["trigger"]["samples"]:
        raise AssemblyError("metadata trigger samples disagree with strict perception evidence")
    if "perception_provenance" in contract.metadata:
        expected_bindings = []
        for ordinal, record in enumerate(incident_records):
            record_digest = hashlib.sha256(
                stable_json(record.to_dict()).encode("utf-8")
            ).hexdigest()
            expected_bindings.append(
                {
                    "captured_at_ms": record.captured_at_ms,
                    "encoded_frame_index": ordinal,
                    "frame_id": record.frame_id,
                    "observation_id": record.observation_id,
                    "perception_record_sha256": record_digest,
                    "sequence": record.sequence,
                }
            )
        if expected_bindings != contract.metadata["perception_provenance"]["frame_bindings"]:
            raise AssemblyError(
                "recorder frame bindings disagree with the supplied strict perception records"
            )
    return incident_records


def _decode_rgb_clip(snapshot: Path, contract: _IncidentContract) -> list[Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise AssemblyError("NumPy and OpenCV are required for offline clip decoding") from error

    with snapshot.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"AVI ":
        raise AssemblyError("incident clip bytes are not a RIFF AVI container")

    capture = cv2.VideoCapture(str(snapshot))
    if not capture.isOpened():
        capture.release()
        raise AssemblyError("OpenCV could not open the snapshotted incident clip")
    try:
        reported_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fourcc = round(capture.get(cv2.CAP_PROP_FOURCC))
        reported_codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        if (reported_width, reported_height) != (contract.width, contract.height):
            raise AssemblyError("OpenCV clip dimensions disagree with metadata")
        if reported_count != contract.frame_count:
            raise AssemblyError("OpenCV reported frame count disagrees with metadata")
        if reported_codec != contract.codec:
            raise AssemblyError("OpenCV clip codec disagrees with metadata")
        if not math.isfinite(reported_fps) or not math.isclose(
            reported_fps,
            contract.fps,
            rel_tol=1e-6,
            abs_tol=1e-3,
        ):
            raise AssemblyError("OpenCV clip FPS disagrees with metadata")

        frames: list[Any] = []
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if (
                not isinstance(bgr, np.ndarray)
                or bgr.dtype != np.uint8
                or bgr.shape != (contract.height, contract.width, 3)
            ):
                raise AssemblyError("OpenCV decoded a frame with invalid shape or dtype")
            frames.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            if len(frames) > contract.frame_count:
                raise AssemblyError("OpenCV decoded more frames than metadata declares")
        if len(frames) != contract.frame_count:
            raise AssemblyError("decoded frame count disagrees with metadata")
        return frames
    finally:
        capture.release()


def _select_windows(
    records: Sequence[PerceptionObservation],
    *,
    window_ms: int,
) -> tuple[list[_Window], list[dict[str, Any]]]:
    timestamp_to_ordinal = {
        record.captured_at_ms: ordinal for ordinal, record in enumerate(records)
    }
    windows: list[_Window] = []
    skipped: list[dict[str, Any]] = []
    selected_frame_entries = 0
    for target_ordinal, record in enumerate(records):
        cats = [track for track in record.tracks if track["class"] == "CAT"]
        if len(cats) > 1:
            skipped.append(
                {
                    "captured_at_ms": record.captured_at_ms,
                    "reason": "MULTI_CAT_AMBIGUOUS",
                    "sequence": record.sequence,
                    "track_ids": sorted(track["track_id"] for track in cats),
                }
            )
            continue
        if not cats:
            continue
        cat = cats[0]
        if float(cat["region_evidence"]["approach_overlap"]) <= 0.0:
            skipped.append(
                {
                    "captured_at_ms": record.captured_at_ms,
                    "reason": "NO_APPROACH_EVIDENCE",
                    "sequence": record.sequence,
                    "track_id": cat["track_id"],
                }
            )
            continue
        if record.captured_at_ms < window_ms:
            skipped.append(
                {
                    "captured_at_ms": record.captured_at_ms,
                    "reason": "TARGET_PRECEDES_WINDOW_MS",
                    "sequence": record.sequence,
                    "track_id": cat["track_id"],
                }
            )
            continue
        start_ms = record.captured_at_ms - window_ms
        first_ordinal = timestamp_to_ordinal.get(start_ms)
        if first_ordinal is None or first_ordinal > target_ordinal:
            skipped.append(
                {
                    "captured_at_ms": record.captured_at_ms,
                    "reason": "NO_EXACT_CAUSAL_WINDOW_START",
                    "sequence": record.sequence,
                    "track_id": cat["track_id"],
                    "window_start_captured_at_ms": start_ms,
                }
            )
            continue
        window_frame_count = target_ordinal - first_ordinal + 1
        selected = records[first_ordinal : target_ordinal + 1]
        window_violation: dict[str, Any] | None = None
        for selected_record in selected:
            selected_cats = [track for track in selected_record.tracks if track["class"] == "CAT"]
            if len(selected_cats) > 1:
                window_violation = {
                    "cat_track_ids": sorted(track["track_id"] for track in selected_cats),
                    "offending_captured_at_ms": selected_record.captured_at_ms,
                    "offending_sequence": selected_record.sequence,
                    "reason": "WINDOW_CAT_CARDINALITY_AMBIGUOUS",
                }
                break
            if selected_cats and selected_cats[0]["track_id"] != cat["track_id"]:
                window_violation = {
                    "offending_captured_at_ms": selected_record.captured_at_ms,
                    "offending_sequence": selected_record.sequence,
                    "offending_track_id": selected_cats[0]["track_id"],
                    "reason": "WINDOW_TRACK_IDENTITY_MISMATCH",
                }
                break
        if window_violation is not None:
            skipped.append(
                {
                    "captured_at_ms": record.captured_at_ms,
                    "sequence": record.sequence,
                    "track_id": cat["track_id"],
                    **window_violation,
                }
            )
            continue
        if len(windows) >= MAX_ASSEMBLED_TARGETS:
            raise AssemblyError("eligible targets exceed the offline assembly-target limit")
        if selected_frame_entries + window_frame_count > MAX_SELECTED_FRAME_ENTRIES:
            raise AssemblyError("causal windows exceed the cumulative selected-frame-entry limit")
        selected_frame_entries += window_frame_count
        windows.append(
            _Window(
                target=record,
                track_id=cat["track_id"],
                first_ordinal=first_ordinal,
                last_ordinal=target_ordinal,
                ordinals=tuple(range(first_ordinal, target_ordinal + 1)),
                sequences=tuple(item.sequence for item in selected),
                timestamps_ms=tuple(item.captured_at_ms for item in selected),
            )
        )
    if not windows:
        raise AssemblyError(
            "incident contains no exactly-windowed target with one CAT and approach evidence"
        )
    return windows, skipped


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _numpy_layout(frames: Sequence[Any]) -> tuple[tuple[int, ...], bytes, int]:
    import numpy as np

    if not frames:
        raise AssemblyError("assembled clip must contain at least one frame")
    expected_shape = frames[0].shape
    if frames[0].dtype != np.uint8 or len(expected_shape) != 3 or expected_shape[-1] != 3:
        raise AssemblyError("assembled clip frame is not uint8 HWC RGB")
    for frame in frames:
        if frame.dtype != np.uint8 or frame.shape != expected_shape:
            raise AssemblyError("assembled clip frames do not share uint8 HWC shape")
    header_stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header_stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
            "fortran_order": False,
            "shape": (len(frames), *expected_shape),
        },
    )
    header = header_stream.getvalue()
    serialized_bytes = len(header) + len(frames) * int(frames[0].nbytes)
    if serialized_bytes > MAX_SHADOW_CLIP_BYTES:
        raise AssemblyError("serialized NumPy clip exceeds the shadow inference limit")
    return expected_shape, header, serialized_bytes


def _write_numpy(path: Path, frames: Sequence[Any]) -> tuple[str, int]:
    import numpy as np

    _, header, projected_byte_size = _numpy_layout(frames)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(header)
            for frame in frames:
                contiguous = np.ascontiguousarray(frame)
                stream.write(memoryview(contiguous).cast("B"))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)
    byte_size = path.stat(follow_symlinks=False).st_size
    if byte_size != projected_byte_size:
        raise AssemblyError("serialized NumPy clip byte size disagrees with its exact layout")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest(), byte_size


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(stable_json(record) + "\n" for record in records).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_output_parent(descriptor: int, path: Path) -> None:
    """Require the requested parent path to still name the held directory."""

    held = os.fstat(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise AssemblyError(f"output parent changed during assembly: {error}") from error
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        held.st_dev,
        held.st_ino,
    ):
        raise AssemblyError("output parent changed during assembly")


def _rename_no_replace(
    parent_descriptor: int,
    parent_path: Path,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a directory without Unix rename's overwrite behavior."""

    for name, context in (
        (source_name, "temporary output directory"),
        (destination_name, "output directory"),
    ):
        if not name or name in {".", ".."} or Path(name).name != name:
            raise AssemblyError(f"{context} name is not a confined path component")
    _verify_output_parent(parent_descriptor, parent_path)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AssemblyError("atomic no-overwrite publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise AssemblyError("output directory already exists; refusing overwrite")
    raise AssemblyError(
        f"could not atomically publish output directory: {os.strerror(error_number)}"
    )


def _validate_options(window_ms: int, logical_latency_ms: int) -> tuple[int, int]:
    window = _integer(window_ms, "window_ms")
    latency = _integer(logical_latency_ms, "logical_latency_ms")
    if window > 30_000:
        raise AssemblyError("window_ms must be within [0, 30000]")
    if latency > MAX_LOGICAL_LATENCY_MS:
        raise AssemblyError(
            f"logical_latency_ms must not exceed the shadow timeout ({MAX_LOGICAL_LATENCY_MS} ms)"
        )
    return window, latency


def _runtime_identity() -> dict[str, str]:
    import cv2
    import numpy as np

    return {
        "assembler_version": ASSEMBLER_VERSION,
        "numpy_version": np.__version__,
        "opencv_build_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "opencv_version": cv2.__version__,
        "python_version": platform.python_version(),
    }


def assemble_incident(
    incident_directory: str | Path,
    perception_jsonl: str | Path,
    output_directory: str | Path,
    *,
    window_ms: int,
    logical_latency_ms: int,
) -> AssemblyResult:
    """Validate one incident and atomically publish deterministic shadow inputs."""

    window, latency = _validate_options(window_ms, logical_latency_ms)
    raw_output = Path(output_directory)
    if raw_output.exists() or raw_output.is_symlink():
        raise AssemblyError("output directory already exists; refusing overwrite")
    if not raw_output.name or raw_output.name in {".", ".."}:
        raise AssemblyError("output directory must have a concrete final name")
    try:
        output_parent = raw_output.parent.resolve(strict=True)
    except OSError as error:
        raise AssemblyError(f"output parent does not exist: {error}") from error
    if not output_parent.is_dir():
        raise AssemblyError("output parent is not a directory")
    destination = output_parent / raw_output.name

    output_parent_descriptor: int | None = None
    directory_descriptor: int | None = None
    source_snapshot_directory: Path | None = None
    stage: Path | None = None
    try:
        initial_parent_state = os.stat(output_parent, follow_symlinks=False)
        output_parent_descriptor = os.open(
            output_parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened_parent_state = os.fstat(output_parent_descriptor)
        if not stat.S_ISDIR(opened_parent_state.st_mode) or (
            opened_parent_state.st_dev,
            opened_parent_state.st_ino,
        ) != (initial_parent_state.st_dev, initial_parent_state.st_ino):
            raise AssemblyError("output parent changed while it was being opened")
        _verify_output_parent(output_parent_descriptor, output_parent)
        directory_descriptor, real_incident, contract = _open_incident(Path(incident_directory))
        if output_parent == real_incident or real_incident in output_parent.parents:
            raise AssemblyError("output directory must not be nested inside the recorder incident")
        source_snapshot_directory = Path(tempfile.mkdtemp(prefix="fw-assembler-source-"))
        os.chmod(source_snapshot_directory, 0o700)
        clip_snapshot = _snapshot_clip(
            directory_descriptor,
            contract,
            source_snapshot_directory,
        )
        perception_input = _load_perceptions(Path(perception_jsonl))
        incident_records = _reconcile_perceptions(perception_input, contract)
        rgb_frames = _decode_rgb_clip(clip_snapshot, contract)
        windows, skipped = _select_windows(incident_records, window_ms=window)

        descriptor_parent = Path(f"/proc/self/fd/{output_parent_descriptor}")
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=descriptor_parent))
        os.chmod(stage, 0o700)
        clips_directory = stage / "clips"
        clips_directory.mkdir(mode=0o700)
        os.chmod(clips_directory, 0o700, follow_symlinks=False)

        request_records: list[dict[str, Any]] = []
        selected_records: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        cumulative_clip_bytes = 0
        for request_sequence, selected in enumerate(windows):
            clip_relative = f"clips/clip-{request_sequence:06d}.npy"
            clip_path = stage / clip_relative
            selected_frames = rgb_frames[selected.first_ordinal : selected.last_ordinal + 1]
            projected_clip_bytes = _numpy_layout(selected_frames)[2]
            if cumulative_clip_bytes + projected_clip_bytes > MAX_ASSEMBLED_CLIP_BYTES:
                raise AssemblyError("assembled clips exceed the cumulative output byte limit")
            clip_digest, clip_byte_size = _write_numpy(
                clip_path,
                selected_frames,
            )
            cumulative_clip_bytes += clip_byte_size
            target = selected.target
            request = {
                "captured_at_ms": target.captured_at_ms,
                "clip": {
                    "format": CLIP_FORMAT,
                    "frame_timestamps_ms": list(selected.timestamps_ms),
                    "path": clip_relative,
                    "sha256": clip_digest,
                    "window_end_captured_at_ms": target.captured_at_ms,
                    "window_start_captured_at_ms": selected.timestamps_ms[0],
                },
                "frame_id": target.frame_id,
                "observation_id": target.observation_id,
                "predicted_at_ms": target.captured_at_ms + latency,
                "record_type": "behavior_inference_request",
                "schema_version": SCHEMA_VERSION,
                "sequence": request_sequence,
                "track_id": selected.track_id,
            }
            if request["predicted_at_ms"] > MAX_SAFE_INTEGER:
                raise AssemblyError("logical prediction timestamp exceeds the safe integer limit")
            request_records.append(request)
            selected_records.append(target.to_dict())
            mappings.append(
                {
                    "clip": {
                        "byte_size": clip_byte_size,
                        "format": CLIP_FORMAT,
                        "path": clip_relative,
                        "sha256": clip_digest,
                    },
                    "request_sequence": request_sequence,
                    "sampled_captured_at_ms": list(selected.timestamps_ms),
                    "sampled_decoded_ordinals": list(selected.ordinals),
                    "sampled_perception_sequences": list(selected.sequences),
                    "target": {
                        "captured_at_ms": target.captured_at_ms,
                        "frame_id": target.frame_id,
                        "observation_id": target.observation_id,
                        "perception_sequence": target.sequence,
                        "track_id": selected.track_id,
                    },
                }
            )

        try:
            parse_inference_requests(request_records)
            parse_perception_stream(selected_records)
        except ContractError as error:
            raise AssemblyError(
                f"derived shadow stream failed its own contract: {error}"
            ) from error

        requests_bytes = _jsonl_bytes(request_records)
        perceptions_bytes = _jsonl_bytes(selected_records)
        incident_perceptions_bytes = _jsonl_bytes(record.to_dict() for record in incident_records)
        _write_exclusive(stage / REQUESTS_FILENAME, requests_bytes)
        _write_exclusive(stage / PERCEPTIONS_FILENAME, perceptions_bytes)
        _write_exclusive(stage / INCIDENT_PERCEPTIONS_FILENAME, incident_perceptions_bytes)
        requests_sha = hashlib.sha256(requests_bytes).hexdigest()
        perceptions_sha = hashlib.sha256(perceptions_bytes).hexdigest()
        incident_perceptions_sha = hashlib.sha256(incident_perceptions_bytes).hexdigest()
        runtime_identity = _runtime_identity()
        assembly_identity = {
            "assembler_contract_version": SCHEMA_VERSION,
            "encoded_clip_sha256": contract.clip_sha256,
            "incident_metadata_sha256": contract.metadata_sha256,
            "incident_perceptions_sha256": incident_perceptions_sha,
            "logical_latency_ms": latency,
            "mapping_sha256": hashlib.sha256(stable_json(mappings).encode("utf-8")).hexdigest(),
            "perception_jsonl_sha256": perception_input.raw_sha256,
            "requests_sha256": requests_sha,
            "runtime": runtime_identity,
            "selected_perceptions_sha256": perceptions_sha,
            "window_ms": window,
        }
        assembly_id = hashlib.sha256(stable_json(assembly_identity).encode("utf-8")).hexdigest()
        provenance = {
            "assembly_id": assembly_id,
            "config": {
                "clip_contract": "FULL_FRAME_RGB_UINT8_THWC_NO_CROP",
                "logical_latency_ms": latency,
                "target_rule": (
                    "TARGET_EXACTLY_ONE_CAT_WITH_POSITIVE_APPROACH_OVERLAP;"
                    "WINDOW_ZERO_OR_SAME_SOLE_CAT"
                ),
                "window_ms": window,
            },
            "inputs": {
                "incident": {
                    "directory_name": real_incident.name,
                    "encoded_clip": {
                        "byte_size": contract.clip_byte_size,
                        "filename": contract.clip_name,
                        "sha256": contract.clip_sha256,
                    },
                    "metadata_sha256": contract.metadata_sha256,
                    "perception_binding": (
                        {
                            "present": True,
                            "record_canonicalization": contract.metadata["perception_provenance"][
                                "record_canonicalization"
                            ],
                            "stream_sha256": contract.metadata["perception_provenance"][
                                "stream_sha256"
                            ],
                            "verified": True,
                        }
                        if "perception_provenance" in contract.metadata
                        else {
                            "present": False,
                            "verified": False,
                        }
                    ),
                    "source": dict(contract.metadata["source"]),
                    "timeline": dict(contract.metadata["timeline"]),
                },
                "perception_jsonl": {
                    "byte_size": perception_input.byte_size,
                    "filename": perception_input.filename,
                    "record_count": len(perception_input.records),
                    "sha256": perception_input.raw_sha256,
                },
            },
            "mapping": mappings,
            "outputs": {
                "behavior_inference_requests": {
                    "byte_size": len(requests_bytes),
                    "path": REQUESTS_FILENAME,
                    "record_count": len(request_records),
                    "sha256": requests_sha,
                },
                "selected_perceptions": {
                    "byte_size": len(perceptions_bytes),
                    "path": PERCEPTIONS_FILENAME,
                    "record_count": len(selected_records),
                    "sha256": perceptions_sha,
                },
                "incident_perceptions": {
                    "byte_size": len(incident_perceptions_bytes),
                    "path": INCIDENT_PERCEPTIONS_FILENAME,
                    "record_count": len(incident_records),
                    "sha256": incident_perceptions_sha,
                },
            },
            "record_type": PROVENANCE_RECORD_TYPE,
            "runtime": runtime_identity,
            "schema_version": SCHEMA_VERSION,
            "skipped_targets": skipped,
            "warnings": [
                _STRUCTURAL_WARNING,
                _FULL_FRAME_WARNING,
                _QUALITY_WARNING,
                _LOSSY_WARNING,
            ],
        }
        _write_exclusive(
            stage / PROVENANCE_FILENAME,
            (stable_json(provenance) + "\n").encode("utf-8"),
        )
        _fsync_directory(clips_directory)
        _fsync_directory(stage)
        _rename_no_replace(
            output_parent_descriptor,
            output_parent,
            stage.name,
            destination.name,
        )
        stage = None
        os.fsync(output_parent_descriptor)
        _verify_output_parent(output_parent_descriptor, output_parent)
        return AssemblyResult(
            output_directory=destination,
            requests_path=destination / REQUESTS_FILENAME,
            perceptions_path=destination / PERCEPTIONS_FILENAME,
            incident_perceptions_path=destination / INCIDENT_PERCEPTIONS_FILENAME,
            provenance_path=destination / PROVENANCE_FILENAME,
            request_count=len(request_records),
            skipped_target_count=len(skipped),
        )
    except OSError as error:
        raise AssemblyError(f"offline assembly failed: {error}") from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        if source_snapshot_directory is not None:
            with suppress(OSError):
                shutil.rmtree(source_snapshot_directory)
        if stage is not None:
            with suppress(OSError):
                shutil.rmtree(stage)
        if output_parent_descriptor is not None:
            os.close(output_parent_descriptor)
