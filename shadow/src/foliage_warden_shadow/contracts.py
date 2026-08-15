"""Strict, deterministic JSONL contracts at the shadow integration boundary."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

SCHEMA_VERSION = 1
BEHAVIOR_LABELS = (
    "PASSING",
    "SNIFFING",
    "EATING",
    "DIGGING",
    "OTHER",
    "UNKNOWN",
)
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
MAX_SAFE_INTEGER = 9_007_199_254_740_991
JsonObject = dict[str, Any]
T = TypeVar("T")


class ContractError(ValueError):
    """A stream record is ambiguous, unsafe, or not the supported version."""


@dataclass(frozen=True, slots=True)
class BehaviorPrediction:
    sequence: int
    observation_id: str
    frame_id: str
    track_id: str
    captured_at_ms: int
    predicted_at_ms: int
    model_id: str
    model_sha256: str
    config_id: str
    config_sha256: str
    predicted_label: str
    probabilities: dict[str, float]

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.observation_id, self.frame_id, self.track_id)

    def to_dict(self) -> JsonObject:
        return {
            "config": {"id": self.config_id, "sha256": self.config_sha256},
            "captured_at_ms": self.captured_at_ms,
            "frame_id": self.frame_id,
            "mode": "OBSERVE_ONLY",
            "model": {"id": self.model_id, "sha256": self.model_sha256},
            "observation_id": self.observation_id,
            "predicted_at_ms": self.predicted_at_ms,
            "predicted_label": self.predicted_label,
            "probabilities": dict(self.probabilities),
            "record_type": "behavior_prediction",
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "track_id": self.track_id,
            "would_action": False,
        }


@dataclass(frozen=True, slots=True)
class PerceptionObservation:
    sequence: int
    observation_id: str
    frame_id: str
    captured_at_ms: int
    camera_id: str
    record: JsonObject

    @property
    def tracks(self) -> list[JsonObject]:
        return self.record["observation"]["tracks"]

    def policy_observation(self) -> JsonObject:
        return deepcopy(self.record["observation"])

    def to_dict(self) -> JsonObject:
        return deepcopy(self.record)


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_stream_sha256(records: Iterable[Any]) -> str:
    from hashlib import sha256

    digest = sha256()
    for record in records:
        value = record.to_dict() if hasattr(record, "to_dict") else record
        digest.update(stable_json(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ContractError(f"{context} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{context} must be a canonical identifier")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ContractError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ContractError(f"{context} must be a non-negative safe integer")
    return value


def _schema_version(value: Any, context: str) -> None:
    # JSON/Python numeric equality treats 1.0 as equal to 1. Wire versions are
    # intentionally stricter so a producer cannot silently change number types.
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ContractError(f"{context} must be the integer 1")


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{context} must be a boolean")
    return value


def _probability(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContractError(f"{context} must be finite and within [0, 1]")
    return result


def _identity(value: Any, context: str) -> tuple[str, str]:
    item = _mapping(value, context)
    _exact_keys(item, {"id", "sha256"}, context)
    return _identifier(item["id"], f"{context}.id"), _sha256(item["sha256"], f"{context}.sha256")


def parse_behavior_prediction(raw: Mapping[str, Any]) -> BehaviorPrediction:
    data = _mapping(raw, "behavior_prediction")
    expected = {
        "captured_at_ms",
        "config",
        "frame_id",
        "mode",
        "model",
        "observation_id",
        "predicted_at_ms",
        "predicted_label",
        "probabilities",
        "record_type",
        "schema_version",
        "sequence",
        "track_id",
        "would_action",
    }
    _exact_keys(data, expected, "behavior_prediction")
    _schema_version(data["schema_version"], "behavior_prediction.schema_version")
    if data["record_type"] != "behavior_prediction":
        raise ContractError("behavior_prediction.record_type is invalid")
    if data["mode"] != "OBSERVE_ONLY" or data["would_action"] is not False:
        raise ContractError("behavior_prediction must be OBSERVE_ONLY with would_action=false")

    probabilities_raw = _mapping(data["probabilities"], "behavior_prediction.probabilities")
    _exact_keys(
        probabilities_raw,
        set(BEHAVIOR_LABELS),
        "behavior_prediction.probabilities",
    )
    probabilities = {
        label: _probability(probabilities_raw[label], f"behavior_prediction.probabilities.{label}")
        for label in BEHAVIOR_LABELS
    }
    total = math.fsum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError(f"behavior_prediction probabilities must sum to 1.0; got {total:.9f}")
    predicted_label = data["predicted_label"]
    if predicted_label not in BEHAVIOR_LABELS:
        raise ContractError("behavior_prediction.predicted_label is not in the six-label schema")
    argmax = max(BEHAVIOR_LABELS, key=probabilities.__getitem__)
    if predicted_label != argmax:
        raise ContractError(
            "behavior_prediction.predicted_label must equal deterministic six-label argmax "
            f"{argmax}"
        )

    captured_at_ms = _integer(data["captured_at_ms"], "behavior_prediction.captured_at_ms")
    predicted_at_ms = _integer(data["predicted_at_ms"], "behavior_prediction.predicted_at_ms")
    if predicted_at_ms < captured_at_ms:
        raise ContractError("behavior_prediction.predicted_at_ms precedes captured_at_ms")
    model_id, model_sha256 = _identity(data["model"], "behavior_prediction.model")
    config_id, config_sha256 = _identity(data["config"], "behavior_prediction.config")
    return BehaviorPrediction(
        sequence=_integer(data["sequence"], "behavior_prediction.sequence"),
        observation_id=_identifier(data["observation_id"], "behavior_prediction.observation_id"),
        frame_id=_identifier(data["frame_id"], "behavior_prediction.frame_id"),
        track_id=_identifier(data["track_id"], "behavior_prediction.track_id"),
        captured_at_ms=captured_at_ms,
        predicted_at_ms=predicted_at_ms,
        model_id=model_id,
        model_sha256=model_sha256,
        config_id=config_id,
        config_sha256=config_sha256,
        predicted_label=predicted_label,
        probabilities=probabilities,
    )


def _validate_bbox(value: Any, context: str) -> None:
    bbox = _mapping(value, context)
    _exact_keys(bbox, {"height", "width", "x", "y"}, context)
    x = _probability(bbox["x"], f"{context}.x")
    y = _probability(bbox["y"], f"{context}.y")
    width = _probability(bbox["width"], f"{context}.width")
    height = _probability(bbox["height"], f"{context}.height")
    if width <= 0.0 or height <= 0.0 or x + width > 1.0 or y + height > 1.0:
        raise ContractError(f"{context} must be a positive box contained in normalized image space")


def _validate_unknown_behavior(value: Any, context: str) -> None:
    behavior = _mapping(value, context)
    _exact_keys(behavior, {"label", "raw_label", "scores"}, context)
    if behavior["label"] != "UNKNOWN" or behavior["raw_label"] != "OTHER_UNKNOWN":
        raise ContractError(f"{context} must use the perception UNKNOWN sentinel")
    scores = _mapping(behavior["scores"], f"{context}.scores")
    expected = {"CLEAR": 0.0, "DIGGING": 0.0, "EATING": 0.0, "UNKNOWN": 1.0}
    _exact_keys(scores, set(expected), f"{context}.scores")
    actual = {name: _probability(scores[name], f"{context}.scores.{name}") for name in expected}
    if actual != expected:
        raise ContractError(f"{context}.scores must be the fail-closed UNKNOWN sentinel")


def _validate_track(value: Any, index: int) -> tuple[str, str]:
    context = f"perception track {index}"
    track = _mapping(value, context)
    track_id = _identifier(track.get("track_id"), f"{context}.track_id")
    object_class = track.get("class")
    common = {
        "ambiguous",
        "bbox",
        "class",
        "detection_confidence",
        "track_age_ms",
        "track_id",
        "track_quality",
    }
    expected = common
    if object_class == "CAT":
        expected = common | {
            "aim_preset_id",
            "behavior",
            "no_fire_intersection",
            "region_evidence",
            "zone_id",
        }
    elif object_class != "PERSON":
        raise ContractError(f"{context} class must be CAT or PERSON")
    _exact_keys(track, expected, context)
    _probability(track["detection_confidence"], f"{context}.detection_confidence")
    _probability(track["track_quality"], f"{context}.track_quality")
    _integer(track["track_age_ms"], f"{context}.track_age_ms")
    _boolean(track["ambiguous"], f"{context}.ambiguous")
    _validate_bbox(track["bbox"], f"{context}.bbox")
    if object_class == "CAT":
        _validate_unknown_behavior(track["behavior"], f"{context}.behavior")
        if track["aim_preset_id"] is not None:
            raise ContractError(f"{context}.aim_preset_id must be null at the perception boundary")
        if track["zone_id"] is not None:
            _identifier(track["zone_id"], f"{context}.zone_id")
        _boolean(track["no_fire_intersection"], f"{context}.no_fire_intersection")
        evidence = _mapping(track["region_evidence"], f"{context}.region_evidence")
        evidence_fields = {
            "approach_overlap",
            "foliage_overlap",
            "motion_score",
            "soil_overlap",
        }
        _exact_keys(evidence, evidence_fields, f"{context}.region_evidence")
        for name in evidence_fields:
            _probability(evidence[name], f"{context}.region_evidence.{name}")
    return track_id, object_class


def _validate_zone_evidence(value: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ContractError("perception_observation.zone_evidence must be a list")
    evidence_by_track: dict[str, Mapping[str, Any]] = {}
    for index, raw_item in enumerate(value):
        context = f"zone_evidence[{index}]"
        item = _mapping(raw_item, context)
        _exact_keys(
            item,
            {"no_fire_overlap", "overlaps", "track_age_frames", "track_id"},
            context,
        )
        track_id = _identifier(item["track_id"], f"{context}.track_id")
        if track_id in evidence_by_track:
            raise ContractError("zone_evidence must contain each track exactly once")
        evidence_by_track[track_id] = item
        _probability(item["no_fire_overlap"], f"{context}.no_fire_overlap")
        _integer(item["track_age_frames"], f"{context}.track_age_frames")
        overlaps = item["overlaps"]
        if not isinstance(overlaps, list):
            raise ContractError(f"{context}.overlaps must be a list")
        zone_ids: list[str] = []
        for overlap_index, raw_overlap in enumerate(overlaps):
            overlap_context = f"{context}.overlaps[{overlap_index}]"
            overlap = _mapping(raw_overlap, overlap_context)
            _exact_keys(overlap, {"overlap", "zone_id", "zone_type"}, overlap_context)
            zone_ids.append(_identifier(overlap["zone_id"], f"{overlap_context}.zone_id"))
            _probability(overlap["overlap"], f"{overlap_context}.overlap")
            if overlap["zone_type"] not in {"approach", "foliage", "soil", "no_fire"}:
                raise ContractError(f"{overlap_context}.zone_type is invalid")
        if len(zone_ids) != len(set(zone_ids)):
            raise ContractError(f"{context}.overlaps contains a duplicate zone")
    return evidence_by_track


def _reconcile_cat_zone_evidence(
    track: Mapping[str, Any], evidence: Mapping[str, Any], context: str
) -> None:
    """Require the diagnostic and policy views of geometry to be identical.

    The policy consumes fields on the cat track, while the perception stream also
    carries a detailed per-zone view. Accepting contradictions would let the less
    conservative copy win at the policy boundary, so disagreement rejects the
    whole record instead of guessing which producer field is authoritative.
    """

    maxima = {zone_type: 0.0 for zone_type in ("approach", "foliage", "soil", "no_fire")}
    approach: list[tuple[float, str]] = []
    for raw_overlap in evidence["overlaps"]:
        overlap = float(raw_overlap["overlap"])
        zone_type = raw_overlap["zone_type"]
        maxima[zone_type] = max(maxima[zone_type], overlap)
        if zone_type == "approach" and overlap > 0.0:
            approach.append((overlap, raw_overlap["zone_id"]))

    if float(evidence["no_fire_overlap"]) != maxima["no_fire"]:
        raise ContractError(f"{context} zone_evidence.no_fire_overlap disagrees with overlaps")

    approach.sort(key=lambda item: (-item[0], item[1]))
    derived_zone_id = approach[0][1] if approach else None
    if track["zone_id"] != derived_zone_id:
        raise ContractError(f"{context}.zone_id disagrees with zone_evidence approach overlaps")

    region = track["region_evidence"]
    for field, zone_type in (
        ("approach_overlap", "approach"),
        ("foliage_overlap", "foliage"),
        ("soil_overlap", "soil"),
    ):
        if float(region[field]) != maxima[zone_type]:
            raise ContractError(f"{context}.region_evidence.{field} disagrees with zone_evidence")

    expected_no_fire = maxima["no_fire"] > 0.0
    if track["no_fire_intersection"] is not expected_no_fire:
        raise ContractError(f"{context}.no_fire_intersection disagrees with zone_evidence")


def parse_perception_observation(raw: Mapping[str, Any]) -> PerceptionObservation:
    data = _mapping(raw, "perception_observation")
    expected = {
        "behavior",
        "cat_count",
        "frame",
        "mode",
        "model",
        "observation",
        "person_present",
        "record_type",
        "schema_version",
        "sequence",
        "source",
        "would_action",
        "zone_evidence",
    }
    _exact_keys(data, expected, "perception_observation")
    _schema_version(data["schema_version"], "perception_observation.schema_version")
    if data["record_type"] != "perception_observation":
        raise ContractError("perception_observation.record_type is invalid")
    if (
        data["mode"] != "OBSERVE_ONLY"
        or data["would_action"] is not False
        or data["behavior"] != "UNKNOWN"
    ):
        raise ContractError("perception_observation must be observe-only and behavior UNKNOWN")
    sequence = _integer(data["sequence"], "perception_observation.sequence")

    model = _mapping(data["model"], "perception_observation.model")
    if set(model) not in ({"id"}, {"id", "sha256"}):
        raise ContractError("perception_observation.model has invalid fields")
    _identifier(model.get("id"), "perception_observation.model.id")
    if "sha256" in model:
        _sha256(model["sha256"], "perception_observation.model.sha256")
    source = _mapping(data["source"], "perception_observation.source")
    _exact_keys(source, {"kind", "name"}, "perception_observation.source")
    if not all(isinstance(source[key], str) and source[key] for key in ("kind", "name")):
        raise ContractError("perception_observation.source values must be non-empty strings")

    frame = _mapping(data["frame"], "perception_observation.frame")
    _exact_keys(frame, {"height", "index", "width"}, "perception_observation.frame")
    frame_index = _integer(frame["index"], "perception_observation.frame.index")
    if frame_index != sequence:
        raise ContractError("perception frame.index must equal sequence")
    for dimension in ("height", "width"):
        if _integer(frame[dimension], f"perception_observation.frame.{dimension}") <= 0:
            raise ContractError(f"perception_observation.frame.{dimension} must be positive")

    observation = _mapping(data["observation"], "perception_observation.observation")
    _exact_keys(
        observation,
        {"camera_id", "captured_at_ms", "frame_id", "observation_id", "tracks"},
        "perception_observation.observation",
    )
    tracks = observation["tracks"]
    if not isinstance(tracks, list) or len(tracks) > 64:
        raise ContractError(
            "perception_observation.observation.tracks must be a list of at most 64"
        )
    track_ids: list[str] = []
    cats = 0
    people = 0
    for index, raw_track in enumerate(tracks):
        track_id, object_class = _validate_track(raw_track, index)
        track_ids.append(track_id)
        if object_class == "CAT":
            cats += 1
        else:
            people += 1
    if len(track_ids) != len(set(track_ids)):
        raise ContractError("perception observation contains duplicate track IDs")
    cat_count = _integer(data["cat_count"], "perception_observation.cat_count")
    person_present = _boolean(data["person_present"], "perception_observation.person_present")
    if cat_count != cats or person_present is not (people > 0):
        raise ContractError("perception top-level cat/person summary disagrees with tracks")
    evidence_by_track = _validate_zone_evidence(data["zone_evidence"])
    if sorted(evidence_by_track) != sorted(track_ids):
        raise ContractError("zone_evidence must contain each track exactly once")
    for index, raw_track in enumerate(tracks):
        if raw_track["class"] == "CAT":
            _reconcile_cat_zone_evidence(
                raw_track,
                evidence_by_track[raw_track["track_id"]],
                f"perception track {index}",
            )

    return PerceptionObservation(
        sequence=sequence,
        observation_id=_identifier(
            observation["observation_id"], "perception_observation.observation_id"
        ),
        frame_id=_identifier(observation["frame_id"], "perception_observation.frame_id"),
        captured_at_ms=_integer(
            observation["captured_at_ms"], "perception_observation.captured_at_ms"
        ),
        camera_id=_identifier(observation["camera_id"], "perception_observation.camera_id"),
        record=deepcopy(dict(data)),
    )


def _validate_perception_order(records: list[PerceptionObservation]) -> None:
    order = [(record.captured_at_ms, record.sequence) for record in records]
    if order != sorted(order) or len(order) != len(set(order)):
        raise ContractError(
            "perception stream must be strictly ordered by (captured_at_ms, sequence)"
        )
    sequences = [record.sequence for record in records]
    if sequences != sorted(sequences) or len(set(sequences)) != len(records):
        raise ContractError("perception sequence values must be strictly increasing")
    if len({record.observation_id for record in records}) != len(records):
        raise ContractError("perception stream contains a duplicate observation_id")
    if len({record.frame_id for record in records}) != len(records):
        raise ContractError("perception stream contains a duplicate frame_id")
    if len({record.camera_id for record in records}) > 1:
        raise ContractError("perception stream contains more than one camera_id")


def _validate_behavior_order(records: list[BehaviorPrediction]) -> None:
    order = [(record.predicted_at_ms, record.sequence) for record in records]
    if order != sorted(order) or len(order) != len(set(order)):
        raise ContractError(
            "behavior stream must be strictly ordered by (predicted_at_ms, sequence)"
        )
    sequences = [record.sequence for record in records]
    if sequences != sorted(sequences) or len(set(sequences)) != len(records):
        raise ContractError("behavior sequence values must be strictly increasing")
    if len({record.key for record in records}) != len(records):
        raise ContractError("behavior stream contains a duplicate observation/frame/track key")
    identities = {
        (
            record.model_id,
            record.model_sha256,
            record.config_id,
            record.config_sha256,
        )
        for record in records
    }
    if len(identities) > 1:
        raise ContractError("behavior stream mixes model or inference-config identities")


def parse_perception_stream(
    values: Iterable[Mapping[str, Any]],
) -> list[PerceptionObservation]:
    records = [parse_perception_observation(value) for value in values]
    if not records:
        raise ContractError("perception stream is empty")
    _validate_perception_order(records)
    return records


def parse_behavior_stream(values: Iterable[Mapping[str, Any]]) -> list[BehaviorPrediction]:
    records = [parse_behavior_prediction(value) for value in values]
    _validate_behavior_order(records)
    return records


def read_jsonl(path: str | Path, parser: Callable[[Iterable[Mapping[str, Any]]], T]) -> T:
    source = Path(path)
    rows: list[Mapping[str, Any]] = []
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ContractError(f"{source}:{line_number}: blank JSONL lines are forbidden")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ContractError(f"{source}:{line_number}: record must be an object")
                rows.append(value)
    except json.JSONDecodeError as error:
        raise ContractError(f"{source}:{error.lineno}: invalid JSON: {error.msg}") from error
    except OSError as error:
        raise ContractError(f"{source}: {error}") from error
    try:
        return parser(rows)
    except ContractError as error:
        raise ContractError(f"{source}: {error}") from error
