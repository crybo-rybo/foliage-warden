"""Fail-closed behavior fusion into canonical simulator observations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from foliage_warden_sim.validation import validate_runtime_semantics

from .contracts import (
    BehaviorPrediction,
    ContractError,
    JsonObject,
    PerceptionObservation,
    canonical_stream_sha256,
)

MAX_AUDIT_AGE_MS = 30_000


@dataclass(frozen=True, slots=True)
class FusionOptions:
    """Timing bounds for a deterministic asynchronous behavior join."""

    prediction_timeout_ms: int = 50
    max_prediction_latency_ms: int = 250

    def __post_init__(self) -> None:
        for name, value in (
            ("prediction_timeout_ms", self.prediction_timeout_ms),
            ("max_prediction_latency_ms", self.max_prediction_latency_ms),
        ):
            valid = (
                not isinstance(value, bool)
                and isinstance(value, int)
                and 0 <= value <= MAX_AUDIT_AGE_MS
            )
            if not valid:
                raise ContractError(f"{name} must be an integer within [0, {MAX_AUDIT_AGE_MS}]")


@dataclass(frozen=True, slots=True)
class FusedFrame:
    delivery_at_ms: int
    perception_sequence: int
    observation: JsonObject


@dataclass(frozen=True, slots=True)
class FusionDiagnostic:
    status: str
    observation_id: str
    frame_id: str
    track_id: str
    perception_sequence: int | None = None
    prediction_sequence: int | None = None
    latency_ms: int | None = None

    def to_dict(self) -> JsonObject:
        result: JsonObject = {
            "frame_id": self.frame_id,
            "observation_id": self.observation_id,
            "status": self.status,
            "track_id": self.track_id,
        }
        if self.perception_sequence is not None:
            result["perception_sequence"] = self.perception_sequence
        if self.prediction_sequence is not None:
            result["prediction_sequence"] = self.prediction_sequence
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        return result


@dataclass(frozen=True, slots=True)
class FusionResult:
    frames: tuple[FusedFrame, ...]
    diagnostics: tuple[FusionDiagnostic, ...]
    perception_sha256: str
    behavior_sha256: str
    behavior_identity: JsonObject | None

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for diagnostic in self.diagnostics:
            counts[diagnostic.status] = counts.get(diagnostic.status, 0) + 1
        return dict(sorted(counts.items()))


def _unknown_behavior() -> JsonObject:
    return {
        "label": "UNKNOWN",
        "raw_label": "OTHER_UNKNOWN",
        "scores": {
            "CLEAR": 0.0,
            "DIGGING": 0.0,
            "EATING": 0.0,
            "UNKNOWN": 1.0,
        },
    }


def _canonical_behavior(prediction: BehaviorPrediction) -> JsonObject:
    probabilities = prediction.probabilities
    label_map = {
        "PASSING": "CLEAR",
        "SNIFFING": "CLEAR",
        "EATING": "EATING",
        "DIGGING": "DIGGING",
        # OTHER is not evidence of a safe/clear scene. It joins UNKNOWN.
        "OTHER": "UNKNOWN",
        "UNKNOWN": "UNKNOWN",
    }
    raw_label = (
        prediction.predicted_label
        if prediction.predicted_label in {"PASSING", "SNIFFING", "EATING", "DIGGING"}
        else "OTHER_UNKNOWN"
    )
    return {
        "label": label_map[prediction.predicted_label],
        "model_id": prediction.model_id,
        "raw_label": raw_label,
        "scores": {
            "CLEAR": probabilities["PASSING"] + probabilities["SNIFFING"],
            "DIGGING": probabilities["DIGGING"],
            "EATING": probabilities["EATING"],
            "UNKNOWN": probabilities["OTHER"] + probabilities["UNKNOWN"],
        },
    }


def _safe_preset(config: JsonObject, zone_id: Any) -> str | None:
    if not isinstance(zone_id, str):
        return None
    candidates = sorted(
        preset["id"] for preset in config["scene"]["aim_presets"] if preset["zone_id"] == zone_id
    )
    return candidates[0] if len(candidates) == 1 else None


def _validate_config(config: JsonObject) -> None:
    try:
        validate_runtime_semantics(config, "shadow runtime config")
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"unsafe shadow runtime config: {error}") from error
    actuator = config["actuator"]
    if actuator["backend"] != "MOCK" or actuator["allow_physical_effects"]:
        raise ContractError("shadow execution requires MOCK with physical effects disabled")


def _validate_zone_identities(perceptions: list[PerceptionObservation], config: JsonObject) -> None:
    configured_types = {zone["id"]: zone["type"] for zone in config["scene"]["zones"]}
    for perception in perceptions:
        for evidence in perception.record["zone_evidence"]:
            for overlap in evidence["overlaps"]:
                zone_id = overlap["zone_id"]
                configured_type = configured_types.get(zone_id)
                if configured_type is None:
                    raise ContractError(
                        f"observation {perception.observation_id!r} references unknown zone "
                        f"{zone_id!r}"
                    )
                if overlap["zone_type"] != configured_type:
                    raise ContractError(
                        f"observation {perception.observation_id!r} describes zone {zone_id!r} "
                        f"as {overlap['zone_type']!r}; runtime config says {configured_type!r}"
                    )


def fuse_streams(
    perceptions: list[PerceptionObservation],
    predictions: list[BehaviorPrediction],
    runtime_config: JsonObject,
    *,
    options: FusionOptions | None = None,
) -> FusionResult:
    """Join only exact identities; missing, late, and mismatched evidence becomes UNKNOWN."""

    if not perceptions:
        raise ContractError("perception stream is empty")
    options = options or FusionOptions()
    config = deepcopy(runtime_config)
    _validate_config(config)
    _validate_zone_identities(perceptions, config)
    expected_camera = config["camera"]["camera_id"]
    if any(record.camera_id != expected_camera for record in perceptions):
        raise ContractError(f"perception camera_id must be {expected_camera!r}")

    predictions_by_key = {prediction.key: prediction for prediction in predictions}
    consumed: set[tuple[str, str, str]] = set()
    frames: list[FusedFrame] = []
    diagnostics: list[FusionDiagnostic] = []
    previous_delivery_at_ms = 0

    for perception in perceptions:
        observation = perception.policy_observation()
        deadlines = [perception.captured_at_ms]
        for track in observation["tracks"]:
            if track["class"] != "CAT":
                continue
            key = (perception.observation_id, perception.frame_id, track["track_id"])
            prediction = predictions_by_key.get(key)
            status: str
            behavior: JsonObject
            prediction_sequence: int | None = None
            latency_ms: int | None = None
            if prediction is None:
                status = "MISSING"
                behavior = _unknown_behavior()
                deadlines.append(perception.captured_at_ms + options.prediction_timeout_ms)
            else:
                consumed.add(key)
                prediction_sequence = prediction.sequence
                latency_ms = prediction.predicted_at_ms - perception.captured_at_ms
                if prediction.captured_at_ms != perception.captured_at_ms:
                    status = "CAPTURE_MISMATCH"
                    behavior = _unknown_behavior()
                    deadlines.append(perception.captured_at_ms + options.prediction_timeout_ms)
                elif latency_ms > options.prediction_timeout_ms:
                    status = "TIMED_OUT"
                    behavior = _unknown_behavior()
                    deadlines.append(perception.captured_at_ms + options.prediction_timeout_ms)
                elif latency_ms > options.max_prediction_latency_ms:
                    status = "LATE"
                    behavior = _unknown_behavior()
                    deadlines.append(prediction.predicted_at_ms)
                else:
                    status = "FUSED"
                    behavior = _canonical_behavior(prediction)
                    deadlines.append(prediction.predicted_at_ms)
            track["behavior"] = behavior
            track["aim_preset_id"] = _safe_preset(config, track.get("zone_id"))
            diagnostics.append(
                FusionDiagnostic(
                    status=status,
                    observation_id=perception.observation_id,
                    frame_id=perception.frame_id,
                    track_id=track["track_id"],
                    perception_sequence=perception.sequence,
                    prediction_sequence=prediction_sequence,
                    latency_ms=latency_ms,
                )
            )

        # Preserve capture order even when behavior completions cross. A later
        # frame may wait behind an earlier deadline, but an older observation is
        # never replayed after a newer one and therefore cannot resurrect stale
        # evidence in the temporal policy.
        delivery_at_ms = max(max(deadlines), previous_delivery_at_ms)
        previous_delivery_at_ms = delivery_at_ms
        if delivery_at_ms - perception.captured_at_ms > MAX_AUDIT_AGE_MS:
            raise ContractError(
                f"observation {perception.observation_id!r} exceeds the 30-second "
                "replay audit bound"
            )
        frames.append(FusedFrame(delivery_at_ms, perception.sequence, observation))

    for prediction in predictions:
        if prediction.key in consumed:
            continue
        diagnostics.append(
            FusionDiagnostic(
                status="UNMATCHED_PREDICTION",
                observation_id=prediction.observation_id,
                frame_id=prediction.frame_id,
                track_id=prediction.track_id,
                prediction_sequence=prediction.sequence,
                latency_ms=prediction.predicted_at_ms - prediction.captured_at_ms,
            )
        )

    frames.sort(
        key=lambda item: (
            item.delivery_at_ms,
            item.perception_sequence,
            item.observation["observation_id"],
        )
    )
    diagnostics.sort(
        key=lambda item: (
            item.prediction_sequence
            if item.prediction_sequence is not None
            else 9_007_199_254_740_991,
            item.perception_sequence
            if item.perception_sequence is not None
            else 9_007_199_254_740_991,
            item.observation_id,
            item.frame_id,
            item.track_id,
            item.status,
        )
    )
    identity = None
    if predictions:
        first = predictions[0]
        identity = {
            "config": {"id": first.config_id, "sha256": first.config_sha256},
            "model": {"id": first.model_id, "sha256": first.model_sha256},
        }
    return FusionResult(
        frames=tuple(frames),
        diagnostics=tuple(diagnostics),
        perception_sha256=canonical_stream_sha256(perceptions),
        behavior_sha256=canonical_stream_sha256(predictions),
        behavior_identity=identity,
    )


def assemble_scenario(
    fusion: FusionResult,
    runtime_config: JsonObject,
    config_path: str | Path,
    *,
    scenario_id: str,
) -> JsonObject:
    """Build a schema-v1 simulator replay with mock transport and placeholder assertions."""

    from .contracts import _identifier

    scenario_id = _identifier(scenario_id, "scenario_id")
    _validate_config(runtime_config)
    timeline: list[JsonObject] = [
        {
            "at_ms": 0,
            "event_id": "shadow-arm",
            "operation": "ARM",
            "sequence": 0,
            "type": "CONTROL",
        }
    ]
    for index, frame in enumerate(fusion.frames, 1):
        timeline.append(
            {
                "at_ms": frame.delivery_at_ms,
                "event_id": f"shadow-observation-{index:06d}",
                "observation": frame.observation,
                "sequence": index,
                "type": "OBSERVATION",
            }
        )
    final_at_ms = max(item["at_ms"] for item in timeline) + 10
    timeline.append(
        {
            "at_ms": final_at_ms,
            "event_id": "shadow-settle",
            "sequence": len(timeline),
            "type": "TICK",
        }
    )
    timeline.sort(key=lambda item: (item["at_ms"], item["sequence"], item["event_id"]))
    return {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "title": "Observe-only perception and behavior shadow replay",
        "description": (
            "Versioned deterministic replay assembled from strict perception and behavior JSONL. "
            "It uses only the virtual clock and mock actuator; no physical effect is possible."
        ),
        "requirement_ids": ["REQ-MOCK-ONLY", "REQ-SHADOW-FUSION"],
        "tags": ["behavior", "observe-only", "shadow"],
        "config_ref": str(Path(config_path).resolve()),
        "initial_conditions": {
            "clock_ms": 0,
            "policy_state": "DISARMED",
            "armed": False,
            "camera_status": "CONNECTED",
            "actuator_status": "READY",
            "command_id_seed": 1,
        },
        "actuator_script": {
            "default_response": "ACK",
            "ack_delay_ms": 1,
            "deduplicate_command_ids": True,
            "overrides": [],
        },
        "timeline": timeline,
        # The runner replaces these replay-derived fields, then executes and verifies again.
        "expectations": {
            "explanation": "Provisional replay expectations; finalized before output.",
            "final_state": "MONITORING",
            "exact_counts": {},
            "required_states": [],
            "forbidden_states": ["FAULT"],
            "required_reason_codes": [],
            "forbidden_commands": [],
            "expected_action_sequence": [],
            "invariants": ["STARTS_DISARMED", "NO_PHYSICAL_EFFECTS"],
        },
    }
