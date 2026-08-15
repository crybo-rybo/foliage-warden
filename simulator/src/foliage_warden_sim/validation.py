"""JSON Schema and cross-field validation for deterministic simulation inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import rfc8785
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from .resources import contract_root_for
from .types import JsonObject, ScheduledInput

MAX_SAFE_INTEGER = 9_007_199_254_740_991


class ContractError(ValueError):
    """A config or scenario is structurally or semantically unsafe."""


@dataclass(frozen=True, slots=True)
class LoadedContracts:
    scenario_path: Path
    config_path: Path
    scenario: JsonObject
    config: JsonObject
    config_sha256: str
    inputs: tuple[ScheduledInput, ...]


def _read_object(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{path}: JSON root must be an object")
    return value


def _registry(schema_dir: Path) -> tuple[Registry, dict[str, JsonObject]]:
    registry = Registry()
    schemas: dict[str, JsonObject] = {}
    paths = sorted(schema_dir.glob("*.schema.json"))
    if not paths:
        raise ContractError(f"no JSON schemas found in {schema_dir}")
    for path in paths:
        schema = _read_object(path)
        try:
            Draft202012Validator.check_schema(schema)
            registry = registry.with_resource(
                schema["$id"], Resource.from_contents(schema)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(f"invalid schema {path}: {error}") from error
        schemas[path.name] = schema
    return registry, schemas


def _validate_schema(
    instance: JsonObject,
    schema: JsonObject,
    registry: Registry,
    path: Path,
) -> None:
    errors = sorted(
        Draft202012Validator(schema, registry=registry).iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    details: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise ContractError(f"{path}:\n  " + "\n  ".join(details))


def _orientation(a: JsonObject, b: JsonObject, c: JsonObject) -> float:
    return (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])


def _on_segment(a: JsonObject, b: JsonObject, p: JsonObject) -> bool:
    epsilon = 1e-12
    return (
        min(a["x"], b["x"]) - epsilon <= p["x"] <= max(a["x"], b["x"]) + epsilon
        and min(a["y"], b["y"]) - epsilon <= p["y"] <= max(a["y"], b["y"]) + epsilon
        and abs(_orientation(a, b, p)) <= epsilon
    )


def _segments_intersect(
    a: JsonObject, b: JsonObject, c: JsonObject, d: JsonObject
) -> bool:
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if ((ab_c > 0 > ab_d) or (ab_d > 0 > ab_c)) and (
        (cd_a > 0 > cd_b) or (cd_b > 0 > cd_a)
    ):
        return True
    return any(
        abs(value) <= 1e-12 and _on_segment(first, second, point)
        for value, first, second, point in (
            (ab_c, a, b, c),
            (ab_d, a, b, d),
            (cd_a, c, d, a),
            (cd_b, c, d, b),
        )
    )


def _validate_polygon(points: list[JsonObject], context: str) -> None:
    twice_area = sum(
        point["x"] * points[(index + 1) % len(points)]["y"]
        - points[(index + 1) % len(points)]["x"] * point["y"]
        for index, point in enumerate(points)
    )
    if abs(twice_area) < 0.0002:
        raise ContractError(f"{context}: polygon has negligible area")
    edges = [
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    for left, (a, b) in enumerate(edges):
        for right, (c, d) in enumerate(edges):
            if right <= left or right == left + 1:
                continue
            if left == 0 and right == len(edges) - 1:
                continue
            if _segments_intersect(a, b, c, d):
                raise ContractError(f"{context}: polygon self-intersects")


def _unique(values: Iterable[str], context: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ContractError(f"{context}: duplicate ID {value!r}")
        seen.add(value)
    return seen


def validate_runtime_semantics(config: JsonObject, context: str = "config") -> None:
    zones = config["scene"]["zones"]
    zone_ids = _unique((zone["id"] for zone in zones), f"{context}.scene.zones")
    zone_by_id = {zone["id"]: zone for zone in zones}
    for zone in zones:
        _validate_polygon(zone["points"], f"{context}.scene.zones[{zone['id']}]")
    evidence_by_plant = {
        (zone.get("plant_id"), zone["type"])
        for zone in zones
        if zone["type"] in {"foliage", "soil"}
    }
    for zone in zones:
        if zone["type"] != "approach":
            continue
        plant_id = zone.get("plant_id")
        if plant_id is None or any(
            (plant_id, evidence_type) not in evidence_by_plant
            for evidence_type in ("foliage", "soil")
        ):
            raise ContractError(
                f"{context}: approach zone {zone['id']!r} requires foliage and soil evidence zones"
            )

    presets = config["scene"]["aim_presets"]
    _unique((preset["id"] for preset in presets), f"{context}.scene.aim_presets")
    for preset in presets:
        zone_id = preset["zone_id"]
        if zone_id not in zone_ids:
            raise ContractError(f"{context}: preset {preset['id']!r} has no zone")
        if zone_by_id[zone_id]["type"] != "approach":
            raise ContractError(
                f"{context}: preset {preset['id']!r} must reference an approach zone"
            )

    burst = config["actuator"]["burst"]
    if burst["duration_ms"] > burst["hardware_max_duration_ms"]:
        raise ContractError(f"{context}: burst duration exceeds hardware clamp")

    runtime = config["runtime"]
    actuator = config["actuator"]
    if runtime["mode"] != "LIVE" and (
        actuator["allow_physical_effects"] or actuator["backend"] == "SERIAL"
    ):
        raise ContractError(f"{context}: simulation exposes a physical actuator path")
    if runtime["mode"] != "SIMULATION":
        raise ContractError(
            f"{context}: reference scenario runner requires SIMULATION mode"
        )
    if config["clock"]["source"] != "VIRTUAL":
        raise ContractError(
            f"{context}: reference scenario runner requires a VIRTUAL clock"
        )
    if config["camera"]["source"] != "SCRIPTED":
        raise ContractError(
            f"{context}: reference scenario runner requires a SCRIPTED camera"
        )
    if actuator["backend"] != "MOCK":
        raise ContractError(
            f"{context}: reference scenario runner requires the MOCK backend"
        )
    if actuator["allow_physical_effects"]:
        raise ContractError(f"{context}: physical effects must be disabled")


def _tracks(event: JsonObject) -> Iterable[JsonObject]:
    if event["type"] == "OBSERVATION":
        return event["observation"]["tracks"]
    if event["type"] == "OBSERVATION_SERIES":
        return event["template"]["tracks"]
    return ()


def validate_scenario_semantics(
    scenario: JsonObject, context: str = "scenario"
) -> None:
    timeline = scenario["timeline"]
    _unique((event["event_id"] for event in timeline), f"{context}.timeline")
    order_keys = [(event["at_ms"], event["sequence"]) for event in timeline]
    if len(order_keys) != len(set(order_keys)):
        raise ContractError(f"{context}: duplicate (at_ms, sequence) keys")
    if order_keys != sorted(order_keys):
        raise ContractError(f"{context}: timeline is not in deterministic order")

    observation_count = 0
    for event in timeline:
        if event["type"] == "OBSERVATION_SERIES":
            observation_count += event["count"]
            last_delivery = event["at_ms"] + (event["count"] - 1) * event["interval_ms"]
            if last_delivery > MAX_SAFE_INTEGER:
                raise ContractError(
                    f"{context}.{event['event_id']}: delivery time overflows"
                )
            if event["at_ms"] + event["capture_offset_ms"] < 0:
                raise ContractError(
                    f"{context}.{event['event_id']}: generated capture time is negative"
                )
        elif event["type"] == "OBSERVATION":
            observation_count += 1
            if event["observation"]["captured_at_ms"] > event["at_ms"]:
                raise ContractError(
                    f"{context}.{event['event_id']}: capture time is in the future"
                )
            if event["at_ms"] - event["observation"]["captured_at_ms"] > 30_000:
                raise ContractError(
                    f"{context}.{event['event_id']}: frame age exceeds the audit contract"
                )

        tracks = list(_tracks(event))
        _unique(
            (track["track_id"] for track in tracks),
            f"{context}.{event['event_id']}.tracks",
        )
        for track in tracks:
            bbox = track["bbox"]
            if (
                bbox["x"] + bbox["width"] > 1.0 + 1e-12
                or bbox["y"] + bbox["height"] > 1.0 + 1e-12
            ):
                raise ContractError(
                    f"{context}.{event['event_id']}.{track['track_id']}: bbox exceeds image"
                )
            behavior = track.get("behavior")
            if behavior is not None:
                total = sum(behavior["scores"].values())
                if not 0.99 <= total <= 1.01:
                    raise ContractError(
                        f"{context}.{event['event_id']}.{track['track_id']}: "
                        f"behavior scores sum to {total}"
                    )

    counts = scenario["expectations"]["exact_counts"]
    if counts.get("physical_bursts", 0) != 0:
        raise ContractError(f"{context}: simulation may not expect a physical burst")
    if counts.get("automatic_retries", 0) != 0:
        raise ContractError(f"{context}: simulation may not expect a BURST retry")

    seed = scenario["initial_conditions"]["command_id_seed"]
    if seed + observation_count * 2 > MAX_SAFE_INTEGER:
        raise ContractError(f"{context}: command-ID seed can overflow after expansion")


def expand_timeline(scenario: JsonObject) -> tuple[ScheduledInput, ...]:
    expanded: list[ScheduledInput] = []
    for source in scenario["timeline"]:
        if source["type"] != "OBSERVATION_SERIES":
            expanded.append(
                ScheduledInput(
                    source["at_ms"],
                    source["sequence"],
                    source["event_id"],
                    deepcopy(source),
                )
            )
            continue

        for index in range(source["count"]):
            at_ms = source["at_ms"] + index * source["interval_ms"]
            captured_at_ms = at_ms + source["capture_offset_ms"]
            if not 0 <= captured_at_ms <= at_ms <= MAX_SAFE_INTEGER:
                raise ContractError(
                    f"{scenario['scenario_id']}.{source['event_id']}: invalid generated time"
                )
            generated_id = f"{source['id_prefix']}-{index:06d}"
            observation = deepcopy(source["template"])
            observation.update(
                {
                    "observation_id": generated_id,
                    "frame_id": generated_id,
                    "captured_at_ms": captured_at_ms,
                }
            )
            for track in observation["tracks"]:
                initial_age = track.pop("initial_track_age_ms")
                track_age = initial_age + index * source["interval_ms"]
                if track_age > MAX_SAFE_INTEGER:
                    raise ContractError(
                        f"{scenario['scenario_id']}.{source['event_id']}: track age overflows"
                    )
                track["track_age_ms"] = track_age
            payload: JsonObject = {
                "event_id": generated_id,
                "at_ms": at_ms,
                "sequence": source["sequence"],
                "type": "OBSERVATION",
                "observation": observation,
            }
            expanded.append(
                ScheduledInput(at_ms, source["sequence"], generated_id, payload)
            )

    expanded.sort(key=lambda item: (item.at_ms, item.sequence, item.event_id))
    event_ids: set[str] = set()
    order_keys: set[tuple[int, int]] = set()
    observation_ids: set[str] = set()
    frame_ids: set[str] = set()
    for event in expanded:
        if event.event_id in event_ids:
            raise ContractError(
                f"expanded timeline has duplicate event ID {event.event_id!r}"
            )
        event_ids.add(event.event_id)
        key = (event.at_ms, event.sequence)
        if key in order_keys:
            raise ContractError(f"expanded timeline has duplicate order key {key}")
        order_keys.add(key)
        if event.payload["type"] == "OBSERVATION":
            observation = event.payload["observation"]
            observation_id = observation["observation_id"]
            frame_id = observation["frame_id"]
            if observation_id in observation_ids:
                raise ContractError(
                    f"expanded timeline has duplicate observation ID {observation_id!r}"
                )
            if frame_id in frame_ids:
                raise ContractError(
                    f"expanded timeline has duplicate frame ID {frame_id!r}"
                )
            observation_ids.add(observation_id)
            frame_ids.add(frame_id)
    return tuple(expanded)


def load_contracts(
    scenario_path: str | Path,
    *,
    config_path: str | Path | None = None,
    schema_dir: str | Path | None = None,
) -> LoadedContracts:
    scenario_path = Path(scenario_path).resolve()
    scenario = _read_object(scenario_path)
    contract_root = contract_root_for(scenario_path)
    schemas_path = (
        Path(schema_dir).resolve() if schema_dir else contract_root / "schemas"
    )
    registry, schemas = _registry(schemas_path)
    try:
        scenario_schema = schemas["scenario.schema.json"]
        config_schema = schemas["runtime-config.schema.json"]
    except KeyError as error:
        raise ContractError(
            f"required schema is missing from {schemas_path}"
        ) from error
    _validate_schema(scenario, scenario_schema, registry, scenario_path)
    validate_scenario_semantics(scenario, str(scenario_path))

    resolved_config = (
        Path(config_path).resolve()
        if config_path is not None
        else (scenario_path.parent / scenario["config_ref"]).resolve()
    )
    if config_path is None and not resolved_config.is_relative_to(contract_root):
        raise ContractError(
            "scenario config_ref must resolve inside its trusted contract root"
        )
    config = _read_object(resolved_config)
    _validate_schema(config, config_schema, registry, resolved_config)
    validate_runtime_semantics(config, str(resolved_config))
    inputs = expand_timeline(scenario)
    expected_camera_id = config["camera"]["camera_id"]
    for item in inputs:
        if item.payload["type"] != "OBSERVATION":
            continue
        actual_camera_id = item.payload["observation"]["camera_id"]
        if actual_camera_id != expected_camera_id:
            raise ContractError(
                f"{scenario_path}: event {item.event_id!r} uses camera {actual_camera_id!r}; "
                f"expected {expected_camera_id!r}"
            )
    digest = sha256(rfc8785.dumps(config)).hexdigest()
    return LoadedContracts(
        scenario_path=scenario_path,
        config_path=resolved_config,
        scenario=scenario,
        config=config,
        config_sha256=digest,
        inputs=inputs,
    )
