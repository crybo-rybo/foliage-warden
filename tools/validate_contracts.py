"""Validate JSON contracts and cross-field safety semantics."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"


class ContractError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path.relative_to(ROOT)}: root must be an object")
    return value


def schema_registry(schemas: Iterable[dict[str, Any]]) -> Registry:
    registry = Registry()
    for schema in schemas:
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def validate_instance(
    path: Path,
    instance: dict[str, Any],
    schema: dict[str, Any],
    registry: Registry,
) -> None:
    validator = Draft202012Validator(schema, registry=registry)
    errors = sorted(
        validator.iter_errors(instance), key=lambda item: list(item.absolute_path)
    )
    if not errors:
        return
    details = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise ContractError(f"{path.relative_to(ROOT)}:\n  " + "\n  ".join(details))


def orientation(a: dict[str, float], b: dict[str, float], c: dict[str, float]) -> float:
    return (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])


def on_segment(a: dict[str, float], b: dict[str, float], p: dict[str, float]) -> bool:
    epsilon = 1e-12
    return (
        min(a["x"], b["x"]) - epsilon <= p["x"] <= max(a["x"], b["x"]) + epsilon
        and min(a["y"], b["y"]) - epsilon <= p["y"] <= max(a["y"], b["y"]) + epsilon
        and abs(orientation(a, b, p)) <= epsilon
    )


def segments_intersect(
    a: dict[str, float],
    b: dict[str, float],
    c: dict[str, float],
    d: dict[str, float],
) -> bool:
    ab_c, ab_d = orientation(a, b, c), orientation(a, b, d)
    cd_a, cd_b = orientation(c, d, a), orientation(c, d, b)
    if ((ab_c > 0 > ab_d) or (ab_d > 0 > ab_c)) and (
        (cd_a > 0 > cd_b) or (cd_b > 0 > cd_a)
    ):
        return True
    return any(
        (
            abs(value) <= 1e-12,
            on_segment(first, second, point),
        )
        == (True, True)
        for value, first, second, point in (
            (ab_c, a, b, c),
            (ab_d, a, b, d),
            (cd_a, c, d, a),
            (cd_b, c, d, b),
        )
    )


def validate_polygon(points: list[dict[str, float]], context: str) -> None:
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
    for left_index, (a, b) in enumerate(edges):
        for right_index, (c, d) in enumerate(edges):
            if right_index <= left_index:
                continue
            if right_index in (left_index + 1, (left_index - 1) % len(edges)):
                continue
            if left_index == 0 and right_index == len(edges) - 1:
                continue
            if segments_intersect(a, b, c, d):
                raise ContractError(f"{context}: polygon self-intersects")


def unique(values: Iterable[str], context: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ContractError(f"{context}: duplicate ID {value!r}")
        seen.add(value)
    return seen


def validate_runtime_semantics(config: dict[str, Any], context: str) -> None:
    scene = config["scene"]
    zones = scene["zones"]
    zone_ids = unique((zone["id"] for zone in zones), f"{context}.scene.zones")
    zone_by_id = {zone["id"]: zone for zone in zones}
    for zone in zones:
        validate_polygon(zone["points"], f"{context}.scene.zones[{zone['id']}]")
    unique(
        (preset["id"] for preset in scene["aim_presets"]),
        f"{context}.scene.aim_presets",
    )
    for preset in scene["aim_presets"]:
        if preset["zone_id"] not in zone_ids:
            raise ContractError(
                f"{context}: aim preset {preset['id']!r} references a missing zone"
            )
        if zone_by_id[preset["zone_id"]]["type"] != "approach":
            raise ContractError(
                f"{context}: aim preset {preset['id']!r} must reference an approach zone"
            )
    burst = config["actuator"]["burst"]
    if burst["duration_ms"] > burst["hardware_max_duration_ms"]:
        raise ContractError(f"{context}: burst duration exceeds the hardware clamp")
    if config["runtime"]["mode"] != "LIVE":
        actuator = config["actuator"]
        if actuator["allow_physical_effects"] or actuator["backend"] == "SERIAL":
            raise ContractError(
                f"{context}: non-LIVE mode exposes a physical actuator path"
            )


def observation_tracks(event: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if event["type"] == "OBSERVATION":
        return event["observation"]["tracks"]
    if event["type"] == "OBSERVATION_SERIES":
        return event["template"]["tracks"]
    return ()


def validate_scenario_semantics(scenario: dict[str, Any], path: Path) -> None:
    context = str(path.relative_to(ROOT))
    timeline = scenario["timeline"]
    unique((event["event_id"] for event in timeline), f"{context}.timeline")
    order_keys = [(event["at_ms"], event["sequence"]) for event in timeline]
    if len(order_keys) != len(set(order_keys)):
        raise ContractError(f"{context}: timeline has duplicate (at_ms, sequence) keys")
    if order_keys != sorted(order_keys):
        raise ContractError(f"{context}: timeline is not in deterministic order")

    for event in timeline:
        if event["type"] == "OBSERVATION_SERIES":
            first_capture = event["at_ms"] + event["capture_offset_ms"]
            if first_capture < 0:
                raise ContractError(
                    f"{context}.{event['event_id']}: generated capture time is negative"
                )
        for track in observation_tracks(event):
            bbox = track["bbox"]
            if (
                bbox["x"] + bbox["width"] > 1.0 + 1e-12
                or bbox["y"] + bbox["height"] > 1.0 + 1e-12
            ):
                raise ContractError(
                    f"{context}.{event['event_id']}.{track['track_id']}: bbox exceeds image"
                )
            if "behavior" not in track:
                continue
            scores = track["behavior"]["scores"]
            total = sum(scores.values())
            if not 0.99 <= total <= 1.01:
                raise ContractError(
                    f"{context}.{event['event_id']}.{track['track_id']}: behavior scores sum to {total}"
                )

    counts = scenario["expectations"]["exact_counts"]
    if counts.get("physical_bursts", 0) != 0:
        raise ContractError(f"{context}: simulation may not expect a physical burst")
    if counts.get("automatic_retries", 0) != 0:
        raise ContractError(f"{context}: simulation may not expect an automatic retry")
    config_path = (path.parent / scenario["config_ref"]).resolve()
    if not config_path.is_relative_to(ROOT) or not config_path.is_file():
        raise ContractError(
            f"{context}: config_ref does not resolve inside the repository"
        )


def main() -> int:
    try:
        schema_paths = sorted(SCHEMA_DIR.glob("*.schema.json"))
        schemas = [load_json(path) for path in schema_paths]
        for schema in schemas:
            Draft202012Validator.check_schema(schema)
        by_name = {
            path.name: schema
            for path, schema in zip(schema_paths, schemas, strict=True)
        }
        registry = schema_registry(schemas)

        config_path = ROOT / "config" / "simulation-safe.example.json"
        config = load_json(config_path)
        validate_instance(
            config_path, config, by_name["runtime-config.schema.json"], registry
        )
        validate_runtime_semantics(config, str(config_path.relative_to(ROOT)))

        scenario_paths = sorted((ROOT / "scenarios").glob("*.json"))
        for path in scenario_paths:
            scenario = load_json(path)
            validate_instance(path, scenario, by_name["scenario.schema.json"], registry)
            validate_scenario_semantics(scenario, path)
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        print(f"contract validation failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"validated {len(schemas)} schemas, 1 runtime config, and {len(scenario_paths)} scenarios"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
