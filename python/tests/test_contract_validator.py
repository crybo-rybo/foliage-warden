from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from tools.validate_contracts import (
    ContractError,
    validate_polygon,
    validate_runtime_semantics,
    validate_scenario_semantics,
)

ROOT = Path(__file__).resolve().parents[2]


def fixture(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_checked_in_config_and_scenario_pass_semantics() -> None:
    config = fixture("config/simulation-safe.example.json")
    scenario_path = ROOT / "scenarios/03-eating-persistence.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    validate_runtime_semantics(config, "config")
    validate_scenario_semantics(scenario, scenario_path)


def test_self_intersecting_calibration_polygon_is_rejected() -> None:
    bowtie = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.9},
        {"x": 0.9, "y": 0.1},
        {"x": 0.1, "y": 0.9},
    ]
    with pytest.raises(ContractError, match="negligible area|self-intersects"):
        validate_polygon(bowtie, "scene.zone")


def test_non_live_config_cannot_expose_serial_backend() -> None:
    config = fixture("config/simulation-safe.example.json")
    config["actuator"]["backend"] = "SERIAL"
    with pytest.raises(ContractError, match="physical actuator path"):
        validate_runtime_semantics(config, "config")


def test_scenario_cannot_expect_physical_effect_or_colliding_order() -> None:
    scenario_path = ROOT / "scenarios/01-clear-pass.json"
    original = json.loads(scenario_path.read_text(encoding="utf-8"))
    physical = deepcopy(original)
    physical["expectations"]["exact_counts"]["physical_bursts"] = 1
    with pytest.raises(ContractError, match="physical burst"):
        validate_scenario_semantics(physical, scenario_path)

    collision = deepcopy(original)
    collision["timeline"][1]["at_ms"] = collision["timeline"][0]["at_ms"]
    collision["timeline"][1]["sequence"] = collision["timeline"][0]["sequence"]
    with pytest.raises(ContractError, match="duplicate .* keys"):
        validate_scenario_semantics(collision, scenario_path)
