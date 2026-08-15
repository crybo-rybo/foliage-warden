from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from conftest import read_json, write_json

from foliage_warden_sim.engine import Simulator
from foliage_warden_sim.validation import load_contracts


@pytest.mark.parametrize(
    ("reason", "mutate"),
    [
        ("AMBIGUOUS_TRACK", lambda track: track.__setitem__("ambiguous", True)),
        ("TRACK_TOO_YOUNG", lambda track: track.__setitem__("initial_track_age_ms", 0)),
        ("NO_SAFE_PRESET", lambda track: track.__setitem__("aim_preset_id", None)),
        (
            "BEHAVIOR_UNKNOWN",
            lambda track: (
                track["behavior"].__setitem__("label", "UNKNOWN"),
                track["behavior"].__setitem__("raw_label", "OTHER_UNKNOWN"),
                track["behavior"].__setitem__(
                    "scores",
                    {"CLEAR": 0.02, "EATING": 0.01, "DIGGING": 0.01, "UNKNOWN": 0.96},
                ),
            ),
        ),
    ],
)
def test_additional_fail_closed_observation_interlocks(
    tmp_path: Path,
    scenario_dir: Path,
    config: Path,
    schemas: Path,
    reason: str,
    mutate,
) -> None:
    scenario = deepcopy(read_json(scenario_dir / "03-eating-persistence.json"))
    track = scenario["timeline"][1]["template"]["tracks"][0]
    mutate(track)
    path = write_json(tmp_path / f"{reason.lower()}.json", scenario)
    result = Simulator(
        load_contracts(path, config_path=config, schema_dir=schemas)
    ).run()
    assert reason in result.reason_codes
    assert result.counts["would_burst_decisions"] == 0
    assert not result.action_sequence


@pytest.mark.parametrize("field_path", [("enabled",), ("burst", "enabled")])
def test_disabled_actuation_blocks_policy_commands(
    tmp_path: Path,
    scenario_dir: Path,
    config: Path,
    schemas: Path,
    field_path: tuple[str, ...],
) -> None:
    disabled = read_json(config)
    target = disabled["actuator"]
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = False
    config_path = write_json(tmp_path / "disabled-config.json", disabled)
    scenario = scenario_dir / "03-eating-persistence.json"
    result = Simulator(
        load_contracts(scenario, config_path=config_path, schema_dir=schemas)
    ).run()
    assert "ACTUATION_DISABLED" in result.reason_codes
    assert not result.action_sequence


def test_mock_adapter_denies_duration_above_hardware_clamp(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = read_json(scenario_dir / "01-clear-pass.json")
    scenario["timeline"].extend(
        [
            {
                "event_id": "inject-overlong-burst",
                "at_ms": 2000,
                "sequence": 0,
                "type": "INJECT_ACTION",
                "action": {"command_id": 9999, "command": "BURST", "duration_ms": 200},
            },
            {"event_id": "settle-denial", "at_ms": 2300, "sequence": 0, "type": "TICK"},
        ]
    )
    path = write_json(tmp_path / "overlong-burst.json", scenario)
    result = Simulator(
        load_contracts(path, config_path=config, schema_dir=schemas)
    ).run()
    assert result.action_sequence[-1]["result"] == "DENIED"
    assert "COMMAND_DENIED" in result.reason_codes
    assert result.counts["physical_bursts"] == 0
