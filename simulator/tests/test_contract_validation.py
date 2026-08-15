from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from conftest import read_json, write_json

from foliage_warden_sim.validation import ContractError, load_contracts


def test_series_expansion_is_exact_and_zero_padded(
    scenario_dir: Path, schemas: Path
) -> None:
    contracts = load_contracts(
        scenario_dir / "03-eating-persistence.json", schema_dir=schemas
    )
    observations = [
        item for item in contracts.inputs if item.payload["type"] == "OBSERVATION"
    ]
    assert [item.at_ms for item in observations] == [100, 500, 900, 1300]
    assert [item.event_id for item in observations] == [
        "eating-000000",
        "eating-000001",
        "eating-000002",
        "eating-000003",
    ]
    assert observations[-1].payload["observation"]["tracks"][0]["track_age_ms"] == 2100


def test_future_capture_is_rejected(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = read_json(scenario_dir / "07-stale-frame.json")
    scenario["timeline"][1]["observation"]["captured_at_ms"] = 2001
    path = write_json(tmp_path / "future.json", scenario)
    with pytest.raises(ContractError, match="future"):
        load_contracts(path, config_path=config, schema_dir=schemas)


def test_bad_behavior_total_is_rejected(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = read_json(scenario_dir / "03-eating-persistence.json")
    scores = scenario["timeline"][1]["template"]["tracks"][0]["behavior"]["scores"]
    scores["EATING"] = 0.5
    path = write_json(tmp_path / "bad-scores.json", scenario)
    with pytest.raises(ContractError, match="scores sum"):
        load_contracts(path, config_path=config, schema_dir=schemas)


def test_collision_created_only_after_series_expansion_is_rejected(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = read_json(scenario_dir / "01-clear-pass.json")
    scenario["timeline"].append(
        {
            "event_id": "collides-with-generated-input",
            "at_ms": 400,
            "sequence": 0,
            "type": "TICK",
        }
    )
    path = write_json(tmp_path / "collision.json", scenario)
    with pytest.raises(
        ContractError, match="expanded timeline has duplicate order key"
    ):
        load_contracts(path, config_path=config, schema_dir=schemas)


def test_physical_or_non_mock_configuration_is_rejected(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    unsafe = read_json(config)
    unsafe["actuator"]["backend"] = "SERIAL"
    unsafe_path = write_json(tmp_path / "unsafe.json", unsafe)
    with pytest.raises(ContractError, match="physical actuator path"):
        load_contracts(
            scenario_dir / "01-clear-pass.json",
            config_path=unsafe_path,
            schema_dir=schemas,
        )


def test_unknown_scenario_field_fails_closed(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = deepcopy(read_json(scenario_dir / "01-clear-pass.json"))
    scenario["allow_anything"] = True
    path = write_json(tmp_path / "unknown.json", scenario)
    with pytest.raises(ContractError, match="Additional properties"):
        load_contracts(path, config_path=config, schema_dir=schemas)
