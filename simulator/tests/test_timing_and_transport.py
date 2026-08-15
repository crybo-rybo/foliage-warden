from __future__ import annotations

from pathlib import Path

from conftest import read_json, write_json

from foliage_warden_sim.engine import Simulator
from foliage_warden_sim.validation import load_contracts


def modified_positive(
    tmp_path: Path,
    scenario_dir: Path,
    config: Path,
    schemas: Path,
    *,
    overrides: list[dict] | None = None,
    extra_timeline: list[dict] | None = None,
    ack_delay_ms: int = 20,
) -> Simulator:
    scenario = read_json(scenario_dir / "03-eating-persistence.json")
    scenario["actuator_script"]["ack_delay_ms"] = ack_delay_ms
    scenario["actuator_script"]["overrides"] = overrides or []
    timeline = scenario["timeline"][:-1]
    timeline.extend(extra_timeline or [])
    timeline.append(scenario["timeline"][-1])
    timeline.sort(key=lambda item: (item["at_ms"], item["sequence"], item["event_id"]))
    scenario["timeline"] = timeline
    path = write_json(tmp_path / "modified.json", scenario)
    return Simulator(load_contracts(path, config_path=config, schema_dir=schemas))


def test_external_health_change_wins_ack_deadline_tie(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    simulator = modified_positive(
        tmp_path,
        scenario_dir,
        config,
        schemas,
        extra_timeline=[
            {
                "event_id": "not-ready-at-ack",
                "at_ms": 1320,
                "sequence": 0,
                "type": "ACTUATOR_STATUS",
                "status": "NOT_READY",
            }
        ],
    )
    result = simulator.run()
    assert result.final_state == "MONITORING"
    assert result.counts["would_burst_decisions"] == 0
    assert [item["command"] for item in result.action_sequence] == ["GOTO_PRESET"]
    assert "HARDWARE_NOT_READY" in result.reason_codes


def test_goto_denial_faults_without_burst(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    simulator = modified_positive(
        tmp_path,
        scenario_dir,
        config,
        schemas,
        overrides=[{"command": "GOTO_PRESET", "occurrence": 1, "response": "DENIED"}],
    )
    result = simulator.run()
    assert result.final_state == "FAULT"
    assert result.action_sequence[-1]["result"] == "DENIED"
    assert result.counts["burst_commands_issued"] == 0
    assert "COMMAND_DENIED" in result.reason_codes


def test_transport_error_faults_and_does_not_retry_burst(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    simulator = modified_positive(
        tmp_path,
        scenario_dir,
        config,
        schemas,
        overrides=[
            {"command": "BURST", "occurrence": 1, "response": "TRANSPORT_ERROR"}
        ],
    )
    result = simulator.run()
    assert result.final_state == "FAULT"
    assert result.action_sequence[-1]["result"] == "TRANSPORT_ERROR"
    assert result.counts["burst_commands_issued"] == 1
    assert result.counts["automatic_retries"] == 0


def test_process_restart_cancels_pending_mock_callback(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    simulator = modified_positive(
        tmp_path,
        scenario_dir,
        config,
        schemas,
        ack_delay_ms=200,
        extra_timeline=[
            {
                "event_id": "restart-before-ack",
                "at_ms": 1400,
                "sequence": 0,
                "type": "PROCESS_RESTART",
                "restart_id": "restart-before-ack",
            }
        ],
    )
    result = simulator.run()
    assert result.final_state == "DISARMED"
    assert not result.action_sequence
    assert result.counts["burst_commands_issued"] == 0
    assert "PROCESS_RESTART" in result.reason_codes
    pending = [
        record for record in result.audit_records if record["outcome"] == "PENDING"
    ]
    assert len(pending) == 1
    assert pending[0]["action"]["command"]["command"] == "GOTO_PRESET"


def test_ack_wins_timeout_at_identical_internal_deadline(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    simulator = modified_positive(
        tmp_path,
        scenario_dir,
        config,
        schemas,
        ack_delay_ms=250,
        extra_timeline=[
            {"event_id": "burst-ack-tie", "at_ms": 1800, "sequence": 0, "type": "TICK"}
        ],
    )
    result = simulator.run()
    assert result.final_state == "COOLDOWN"
    assert [item["result"] for item in result.action_sequence] == ["ACK", "ACK"]
    assert "COMMAND_ACK_TIMEOUT" not in result.reason_codes


def test_track_gaps_break_harmful_persistence(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = read_json(scenario_dir / "03-eating-persistence.json")
    series = scenario["timeline"][1]
    series["interval_ms"] = 600
    scenario["timeline"] = scenario["timeline"][:2]
    path = write_json(tmp_path / "gapped.json", scenario)
    simulator = Simulator(load_contracts(path, config_path=config, schema_dir=schemas))
    result = simulator.run()
    assert result.counts["ready_transitions"] == 0
    assert not result.action_sequence
    assert "BEHAVIOR_NOT_PERSISTENT" in result.reason_codes


def test_explicit_clear_and_elapsed_cooldown_allow_a_new_incident(
    tmp_path: Path, scenario_dir: Path, config: Path, schemas: Path
) -> None:
    scenario = read_json(scenario_dir / "03-eating-persistence.json")
    clear_source = read_json(scenario_dir / "01-clear-pass.json")["timeline"][1]
    clear_source["event_id"] = "explicit-clear"
    clear_source["at_ms"] = 2000
    clear_source["interval_ms"] = 1000
    clear_source["count"] = 2
    clear_source["id_prefix"] = "explicit-clear"

    second = read_json(scenario_dir / "03-eating-persistence.json")["timeline"][1]
    second["event_id"] = "second-incident"
    second["at_ms"] = 32000
    second["id_prefix"] = "second-incident"
    scenario["timeline"] = [
        scenario["timeline"][0],
        scenario["timeline"][1],
        clear_source,
        second,
        {"event_id": "settle-second", "at_ms": 33500, "sequence": 0, "type": "TICK"},
    ]
    path = write_json(tmp_path / "two-incidents.json", scenario)
    simulator = Simulator(load_contracts(path, config_path=config, schema_dir=schemas))
    result = simulator.run()
    assert result.counts["burst_commands_issued"] == 2
    assert result.counts["burst_commands_acked"] == 2
    burst_ids = [
        item["incident_id"]
        for item in simulator.command_attempts
        if item["command"]["command"] == "BURST"
    ]
    assert burst_ids == ["incident-000001", "incident-000002"]
