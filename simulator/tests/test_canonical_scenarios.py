from __future__ import annotations

from pathlib import Path

import pytest

from foliage_warden_sim import run_scenario


def canonical_paths(scenario_dir: Path) -> list[Path]:
    return sorted(scenario_dir.glob("*.json"))


def test_all_fourteen_canonical_scenarios_pass(scenario_dir: Path) -> None:
    paths = canonical_paths(scenario_dir)
    assert len(paths) == 14
    results = [run_scenario(path) for path in paths]
    assert all(result.passed for result in results)
    assert all(result.deterministic_replay_verified for result in results)


@pytest.mark.parametrize(
    ("filename", "final_state", "actions"),
    [
        ("01-clear-pass.json", "MONITORING", []),
        ("03-eating-persistence.json", "COOLDOWN", ["GOTO_PRESET", "BURST"]),
        ("04-digging-persistence.json", "COOLDOWN", ["GOTO_PRESET", "BURST"]),
        ("11-missing-burst-ack.json", "FAULT", ["GOTO_PRESET", "BURST"]),
        (
            "12-duplicate-command-id.json",
            "COOLDOWN",
            ["GOTO_PRESET", "BURST", "BURST"],
        ),
        ("14-camera-loss-restart.json", "DISARMED", []),
    ],
)
def test_representative_policy_traces(
    scenario_dir: Path, filename: str, final_state: str, actions: list[str]
) -> None:
    result = run_scenario(scenario_dir / filename)
    assert result.final_state == final_state
    assert [item["command"] for item in result.action_sequence] == actions


def test_missing_burst_ack_never_retries(scenario_dir: Path) -> None:
    result = run_scenario(scenario_dir / "11-missing-burst-ack.json")
    assert result.counts["burst_commands_issued"] == 1
    assert result.counts["burst_commands_acked"] == 0
    assert result.counts["automatic_retries"] == 0
    assert result.action_sequence[-1]["result"] == "TIMEOUT"


def test_continuous_harmful_incident_remains_latched(scenario_dir: Path) -> None:
    result = run_scenario(scenario_dir / "13-continuous-incident-cooldown.json")
    assert result.final_clock_ms > 30_000
    assert result.final_state == "COOLDOWN"
    assert result.counts["burst_commands_issued"] == 1
    assert "INCIDENT_ALREADY_ACTIONED" in result.reason_codes
