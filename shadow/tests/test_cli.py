from __future__ import annotations

import json
from pathlib import Path

from foliage_warden_eval.schemas import parse_replay_record
from support import predictions_for, series

from foliage_warden_shadow.cli import main
from foliage_warden_shadow.contracts import stable_json


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(stable_json(record) + "\n" for record in records), encoding="utf-8")


def _execute(tmp_path: Path, config_path: Path, suffix: str) -> tuple[bytes, bytes, bytes, dict]:
    perceptions = series(4)
    predictions = predictions_for(perceptions, "EATING")
    perception_path = tmp_path / "perception.jsonl"
    behavior_path = tmp_path / "behavior.jsonl"
    _write_jsonl(perception_path, perceptions)
    _write_jsonl(behavior_path, predictions)
    scenario_path = tmp_path / f"scenario-{suffix}.json"
    replay_path = tmp_path / f"replay-{suffix}.jsonl"
    summary_path = tmp_path / f"summary-{suffix}.json"
    fusion_path = tmp_path / f"fusion-{suffix}.jsonl"

    status = main(
        [
            str(perception_path),
            str(behavior_path),
            "--config",
            str(config_path),
            "--scenario-id",
            "shadow-cli-byte-stable",
            "--scenario-out",
            str(scenario_path),
            "--evaluator-jsonl",
            str(replay_path),
            "--fusion-jsonl",
            str(fusion_path),
            "--summary",
            str(summary_path),
        ]
    )
    assert status == 0
    return (
        scenario_path.read_bytes(),
        replay_path.read_bytes(),
        fusion_path.read_bytes(),
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


def test_cli_outputs_are_byte_stable_and_evaluator_ready(
    tmp_path: Path,
    config_path: Path,
    capsys,
) -> None:
    first = _execute(tmp_path, config_path, "first")
    first_stdout = capsys.readouterr().out
    second = _execute(tmp_path, config_path, "second")
    second_stdout = capsys.readouterr().out

    assert first[:3] == second[:3]
    assert first[3] == second[3]
    assert first_stdout == second_stdout
    assert first[3]["mode"] == "OBSERVE_ONLY"
    assert first[3]["actuator"] == {"backend": "MOCK", "physical_effect_possible": False}
    assert first[3]["safety"]["violation_count"] == 0
    replay_lines = first[1].decode().splitlines()
    assert replay_lines
    assert all(parse_replay_record(json.loads(line)) for line in replay_lines)


def test_shadow_source_has_no_physical_adapter_imports(repository: Path) -> None:
    source = repository / "shadow" / "src"
    forbidden = ("import serial", "import socket", "import gpiod", "import RPi.GPIO")
    text = "\n".join(path.read_text(encoding="utf-8") for path in source.rglob("*.py"))
    assert not any(token in text for token in forbidden)
