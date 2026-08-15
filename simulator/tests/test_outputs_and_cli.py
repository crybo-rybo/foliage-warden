from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from foliage_warden_sim.cli import main
from foliage_warden_sim.engine import run_scenario


def schema_validator(schema_dir: Path, name: str) -> Draft202012Validator:
    registry = Registry()
    schemas: dict[str, dict] = {}
    for path in sorted(schema_dir.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schemas[path.name] = schema
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return Draft202012Validator(schemas[name], registry=registry)


def test_canonical_records_validate_against_repository_schemas(
    scenario_dir: Path, schemas: Path
) -> None:
    result = run_scenario(scenario_dir / "12-duplicate-command-id.json")
    event_validator = schema_validator(schemas, "event-record.schema.json")
    audit_validator = schema_validator(schemas, "action-audit.schema.json")
    for record in result.event_records:
        event_validator.validate(record)
    for record in result.audit_records:
        audit_validator.validate(record)
    assert any(record["decision"] == "SUPPRESS" for record in result.audit_records)
    unique_dispatches = [
        record
        for record in result.audit_records
        if record["decision"] == "DISPATCH" and record["outcome"] != "DUPLICATE"
    ]
    assert unique_dispatches
    assert all(record["safety"]["all_clear"] for record in unique_dispatches)


def test_evaluator_records_are_directly_parseable(
    scenario_dir: Path, repository: Path
) -> None:
    sys.path.insert(0, str(repository / "python" / "src"))
    try:
        from foliage_warden_eval.safety import check_safety
        from foliage_warden_eval.schemas import (
            ActionRecord,
            PredictionEvent,
            SessionRecord,
            parse_replay_record,
        )

        results = [run_scenario(path) for path in sorted(scenario_dir.glob("*.json"))]
        records = [record for result in results for record in result.evaluator_records]
        parsed = [parse_replay_record(record) for record in records]
        report = check_safety(
            (record for record in parsed if isinstance(record, SessionRecord)),
            (record for record in parsed if isinstance(record, PredictionEvent)),
            (record for record in parsed if isinstance(record, ActionRecord)),
        )
    finally:
        sys.path.pop(0)
    assert len(parsed) == len(records)
    assert report.passed


def test_cli_trace_is_byte_stable_and_comparable(
    tmp_path: Path, scenario_dir: Path, capsys
) -> None:
    trace = tmp_path / "trace.json"
    args = [str(scenario_dir / "03-eating-persistence.json"), "--trace", str(trace)]
    assert main(args) == 0
    first = trace.read_bytes()
    capsys.readouterr()
    assert (
        main(
            [
                str(scenario_dir / "03-eating-persistence.json"),
                "--compare-trace",
                str(trace),
            ]
        )
        == 0
    )
    capsys.readouterr()
    second_path = tmp_path / "second.json"
    assert (
        main(
            [
                str(scenario_dir / "03-eating-persistence.json"),
                "--trace",
                str(second_path),
            ]
        )
        == 0
    )
    assert first == second_path.read_bytes()


def test_cli_returns_nonzero_on_trace_mismatch(
    tmp_path: Path, scenario_dir: Path, capsys
) -> None:
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}\n", encoding="utf-8")
    assert (
        main(
            [
                str(scenario_dir / "03-eating-persistence.json"),
                "--compare-trace",
                str(wrong),
            ]
        )
        == 1
    )
    assert "trace mismatch" in capsys.readouterr().err


def test_simulator_source_has_no_physical_io_imports(repository: Path) -> None:
    source = repository / "simulator" / "src"
    forbidden = ("import serial", "import socket", "import gpiod", "import RPi.GPIO")
    text = "\n".join(path.read_text(encoding="utf-8") for path in source.rglob("*.py"))
    assert not any(token in text for token in forbidden)
