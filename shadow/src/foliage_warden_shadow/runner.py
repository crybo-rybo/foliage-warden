"""Execute finalized shadow replays through the existing mock-only simulator."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from foliage_warden_eval.safety import check_safety
from foliage_warden_eval.schemas import (
    ActionRecord,
    PredictionEvent,
    SessionRecord,
    parse_replay_record,
)
from foliage_warden_sim.engine import RunResult, run_scenario

from .contracts import (
    BehaviorPrediction,
    ContractError,
    JsonObject,
    PerceptionObservation,
    stable_json,
)
from .fusion import FusionOptions, FusionResult, assemble_scenario, fuse_streams


@dataclass(frozen=True, slots=True)
class ShadowRun:
    scenario: JsonObject
    fusion: FusionResult
    simulation: RunResult
    evaluator_records: tuple[JsonObject, ...]
    safety: JsonObject
    summary: JsonObject


def _write_text_atomic(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, destination)


def write_json(path: str | Path, value: Any) -> None:
    _write_text_atomic(path, stable_json(value) + "\n")


def write_jsonl(path: str | Path, records: Iterable[JsonObject]) -> None:
    _write_text_atomic(path, "".join(stable_json(record) + "\n" for record in records))


def _default_scenario_id(
    fusion: FusionResult,
    runtime_config: JsonObject,
) -> str:
    payload = {
        "behavior_sha256": fusion.behavior_sha256,
        "config": runtime_config,
        "perception_sha256": fusion.perception_sha256,
        "schema_version": 1,
    }
    digest = sha256(stable_json(payload).encode("utf-8")).hexdigest()
    return f"shadow-{digest[:24]}"


def _verified_config_snapshot(runtime_config: JsonObject, config_path: str | Path) -> str:
    source = Path(config_path)
    try:
        on_disk = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{source}: {error}") from error
    if not isinstance(on_disk, dict):
        raise ContractError(f"{source}: runtime config root must be an object")
    in_memory_text = stable_json(runtime_config)
    if stable_json(on_disk) != in_memory_text:
        raise ContractError(
            "runtime_config does not exactly match config_path; refusing split-brain replay"
        )
    return in_memory_text + "\n"


def _finalize_expectations(scenario: JsonObject, result: RunResult) -> None:
    scenario["expectations"] = {
        "explanation": (
            "Replay-derived exact outputs plus independent mock-only invariants. "
            "These expectations make this assembled trace reproducible; they are not "
            "model acceptance criteria."
        ),
        "final_state": result.final_state,
        "exact_counts": dict(result.counts),
        "required_states": [],
        "forbidden_states": ["FAULT"],
        "required_reason_codes": [],
        "forbidden_commands": [],
        "expected_action_sequence": [dict(item) for item in result.action_sequence],
        "invariants": [
            "STARTS_DISARMED",
            "NO_PHYSICAL_EFFECTS",
            "ONE_BURST_PER_INCIDENT",
            "BURST_NOT_RETRIED",
            "COMMAND_IDS_DEDUPLICATED",
            "ACTIONS_AUDITED",
            "DETERMINISTIC_REPLAY",
        ],
    }


def _evaluate_safety(records: tuple[JsonObject, ...]) -> JsonObject:
    parsed = [parse_replay_record(record) for record in records]
    report = check_safety(
        (record for record in parsed if isinstance(record, SessionRecord)),
        (record for record in parsed if isinstance(record, PredictionEvent)),
        (record for record in parsed if isinstance(record, ActionRecord)),
    )
    result = report.to_dict()
    result["violation_count"] = len(report.violations)
    return result


def _assert_mock_only(result: RunResult, safety: JsonObject) -> None:
    if result.counts["physical_bursts"] != 0:
        raise ContractError("shadow simulator reported a physical burst")
    for audit in result.audit_records:
        action = audit.get("action")
        if isinstance(action, dict) and action.get("physical_effect_possible") is not False:
            raise ContractError("shadow audit did not prove physical_effect_possible=false")
    if not safety["passed"]:
        raise ContractError("evaluator safety checks found a shadow replay violation")
    if not result.passed or not result.deterministic_replay_verified:
        raise ContractError("final shadow scenario did not pass deterministic replay assertions")


def execute_shadow(
    perceptions: list[PerceptionObservation],
    predictions: list[BehaviorPrediction],
    runtime_config: JsonObject,
    config_path: str | Path,
    *,
    options: FusionOptions | None = None,
    scenario_id: str | None = None,
    scenario_out: str | Path | None = None,
) -> ShadowRun:
    """Fuse, assemble, execute twice, and safety-check a versioned shadow replay."""

    config = deepcopy(runtime_config)
    config_snapshot = _verified_config_snapshot(config, config_path)
    fusion = fuse_streams(perceptions, predictions, config, options=options)
    selected_id = scenario_id or _default_scenario_id(fusion, config)
    scenario = assemble_scenario(
        fusion,
        config,
        config_path,
        scenario_id=selected_id,
    )

    with tempfile.TemporaryDirectory(prefix="foliage-warden-shadow-") as directory:
        snapshot_path = Path(directory) / "runtime-config.json"
        _write_text_atomic(snapshot_path, config_snapshot)
        provisional_path = Path(directory) / "provisional.json"
        write_json(provisional_path, scenario)
        probe = run_scenario(
            provisional_path,
            config_path=snapshot_path,
            verify_determinism=False,
        )
        _finalize_expectations(scenario, probe)
        final_path = Path(directory) / "scenario.json"
        write_json(final_path, scenario)
        result = run_scenario(final_path, config_path=snapshot_path, verify_determinism=True)

    evaluator_records = tuple(deepcopy(record) for record in result.evaluator_records)
    safety = _evaluate_safety(evaluator_records)
    _assert_mock_only(result, safety)
    if scenario_out is not None:
        write_json(scenario_out, scenario)

    summary: JsonObject = {
        "actuator": {
            "backend": "MOCK",
            "physical_effect_possible": False,
        },
        "behavior_identity": fusion.behavior_identity,
        "counts": dict(result.counts),
        "deterministic_replay_verified": result.deterministic_replay_verified,
        "final_state": result.final_state,
        "fusion": {
            "behavior_sha256": fusion.behavior_sha256,
            "diagnostics": [item.to_dict() for item in fusion.diagnostics],
            "frame_count": len(fusion.frames),
            "perception_sha256": fusion.perception_sha256,
            "status_counts": fusion.status_counts,
        },
        "mode": "OBSERVE_ONLY",
        "passed": result.passed and safety["passed"],
        "record_type": "shadow_run_summary",
        "safety": safety,
        "scenario_id": result.scenario_id,
        "schema_version": 1,
        "simulator": result.summary(),
    }
    return ShadowRun(
        scenario=deepcopy(scenario),
        fusion=fusion,
        simulation=result,
        evaluator_records=evaluator_records,
        safety=safety,
        summary=summary,
    )
