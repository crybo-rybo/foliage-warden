from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pytest
from support import cat_track, person_track, predictions_for, series

from foliage_warden_shadow.contracts import (
    ContractError,
    parse_behavior_stream,
    parse_perception_stream,
)
from foliage_warden_shadow.fusion import FusionOptions
from foliage_warden_shadow.runner import execute_shadow


@dataclass(frozen=True)
class Case:
    perceptions: list[dict]
    predictions: list[dict]
    options: FusionOptions
    reason: str
    bursts: int
    diagnostic_status: str


def _case(name: str) -> Case:
    options = FusionOptions()
    if name == "eating":
        perceptions = series(4)
        return Case(
            perceptions,
            predictions_for(perceptions, "EATING"),
            options,
            "EATING_CONFIRMED",
            1,
            "FUSED",
        )
    if name == "digging":
        perceptions = series(4)
        return Case(
            perceptions,
            predictions_for(perceptions, "DIGGING"),
            options,
            "DIGGING_CONFIRMED",
            1,
            "FUSED",
        )
    if name == "sniffing":
        perceptions = series(4)
        return Case(
            perceptions,
            predictions_for(perceptions, "SNIFFING"),
            options,
            "BEHAVIOR_CLEAR",
            0,
            "FUSED",
        )
    if name == "missing":
        return Case(series(4), [], options, "BEHAVIOR_UNKNOWN", 0, "MISSING")
    if name == "late":
        perceptions = series(4)
        predictions = predictions_for(perceptions, "EATING", latency_ms=300)
        return Case(perceptions, predictions, options, "BEHAVIOR_UNKNOWN", 0, "TIMED_OUT")
    if name == "mismatched":
        perceptions = series(4)
        predictions = predictions_for(perceptions, "EATING")
        for prediction in predictions:
            prediction["frame_id"] += ":mismatch"
        return Case(perceptions, predictions, options, "BEHAVIOR_UNKNOWN", 0, "MISSING")
    if name == "person":
        perceptions = series(4, tracks=[cat_track(), person_track()])
        return Case(
            perceptions,
            predictions_for(perceptions, "EATING"),
            options,
            "PERSON_PRESENT",
            0,
            "FUSED",
        )
    if name == "multiple-cats":
        perceptions = series(4, tracks=[cat_track("cat-a"), cat_track("cat-b")])
        return Case(
            perceptions,
            predictions_for(perceptions, "EATING"),
            options,
            "MULTIPLE_CATS",
            0,
            "FUSED",
        )
    if name == "stale":
        perceptions = series(4)
        predictions = predictions_for(perceptions, "EATING", latency_ms=300)
        return Case(
            perceptions,
            predictions,
            FusionOptions(prediction_timeout_ms=1_000, max_prediction_latency_ms=1_000),
            "FRAME_STALE",
            0,
            "FUSED",
        )
    if name == "no-fire":
        perceptions = series(4, tracks=[cat_track(no_fire=True)])
        return Case(
            perceptions,
            predictions_for(perceptions, "EATING"),
            options,
            "NO_FIRE_INTERSECTION",
            0,
            "FUSED",
        )
    if name == "weak-track":
        perceptions = series(4, tracks=[cat_track(quality=0.5)])
        return Case(
            perceptions,
            predictions_for(perceptions, "EATING"),
            options,
            "POOR_TRACK",
            0,
            "FUSED",
        )
    if name == "one-burst-latch":
        perceptions = series(7)
        return Case(
            perceptions,
            predictions_for(perceptions, "EATING"),
            options,
            "INCIDENT_ALREADY_ACTIONED",
            1,
            "FUSED",
        )
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    [
        "eating",
        "digging",
        "sniffing",
        "missing",
        "late",
        "mismatched",
        "person",
        "multiple-cats",
        "stale",
        "no-fire",
        "weak-track",
        "one-burst-latch",
    ],
)
def test_synthetic_shadow_cases_are_fail_closed_and_evaluator_safe(
    name: str,
    runtime_config: dict,
    config_path,
    tmp_path,
) -> None:
    case = _case(name)
    run = execute_shadow(
        parse_perception_stream(deepcopy(case.perceptions)),
        parse_behavior_stream(deepcopy(case.predictions)),
        runtime_config,
        config_path,
        options=case.options,
        scenario_id=f"shadow-fixture-{name}",
        scenario_out=tmp_path / f"{name}.scenario.json",
    )

    assert run.summary["passed"] is True
    assert run.safety["passed"] is True
    assert run.safety["violation_count"] == 0
    assert run.safety["violations"] == []
    assert run.simulation.counts["would_burst_decisions"] == case.bursts
    assert run.simulation.counts["burst_commands_issued"] == case.bursts
    assert run.simulation.counts["physical_bursts"] == 0
    assert run.simulation.counts["automatic_retries"] == 0
    assert case.reason in run.simulation.reason_codes
    assert case.diagnostic_status in run.fusion.status_counts
    assert all(
        audit.get("action", {}).get("physical_effect_possible") is False
        for audit in run.simulation.audit_records
        if "action" in audit
    )

    if name == "mismatched":
        assert run.fusion.status_counts == {"MISSING": 4, "UNMATCHED_PREDICTION": 4}
    if case.bursts == 0:
        assert not any(item["command"] == "BURST" for item in run.simulation.action_sequence)


def test_continuous_incident_has_exactly_one_mock_burst(
    runtime_config: dict,
    config_path,
) -> None:
    case = _case("one-burst-latch")
    run = execute_shadow(
        parse_perception_stream(case.perceptions),
        parse_behavior_stream(case.predictions),
        runtime_config,
        config_path,
        scenario_id="shadow-one-burst-proof",
    )

    commands = [item["command"] for item in run.simulation.action_sequence]
    assert commands == ["GOTO_PRESET", "BURST"]
    assert run.simulation.counts["would_burst_decisions"] == 1
    assert run.simulation.counts["burst_commands_issued"] == 1
    assert "INCIDENT_ALREADY_ACTIONED" in run.simulation.reason_codes
    assert "COOLDOWN_ACTIVE" in run.simulation.reason_codes


def test_execute_rejects_in_memory_and_on_disk_config_split_brain(
    runtime_config: dict,
    config_path,
) -> None:
    perceptions = series(4)
    predictions = predictions_for(perceptions, "EATING")
    changed = deepcopy(runtime_config)
    changed["perception"]["behavior"]["eating_min_probability"] = 0.99

    with pytest.raises(ContractError, match="does not exactly match config_path"):
        execute_shadow(
            parse_perception_stream(perceptions),
            parse_behavior_stream(predictions),
            changed,
            config_path,
            scenario_id="shadow-split-brain-rejected",
        )
