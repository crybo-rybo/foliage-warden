from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from support import behavior_prediction, perception_record

from foliage_warden_shadow.contracts import (
    ContractError,
    parse_behavior_prediction,
    parse_behavior_stream,
    parse_perception_observation,
    parse_perception_stream,
    stable_json,
)


def test_behavior_contract_requires_all_six_probabilities_and_pinned_identity() -> None:
    perception = perception_record(0, captured_at_ms=100)
    raw = behavior_prediction(perception, "cat-a", "EATING", sequence=0)

    parsed = parse_behavior_prediction(raw)

    assert parsed.key == (
        "camera-1:observation:00000000",
        "camera-1:frame:00000000",
        "cat-a",
    )
    assert parsed.predicted_label == "EATING"
    assert parsed.to_dict() == raw


def test_behavior_record_validates_against_versioned_json_schema(repository: Path) -> None:
    schema = json.loads(
        (repository / "shadow" / "schemas" / "behavior-prediction.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    raw = behavior_prediction(
        perception_record(0, captured_at_ms=100),
        "cat-a",
        "EATING",
        sequence=0,
    )
    Draft202012Validator(schema).validate(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.__setitem__("would_action", True), "OBSERVE_ONLY"),
        (lambda item: item.__setitem__("schema_version", 2), "schema_version"),
        (lambda item: item.__setitem__("schema_version", True), "schema_version"),
        (lambda item: item.__setitem__("schema_version", 1.0), "integer 1"),
        (lambda item: item["probabilities"].pop("OTHER"), "missing field"),
        (lambda item: item["probabilities"].__setitem__("UNKNOWN", float("nan")), "finite"),
        (lambda item: item.__setitem__("predicted_label", "PASSING"), "argmax"),
        (lambda item: item["model"].__setitem__("sha256", "bad"), "SHA-256"),
        (lambda item: item.__setitem__("extra", True), "unknown field"),
    ],
)
def test_behavior_contract_rejects_ambiguous_or_unsafe_records(mutation, message: str) -> None:
    raw = behavior_prediction(
        perception_record(0, captured_at_ms=100),
        "cat-a",
        "EATING",
        sequence=0,
    )
    mutation(raw)
    with pytest.raises(ContractError, match=message):
        parse_behavior_prediction(raw)


def test_streams_reject_duplicate_identity_and_non_deterministic_order() -> None:
    first = perception_record(0, captured_at_ms=100)
    second = perception_record(1, captured_at_ms=500)
    first_prediction = behavior_prediction(first, "cat-a", "EATING", sequence=0)
    duplicate = deepcopy(first_prediction)
    duplicate["sequence"] = 1
    duplicate["predicted_at_ms"] += 1
    with pytest.raises(ContractError, match="duplicate observation/frame/track"):
        parse_behavior_stream([first_prediction, duplicate])

    with pytest.raises(ContractError, match="strictly ordered"):
        parse_perception_stream([second, first])


def test_perception_contract_requires_unknown_unarmed_shell_and_consistent_identity() -> None:
    raw = perception_record(0, captured_at_ms=100)
    parsed = parse_perception_observation(raw)
    assert parsed.policy_observation() == raw["observation"]

    raw["observation"]["tracks"][0]["aim_preset_id"] = "unsafe-prepopulation"
    with pytest.raises(ContractError, match="aim_preset_id must be null"):
        parse_perception_observation(raw)


def test_perception_schema_version_rejects_float_equivalence() -> None:
    raw = perception_record(0, captured_at_ms=100)
    raw["schema_version"] = 1.0
    with pytest.raises(ContractError, match="integer 1"):
        parse_perception_observation(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda item: item["observation"]["tracks"][0].__setitem__("no_fire_intersection", True),
            "no_fire_intersection disagrees",
        ),
        (
            lambda item: item["observation"]["tracks"][0]["region_evidence"].__setitem__(
                "foliage_overlap", 0.1
            ),
            "foliage_overlap disagrees",
        ),
        (
            lambda item: item["observation"]["tracks"][0].__setitem__("zone_id", None),
            "zone_id disagrees",
        ),
        (
            lambda item: item["zone_evidence"][0].__setitem__("no_fire_overlap", 0.4),
            "no_fire_overlap disagrees",
        ),
    ],
)
def test_perception_rejects_contradictory_policy_and_zone_evidence(mutation, message: str) -> None:
    raw = perception_record(0, captured_at_ms=100)
    mutation(raw)
    with pytest.raises(ContractError, match=message):
        parse_perception_observation(raw)


def test_no_fire_zone_evidence_cannot_be_downgraded_at_policy_boundary() -> None:
    raw = perception_record(0, captured_at_ms=100)
    raw["zone_evidence"][0]["no_fire_overlap"] = 0.2
    raw["zone_evidence"][0]["overlaps"][3]["overlap"] = 0.2
    assert raw["observation"]["tracks"][0]["no_fire_intersection"] is False

    with pytest.raises(ContractError, match="no_fire_intersection disagrees"):
        parse_perception_observation(raw)


def test_stable_json_is_byte_stable_and_refuses_non_finite_values() -> None:
    assert stable_json({"z": 1, "a": 2}) == '{"a":2,"z":1}'
    with pytest.raises(ValueError, match="Out of range"):
        stable_json({"probability": float("inf")})
