from __future__ import annotations

from copy import deepcopy

import pytest
from support import behavior_prediction, perception_record, probabilities

from foliage_warden_shadow.contracts import (
    ContractError,
    parse_behavior_stream,
    parse_perception_stream,
)
from foliage_warden_shadow.fusion import FusionOptions, fuse_streams


def _fused_label(raw_label: str, runtime_config: dict) -> dict:
    perception = perception_record(0, captured_at_ms=100)
    prediction = behavior_prediction(perception, "cat-a", raw_label, sequence=0)
    result = fuse_streams(
        parse_perception_stream([perception]),
        parse_behavior_stream([prediction]),
        runtime_config,
    )
    return result.frames[0].observation["tracks"][0]["behavior"]


def test_six_labels_map_to_four_policy_labels_conservatively(runtime_config: dict) -> None:
    assert _fused_label("PASSING", runtime_config)["label"] == "CLEAR"
    assert _fused_label("SNIFFING", runtime_config)["label"] == "CLEAR"
    assert _fused_label("EATING", runtime_config)["label"] == "EATING"
    assert _fused_label("DIGGING", runtime_config)["label"] == "DIGGING"
    other = _fused_label("OTHER", runtime_config)
    unknown = _fused_label("UNKNOWN", runtime_config)
    assert other["label"] == unknown["label"] == "UNKNOWN"
    assert other["raw_label"] == unknown["raw_label"] == "OTHER_UNKNOWN"
    assert other["scores"]["UNKNOWN"] == 0.96


def test_clear_and_unknown_scores_are_probability_conserving(runtime_config: dict) -> None:
    perception = perception_record(0, captured_at_ms=100)
    prediction = behavior_prediction(perception, "cat-a", "PASSING", sequence=0)
    prediction["probabilities"] = {
        "PASSING": 0.5,
        "SNIFFING": 0.2,
        "EATING": 0.1,
        "DIGGING": 0.05,
        "OTHER": 0.1,
        "UNKNOWN": 0.05,
    }
    fused = fuse_streams(
        parse_perception_stream([perception]),
        parse_behavior_stream([prediction]),
        runtime_config,
    )
    behavior = fused.frames[0].observation["tracks"][0]["behavior"]
    assert behavior["scores"] == {
        "CLEAR": 0.7,
        "DIGGING": 0.05,
        "EATING": 0.1,
        "UNKNOWN": 0.15000000000000002,
    }
    assert sum(behavior["scores"].values()) == 1.0


def test_exact_identity_is_required_and_unmatched_prediction_is_audited(
    runtime_config: dict,
) -> None:
    perception = perception_record(0, captured_at_ms=100)
    mismatch = behavior_prediction(
        perception,
        "cat-a",
        "EATING",
        sequence=0,
        frame_id="camera-1:frame:wrong",
    )
    result = fuse_streams(
        parse_perception_stream([perception]),
        parse_behavior_stream([mismatch]),
        runtime_config,
    )
    assert result.frames[0].observation["tracks"][0]["behavior"]["label"] == "UNKNOWN"
    assert result.status_counts == {"MISSING": 1, "UNMATCHED_PREDICTION": 1}
    assert [item.status for item in result.diagnostics] == [
        "UNMATCHED_PREDICTION",
        "MISSING",
    ]


def test_fixture_probability_helper_has_exact_six_way_total() -> None:
    assert sum(probabilities("EATING").values()) == 1.0


def test_non_mock_or_physical_runtime_configuration_is_rejected(runtime_config: dict) -> None:
    perception = perception_record(0, captured_at_ms=100)
    parsed_perception = parse_perception_stream([perception])
    parsed_behavior = parse_behavior_stream([])
    unsafe = deepcopy(runtime_config)
    unsafe["actuator"]["backend"] = "SERIAL"
    unsafe["actuator"]["allow_physical_effects"] = True

    with pytest.raises(ContractError, match="physical actuator path"):
        fuse_streams(parsed_perception, parsed_behavior, unsafe)


def test_prediction_timeout_and_max_latency_are_independent_fail_closed_bounds(
    runtime_config: dict,
) -> None:
    perception = perception_record(0, captured_at_ms=100)
    timed_out = behavior_prediction(perception, "cat-a", "EATING", sequence=0, latency_ms=10)
    result = fuse_streams(
        parse_perception_stream([perception]),
        parse_behavior_stream([timed_out]),
        runtime_config,
        options=FusionOptions(prediction_timeout_ms=0),
    )
    assert result.diagnostics[0].status == "TIMED_OUT"
    assert result.frames[0].delivery_at_ms == 100
    assert result.frames[0].observation["tracks"][0]["behavior"]["label"] == "UNKNOWN"

    over_latency = behavior_prediction(perception, "cat-a", "EATING", sequence=0, latency_ms=300)
    result = fuse_streams(
        parse_perception_stream([perception]),
        parse_behavior_stream([over_latency]),
        runtime_config,
        options=FusionOptions(prediction_timeout_ms=500, max_prediction_latency_ms=250),
    )
    assert result.diagnostics[0].status == "LATE"
    assert result.frames[0].delivery_at_ms == 400
    assert result.frames[0].observation["tracks"][0]["behavior"]["label"] == "UNKNOWN"


def test_crossed_prediction_arrivals_never_reorder_captured_frames(
    runtime_config: dict,
) -> None:
    first = perception_record(0, captured_at_ms=100)
    second = perception_record(1, captured_at_ms=200)
    second_prediction = behavior_prediction(second, "cat-a", "EATING", sequence=0, latency_ms=10)
    first_prediction = behavior_prediction(first, "cat-a", "EATING", sequence=1, latency_ms=200)

    result = fuse_streams(
        parse_perception_stream([first, second]),
        parse_behavior_stream([second_prediction, first_prediction]),
        runtime_config,
        options=FusionOptions(prediction_timeout_ms=500),
    )

    assert [frame.observation["captured_at_ms"] for frame in result.frames] == [100, 200]
    assert [frame.delivery_at_ms for frame in result.frames] == [300, 300]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("zone_id", "unknown-zone", "unknown zone"),
        ("zone_type", "soil", "runtime config says 'approach'"),
    ],
)
def test_zone_evidence_must_use_runtime_config_identities(
    runtime_config: dict, field: str, value: str, message: str
) -> None:
    perception = perception_record(0, captured_at_ms=100)
    perception["zone_evidence"][0]["overlaps"][0][field] = value
    if field == "zone_id":
        perception["observation"]["tracks"][0]["zone_id"] = value
    if field == "zone_type":
        # Keep the internally duplicated maxima consistent so this reaches the
        # runtime-config identity check rather than failing during parsing.
        track = perception["observation"]["tracks"][0]
        track["region_evidence"]["approach_overlap"] = 0.0
        track["region_evidence"]["soil_overlap"] = 0.91
        track["zone_id"] = None

    with pytest.raises(ContractError, match=message):
        fuse_streams(parse_perception_stream([perception]), [], runtime_config)
