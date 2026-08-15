#!/usr/bin/env bash
set -euo pipefail
export UV_LOCKED=1

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bridge_directory="$(mktemp -d -t foliage-warden-bridge-smoke.XXXXXX)"

cleanup() {
  rm -rf -- "$bridge_directory"
}
trap cleanup EXIT

bash "$repository_root/training/scripts/smoke.sh" "$bridge_directory"

PYTHONPATH="$repository_root/shadow/tests" \
  uv run --project "$repository_root/shadow" --group dev --group inference-test \
  python - "$bridge_directory" <<'PY'
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from support import cat_track, perception_record

from foliage_warden_shadow.contracts import stable_json

root = Path(sys.argv[1])
clip = sorted((root / "data" / "clips" / "test").glob("*.npz"))[0]
with np.load(clip, allow_pickle=False) as loaded:
    frame_count = loaded["frames"].shape[0]
if frame_count != 4:
    raise RuntimeError(f"expected four synthetic clip frames, got {frame_count}")

perception = perception_record(0, captured_at_ms=100, tracks=[cat_track("cat-a")])
observation = perception["observation"]
request = {
    "captured_at_ms": 100,
    "clip": {
        "format": "NUMPY_RGB_UINT8_THWC",
        "frame_timestamps_ms": [0, 33, 66, 100],
        "path": clip.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "window_end_captured_at_ms": 100,
        "window_start_captured_at_ms": 0,
    },
    "frame_id": observation["frame_id"],
    "observation_id": observation["observation_id"],
    "predicted_at_ms": 110,
    "record_type": "behavior_inference_request",
    "schema_version": 1,
    "sequence": 0,
    "track_id": "cat-a",
}
(root / "perception.jsonl").write_text(stable_json(perception) + "\n", encoding="utf-8")
(root / "requests.jsonl").write_text(stable_json(request) + "\n", encoding="utf-8")
PY

model_digest="$(sha256sum "$bridge_directory/run/behavior.onnx" | cut -d ' ' -f 1)"
uv run --project "$repository_root/shadow" --extra inference \
  foliage-warden-shadow-infer \
  "$bridge_directory/perception.jsonl" "$bridge_directory/requests.jsonl" \
  --model "$bridge_directory/run/behavior.onnx" \
  --metadata "$bridge_directory/run/behavior.metadata.json" \
  --expected-onnx-sha256 "$model_digest" \
  --window-ms 100 \
  --logical-latency-ms 10 \
  --output "$bridge_directory/behavior.jsonl"

uv run --project "$repository_root/shadow" --group dev foliage-warden-shadow \
  "$bridge_directory/perception.jsonl" "$bridge_directory/behavior.jsonl" \
  --config "$repository_root/config/simulation-safe.example.json" \
  --scenario-out "$bridge_directory/shadow-scenario.json" \
  --summary "$bridge_directory/shadow-summary.json" \
  --evaluator-jsonl "$bridge_directory/shadow-evaluator.jsonl" \
  --audit-jsonl "$bridge_directory/shadow-audit.jsonl" >/dev/null

uv run --project "$repository_root/shadow" --group dev \
  python - "$bridge_directory" "$model_digest" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from foliage_warden_eval.safety import check_safety
from foliage_warden_eval.schemas import (
    ActionRecord,
    PredictionEvent,
    SessionRecord,
    parse_replay_record,
)
from foliage_warden_shadow.contracts import BEHAVIOR_LABELS, stable_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"behavior bridge verification failed: {message}")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require(bool(line.strip()), f"{path.name}:{line_number} is blank")
        value = json.loads(line)
        require(isinstance(value, dict), f"{path.name}:{line_number} must be an object")
        records.append(value)
    return records


root = Path(sys.argv[1])
model_digest = sys.argv[2]
summary = read_json(root / "shadow-summary.json")
behavior_records = read_jsonl(root / "behavior.jsonl")
scenario = read_json(root / "shadow-scenario.json")
audit_records = read_jsonl(root / "shadow-audit.jsonl")
evaluator_records = read_jsonl(root / "shadow-evaluator.jsonl")

require(summary.get("passed") is True, "mock replay did not pass")
require(summary.get("mode") == "OBSERVE_ONLY", "replay was not observe-only")
require(
    summary.get("actuator") == {"backend": "MOCK", "physical_effect_possible": False},
    "replay did not use the no-physical-effect MOCK actuator",
)
require(
    summary.get("behavior_identity", {}).get("model", {}).get("sha256") == model_digest,
    "summary model digest does not match exported ONNX bytes",
)
require(
    summary.get("fusion", {}).get("status_counts") == {"FUSED": 1},
    "prediction was missing, late, mismatched, or otherwise replaced during fusion",
)
require(summary.get("safety", {}).get("passed") is True, "evaluator safety failed")
require(
    summary.get("safety", {}).get("violation_count") == 0,
    "evaluator reported a safety violation",
)
require(summary.get("counts", {}).get("physical_bursts") == 0, "physical burst was reported")
require(
    summary.get("deterministic_replay_verified") is True,
    "deterministic mock replay was not verified",
)

require(len(behavior_records) == 1, "inference must emit exactly one prediction")
prediction = behavior_records[0]
require(prediction.get("record_type") == "behavior_prediction", "inference record type is invalid")
require(
    prediction.get("model", {}).get("sha256") == model_digest,
    "prediction model digest does not match exported ONNX bytes",
)
predicted_label = prediction.get("predicted_label")
require(
    predicted_label in {"PASSING", "SNIFFING", "EATING", "DIGGING"},
    "synthetic smoke inference resolved to OTHER/UNKNOWN; concrete propagation is unproven",
)
probabilities = prediction.get("probabilities")
require(isinstance(probabilities, dict), "prediction probabilities are missing")
require(set(probabilities) == set(BEHAVIOR_LABELS), "prediction label order contract is incomplete")
require(
    math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6),
    "prediction probabilities do not sum to one",
)
require(
    max(BEHAVIOR_LABELS, key=probabilities.__getitem__) == predicted_label,
    "prediction label is not the fixed-order argmax",
)

policy_label = {
    "PASSING": "CLEAR",
    "SNIFFING": "CLEAR",
    "EATING": "EATING",
    "DIGGING": "DIGGING",
}[predicted_label]
expected_behavior = {
    "label": policy_label,
    "model_id": prediction["model"]["id"],
    "raw_label": predicted_label,
    "scores": {
        "CLEAR": probabilities["PASSING"] + probabilities["SNIFFING"],
        "DIGGING": probabilities["DIGGING"],
        "EATING": probabilities["EATING"],
        "UNKNOWN": probabilities["OTHER"] + probabilities["UNKNOWN"],
    },
}

scenario_observations = [
    item["observation"] for item in scenario.get("timeline", []) if item.get("type") == "OBSERVATION"
]
require(len(scenario_observations) == 1, "scenario must contain exactly one fused observation")
scenario_tracks = scenario_observations[0].get("tracks", [])
scenario_cat = next(
    (track for track in scenario_tracks if track.get("track_id") == prediction.get("track_id")),
    None,
)
require(scenario_cat is not None, "inferred cat track is absent from the fused scenario")
require(
    scenario_cat.get("behavior") == expected_behavior,
    "ONNX label/model/scores did not reach the fused scenario exactly",
)

matching_audits = [
    record
    for record in audit_records
    if record.get("evidence", {}).get("behavior") == expected_behavior
    and prediction["model"]["id"] in record.get("evidence", {}).get("model_ids", [])
]
require(len(matching_audits) == 1, "ONNX behavior did not reach exactly one policy audit record")
require(
    matching_audits[0].get("decision") == "SUPPRESS",
    "single low-confidence smoke observation must remain suppressed",
)

parsed_evaluator = [parse_replay_record(record) for record in evaluator_records]
raw_sessions = [record for record in evaluator_records if record.get("record_type") == "session"]
sessions = [record for record in parsed_evaluator if isinstance(record, SessionRecord)]
simulator_predictions = [
    record for record in parsed_evaluator if isinstance(record, PredictionEvent)
]
actions = [record for record in parsed_evaluator if isinstance(record, ActionRecord)]
require(len(sessions) == 1, "simulator evaluator output must contain exactly one session")
require(len(raw_sessions) == 1, "raw simulator evaluator output must contain exactly one session")
simulator_safety = check_safety(sessions, simulator_predictions, actions)
require(simulator_safety.passed, "emitted simulator evaluator records failed safety checks")
require(not simulator_safety.violations, "emitted simulator evaluator records contain violations")
for event in simulator_predictions:
    require(event.model_id == prediction["model"]["id"], "simulator evaluator model ID drifted")
    require(event.behavior.value == policy_label, "simulator evaluator behavior drifted")

# One intentionally low-confidence observation normally creates no simulator
# incident/prediction_event. Exercise the evaluator's typed prediction boundary
# separately with a fixture derived exactly from the inference record. This is
# not represented as a simulator incident or as model-performance evidence.
fixture_event = {
    "behavior": predicted_label,
    "cat_count": 1,
    "config_hash": prediction["config"]["sha256"],
    "end_ms": prediction["predicted_at_ms"],
    "event_id": "offline-inference-fixture-000001",
    "hardware_ready": True,
    "incident_id": None,
    "metadata": {
        "fixture_only": True,
        "not_a_simulator_incident": True,
        "source_record_type": "behavior_prediction",
    },
    "model_id": prediction["model"]["id"],
    "no_fire_intersection": False,
    "person_present": False,
    "ready_ms": None,
    "record_type": "prediction_event",
    "schema_version": 1,
    "score": probabilities[predicted_label],
    "session_id": sessions[0].session_id,
    "stale_input": False,
    "start_ms": prediction["captured_at_ms"],
    "track_id": prediction["track_id"],
    "track_lost": False,
    "would_action": False,
    "zone_id": scenario_cat["zone_id"],
}
fixture_path = root / "inference-evaluator-fixture.jsonl"
fixture_path.write_text(
    stable_json(raw_sessions[0]) + "\n" + stable_json(fixture_event) + "\n",
    encoding="utf-8",
)
parsed_fixture = parse_replay_record(fixture_event)
require(isinstance(parsed_fixture, PredictionEvent), "evaluator fixture did not parse")
require(parsed_fixture.model_id == prediction["model"]["id"], "fixture model ID drifted")
require(parsed_fixture.behavior.value == predicted_label, "fixture behavior drifted")
require(parsed_fixture.score == probabilities[predicted_label], "fixture score drifted")
fixture_safety = check_safety(sessions, [parsed_fixture], [])
require(fixture_safety.passed, "inference-derived evaluator fixture failed safety checks")
require(not fixture_safety.violations, "inference-derived evaluator fixture has violations")

print(
    json.dumps(
        {
            "evaluator_fixture_only_predictions": 1,
            "fused_predictions": summary["fusion"]["status_counts"]["FUSED"],
            "mode": "OBSERVE_ONLY",
            "onnx_predicted_label": predicted_label,
            "passed": summary["passed"],
            "physical_bursts": summary["counts"]["physical_bursts"],
            "policy_audit_matches": len(matching_audits),
            "record_type": "behavior_bridge_verification",
            "safety_violations": summary["safety"]["violation_count"],
            "simulator_evaluator_prediction_events": len(simulator_predictions),
            "simulator_evaluator_sessions": len(sessions),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
