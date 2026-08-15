from __future__ import annotations

# Native ONNX Runtime wheels do not cover every Python in the base matrix.
# ruff: noqa: E402, I001

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="controlled shadow inference dependencies not installed")
onnx = pytest.importorskip("onnx", reason="controlled shadow inference dependencies not installed")
from onnx import TensorProto, helper, numpy_helper

from foliage_warden_assembler import assemble_incident
from foliage_warden_shadow.contracts import (
    BEHAVIOR_LABELS,
    parse_perception_stream,
    read_jsonl,
    stable_json,
)
from foliage_warden_shadow.inference import (
    infer_behavior_predictions,
    parse_inference_requests,
)
from foliage_warden_shadow.runner import execute_shadow

from support import (
    cat_track,
    file_sha256,
    perception_record,
    person_track,
    write_recorder_incident,
)


def _label_schema_id() -> str:
    payload = json.dumps({"version": 1, "labels": BEHAVIOR_LABELS}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_export(directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    model_path = directory / "behavior.onnx"
    metadata_path = directory / "behavior.metadata.json"
    model_config = {
        "dropout": 0.1,
        "feature_dim": 8,
        "gru_layers": 1,
        "hidden_dim": 8,
        "image_size": 16,
        "num_classes": 6,
        "num_frames": 2,
    }
    architecture = "temporal-cnn-gru-v1"
    model_config_id = hashlib.sha256(
        stable_json({"model_architecture": architecture, "model_config": model_config}).encode(
            "utf-8"
        )
    ).hexdigest()
    artifact_id = hashlib.sha256(b"assembler-controlled-artifact").hexdigest()
    training_config_id = hashlib.sha256(b"assembler-controlled-config").hexdigest()
    training_manifest_sha = hashlib.sha256(b"assembler-controlled-manifest").hexdigest()

    input_width = 2 * 3 * 16 * 16
    weights = np.zeros((input_width, 6), dtype=np.float32)
    bias = np.asarray([-2.0, -2.0, 6.0, -2.0, -2.0, -2.0], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["frames"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weights", "bias"], ["logits"]),
        ],
        "assembler-shadow-integration",
        [helper.make_tensor_value_info("frames", TensorProto.FLOAT, ["batch", 2, 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(
        graph,
        producer_name="foliage-warden-assembler-test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    embedded = {
        "foliage_warden.artifact_id": artifact_id,
        "foliage_warden.input_color": "RGB",
        "foliage_warden.input_layout": "N,T,C,H,W",
        "foliage_warden.label_schema_id": _label_schema_id(),
        "foliage_warden.labels": json.dumps(list(BEHAVIOR_LABELS), separators=(",", ":")),
        "foliage_warden.model_architecture": architecture,
        "foliage_warden.model_config_id": model_config_id,
        "foliage_warden.rgb_mean": json.dumps([0.5, 0.5, 0.5]),
        "foliage_warden.rgb_std": json.dumps([0.5, 0.5, 0.5]),
        "foliage_warden.training_config_id": training_config_id,
        "foliage_warden.training_manifest_sha256": training_manifest_sha,
    }
    helper.set_model_props(model, embedded)
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    metadata = {
        "artifact_id": artifact_id,
        "checkpoint": str(directory / "controlled-checkpoint.pt"),
        "checkpoint_sha256": hashlib.sha256(b"controlled-checkpoint").hexdigest(),
        "export_format_version": 1,
        "format": "ONNX",
        "input": {
            "color": "RGB",
            "dtype": "float32",
            "layout": "N,T,C,H,W",
            "mean": [0.5, 0.5, 0.5],
            "name": "frames",
            "range_before_normalization": [0.0, 1.0],
            "shape": ["batch", 2, 3, 16, 16],
            "std": [0.5, 0.5, 0.5],
        },
        "label_schema_id": _label_schema_id(),
        "labels": list(BEHAVIOR_LABELS),
        "model_architecture": architecture,
        "model_config": model_config,
        "model_config_id": model_config_id,
        "onnx": str(model_path),
        "onnx_runtime_parity": {
            "atol": 1e-5,
            "batch_sizes": [1, 2],
            "max_absolute_error": 0.0,
            "rtol": 1e-4,
        },
        "onnx_sha256": file_sha256(model_path),
        "opset": 17,
        "output": {
            "label_order": list(BEHAVIOR_LABELS),
            "name": "logits",
            "shape": ["batch", 6],
        },
        "tensor_rt_note": "Controlled CPU-only assembler integration fixture.",
        "training_config_id": training_config_id,
        "training_manifest_sha256": training_manifest_sha,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path, metadata_path, metadata


def test_assembled_requests_feed_real_shadow_onnx_inference(tmp_path: Path) -> None:
    records = [
        perception_record(10, 100, tracks=[cat_track("cat-a", approach_overlap=0.0)]),
        perception_record(20, 200, tracks=[cat_track("cat-a")]),
        perception_record(30, 225),
        perception_record(35, 250, tracks=[person_track()]),
        perception_record(40, 300, tracks=[cat_track("cat-a"), cat_track("cat-b")]),
    ]
    incident, source_perceptions, _ = write_recorder_incident(tmp_path, records)
    assembled = assemble_incident(
        incident,
        source_perceptions,
        tmp_path / "assembled",
        window_ms=100,
        logical_latency_ms=10,
    )
    model, metadata_path, metadata = _make_export(tmp_path)

    selected_perceptions = read_jsonl(assembled.perceptions_path, parse_perception_stream)
    incident_perceptions = read_jsonl(assembled.incident_perceptions_path, parse_perception_stream)
    requests = read_jsonl(assembled.requests_path, parse_inference_requests)
    predictions = infer_behavior_predictions(
        selected_perceptions,
        requests,
        manifest_directory=assembled.output_directory,
        model_path=model,
        metadata_path=metadata_path,
        expected_onnx_sha256=file_sha256(model),
        logical_latency_ms=10,
        window_ms=100,
    )

    assert len(selected_perceptions) == len(requests) == len(predictions) == 1
    assert [prediction.key for prediction in predictions] == [request.key for request in requests]
    assert {prediction.predicted_label for prediction in predictions} == {"EATING"}
    assert {prediction.model_id for prediction in predictions} == {metadata["artifact_id"]}
    assert [prediction.predicted_at_ms for prediction in predictions] == [210]
    assert all(
        abs(sum(prediction.probabilities.values()) - 1.0) < 1e-9 for prediction in predictions
    )

    repository = Path(__file__).resolve().parents[2]
    config_path = repository / "config" / "simulation-safe.example.json"
    runtime_config = json.loads(config_path.read_text(encoding="utf-8"))
    run = execute_shadow(incident_perceptions, predictions, runtime_config, config_path)

    assert [record.sequence for record in incident_perceptions] == [10, 20, 30, 35, 40]
    assert incident_perceptions[2].record["person_present"] is False
    assert incident_perceptions[3].record["person_present"] is True
    assert run.fusion.status_counts == {"FUSED": 1, "MISSING": 3}
    assert [diagnostic.status for diagnostic in run.fusion.diagnostics] == [
        "FUSED",
        "MISSING",
        "MISSING",
        "MISSING",
    ]
    assert run.fusion.behavior_identity == {
        "config": {
            "id": predictions[0].config_id,
            "sha256": predictions[0].config_sha256,
        },
        "model": {
            "id": metadata["artifact_id"],
            "sha256": file_sha256(model),
        },
    }
    assert [frame.perception_sequence for frame in run.fusion.frames] == [10, 20, 30, 35, 40]
    frames_by_sequence = {
        frame.perception_sequence: frame.observation for frame in run.fusion.frames
    }
    assert frames_by_sequence[30]["tracks"] == []
    assert [track["class"] for track in frames_by_sequence[35]["tracks"]] == ["PERSON"]
    behavior_by_key = {
        (frame.perception_sequence, track["track_id"]): track["behavior"]
        for frame in run.fusion.frames
        for track in frame.observation["tracks"]
        if track["class"] == "CAT"
    }
    fused = behavior_by_key[(20, "cat-a")]
    assert fused["label"] == "EATING"
    assert fused["raw_label"] == "EATING"
    assert fused["model_id"] == metadata["artifact_id"]
    assert fused["scores"]["EATING"] == predictions[0].probabilities["EATING"]
    assert fused["scores"]["UNKNOWN"] == (
        predictions[0].probabilities["OTHER"] + predictions[0].probabilities["UNKNOWN"]
    )
    for key in ((10, "cat-a"), (40, "cat-a"), (40, "cat-b")):
        assert behavior_by_key[key] == {
            "label": "UNKNOWN",
            "raw_label": "OTHER_UNKNOWN",
            "scores": {
                "CLEAR": 0.0,
                "DIGGING": 0.0,
                "EATING": 0.0,
                "UNKNOWN": 1.0,
            },
        }
    evaluator_predictions = [
        record for record in run.evaluator_records if record["record_type"] == "prediction_event"
    ]
    assert len(evaluator_predictions) == 1
    assert {record["behavior"] for record in evaluator_predictions} == {"EATING"}
    assert {record["model_id"] for record in evaluator_predictions} == {metadata["artifact_id"]}
    assert evaluator_predictions[0]["score"] == predictions[0].probabilities["EATING"]
    assert run.summary["counts"]["physical_bursts"] == 0
    assert "PERSON_PRESENT" in run.simulation.reason_codes
    assert run.summary["safety"]["passed"] is True
    assert run.summary["safety"]["violation_count"] == 0
    assert run.summary["passed"] is True
    assert run.summary["deterministic_replay_verified"] is True
    assert run.summary["actuator"] == {
        "backend": "MOCK",
        "physical_effect_possible": False,
    }
