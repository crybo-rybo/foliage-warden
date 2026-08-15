from __future__ import annotations

# Optional native inference wheels do not cover every Python in the base test matrix.
# ruff: noqa: E402, I001

import json
import stat
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

cv2 = pytest.importorskip("cv2", reason="offline inference integration dependencies not installed")
np = pytest.importorskip("numpy", reason="offline inference integration dependencies not installed")
onnx = pytest.importorskip(
    "onnx", reason="offline inference integration dependencies not installed"
)
from onnx import TensorProto, helper, numpy_helper

from foliage_warden_shadow.contracts import (
    BEHAVIOR_LABELS,
    ContractError,
    parse_behavior_stream,
    parse_perception_stream,
    read_jsonl,
    stable_json,
)
from foliage_warden_shadow.infer_cli import main as infer_main
from foliage_warden_shadow.inference import (
    CLIP_FORMAT,
    infer_behavior_predictions,
    parse_inference_requests,
)
from foliage_warden_shadow.runner import execute_shadow

from support import cat_track, perception_record


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _label_schema_id() -> str:
    payload = json.dumps({"version": 1, "labels": BEHAVIOR_LABELS}, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(stable_json(record) + "\n" for record in records), encoding="utf-8")


def _make_export(directory: Path) -> tuple[Path, Path, np.ndarray, np.ndarray, dict[str, Any]]:
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
    model_config_id = sha256(
        stable_json({"model_architecture": architecture, "model_config": model_config}).encode()
    ).hexdigest()
    artifact_id = sha256(b"test-artifact").hexdigest()
    training_config_id = sha256(b"test-training-config").hexdigest()
    training_manifest_sha = sha256(b"test-training-manifest").hexdigest()

    input_width = 2 * 3 * 16 * 16
    weights = np.linspace(-0.004, 0.004, input_width * 6, dtype=np.float32).reshape(input_width, 6)
    bias = np.asarray([-2.0, -2.0, 6.0, -2.0, -2.0, -2.0], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["frames"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "weights", "bias"], ["logits"]),
        ],
        "shadow-inference-contract-test",
        [helper.make_tensor_value_info("frames", TensorProto.FLOAT, ["batch", 2, 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(
        graph,
        producer_name="foliage-warden-test",
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
        "checkpoint": str((directory / "checkpoint.pt").resolve()),
        "checkpoint_sha256": sha256(b"test-checkpoint").hexdigest(),
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
        "onnx": str(model_path.resolve()),
        "onnx_runtime_parity": {
            "atol": 1e-5,
            "batch_sizes": [1, 2],
            "max_absolute_error": 0.0,
            "rtol": 1e-4,
        },
        "onnx_sha256": _file_sha256(model_path),
        "opset": 17,
        "output": {
            "label_order": list(BEHAVIOR_LABELS),
            "name": "logits",
            "shape": ["batch", 6],
        },
        "tensor_rt_note": "Test fixture; CPU inference is the adapter contract.",
        "training_config_id": training_config_id,
        "training_manifest_sha256": training_manifest_sha,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model_path, metadata_path, weights, bias, metadata


def _make_inputs(directory: Path) -> dict[str, Any]:
    perception = perception_record(0, captured_at_ms=100, tracks=[cat_track("cat-a")])
    perception_path = directory / "perception.jsonl"
    _write_jsonl(perception_path, [perception])
    clips = directory / "clips"
    clips.mkdir()
    frames = np.zeros((3, 18, 20, 3), dtype=np.uint8)
    frames[0, :, :, 0] = 32
    frames[1, :, :, 1] = 128
    frames[2, :, :, 2] = 224
    clip_path = clips / "cat-a.npy"
    np.save(clip_path, frames, allow_pickle=False)
    observation = perception["observation"]
    request = {
        "captured_at_ms": 100,
        "clip": {
            "format": CLIP_FORMAT,
            "frame_timestamps_ms": [0, 50, 100],
            "path": "clips/cat-a.npy",
            "sha256": _file_sha256(clip_path),
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
    request_path = directory / "requests.jsonl"
    _write_jsonl(request_path, [request])
    return {
        "clip_path": clip_path,
        "frames": frames,
        "perception": perception,
        "perception_path": perception_path,
        "request": request,
        "request_path": request_path,
    }


def _cli_args(
    inputs: dict[str, Any], model: Path, metadata: Path, output: Path | None = None
) -> list[str]:
    args = [
        str(inputs["perception_path"]),
        str(inputs["request_path"]),
        "--model",
        str(model),
        "--metadata",
        str(metadata),
        "--expected-onnx-sha256",
        _file_sha256(model),
        "--logical-latency-ms",
        "10",
        "--window-ms",
        "100",
    ]
    if output is not None:
        args.extend(["--output", str(output)])
    return args


def test_real_onnx_inference_is_numerically_correct_and_byte_stable(
    tmp_path: Path,
    config_path: Path,
    runtime_config: dict[str, Any],
) -> None:
    model, metadata_path, weights, bias, metadata = _make_export(tmp_path)
    inputs = _make_inputs(tmp_path)
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    assert infer_main(_cli_args(inputs, model, metadata_path, first_output)) == 0
    assert infer_main(_cli_args(inputs, model, metadata_path, second_output)) == 0
    assert first_output.read_bytes() == second_output.read_bytes()
    assert stat.S_IMODE(first_output.stat().st_mode) == 0o600

    predictions = read_jsonl(first_output, parse_behavior_stream)
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.model_id == metadata["artifact_id"]
    assert prediction.model_sha256 == _file_sha256(model)
    assert prediction.captured_at_ms == 100
    assert prediction.predicted_at_ms == 110

    sampled = inputs["frames"][[0, 2]]
    resized = np.stack(
        [cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA) for frame in sampled]
    )
    tensor = resized.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    tensor = (tensor - np.float32(0.5)) / np.float32(0.5)
    logits = tensor.reshape(1, -1) @ weights + bias
    exponentials = np.exp(logits[0].astype(np.float64) - np.max(logits[0]))
    expected = exponentials / exponentials.sum(dtype=np.float64)
    assert list(prediction.probabilities) == list(BEHAVIOR_LABELS)
    np.testing.assert_allclose(
        list(prediction.probabilities.values()), expected, rtol=1e-6, atol=1e-8
    )
    assert prediction.predicted_label == BEHAVIOR_LABELS[int(np.argmax(expected))]
    assert prediction.predicted_label == "EATING"

    perceptions = read_jsonl(inputs["perception_path"], parse_perception_stream)
    run = execute_shadow(perceptions, predictions, runtime_config, config_path)
    assert run.fusion.status_counts == {"FUSED": 1}
    assert [diagnostic.status for diagnostic in run.fusion.diagnostics] == ["FUSED"]
    assert run.fusion.behavior_identity == {
        "config": {"id": prediction.config_id, "sha256": prediction.config_sha256},
        "model": {"id": prediction.model_id, "sha256": prediction.model_sha256},
    }
    fused_behavior = run.fusion.frames[0].observation["tracks"][0]["behavior"]
    assert fused_behavior["label"] == "EATING"
    assert fused_behavior["raw_label"] == "EATING"
    assert fused_behavior["model_id"] == prediction.model_id
    assert fused_behavior["scores"]["EATING"] == prediction.probabilities["EATING"]
    assert fused_behavior["scores"]["UNKNOWN"] == (
        prediction.probabilities["OTHER"] + prediction.probabilities["UNKNOWN"]
    )
    scenario_observations = [
        event["observation"] for event in run.scenario["timeline"] if event["type"] == "OBSERVATION"
    ]
    assert len(scenario_observations) == 1
    assert scenario_observations[0]["tracks"][0]["behavior"] == fused_behavior
    evaluator_predictions = [
        record for record in run.evaluator_records if record["record_type"] == "prediction_event"
    ]
    assert len(evaluator_predictions) == 1
    assert evaluator_predictions[0]["behavior"] == "EATING"
    assert evaluator_predictions[0]["model_id"] == prediction.model_id
    assert evaluator_predictions[0]["score"] == prediction.probabilities["EATING"]
    assert run.summary["passed"] is True
    assert run.summary["deterministic_replay_verified"] is True
    assert run.summary["safety"]["passed"] is True
    assert run.summary["counts"]["physical_bursts"] == 0
    assert run.summary["actuator"] == {
        "backend": "MOCK",
        "physical_effect_possible": False,
    }


def test_stdout_is_empty_until_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model, metadata_path, _, _, _ = _make_export(tmp_path)
    inputs = _make_inputs(tmp_path)
    inputs["request"]["clip"]["sha256"] = "f" * 64
    _write_jsonl(inputs["request_path"], [inputs["request"]])

    assert infer_main(_cli_args(inputs, model, metadata_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "clip SHA-256 does not match manifest" in captured.err


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda request: request.__setitem__("schema_version", 1.0),
            "schema_version must be the integer 1",
        ),
        (
            lambda request: request.__setitem__("track_id", "cat-missing"),
            "non-CAT or unknown track",
        ),
        (
            lambda request: request.__setitem__("predicted_at_ms", 111),
            "does not match logical_latency_ms",
        ),
        (
            lambda request: request["clip"].__setitem__("frame_timestamps_ms", [0, 50, 101]),
            "must end exactly at captured_at_ms",
        ),
        (
            lambda request: request["clip"].__setitem__("path", "../cat-a.npy"),
            "normalized relative path",
        ),
        (
            lambda request: request["clip"].__setitem__("path", "./clips/cat-a.npy"),
            "not canonical POSIX text",
        ),
        (
            lambda request: request["clip"].__setitem__("path", "clips/./cat-a.npy"),
            "not canonical POSIX text",
        ),
        (
            lambda request: request["clip"].__setitem__("path", "clips//cat-a.npy"),
            "not canonical POSIX text",
        ),
    ],
)
def test_manifest_disagreement_fails_closed(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    model, metadata_path, _, _, _ = _make_export(tmp_path)
    inputs = _make_inputs(tmp_path)
    mutation(inputs["request"])
    _write_jsonl(inputs["request_path"], [inputs["request"]])

    perceptions = read_jsonl(inputs["perception_path"], parse_perception_stream)
    try:
        requests = read_jsonl(inputs["request_path"], parse_inference_requests)
        infer_behavior_predictions(
            perceptions,
            requests,
            manifest_directory=inputs["request_path"].parent,
            model_path=model,
            metadata_path=metadata_path,
            expected_onnx_sha256=_file_sha256(model),
            logical_latency_ms=10,
            window_ms=100,
        )
    except ContractError as error:
        assert message in str(error)
    else:  # pragma: no cover - keeps each negative case explicit.
        pytest.fail("invalid inference request was accepted")


def test_missing_cat_request_and_wrong_clip_dtype_fail_closed(tmp_path: Path) -> None:
    model, metadata_path, _, _, _ = _make_export(tmp_path)
    inputs = _make_inputs(tmp_path)
    perceptions = read_jsonl(inputs["perception_path"], parse_perception_stream)

    with pytest.raises(ContractError, match="missing CAT track"):
        infer_behavior_predictions(
            perceptions,
            [],
            manifest_directory=tmp_path,
            model_path=model,
            metadata_path=metadata_path,
            expected_onnx_sha256=_file_sha256(model),
            logical_latency_ms=10,
            window_ms=100,
        )

    np.save(inputs["clip_path"], inputs["frames"].astype(np.float32), allow_pickle=False)
    inputs["request"]["clip"]["sha256"] = _file_sha256(inputs["clip_path"])
    requests = parse_inference_requests([inputs["request"]])
    with pytest.raises(ContractError, match="exact uint8 dtype"):
        infer_behavior_predictions(
            perceptions,
            requests,
            manifest_directory=tmp_path,
            model_path=model,
            metadata_path=metadata_path,
            expected_onnx_sha256=_file_sha256(model),
            logical_latency_ms=10,
            window_ms=100,
        )


def test_npz_clip_and_replay_timing_are_part_of_config_identity(tmp_path: Path) -> None:
    model, metadata_path, _, _, _ = _make_export(tmp_path)
    inputs = _make_inputs(tmp_path)
    npz_path = tmp_path / "clips" / "cat-a.npz"
    np.savez_compressed(npz_path, frames=inputs["frames"])
    inputs["request"]["clip"]["path"] = "clips/cat-a.npz"
    inputs["request"]["clip"]["sha256"] = _file_sha256(npz_path)
    perceptions = parse_perception_stream([inputs["perception"]])
    requests = parse_inference_requests([inputs["request"]])

    first = infer_behavior_predictions(
        perceptions,
        requests,
        manifest_directory=tmp_path,
        model_path=model,
        metadata_path=metadata_path,
        expected_onnx_sha256=_file_sha256(model),
        logical_latency_ms=10,
        window_ms=100,
    )
    changed_request = deepcopy(inputs["request"])
    changed_request["predicted_at_ms"] = 120
    changed_request["clip"]["frame_timestamps_ms"] = [50, 75, 100]
    changed_request["clip"]["window_start_captured_at_ms"] = 50
    second = infer_behavior_predictions(
        perceptions,
        parse_inference_requests([changed_request]),
        manifest_directory=tmp_path,
        model_path=model,
        metadata_path=metadata_path,
        expected_onnx_sha256=_file_sha256(model),
        logical_latency_ms=20,
        window_ms=50,
    )
    assert first[0].config_sha256 != second[0].config_sha256


def test_external_digest_sidecar_and_embedded_metadata_are_independent_locks(
    tmp_path: Path,
) -> None:
    model, metadata_path, _, _, metadata = _make_export(tmp_path)
    inputs = _make_inputs(tmp_path)
    perceptions = parse_perception_stream([inputs["perception"]])
    requests = parse_inference_requests([inputs["request"]])

    with pytest.raises(ContractError, match="expected-onnx-sha256"):
        infer_behavior_predictions(
            perceptions,
            requests,
            manifest_directory=tmp_path,
            model_path=model,
            metadata_path=metadata_path,
            expected_onnx_sha256="0" * 64,
            logical_latency_ms=10,
            window_ms=100,
        )

    metadata["onnx_sha256"] = "1" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ContractError, match="export metadata"):
        infer_behavior_predictions(
            perceptions,
            requests,
            manifest_directory=tmp_path,
            model_path=model,
            metadata_path=metadata_path,
            expected_onnx_sha256=_file_sha256(model),
            logical_latency_ms=10,
            window_ms=100,
        )

    model, metadata_path, _, _, metadata = _make_export(tmp_path)
    changed = onnx.load(model)
    for item in changed.metadata_props:
        if item.key == "foliage_warden.input_color":
            item.value = "BGR"
    onnx.save(changed, model)
    metadata["onnx_sha256"] = _file_sha256(model)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ContractError, match="embedded metadata disagrees"):
        infer_behavior_predictions(
            perceptions,
            requests,
            manifest_directory=tmp_path,
            model_path=model,
            metadata_path=metadata_path,
            expected_onnx_sha256=_file_sha256(model),
            logical_latency_ms=10,
            window_ms=100,
        )


def test_inference_surface_has_no_camera_network_or_action_api() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "foliage_warden_shadow"
    source = (package / "inference.py").read_text(encoding="utf-8")
    source += (package / "infer_cli.py").read_text(encoding="utf-8")
    for forbidden in (
        "VideoCapture",
        "urlopen",
        "requests.",
        "import socket",
        "import serial",
        "import gpio",
        "import time",
        "datetime",
        "physical_effect",
    ):
        assert forbidden not in source
