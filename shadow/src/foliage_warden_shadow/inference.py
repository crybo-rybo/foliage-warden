"""Deterministic, offline ONNX-to-shadow behavior inference.

This module intentionally accepts only pre-extracted NumPy RGB clips. It has no
camera, video-decoder, network, policy, or actuator surface.
"""

from __future__ import annotations

import io
import json
import math
import platform
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

from .contracts import (
    BEHAVIOR_LABELS,
    IDENTIFIER,
    MAX_SAFE_INTEGER,
    SCHEMA_VERSION,
    SHA256,
    BehaviorPrediction,
    ContractError,
    PerceptionObservation,
    parse_behavior_stream,
    stable_json,
)

REQUEST_RECORD_TYPE = "behavior_inference_request"
CLIP_FORMAT = "NUMPY_RGB_UINT8_THWC"
ADAPTER_CONFIG_ID = "offline-onnx-rgb-clip-v1"
MODEL_ARCHITECTURE = "temporal-cnn-gru-v1"
MAX_CLIP_BYTES = 256 * 1024 * 1024
MAX_MODEL_BYTES = 512 * 1024 * 1024

_EXPORT_FIELDS = {
    "artifact_id",
    "checkpoint",
    "checkpoint_sha256",
    "export_format_version",
    "format",
    "input",
    "label_schema_id",
    "labels",
    "model_architecture",
    "model_config",
    "model_config_id",
    "onnx",
    "onnx_runtime_parity",
    "onnx_sha256",
    "opset",
    "output",
    "tensor_rt_note",
    "training_config_id",
    "training_manifest_sha256",
}
_EMBEDDED_FIELDS = {
    "foliage_warden.artifact_id",
    "foliage_warden.input_color",
    "foliage_warden.input_layout",
    "foliage_warden.label_schema_id",
    "foliage_warden.labels",
    "foliage_warden.model_architecture",
    "foliage_warden.model_config_id",
    "foliage_warden.rgb_mean",
    "foliage_warden.rgb_std",
    "foliage_warden.training_config_id",
    "foliage_warden.training_manifest_sha256",
}


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    sequence: int
    observation_id: str
    frame_id: str
    track_id: str
    captured_at_ms: int
    predicted_at_ms: int
    clip_path: str
    clip_sha256: str
    frame_timestamps_ms: tuple[int, ...]
    window_start_captured_at_ms: int
    window_end_captured_at_ms: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.observation_id, self.frame_id, self.track_id)


@dataclass(frozen=True, slots=True)
class _ModelContract:
    model_id: str
    model_sha256: str
    input_name: str
    output_name: str
    num_frames: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    config_sha256: str
    embedded_metadata: dict[str, str]


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise ContractError(f"{context} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{context} has unknown field(s): {', '.join(unknown)}")


def _integer(value: Any, context: str, *, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if type(value) is not int or not lower <= value <= MAX_SAFE_INTEGER:
        qualifier = "positive" if positive else "non-negative"
        raise ContractError(f"{context} must be a {qualifier} safe integer")
    return value


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{context} must be a canonical identifier")
    return value


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ContractError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context} must be a non-empty string")
    return value


def _finite_float(value: Any, context: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0.0):
        raise ContractError(f"{context} must be a finite non-negative number")
    return result


def _exact_float_triplet(
    value: Any, expected: tuple[float, float, float], context: str
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ContractError(f"{context} must be a three-value list")
    if any(type(item) is not float or not math.isfinite(item) for item in value):
        raise ContractError(f"{context} must contain three finite JSON floats")
    result = tuple(value)
    if result != expected:
        raise ContractError(f"{context} does not match the supported export contract")
    return result  # type: ignore[return-value]


def parse_inference_request(raw: Mapping[str, Any]) -> InferenceRequest:
    data = _object(raw, "behavior_inference_request")
    _exact_fields(
        data,
        {
            "captured_at_ms",
            "clip",
            "frame_id",
            "observation_id",
            "predicted_at_ms",
            "record_type",
            "schema_version",
            "sequence",
            "track_id",
        },
        "behavior_inference_request",
    )
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        raise ContractError("behavior_inference_request.schema_version must be the integer 1")
    if data["record_type"] != REQUEST_RECORD_TYPE:
        raise ContractError("behavior_inference_request.record_type is invalid")
    captured_at_ms = _integer(data["captured_at_ms"], "behavior_inference_request.captured_at_ms")
    predicted_at_ms = _integer(
        data["predicted_at_ms"], "behavior_inference_request.predicted_at_ms"
    )
    if predicted_at_ms < captured_at_ms:
        raise ContractError("behavior_inference_request.predicted_at_ms precedes captured_at_ms")

    clip = _object(data["clip"], "behavior_inference_request.clip")
    _exact_fields(
        clip,
        {
            "format",
            "frame_timestamps_ms",
            "path",
            "sha256",
            "window_end_captured_at_ms",
            "window_start_captured_at_ms",
        },
        "behavior_inference_request.clip",
    )
    if clip["format"] != CLIP_FORMAT:
        raise ContractError(f"behavior_inference_request.clip.format must be {CLIP_FORMAT}")
    clip_path = clip["path"]
    if (
        not isinstance(clip_path, str)
        or not clip_path
        or len(clip_path) > 1024
        or clip_path != clip_path.strip()
        or "\x00" in clip_path
        or "\\" in clip_path
    ):
        raise ContractError("behavior_inference_request.clip.path is not canonical")
    if clip_path != Path(clip_path).as_posix():
        raise ContractError("behavior_inference_request.clip.path is not canonical POSIX text")
    timestamps_raw = clip["frame_timestamps_ms"]
    if not isinstance(timestamps_raw, list) or not 1 <= len(timestamps_raw) <= 100_000:
        raise ContractError(
            "behavior_inference_request.clip.frame_timestamps_ms must contain 1..100000 items"
        )
    timestamps = tuple(
        _integer(value, f"behavior_inference_request.clip.frame_timestamps_ms[{index}]")
        for index, value in enumerate(timestamps_raw)
    )
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise ContractError(
            "behavior_inference_request.clip.frame_timestamps_ms must be strictly increasing"
        )
    if timestamps[-1] != captured_at_ms:
        raise ContractError("behavior_inference_request clip must end exactly at captured_at_ms")
    window_start = _integer(
        clip["window_start_captured_at_ms"],
        "behavior_inference_request.clip.window_start_captured_at_ms",
    )
    window_end = _integer(
        clip["window_end_captured_at_ms"],
        "behavior_inference_request.clip.window_end_captured_at_ms",
    )
    if window_start != timestamps[0] or window_end != timestamps[-1]:
        raise ContractError(
            "behavior_inference_request clip window bounds must equal its first/last timestamp"
        )
    if window_end != captured_at_ms:
        raise ContractError(
            "behavior_inference_request clip window must end exactly at captured_at_ms"
        )
    return InferenceRequest(
        sequence=_integer(data["sequence"], "behavior_inference_request.sequence"),
        observation_id=_identifier(
            data["observation_id"], "behavior_inference_request.observation_id"
        ),
        frame_id=_identifier(data["frame_id"], "behavior_inference_request.frame_id"),
        track_id=_identifier(data["track_id"], "behavior_inference_request.track_id"),
        captured_at_ms=captured_at_ms,
        predicted_at_ms=predicted_at_ms,
        clip_path=clip_path,
        clip_sha256=_digest(clip["sha256"], "behavior_inference_request.clip.sha256"),
        frame_timestamps_ms=timestamps,
        window_start_captured_at_ms=window_start,
        window_end_captured_at_ms=window_end,
    )


def parse_inference_requests(
    values: Iterable[Mapping[str, Any]],
) -> list[InferenceRequest]:
    records = [parse_inference_request(value) for value in values]
    order = [(record.predicted_at_ms, record.sequence) for record in records]
    if order != sorted(order) or len(order) != len(set(order)):
        raise ContractError(
            "inference request stream must be strictly ordered by (predicted_at_ms, sequence)"
        )
    sequences = [record.sequence for record in records]
    if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
        raise ContractError("inference request sequence values must be strictly increasing")
    if len({record.key for record in records}) != len(records):
        raise ContractError(
            "inference request stream contains a duplicate observation/frame/track key"
        )
    return records


def _format_key(key: tuple[str, str, str]) -> str:
    return "/".join(key)


def _bind_requests(
    perceptions: list[PerceptionObservation], requests: list[InferenceRequest]
) -> None:
    expected: dict[tuple[str, str, str], int] = {}
    for perception in perceptions:
        for track in perception.tracks:
            if track["class"] == "CAT":
                expected[(perception.observation_id, perception.frame_id, track["track_id"])] = (
                    perception.captured_at_ms
                )
    actual = {request.key: request for request in requests}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    disagreement = []
    if missing:
        disagreement.append(
            "missing CAT track(s): " + ", ".join(_format_key(key) for key in missing)
        )
    if extra:
        disagreement.append(
            "contains non-CAT or unknown track(s): " + ", ".join(_format_key(key) for key in extra)
        )
    if disagreement:
        raise ContractError("inference request manifest " + "; ".join(disagreement))
    for key, request in actual.items():
        if request.captured_at_ms != expected[key]:
            raise ContractError(
                f"inference request {_format_key(key)} capture timestamp disagrees with perception"
            )


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"export metadata contains forbidden JSON constant {value}")


def _read_metadata(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ContractError(f"{path}: invalid JSON: {error.msg}") from error
    except OSError as error:
        raise ContractError(f"{path}: {error}") from error
    return _object(value, "export metadata")


def _label_schema_id() -> str:
    payload = json.dumps({"version": 1, "labels": BEHAVIOR_LABELS}, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(payload).hexdigest()


def _validate_model_config(value: Any) -> tuple[Mapping[str, Any], int, int]:
    config = _object(value, "export metadata.model_config")
    _exact_fields(
        config,
        {
            "dropout",
            "feature_dim",
            "gru_layers",
            "hidden_dim",
            "image_size",
            "num_classes",
            "num_frames",
        },
        "export metadata.model_config",
    )
    for field in ("feature_dim", "gru_layers", "hidden_dim", "image_size", "num_frames"):
        _integer(config[field], f"export metadata.model_config.{field}", positive=True)
    if config["image_size"] < 16:
        raise ContractError("export metadata.model_config.image_size must be at least 16")
    if _integer(
        config["num_classes"], "export metadata.model_config.num_classes", positive=True
    ) != len(BEHAVIOR_LABELS):
        raise ContractError("export metadata.model_config.num_classes must be six")
    dropout = _finite_float(config["dropout"], "export metadata.model_config.dropout")
    if type(config["dropout"]) is not float or not 0.0 <= dropout < 1.0:
        raise ContractError("export metadata.model_config.dropout must be a JSON float in [0, 1)")
    return config, config["num_frames"], config["image_size"]  # type: ignore[return-value]


def _validate_export_metadata(
    metadata: Mapping[str, Any],
    model: Path,
    *,
    actual_model_sha256: str,
    expected_onnx_sha256: str,
    logical_latency_ms: int,
    runtime_identity: Mapping[str, Any],
    window_ms: int,
) -> _ModelContract:
    _exact_fields(metadata, _EXPORT_FIELDS, "export metadata")
    if type(metadata["export_format_version"]) is not int or metadata["export_format_version"] != 1:
        raise ContractError("export metadata.export_format_version must be the integer 1")
    if metadata["format"] != "ONNX":
        raise ContractError("export metadata.format must be ONNX")
    opset = _integer(metadata["opset"], "export metadata.opset", positive=True)
    if opset < 17:
        raise ContractError("export metadata.opset must be at least 17")
    if Path(_string(metadata["onnx"], "export metadata.onnx")).name != model.name:
        raise ContractError("export metadata.onnx filename does not match the supplied model")
    _string(metadata["checkpoint"], "export metadata.checkpoint")
    _string(metadata["tensor_rt_note"], "export metadata.tensor_rt_note")
    _digest(metadata["checkpoint_sha256"], "export metadata.checkpoint_sha256")
    declared_model_sha = _digest(metadata["onnx_sha256"], "export metadata.onnx_sha256")
    if actual_model_sha256 != expected_onnx_sha256:
        raise ContractError("supplied ONNX SHA-256 does not match --expected-onnx-sha256")
    if actual_model_sha256 != declared_model_sha:
        raise ContractError("supplied ONNX SHA-256 does not match export metadata")

    model_id = _digest(metadata["artifact_id"], "export metadata.artifact_id")
    labels = metadata["labels"]
    if labels != list(BEHAVIOR_LABELS):
        raise ContractError("export metadata.labels does not match the fixed six-label order")
    label_schema_id = _digest(metadata["label_schema_id"], "export metadata.label_schema_id")
    if label_schema_id != _label_schema_id():
        raise ContractError("export metadata.label_schema_id is inconsistent")
    architecture = _string(metadata["model_architecture"], "export metadata.model_architecture")
    if architecture != MODEL_ARCHITECTURE:
        raise ContractError(f"unsupported model architecture: {architecture}")
    model_config, num_frames, image_size = _validate_model_config(metadata["model_config"])
    expected_model_config_id = sha256(
        stable_json({"model_architecture": architecture, "model_config": model_config}).encode(
            "utf-8"
        )
    ).hexdigest()
    model_config_id = _digest(metadata["model_config_id"], "export metadata.model_config_id")
    if model_config_id != expected_model_config_id:
        raise ContractError("export metadata.model_config_id is inconsistent")
    training_config_id = _digest(
        metadata["training_config_id"], "export metadata.training_config_id"
    )
    training_manifest_sha = _digest(
        metadata["training_manifest_sha256"], "export metadata.training_manifest_sha256"
    )

    input_contract = _object(metadata["input"], "export metadata.input")
    _exact_fields(
        input_contract,
        {
            "color",
            "dtype",
            "layout",
            "mean",
            "name",
            "range_before_normalization",
            "shape",
            "std",
        },
        "export metadata.input",
    )
    input_name = _string(input_contract["name"], "export metadata.input.name")
    if input_name != "frames":
        raise ContractError("export metadata.input.name must be frames")
    input_shape = input_contract["shape"]
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 5
        or input_shape[0] != "batch"
        or any(type(item) is not int for item in input_shape[1:])
        or input_shape != ["batch", num_frames, 3, image_size, image_size]
    ):
        raise ContractError("export metadata.input.shape does not match model_config")
    if (
        input_contract["dtype"] != "float32"
        or input_contract["layout"] != "N,T,C,H,W"
        or input_contract["color"] != "RGB"
    ):
        raise ContractError("export metadata.input tensor contract is unsupported")
    value_range = input_contract["range_before_normalization"]
    if (
        not isinstance(value_range, list)
        or len(value_range) != 2
        or any(type(item) is not float for item in value_range)
        or value_range != [0.0, 1.0]
    ):
        raise ContractError("export metadata.input range must be [0.0, 1.0] JSON floats")
    mean = _exact_float_triplet(input_contract["mean"], (0.5, 0.5, 0.5), "input mean")
    std = _exact_float_triplet(input_contract["std"], (0.5, 0.5, 0.5), "input std")

    output_contract = _object(metadata["output"], "export metadata.output")
    _exact_fields(output_contract, {"label_order", "name", "shape"}, "export metadata.output")
    output_name = _string(output_contract["name"], "export metadata.output.name")
    if output_name != "logits":
        raise ContractError("export metadata.output.name must be logits")
    output_shape = output_contract["shape"]
    if (
        not isinstance(output_shape, list)
        or len(output_shape) != 2
        or output_shape[0] != "batch"
        or type(output_shape[1]) is not int
        or output_shape != ["batch", len(BEHAVIOR_LABELS)]
    ):
        raise ContractError("export metadata.output.shape must be [batch, 6]")
    if output_contract["label_order"] != list(BEHAVIOR_LABELS):
        raise ContractError("export metadata.output.label_order is inconsistent")

    parity = _object(metadata["onnx_runtime_parity"], "export metadata.onnx_runtime_parity")
    _exact_fields(
        parity,
        {"atol", "batch_sizes", "max_absolute_error", "rtol"},
        "export metadata.onnx_runtime_parity",
    )
    if (
        not isinstance(parity["batch_sizes"], list)
        or any(type(item) is not int for item in parity["batch_sizes"])
        or parity["batch_sizes"] != [1, 2]
    ):
        raise ContractError("export metadata parity must cover batch sizes 1 and 2")
    if _finite_float(parity["rtol"], "parity.rtol", non_negative=True) != 1e-4:
        raise ContractError("export metadata parity.rtol is unsupported")
    if _finite_float(parity["atol"], "parity.atol", non_negative=True) != 1e-5:
        raise ContractError("export metadata parity.atol is unsupported")
    _finite_float(parity["max_absolute_error"], "parity.max_absolute_error", non_negative=True)

    embedded_metadata = {
        "foliage_warden.artifact_id": model_id,
        "foliage_warden.input_color": "RGB",
        "foliage_warden.input_layout": "N,T,C,H,W",
        "foliage_warden.label_schema_id": label_schema_id,
        "foliage_warden.labels": json.dumps(list(BEHAVIOR_LABELS), separators=(",", ":")),
        "foliage_warden.model_architecture": architecture,
        "foliage_warden.model_config_id": model_config_id,
        "foliage_warden.rgb_mean": json.dumps(list(mean)),
        "foliage_warden.rgb_std": json.dumps(list(std)),
        "foliage_warden.training_config_id": training_config_id,
        "foliage_warden.training_manifest_sha256": training_manifest_sha,
    }
    config_identity = {
        "adapter": ADAPTER_CONFIG_ID,
        "execution": {
            "graph_optimization": "DISABLED",
            "inter_op_threads": 1,
            "intra_op_threads": 1,
            "provider": "CPUExecutionProvider",
            "sequential": True,
        },
        "export_contract": {
            "label_schema_id": label_schema_id,
            "model_architecture": architecture,
            "model_config_id": model_config_id,
            "training_config_id": training_config_id,
            "training_manifest_sha256": training_manifest_sha,
        },
        "input": dict(input_contract),
        "labels": list(BEHAVIOR_LABELS),
        "preprocessing": {
            "clip_format": CLIP_FORMAT,
            "resize": "opencv-inter-area-v1",
            "temporal_sampling": "numpy-rint-linspace-endpoints-v1",
        },
        "replay_timing": {
            "logical_latency_ms": logical_latency_ms,
            "window_ms": window_ms,
        },
        "runtime_identity": dict(runtime_identity),
        "schema_version": 1,
    }
    return _ModelContract(
        model_id=model_id,
        model_sha256=actual_model_sha256,
        input_name=input_name,
        output_name=output_name,
        num_frames=num_frames,
        image_size=image_size,
        mean=mean,
        std=std,
        config_sha256=sha256(stable_json(config_identity).encode("utf-8")).hexdigest(),
        embedded_metadata=embedded_metadata,
    )


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - exercised from a base-only installation.
        raise ContractError(
            "offline inference dependencies are missing; install foliage-warden-shadow[inference]"
        ) from error
    return np, cv2, ort


def _runtime_identity(np: Any, cv2: Any, ort: Any) -> dict[str, Any]:
    try:
        numpy_build = np.show_config(mode="dicts")
        opencv_build = cv2.getBuildInformation()
        onnxruntime_build = ort.get_build_info()
    except Exception as error:
        raise ContractError(f"could not read native inference build identity: {error}") from error
    if not isinstance(numpy_build, Mapping):
        raise ContractError("NumPy did not provide structured native build identity")
    if not isinstance(opencv_build, str) or not isinstance(onnxruntime_build, str):
        raise ContractError("native inference libraries did not provide build identity text")
    libc_name, libc_version = platform.libc_ver()
    return {
        "native_build_metadata_sha256": {
            "numpy": sha256(stable_json(numpy_build).encode("utf-8")).hexdigest(),
            "onnxruntime": sha256(onnxruntime_build.encode("utf-8")).hexdigest(),
            "opencv": sha256(opencv_build.encode("utf-8")).hexdigest(),
        },
        "packages": {
            "numpy": str(np.__version__),
            "onnxruntime": str(ort.__version__),
            "opencv": str(cv2.__version__),
        },
        "platform": {
            "libc_name": libc_name,
            "libc_version": libc_version,
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": {
            "compiler": platform.python_compiler(),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _build_session(model_bytes: bytes, model_name: Path, contract: _ModelContract, ort: Any) -> Any:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        raise ContractError(f"could not load verified ONNX {model_name} on CPU: {error}") from error
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ContractError("ONNX Runtime did not honor the CPU-only provider contract")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_input_shape = [
        "batch",
        contract.num_frames,
        3,
        contract.image_size,
        contract.image_size,
    ]
    if (
        len(inputs) != 1
        or inputs[0].name != contract.input_name
        or inputs[0].type != "tensor(float)"
        or inputs[0].shape != expected_input_shape
    ):
        raise ContractError("ONNX graph input does not match verified export metadata")
    if (
        len(outputs) != 1
        or outputs[0].name != contract.output_name
        or outputs[0].type != "tensor(float)"
        or outputs[0].shape != ["batch", len(BEHAVIOR_LABELS)]
    ):
        raise ContractError("ONNX graph output does not match verified export metadata")
    embedded = dict(session.get_modelmeta().custom_metadata_map)
    _exact_fields(embedded, _EMBEDDED_FIELDS, "ONNX embedded metadata")
    if embedded != contract.embedded_metadata:
        differing = sorted(
            key for key in _EMBEDDED_FIELDS if embedded.get(key) != contract.embedded_metadata[key]
        )
        raise ContractError(
            "ONNX embedded metadata disagrees with export sidecar: " + ", ".join(differing)
        )
    return session


def _resolve_clip(request: InferenceRequest, manifest_directory: Path) -> Path:
    relative = Path(request.clip_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContractError(f"clip path must be a normalized relative path: {request.clip_path}")
    if relative.suffix not in {".npy", ".npz"}:
        raise ContractError(f"clip path must end in lowercase .npy or .npz: {request.clip_path}")
    try:
        base = manifest_directory.resolve(strict=True)
        resolved = (base / relative).resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as error:
        raise ContractError(f"clip path escapes or does not exist: {request.clip_path}") from error
    if not resolved.is_file():
        raise ContractError(f"clip path is not a regular file: {request.clip_path}")
    return resolved


def _read_bounded_file(path: Path, maximum: int, context: str) -> bytes:
    try:
        with path.open("rb") as stream:
            content = stream.read(maximum + 1)
    except OSError as error:
        raise ContractError(f"{path}: {error}") from error
    if not content or len(content) > maximum:
        raise ContractError(f"{context} size must be within (0, {maximum}] bytes: {path}")
    return content


def _check_clip_container(path: Path, content: bytes) -> None:
    size = len(content)
    if size <= 0 or size > MAX_CLIP_BYTES:
        raise ContractError(f"clip file size must be within (0, {MAX_CLIP_BYTES}] bytes: {path}")
    if path.suffix == ".npz":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) != 1 or entries[0].filename != "frames.npy":
                    raise ContractError("NPZ clip must contain exactly one frames.npy member")
                if entries[0].flag_bits & 0x1:
                    raise ContractError("encrypted NPZ clips are forbidden")
                if entries[0].file_size <= 0 or entries[0].file_size > MAX_CLIP_BYTES:
                    raise ContractError("NPZ frames.npy exceeds the decoded clip size bound")
        except (OSError, zipfile.BadZipFile) as error:
            raise ContractError(f"invalid NPZ clip {path}: {error}") from error


def _load_clip(request: InferenceRequest, manifest_directory: Path, np: Any) -> Any:
    path = _resolve_clip(request, manifest_directory)
    content = _read_bounded_file(path, MAX_CLIP_BYTES, "clip file")
    _check_clip_container(path, content)
    if sha256(content).hexdigest() != request.clip_sha256:
        raise ContractError(f"clip SHA-256 does not match manifest: {request.clip_path}")
    try:
        loaded = np.load(io.BytesIO(content), allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if loaded.files != ["frames"]:
                    raise ContractError("NPZ clip must contain exactly one frames array")
                frames = loaded["frames"]
            finally:
                loaded.close()
        else:
            frames = loaded
    except ContractError:
        raise
    except Exception as error:
        raise ContractError(f"could not decode NumPy clip {request.clip_path}: {error}") from error
    if frames.dtype != np.uint8:
        raise ContractError(f"clip must have exact uint8 dtype: {request.clip_path}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ContractError(f"clip must have exact [T,H,W,3] shape: {request.clip_path}")
    if frames.shape[0] != len(request.frame_timestamps_ms):
        raise ContractError(
            f"clip frame count does not match frame_timestamps_ms: {request.clip_path}"
        )
    if any(type(size) is not int or size <= 0 for size in frames.shape):
        raise ContractError(f"clip dimensions must be positive: {request.clip_path}")
    if frames.nbytes > MAX_CLIP_BYTES:
        raise ContractError(f"decoded clip exceeds {MAX_CLIP_BYTES} bytes: {request.clip_path}")
    return frames


def _preprocess(frames: Any, contract: _ModelContract, np: Any, cv2: Any) -> Any:
    indices = np.rint(np.linspace(0, frames.shape[0] - 1, contract.num_frames)).astype(np.int64)
    try:
        resized = np.stack(
            [
                cv2.resize(
                    frame,
                    (contract.image_size, contract.image_size),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in frames[indices]
            ]
        )
    except Exception as error:
        raise ContractError(f"could not apply deterministic clip resize: {error}") from error
    tensor = resized.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255.0)
    mean = np.asarray(contract.mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(contract.std, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray(((tensor - mean) / std)[None, ...], dtype=np.float32)


def _infer_probabilities(
    session: Any, tensor: Any, contract: _ModelContract, np: Any
) -> dict[str, float]:
    try:
        first = session.run([contract.output_name], {contract.input_name: tensor})[0]
        second = session.run([contract.output_name], {contract.input_name: tensor})[0]
    except Exception as error:
        raise ContractError(f"ONNX Runtime inference failed: {error}") from error
    if first.shape != (1, len(BEHAVIOR_LABELS)) or first.dtype != np.float32:
        raise ContractError("ONNX Runtime returned an invalid logits tensor")
    if not np.isfinite(first).all():
        raise ContractError("ONNX Runtime returned non-finite logits")
    if not np.array_equal(first, second):
        raise ContractError("ONNX Runtime did not return bitwise-repeatable CPU logits")
    logits = first[0].astype(np.float64)
    exponentials = np.exp(logits - np.max(logits))
    values = exponentials / np.sum(exponentials, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ContractError("softmax produced non-finite probabilities")
    return {label: float(values[index]) for index, label in enumerate(BEHAVIOR_LABELS)}


def infer_behavior_predictions(
    perceptions: list[PerceptionObservation],
    requests: list[InferenceRequest],
    *,
    manifest_directory: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    expected_onnx_sha256: str,
    logical_latency_ms: int,
    window_ms: int,
) -> list[BehaviorPrediction]:
    """Verify artifacts, run CPU inference twice, and return strict shadow records."""

    _bind_requests(perceptions, requests)
    expected_onnx_sha = _digest(expected_onnx_sha256, "expected_onnx_sha256")
    logical_latency = _integer(logical_latency_ms, "logical_latency_ms")
    window = _integer(window_ms, "window_ms")
    for request in requests:
        if request.captured_at_ms + logical_latency > MAX_SAFE_INTEGER:
            raise ContractError("logical predicted_at_ms exceeds the maximum safe integer")
        if request.predicted_at_ms != request.captured_at_ms + logical_latency:
            raise ContractError(
                "inference request predicted_at_ms does not match logical_latency_ms"
            )
        if request.captured_at_ms < window:
            raise ContractError("inference request captured_at_ms is earlier than window_ms")
        if request.window_start_captured_at_ms != request.captured_at_ms - window:
            raise ContractError("inference request clip window does not match window_ms")
    model = Path(model_path)
    if model.suffix != ".onnx":
        raise ContractError("model path must end in lowercase .onnx")
    if not model.is_file():
        raise ContractError(f"ONNX model does not exist or is not a file: {model}")
    model_bytes = _read_bounded_file(model, MAX_MODEL_BYTES, "ONNX model")
    actual_model_sha = sha256(model_bytes).hexdigest()
    metadata_file = Path(metadata_path)
    if metadata_file.suffix != ".json" or not metadata_file.is_file():
        raise ContractError(f"export metadata must be an existing .json file: {metadata_file}")
    np, cv2, ort = _load_dependencies()
    contract = _validate_export_metadata(
        _read_metadata(metadata_file),
        model,
        actual_model_sha256=actual_model_sha,
        expected_onnx_sha256=expected_onnx_sha,
        logical_latency_ms=logical_latency,
        runtime_identity=_runtime_identity(np, cv2, ort),
        window_ms=window,
    )
    session = _build_session(model_bytes, model, contract, ort)
    manifest_base = Path(manifest_directory)

    raw_predictions = []
    for request in requests:
        frames = _load_clip(request, manifest_base, np)
        tensor = _preprocess(frames, contract, np, cv2)
        probabilities = _infer_probabilities(session, tensor, contract, np)
        predicted_label = max(BEHAVIOR_LABELS, key=probabilities.__getitem__)
        raw_predictions.append(
            BehaviorPrediction(
                sequence=request.sequence,
                observation_id=request.observation_id,
                frame_id=request.frame_id,
                track_id=request.track_id,
                captured_at_ms=request.captured_at_ms,
                predicted_at_ms=request.predicted_at_ms,
                model_id=contract.model_id,
                model_sha256=contract.model_sha256,
                config_id=ADAPTER_CONFIG_ID,
                config_sha256=contract.config_sha256,
                predicted_label=predicted_label,
                probabilities=probabilities,
            ).to_dict()
        )
    return parse_behavior_stream(raw_predictions)
