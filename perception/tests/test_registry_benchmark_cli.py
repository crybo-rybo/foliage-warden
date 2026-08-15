from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from foliage_warden_perception.benchmark import BenchmarkAccumulator
from foliage_warden_perception.cli import build_parser, main
from foliage_warden_perception.errors import ModelError
from foliage_warden_perception.registry import load_model_spec, resolve_and_verify_model


def _registry(filename: str, digest: str) -> dict[str, object]:
    return {
        "version": 1,
        "models": {
            "yolox_s_opencv_zoo": {
                "description": "test model",
                "filename": filename,
                "format": "onnx",
                "input": {"color": "RGB", "height": 640, "width": 640},
                "license": "Apache-2.0",
                "relevant_classes": {"cat": 15, "person": 0},
                "sha256": digest,
                "source_revision": "opencv/opencv_zoo@test",
                "url": "https://example.invalid/model.onnx",
            }
        },
    }


def test_registry_load_and_checksum_verification(tmp_path: Path) -> None:
    payload = b"small synthetic model placeholder"
    digest = hashlib.sha256(payload).hexdigest()
    model = tmp_path / "model.onnx"
    model.write_bytes(payload)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(_registry(model.name, digest)), encoding="utf-8")

    spec = load_model_spec(registry)

    assert spec.input_width == spec.input_height == 640
    assert spec.relevant_classes == {0: "person", 15: "cat"}
    assert resolve_and_verify_model(registry, spec) == model


def test_model_missing_and_checksum_mismatch_fail_before_inference(tmp_path: Path) -> None:
    digest = "0" * 64
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(_registry("missing.onnx", digest)), encoding="utf-8")
    spec = load_model_spec(registry)
    with pytest.raises(ModelError, match="not found"):
        resolve_and_verify_model(registry, spec)

    model = tmp_path / "different.onnx"
    model.write_bytes(b"wrong bytes")
    with pytest.raises(ModelError, match="checksum mismatch"):
        resolve_and_verify_model(registry, spec, model)


def test_cli_reports_missing_model_without_opencv_or_image_access(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(_registry("missing.onnx", "0" * 64)), encoding="utf-8")

    result = main(["image", str(tmp_path / "also-missing.jpg"), "--registry", str(registry)])

    assert result == 2
    assert "model artifact not found" in capsys.readouterr().err


def test_cli_defaults_to_headless_unbounded_observation() -> None:
    args = build_parser().parse_args(["camera"])

    assert args.max_frames is None
    assert args.output == "-"
    assert args.benchmark is False
    assert not hasattr(args, "display")
    assert not hasattr(args, "record")


def test_benchmark_summary_percentiles_and_fps() -> None:
    benchmark = BenchmarkAccumulator()
    for value in (10.0, 20.0, 30.0):
        benchmark.add("total", value)
        benchmark.add("inference", value / 2)

    report = benchmark.to_dict()

    assert report["frame_count"] == 3
    assert report["effective_fps"] == 50.0
    assert report["stages"]["total"] == {
        "count": 3,
        "max_ms": 30.0,
        "mean_ms": 20.0,
        "min_ms": 10.0,
        "p50_ms": 20.0,
        "p95_ms": 29.0,
    }
