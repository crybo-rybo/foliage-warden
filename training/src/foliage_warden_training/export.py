from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch

from .artifacts import load_checkpoint
from .dataset import RGB_MEAN, RGB_STD
from .runtime import atomic_write_json, file_sha256, set_determinism


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a behavior checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx file")
    parser.add_argument("--metadata-output", type=Path, help="Default: <output>.metadata.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def export_onnx(args: argparse.Namespace) -> dict[str, Any]:
    if args.opset < 17:
        raise ValueError("opset must be at least 17 for the supported TensorRT-oriented export")
    if args.seed < 0:
        raise ValueError("seed cannot be negative")
    if args.output.suffix.lower() != ".onnx":
        raise ValueError("output must use the .onnx extension")
    set_determinism(args.seed, enable_cuda=False)
    model, metadata, _ = load_checkpoint(args.checkpoint)
    model.eval()
    config = model.config
    example = torch.zeros(
        1,
        config.num_frames,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model,
        example,
        temporary,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["frames"],
        output_names=["logits"],
        dynamic_axes={"frames": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx_model = onnx.load(temporary)
    onnx.checker.check_model(onnx_model)
    properties = {
        "foliage_warden.artifact_id": str(metadata["artifact_id"]),
        "foliage_warden.label_schema_id": str(metadata["label_schema_id"]),
        "foliage_warden.labels": json.dumps(metadata["labels"], separators=(",", ":")),
        "foliage_warden.model_architecture": str(metadata["model_architecture"]),
        "foliage_warden.model_config_id": str(metadata["model_config_id"]),
        "foliage_warden.training_config_id": str(metadata["training_config_id"]),
        "foliage_warden.training_manifest_sha256": str(metadata["training_manifest_sha256"]),
        "foliage_warden.input_layout": "N,T,C,H,W",
        "foliage_warden.input_color": "RGB",
        "foliage_warden.rgb_mean": json.dumps(RGB_MEAN),
        "foliage_warden.rgb_std": json.dumps(RGB_STD),
    }
    del onnx_model.metadata_props[:]
    for key, value in properties.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(onnx_model, temporary)

    session = onnxruntime.InferenceSession(
        str(temporary),
        providers=["CPUExecutionProvider"],
    )
    parity_max_absolute_error = 0.0
    parity_generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        for batch_size in (1, 2):
            parity_input = torch.rand(
                batch_size,
                config.num_frames,
                3,
                config.image_size,
                config.image_size,
                generator=parity_generator,
            ).mul_(2.0).sub_(1.0)
            torch_logits = model(parity_input).numpy()
            onnx_logits = session.run(["logits"], {"frames": parity_input.numpy()})[0]
            np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-4, atol=1e-5)
            parity_max_absolute_error = max(
                parity_max_absolute_error,
                float(np.max(np.abs(torch_logits - onnx_logits))),
            )
    temporary.replace(output)

    export_metadata = {
        "export_format_version": 1,
        "format": "ONNX",
        "opset": args.opset,
        "onnx": str(output),
        "onnx_sha256": file_sha256(output),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "artifact_id": metadata["artifact_id"],
        "labels": metadata["labels"],
        "label_schema_id": metadata["label_schema_id"],
        "model_architecture": metadata["model_architecture"],
        "model_config": metadata["model_config"],
        "model_config_id": metadata["model_config_id"],
        "training_config_id": metadata["training_config_id"],
        "training_manifest_sha256": metadata["training_manifest_sha256"],
        "input": {
            "name": "frames",
            "shape": [
                "batch",
                config.num_frames,
                3,
                config.image_size,
                config.image_size,
            ],
            "dtype": "float32",
            "layout": "N,T,C,H,W",
            "color": "RGB",
            "range_before_normalization": [0.0, 1.0],
            "mean": list(RGB_MEAN),
            "std": list(RGB_STD),
        },
        "output": {
            "name": "logits",
            "shape": ["batch", len(metadata["labels"])],
            "label_order": metadata["labels"],
        },
        "tensor_rt_note": (
            "Batch is dynamic; temporal length and spatial dimensions are fixed. Build the "
            "TensorRT engine on the target Jetson and validate numerical parity before deployment."
        ),
        "onnx_runtime_parity": {
            "batch_sizes": [1, 2],
            "rtol": 1e-4,
            "atol": 1e-5,
            "max_absolute_error": parity_max_absolute_error,
        },
    }
    metadata_output = args.metadata_output or output.with_suffix(".metadata.json")
    atomic_write_json(metadata_output, export_metadata)
    return export_metadata


def main() -> None:
    args = build_parser().parse_args()
    metadata = export_onnx(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
