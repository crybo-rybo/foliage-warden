from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .artifacts import load_checkpoint
from .dataset import ClipDataset, seed_worker
from .engine import evaluate_model
from .manifest import load_manifest, manifest_sha256, summarize_records
from .runtime import atomic_write_json, file_sha256, select_device, set_determinism


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a temporal behavior checkpoint")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a JSON report (stdout is always printed)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--omit-predictions",
        action="store_true",
        help="Omit per-clip probabilities from the report",
    )
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.num_workers < 0 or args.seed < 0:
        raise ValueError("batch-size must be positive; num-workers and seed cannot be negative")
    device = select_device(args.device)
    set_determinism(args.seed, enable_cuda=device.type == "cuda")
    model, metadata, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)

    manifest_path = args.manifest.resolve()
    records = load_manifest(manifest_path)
    split_records = [record for record in records if record.split == args.split]
    if not split_records:
        raise ValueError(f"manifest contains no clips in split {args.split!r}")
    dataset = ClipDataset(
        split_records,
        num_frames=model.config.num_frames,
        image_size=model.config.image_size,
        seed=args.seed,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.num_workers > 0,
    )
    metrics, predictions = evaluate_model(
        model,
        loader,
        device=device,
        include_predictions=not args.omit_predictions,
    )
    evaluation_manifest_hash = manifest_sha256(manifest_path)
    report = {
        "report_type": "clip_classification_evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "artifact_id": metadata["artifact_id"],
        "label_schema_id": metadata["label_schema_id"],
        "model_architecture": metadata["model_architecture"],
        "model_config": metadata["model_config"],
        "model_config_id": metadata["model_config_id"],
        "training_config_id": metadata["training_config_id"],
        "training_manifest_sha256": metadata["training_manifest_sha256"],
        "split": args.split,
        "seed": args.seed,
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": evaluation_manifest_hash,
        "matches_training_manifest": evaluation_manifest_hash
        == metadata["training_manifest_sha256"],
        "manifest_summary": summarize_records(records),
        "evaluated_clips_are_entirely_synthetic": all(
            record.metadata.get("synthetic") is True for record in split_records
        ),
        "metrics": metrics,
    }
    if predictions:
        report["predictions"] = predictions
    if report["evaluated_clips_are_entirely_synthetic"]:
        report["performance_warning"] = (
            "Synthetic evaluation verifies pipeline mechanics only and is not evidence of "
            "real-world performance."
        )
    if args.output:
        atomic_write_json(args.output, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
