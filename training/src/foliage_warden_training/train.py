from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import make_training_metadata, save_checkpoint
from .dataset import ClipDataset, compute_class_weights, seed_worker
from .engine import evaluate_model
from .labels import BEHAVIOR_LABELS
from .manifest import load_manifest, manifest_sha256, summarize_records
from .model import ModelConfig, TemporalCnnGru, initialize_weights, parameter_count
from .runtime import atomic_write_json, select_device, set_determinism


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the temporal cat behavior baseline")
    parser.add_argument("--manifest", type=Path, required=True, help="Leakage-safe JSONL manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--uniform-class-weights",
        action="store_true",
        help="Disable inverse-frequency weighting (not recommended for imbalanced real data)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "gru_layers": args.gru_layers,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these options must be positive: {', '.join(invalid)}")
    if args.num_workers < 0 or args.seed < 0 or args.weight_decay < 0 or args.grad_clip < 0:
        raise ValueError("num-workers, seed, weight-decay, and grad-clip cannot be negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix or replace artifacts in non-empty output directory: {output_dir}"
        )
    device = select_device(args.device)
    set_determinism(args.seed, enable_cuda=device.type == "cuda")
    manifest_path = args.manifest.resolve()
    records = load_manifest(manifest_path)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    if not train_records or not val_records:
        raise ValueError("training requires non-empty train and val splits")
    missing_train_labels = [
        label
        for label in BEHAVIOR_LABELS
        if not any(record.label == label for record in train_records)
    ]
    if missing_train_labels:
        raise ValueError(
            "train split must represent every behavior label; missing: "
            + ", ".join(missing_train_labels)
        )

    model_config = ModelConfig(
        num_frames=args.num_frames,
        image_size=args.image_size,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    )
    model = TemporalCnnGru(model_config)
    initialize_weights(model)
    model.to(device)

    train_dataset = ClipDataset(
        train_records,
        num_frames=model_config.num_frames,
        image_size=model_config.image_size,
        seed=args.seed,
        training=True,
    )
    val_dataset = ClipDataset(
        val_records,
        num_frames=model_config.num_frames,
        image_size=model_config.image_size,
        seed=args.seed,
        training=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    class_weights = (
        torch.ones(model_config.num_classes, dtype=torch.float32)
        if args.uniform_class_weights
        else compute_class_weights(train_records)
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    training_config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
        "requested_device": args.device,
        "selected_device": str(device),
        "deterministic_algorithms": True,
        "temporal_sampling": "seeded_stratified_per_clip_per_epoch",
        "class_weighting": "uniform" if args.uniform_class_weights else "inverse_frequency",
        "class_weights": class_weights.tolist(),
        "optimizer": "AdamW",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    summary = summarize_records(records)
    metadata = make_training_metadata(
        manifest_path=manifest_path,
        manifest_hash=manifest_sha256(manifest_path),
        manifest_summary=summary,
        model_config=model_config,
        training_config=training_config,
    )
    metadata["parameters"] = parameter_count(model)
    metadata["dataset_is_entirely_synthetic"] = all(
        record.metadata.get("synthetic") is True for record in records
    )
    if metadata["dataset_is_entirely_synthetic"]:
        metadata["performance_warning"] = (
            "Synthetic clips verify pipeline mechanics only; these metrics say nothing about real "
            "behavior."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "metadata.json", metadata)
    history_path = output_dir / "history.jsonl"
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}

    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_dataset.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            samples = 0
            for batch in train_loader:
                frames = batch["frames"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(frames)
                loss = criterion(logits, labels)
                loss.backward()
                if args.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                batch_size = labels.shape[0]
                running_loss += float(loss.item()) * batch_size
                samples += batch_size

            val_metrics, _ = evaluate_model(model, val_loader, device=device, criterion=criterion)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": running_loss / samples,
                "val": val_metrics,
            }
            history.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
            history.flush()
            score = float(val_metrics["macro_f1_supported_classes"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = epoch_metrics
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    metadata=metadata,
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                metadata=metadata,
                epoch=epoch,
                metrics=epoch_metrics,
            )

    result = {
        "artifact_id": metadata["artifact_id"],
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "dataset_is_entirely_synthetic": metadata["dataset_is_entirely_synthetic"],
    }
    atomic_write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
