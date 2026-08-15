from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .labels import BEHAVIOR_LABELS, LABEL_SCHEMA_VERSION, label_schema_id
from .model import MODEL_ARCHITECTURE, ModelConfig, TemporalCnnGru
from .runtime import canonical_sha256

ARTIFACT_FORMAT_VERSION = 1


def _artifact_identity(
    *,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    manifest_hash: str,
) -> dict[str, Any]:
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "label_schema_id": label_schema_id(),
        "model_architecture": MODEL_ARCHITECTURE,
        "model_config": model_config,
        "training_config": training_config,
        "training_manifest_sha256": manifest_hash,
    }


def make_training_metadata(
    *,
    manifest_path: Path,
    manifest_hash: str,
    manifest_summary: dict[str, Any],
    model_config: ModelConfig,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    model_config_id = canonical_sha256(
        {
            "model_architecture": MODEL_ARCHITECTURE,
            "model_config": model_config.to_dict(),
        }
    )
    training_config_id = canonical_sha256(training_config)
    identity = _artifact_identity(
        model_config=model_config.to_dict(),
        training_config=training_config,
        manifest_hash=manifest_hash,
    )
    return {
        **identity,
        "artifact_id": canonical_sha256(identity),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": list(BEHAVIOR_LABELS),
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "model_config_id": model_config_id,
        "training_config_id": training_config_id,
        "training_manifest": str(manifest_path.resolve()),
        "manifest_summary": manifest_summary,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: TemporalCnnGru,
    metadata: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "model_state_dict": state_dict,
            "metadata": metadata,
            "epoch": epoch,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[TemporalCnnGru, dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - supports the oldest allowed PyTorch.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid checkpoint payload: {checkpoint_path}")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint has no metadata: {checkpoint_path}")
    if metadata.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint artifact format")
    if metadata.get("model_architecture") != MODEL_ARCHITECTURE:
        raise ValueError(f"unsupported model architecture: {metadata.get('model_architecture')!r}")
    if metadata.get("labels") != list(BEHAVIOR_LABELS):
        raise ValueError("checkpoint label order does not match this training package")
    if metadata.get("label_schema_id") != label_schema_id():
        raise ValueError("checkpoint label schema identity does not match this training package")
    config_value = metadata.get("model_config")
    if not isinstance(config_value, dict):
        raise ValueError("checkpoint has no valid model configuration")
    training_config = metadata.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("checkpoint has no valid training configuration")
    manifest_hash = metadata.get("training_manifest_sha256")
    if not isinstance(manifest_hash, str):
        raise ValueError("checkpoint has no valid training manifest identity")
    expected_model_config_id = canonical_sha256(
        {"model_architecture": MODEL_ARCHITECTURE, "model_config": config_value}
    )
    if metadata.get("model_config_id") != expected_model_config_id:
        raise ValueError("checkpoint model configuration identity is inconsistent")
    if metadata.get("training_config_id") != canonical_sha256(training_config):
        raise ValueError("checkpoint training configuration identity is inconsistent")
    expected_artifact_id = canonical_sha256(
        _artifact_identity(
            model_config=config_value,
            training_config=training_config,
            manifest_hash=manifest_hash,
        )
    )
    if metadata.get("artifact_id") != expected_artifact_id:
        raise ValueError("checkpoint artifact identity is inconsistent")
    model = TemporalCnnGru(ModelConfig.from_dict(config_value))
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no model state")
    model.load_state_dict(state_dict, strict=True)
    return model, metadata, checkpoint
