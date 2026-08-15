from __future__ import annotations

from pathlib import Path

import pytest
import torch

from foliage_warden_training.artifacts import (
    load_checkpoint,
    make_training_metadata,
    save_checkpoint,
)
from foliage_warden_training.model import ModelConfig, TemporalCnnGru


def _metadata(tmp_path: Path) -> dict[str, object]:
    return make_training_metadata(
        manifest_path=tmp_path / "manifest.jsonl",
        manifest_hash="a" * 64,
        manifest_summary={"clips": 1},
        model_config=ModelConfig(num_frames=2, image_size=16, feature_dim=8, hidden_dim=8),
        training_config={"seed": 7, "epochs": 1},
    )


def test_checkpoint_round_trip_preserves_identity(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    checkpoint = tmp_path / "model.pt"
    model = TemporalCnnGru(ModelConfig(num_frames=2, image_size=16, feature_dim=8, hidden_dim=8))
    save_checkpoint(checkpoint, model=model, metadata=metadata, epoch=1, metrics={"loss": 1.0})

    loaded, loaded_metadata, payload = load_checkpoint(checkpoint)

    assert loaded_metadata["artifact_id"] == metadata["artifact_id"]
    assert len(loaded_metadata["model_config_id"]) == 64
    assert len(loaded_metadata["training_config_id"]) == 64
    assert payload["epoch"] == 1
    torch.testing.assert_close(loaded.state_dict(), model.state_dict())


def test_checkpoint_rejects_tampered_identity(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    metadata["training_config"]["seed"] = 99
    checkpoint = tmp_path / "model.pt"
    model = TemporalCnnGru(ModelConfig(num_frames=2, image_size=16, feature_dim=8, hidden_dim=8))
    save_checkpoint(checkpoint, model=model, metadata=metadata, epoch=1, metrics={})

    with pytest.raises(ValueError, match="training configuration identity"):
        load_checkpoint(checkpoint)
