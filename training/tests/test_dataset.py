from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from foliage_warden_training.dataset import (
    ClipDataset,
    compute_class_weights,
    sample_frame_indices,
)
from foliage_warden_training.manifest import ClipRecord


def _record(path: Path, clip_id: str = "clip-1", label: str = "PASSING") -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        path=path,
        source_path=path.name,
        label=label,
        split="train",
        session_id="session-1",
        day="2026-08-14",
        staged_safe=False,
        camera_id=None,
        metadata={},
    )


def test_eval_sampling_is_even_and_training_sampling_is_seeded() -> None:
    assert sample_frame_indices(10, 4, seed=1, training=False).tolist() == [0, 3, 6, 9]
    first = sample_frame_indices(100, 8, seed=99, training=True)
    second = sample_frame_indices(100, 8, seed=99, training=True)
    different = sample_frame_indices(100, 8, seed=100, training=True)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_dataset_is_deterministic_for_clip_and_epoch(tmp_path: Path) -> None:
    path = tmp_path / "clip.npz"
    frames = np.stack([np.full((20, 20, 3), value, np.uint8) for value in range(32)])
    np.savez_compressed(path, frames=frames)
    dataset = ClipDataset([_record(path)], num_frames=4, image_size=16, seed=7, training=True)

    first = dataset[0]["frames"]
    second = dataset[0]["frames"]
    assert isinstance(first, torch.Tensor)
    torch.testing.assert_close(first, second)

    dataset.set_epoch(1)
    third = dataset[0]["frames"]
    assert isinstance(third, torch.Tensor)
    assert not torch.equal(first, third)


def test_inverse_frequency_class_weights() -> None:
    records = [
        _record(Path("a"), clip_id="a", label="PASSING"),
        _record(Path("b"), clip_id="b", label="PASSING"),
        _record(Path("c"), clip_id="c", label="EATING"),
    ]

    weights = compute_class_weights(records)

    assert weights.tolist() == [0.75, 0.0, 1.5, 0.0, 0.0, 0.0]
