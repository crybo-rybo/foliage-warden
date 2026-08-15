from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .labels import BEHAVIOR_LABELS, LABEL_TO_INDEX
from .manifest import ClipRecord

RGB_MEAN = (0.5, 0.5, 0.5)
RGB_STD = (0.5, 0.5, 0.5)


def _stable_seed(seed: int, epoch: int, clip_id: str) -> int:
    value = f"{seed}:{epoch}:{clip_id}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "little")


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    *,
    seed: int,
    training: bool,
) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("clip must contain at least one frame")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    if not training:
        return np.rint(np.linspace(0, total_frames - 1, num_frames)).astype(np.int64)

    rng = np.random.default_rng(seed)
    boundaries = np.linspace(0, total_frames, num_frames + 1)
    indices = []
    for start_float, end_float in zip(boundaries[:-1], boundaries[1:], strict=True):
        start = min(int(np.floor(start_float)), total_frames - 1)
        end = min(max(int(np.ceil(end_float)), start + 1), total_frames)
        indices.append(int(rng.integers(start, end)))
    return np.asarray(indices, dtype=np.int64)


def _load_array_clip(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "frames" not in loaded.files:
                raise ValueError(f"NPZ clip must contain a 'frames' array: {path}")
            frames = loaded["frames"]
        finally:
            loaded.close()
        return frames
    return loaded


def _load_video_clip(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video clip: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video clip contains no decodable frames: {path}")
    return np.stack(frames)


def load_clip(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".npy", ".npz"}:
        frames = _load_array_clip(path)
    else:
        frames = _load_video_clip(path)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"clip must have shape [T,H,W,3], got {frames.shape} from {path}")
    if frames.shape[0] == 0:
        raise ValueError(f"clip contains no frames: {path}")
    if frames.dtype != np.uint8:
        if not np.issubdtype(frames.dtype, np.number):
            raise ValueError(f"clip frames must be numeric, got {frames.dtype} from {path}")
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


class ClipDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        records: list[ClipRecord],
        *,
        num_frames: int,
        image_size: int,
        seed: int,
        training: bool,
    ) -> None:
        if not records:
            raise ValueError("dataset requires at least one record")
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.records = records
        self.num_frames = num_frames
        self.image_size = image_size
        self.seed = seed
        self.training = training
        self.epoch = 0
        self._mean = torch.tensor(RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(RGB_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        frames = load_clip(record.path)
        indices = sample_frame_indices(
            frames.shape[0],
            self.num_frames,
            seed=_stable_seed(self.seed, self.epoch, record.clip_id),
            training=self.training,
        )
        sampled = frames[indices]
        resized = np.stack(
            [
                cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                for frame in sampled
            ]
        )
        tensor = torch.from_numpy(resized.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = (tensor - self._mean) / self._std
        return {
            "frames": tensor,
            "label": torch.tensor(LABEL_TO_INDEX[record.label], dtype=torch.long),
            "clip_id": record.clip_id,
        }


def compute_class_weights(records: list[ClipRecord]) -> Tensor:
    counts = Counter(record.label for record in records)
    present = [count for count in counts.values() if count > 0]
    if not present:
        raise ValueError("cannot compute class weights without records")
    total = sum(present)
    class_count = len(present)
    weights = [
        total / (class_count * counts[label]) if counts[label] else 0.0
        for label in BEHAVIOR_LABELS
    ]
    return torch.tensor(weights, dtype=torch.float32)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
