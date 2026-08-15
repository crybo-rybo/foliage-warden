from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .labels import BEHAVIOR_LABELS
from .runtime import atomic_write_json

SPLIT_DAYS = {
    "train": "2099-01-01",
    "val": "2099-01-02",
    "test": "2099-01-03",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate obviously synthetic clips for training-pipeline smoke tests"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clips-per-label", type=int, default=2)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser


def _make_clip(
    *,
    label_index: int,
    clip_index: int,
    frames: int,
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clip = rng.integers(0, 18, (frames, image_size, image_size, 3), dtype=np.uint8)
    colors = np.asarray(
        [
            (220, 50, 50),
            (50, 220, 50),
            (50, 50, 220),
            (220, 180, 40),
            (180, 50, 200),
            (100, 100, 100),
        ],
        dtype=np.uint8,
    )
    color = colors[label_index]
    size = max(4, image_size // 5)
    travel = max(1, image_size - size)
    for frame_index in range(frames):
        phase = frame_index / max(frames - 1, 1)
        if label_index in (0, 1):
            x = int(phase * travel)
            y = image_size // 4 if label_index == 0 else image_size // 2
        elif label_index in (2, 3):
            x = image_size // 2
            frequency = 2 if label_index == 2 else 4
            y = int((0.5 + 0.3 * np.sin(phase * np.pi * frequency)) * travel)
        else:
            x = int(rng.integers(0, travel + 1))
            y = int(rng.integers(0, travel + 1))
        clip[frame_index, y : y + size, x : x + size] = color

    # The border makes these fixtures intentionally unlike camera footage.
    border = 1 + clip_index % 2
    clip[:, :border] = color
    clip[:, -border:] = color
    return clip


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.clips_per_label <= 0 or args.frames <= 0 or args.image_size < 16 or args.seed < 0:
        raise ValueError(
            "clips-per-label and frames must be positive, image-size >= 16, and seed non-negative"
        )
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"
    provenance_path = output_dir / "SYNTHETIC_ONLY.json"
    if manifest_path.exists() or provenance_path.exists():
        raise FileExistsError(
            f"refusing to replace an existing synthetic dataset in {output_dir}; "
            "choose a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for split_index, (split, day) in enumerate(SPLIT_DAYS.items()):
        for label_index, label in enumerate(BEHAVIOR_LABELS):
            for clip_index in range(args.clips_per_label):
                clip_id = f"synthetic-{split}-{label.lower()}-{clip_index:03d}"
                relative_path = Path("clips") / split / f"{clip_id}.npz"
                path = output_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                clip_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                    + split_index
                    + label_index
                    + clip_index
                )
                clip = _make_clip(
                    label_index=label_index,
                    clip_index=clip_index,
                    frames=args.frames,
                    image_size=args.image_size,
                    rng=clip_rng,
                )
                np.savez_compressed(path, frames=clip)
                rows.append(
                    {
                        "clip_id": clip_id,
                        "path": relative_path.as_posix(),
                        "label": label,
                        "split": split,
                        "session_id": f"synthetic-session-{split}-{clip_index:03d}",
                        "day": day,
                        "camera_id": "synthetic-generator-v1",
                        "staged_safe": True,
                        "metadata": {
                            "synthetic": True,
                            "pipeline_verification_only": True,
                        },
                    }
                )

    with manifest_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "synthetic": True,
        "pipeline_verification_only": True,
        "performance_claims_allowed": False,
        "warning": (
            "These generated arrays contain artificial colored patterns. They must never be used "
            "to estimate camera, cat, household, or behavior-classification performance."
        ),
        "seed": args.seed,
        "clips": len(rows),
        "clips_per_label_per_split": args.clips_per_label,
        "frames": args.frames,
        "image_size": args.image_size,
        "labels": list(BEHAVIOR_LABELS),
        "manifest": str(manifest_path),
    }
    atomic_write_json(provenance_path, provenance)
    return provenance


def main() -> None:
    args = build_parser().parse_args()
    result = generate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
