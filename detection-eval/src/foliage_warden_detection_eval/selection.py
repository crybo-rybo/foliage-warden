"""Deterministic, class-aware COCO validation subset selection."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass

from .coco import CocoImage, CocoIndex
from .constants import (
    HARD_NEGATIVE_CATEGORY_NAMES,
    SELECTION_ALGORITHM,
    SELECTION_WEIGHTS,
)
from .errors import DetectionEvalError


@dataclass(frozen=True, slots=True)
class SelectedImage:
    image: CocoImage
    stratum: str


@dataclass(frozen=True, slots=True)
class Selection:
    images: tuple[SelectedImage, ...]
    requested_count: int
    planned_count: int
    seed: int
    targets: dict[str, int]
    available: dict[str, int]

    @property
    def actual(self) -> dict[str, int]:
        counts = {name: 0 for name in SELECTION_WEIGHTS}
        for selected in self.images:
            counts[selected.stratum] += 1
        return counts


def _quota(total: int) -> dict[str, int]:
    weight_total = sum(SELECTION_WEIGHTS.values())
    exact = {name: total * weight / weight_total for name, weight in SELECTION_WEIGHTS.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(
        SELECTION_WEIGHTS,
        key=lambda name: (-(exact[name] - result[name]), -SELECTION_WEIGHTS[name], name),
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def _stable_rank(seed: int, stratum: str, image_id: int) -> tuple[str, int]:
    value = hashlib.sha256(
        f"{SELECTION_ALGORITHM}:{seed}:{stratum}:{image_id}".encode()
    ).hexdigest()
    return value, image_id


def select_images(index: CocoIndex, *, max_images: int, seed: int) -> Selection:
    """Select disjoint positive and negative strata with reproducible quotas."""

    if max_images <= 0:
        raise ValueError("max_images must be positive")
    missing = [
        name
        for name in ("cat", "person", *HARD_NEGATIVE_CATEGORY_NAMES)
        if name not in index.category_ids_by_name
    ]
    if missing:
        raise DetectionEvalError(f"COCO annotations lack required categories: {', '.join(missing)}")

    cat_id = index.category_ids_by_name["cat"]
    person_id = index.category_ids_by_name["person"]
    hard_ids = {index.category_ids_by_name[name] for name in HARD_NEGATIVE_CATEGORY_NAMES}
    categories_by_image: dict[int, set[int]] = defaultdict(set)
    for annotation in index.annotations:
        categories_by_image[annotation.image_id].add(annotation.category_id)

    candidates: dict[str, list[CocoImage]] = {name: [] for name in SELECTION_WEIGHTS}
    for image_id, image in sorted(index.images.items()):
        categories = categories_by_image[image_id]
        if cat_id in categories:
            stratum = "cat_positive"
        elif person_id in categories:
            stratum = "person_positive"
        elif categories & hard_ids:
            stratum = "hard_negative"
        else:
            stratum = "background_negative"
        candidates[stratum].append(image)

    for name, values in candidates.items():
        values.sort(key=lambda image: _stable_rank(seed, name, image.image_id))

    planned = min(max_images, len(index.images))
    targets = _quota(planned)
    chosen: list[SelectedImage] = []
    offsets: dict[str, int] = {}
    for name in SELECTION_WEIGHTS:
        take = min(targets[name], len(candidates[name]))
        chosen.extend(SelectedImage(image=image, stratum=name) for image in candidates[name][:take])
        offsets[name] = take

    # A deficient stratum cannot silently shrink the subset. Refill in weighted
    # priority order, taking the next deterministically ranked candidate each pass.
    priority = sorted(SELECTION_WEIGHTS, key=lambda name: (-SELECTION_WEIGHTS[name], name))
    while len(chosen) < planned:
        progress = False
        for name in priority:
            offset = offsets[name]
            if offset < len(candidates[name]):
                chosen.append(SelectedImage(image=candidates[name][offset], stratum=name))
                offsets[name] += 1
                progress = True
                if len(chosen) == planned:
                    break
        if not progress:
            break

    chosen.sort(key=lambda item: item.image.image_id)
    return Selection(
        images=tuple(chosen),
        requested_count=max_images,
        planned_count=planned,
        seed=seed,
        targets=targets,
        available={name: len(values) for name, values in candidates.items()},
    )
