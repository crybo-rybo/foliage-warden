"""Dependency-free loading of the COCO instances subset used by this harness."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DetectionEvalError


@dataclass(frozen=True, slots=True)
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int
    license_id: int | None


@dataclass(frozen=True, slots=True)
class CocoAnnotation:
    annotation_id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    iscrowd: bool


@dataclass(frozen=True, slots=True)
class CocoIndex:
    images: dict[int, CocoImage]
    annotations: tuple[CocoAnnotation, ...]
    category_ids_by_name: dict[str, int]
    category_names_by_id: dict[int, str]
    licenses: tuple[dict[str, Any], ...]


def _integer(value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DetectionEvalError(f"{context} must be an integer")
    if positive and value <= 0:
        raise DetectionEvalError(f"{context} must be positive")
    return value


def _object_list(root: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = root.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DetectionEvalError(f"COCO {key} must be an array of objects")
    return value


def load_coco(path: Path) -> CocoIndex:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DetectionEvalError(f"COCO annotation file not found: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise DetectionEvalError(f"cannot parse COCO annotations {path}: {error}") from error
    if not isinstance(root, dict):
        raise DetectionEvalError("COCO annotation root must be an object")

    images: dict[int, CocoImage] = {}
    for raw in _object_list(root, "images"):
        image_id = _integer(raw.get("id"), "image.id", positive=True)
        file_name = raw.get("file_name")
        if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
            raise DetectionEvalError(f"image {image_id} has an unsafe file_name")
        if image_id in images:
            raise DetectionEvalError(f"duplicate COCO image id {image_id}")
        license_id = raw.get("license")
        if license_id is not None:
            license_id = _integer(license_id, f"image {image_id}.license")
        images[image_id] = CocoImage(
            image_id=image_id,
            file_name=file_name,
            width=_integer(raw.get("width"), f"image {image_id}.width", positive=True),
            height=_integer(raw.get("height"), f"image {image_id}.height", positive=True),
            license_id=license_id,
        )

    category_ids_by_name: dict[str, int] = {}
    category_names_by_id: dict[int, str] = {}
    for raw in _object_list(root, "categories"):
        category_id = _integer(raw.get("id"), "category.id", positive=True)
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise DetectionEvalError(f"category {category_id} has no name")
        if category_id in category_names_by_id or name in category_ids_by_name:
            raise DetectionEvalError(f"duplicate COCO category {category_id}/{name!r}")
        category_ids_by_name[name] = category_id
        category_names_by_id[category_id] = name

    annotations: list[CocoAnnotation] = []
    annotation_ids: set[int] = set()
    for raw in _object_list(root, "annotations"):
        annotation_id = _integer(raw.get("id"), "annotation.id", positive=True)
        image_id = _integer(
            raw.get("image_id"), f"annotation {annotation_id}.image_id", positive=True
        )
        category_id = _integer(
            raw.get("category_id"), f"annotation {annotation_id}.category_id", positive=True
        )
        if annotation_id in annotation_ids:
            raise DetectionEvalError(f"duplicate COCO annotation id {annotation_id}")
        if image_id not in images:
            raise DetectionEvalError(
                f"annotation {annotation_id} references unknown image {image_id}"
            )
        if category_id not in category_names_by_id:
            raise DetectionEvalError(
                f"annotation {annotation_id} references unknown category {category_id}"
            )
        bbox = raw.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox)
            or any(not math.isfinite(float(value)) for value in bbox)
            or float(bbox[2]) <= 0.0
            or float(bbox[3]) <= 0.0
        ):
            raise DetectionEvalError(f"annotation {annotation_id}.bbox must be positive [x,y,w,h]")
        raw_iscrowd = raw.get("iscrowd", 0)
        if not (
            isinstance(raw_iscrowd, bool)
            or (isinstance(raw_iscrowd, int) and raw_iscrowd in (0, 1))
        ):
            raise DetectionEvalError(f"annotation {annotation_id}.iscrowd must be 0 or 1")
        annotation_ids.add(annotation_id)
        annotations.append(
            CocoAnnotation(
                annotation_id=annotation_id,
                image_id=image_id,
                category_id=category_id,
                bbox=tuple(float(value) for value in bbox),
                iscrowd=bool(raw_iscrowd),
            )
        )

    raw_licenses = root.get("licenses", [])
    if not isinstance(raw_licenses, list) or any(
        not isinstance(item, dict) for item in raw_licenses
    ):
        raise DetectionEvalError("COCO licenses must be an array of objects")
    return CocoIndex(
        images=images,
        annotations=tuple(annotations),
        category_ids_by_name=category_ids_by_name,
        category_names_by_id=category_names_by_id,
        licenses=tuple(raw_licenses),
    )
