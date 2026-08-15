"""COCO subset preparation, manifests, and input verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import ensure_coco_annotations, ensure_remote_image, file_hashes, sha256_file
from .coco import CocoIndex, load_coco
from .constants import (
    COCO_ANNOTATIONS_MD5,
    COCO_ANNOTATIONS_SHA256,
    COCO_ANNOTATIONS_SIZE,
    COCO_ANNOTATIONS_URL,
    COCO_DATASET_NAME,
    COCO_IMAGE_URL_TEMPLATE,
    COCO_INSTANCES_MEMBER,
    COCO_INSTANCES_SHA256,
    HARD_NEGATIVE_CATEGORY_NAMES,
    SELECTION_ALGORITHM,
    SELECTION_WEIGHTS,
)
from .errors import DetectionEvalError
from .selection import Selection, select_images


@dataclass(frozen=True, slots=True)
class PreparedSubset:
    annotation_path: Path
    dataset_root: Path
    image_dir: Path
    index: CocoIndex
    manifest_path: Path
    manifest: dict[str, Any]


def _annotation_path(root: Path) -> Path:
    return root / COCO_INSTANCES_MEMBER


def _stable_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_official_annotation(path: Path) -> None:
    actual = sha256_file(path)
    if actual != COCO_INSTANCES_SHA256:
        raise DetectionEvalError(
            f"{path} is not the pinned COCO 2017 val instances file: SHA-256 "
            f"{actual} != {COCO_INSTANCES_SHA256}"
        )


def _manifest_for(
    selection: Selection,
    index: CocoIndex,
    image_identities: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    for selected in selection.images:
        image = selected.image
        images.append(
            {
                "file_name": image.file_name,
                "height": image.height,
                "id": image.image_id,
                "license_id": image.license_id,
                "stratum": selected.stratum,
                "verified_content": image_identities[image.image_id],
                "width": image.width,
            }
        )
    return {
        "annotations": {
            "archive_md5": COCO_ANNOTATIONS_MD5,
            "archive_sha256": COCO_ANNOTATIONS_SHA256,
            "archive_size": COCO_ANNOTATIONS_SIZE,
            "archive_url": COCO_ANNOTATIONS_URL,
            "instances_member": COCO_INSTANCES_MEMBER,
            "instances_sha256": COCO_INSTANCES_SHA256,
        },
        "categories": {
            "cat": index.category_ids_by_name["cat"],
            "person": index.category_ids_by_name["person"],
        },
        "dataset": COCO_DATASET_NAME,
        "images": images,
        "schema_version": 1,
        "selection": _selection_manifest(selection),
    }


def _selection_manifest(selection: Selection) -> dict[str, Any]:
    return {
        "actual": selection.actual,
        "algorithm": SELECTION_ALGORITHM,
        "available": selection.available,
        "hard_negative_categories": list(HARD_NEGATIVE_CATEGORY_NAMES),
        "planned_count": selection.planned_count,
        "requested_count": selection.requested_count,
        "selected_count": len(selection.images),
        "seed": selection.seed,
        "target_weights": SELECTION_WEIGHTS,
        "targets": selection.targets,
    }


def _recompute_declared_selection(manifest: dict[str, Any], index: CocoIndex) -> Selection:
    raw = manifest.get("selection")
    if not isinstance(raw, dict) or raw.get("algorithm") != SELECTION_ALGORITHM:
        raise DetectionEvalError("subset manifest selection algorithm is unsupported")
    requested_count = raw.get("requested_count")
    seed = raw.get("seed")
    if (
        isinstance(requested_count, bool)
        or not isinstance(requested_count, int)
        or requested_count <= 0
    ):
        raise DetectionEvalError("subset manifest requested_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DetectionEvalError("subset manifest seed must be an integer")

    expected = select_images(index, max_images=requested_count, seed=seed)
    try:
        declared_json = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DetectionEvalError(
            f"subset manifest selection metadata is invalid: {error}"
        ) from error
    expected_json = json.dumps(
        _selection_manifest(expected), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if declared_json != expected_json:
        raise DetectionEvalError(
            "subset manifest selection metadata does not match a fresh deterministic selection"
        )
    return expected


def prepare_subset(
    *,
    cache_dir: Path,
    coco_root: Path | None,
    max_images: int,
    seed: int,
    offline: bool,
    manifest_path: Path | None = None,
) -> PreparedSubset:
    """Select and verify a bounded subset, downloading only selected images."""

    dataset_root = coco_root if coco_root is not None else cache_dir
    if coco_root is not None:
        annotation_path = _annotation_path(coco_root)
        if not annotation_path.is_file():
            raise DetectionEvalError(
                f"existing COCO root lacks {COCO_INSTANCES_MEMBER}: {coco_root}"
            )
        _validate_official_annotation(annotation_path)
    else:
        annotation_path = ensure_coco_annotations(cache_dir, offline=offline)

    index = load_coco(annotation_path)
    selection = select_images(index, max_images=max_images, seed=seed)
    image_dir = dataset_root / "val2017"
    image_identities: dict[int, dict[str, Any]] = {}
    for selected in selection.images:
        image = selected.image
        image_path = image_dir / image.file_name
        if coco_root is None:
            url = COCO_IMAGE_URL_TEMPLATE.format(file_name=image.file_name)
            identity = ensure_remote_image(url, image_path, offline=offline)
            identity_value = identity.to_dict()
            identity_value["verification"] = "TLS S3 Content-Length and simple MD5 ETag"
        else:
            if not image_path.is_file():
                raise DetectionEvalError(f"existing COCO root lacks selected image: {image_path}")
            size, md5, sha256 = file_hashes(image_path)
            identity_value = {
                "md5": md5,
                "sha256": sha256,
                "size": size,
                "url": COCO_IMAGE_URL_TEMPLATE.format(file_name=image.file_name),
                "verification": "local content hash (existing root)",
            }
        image_identities[image.image_id] = identity_value

    manifest = _manifest_for(selection, index, image_identities)
    if manifest_path is None:
        manifest_path = cache_dir / "manifests" / f"coco-val-max{max_images}-seed{seed}.json"
    _stable_write(manifest_path, manifest)
    return PreparedSubset(
        annotation_path=annotation_path,
        dataset_root=dataset_root,
        image_dir=image_dir,
        index=index,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_prepared_subset(manifest_path: Path, dataset_root: Path) -> PreparedSubset:
    """Load a manifest and reject changed annotation or image bytes."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DetectionEvalError(f"cannot read subset manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise DetectionEvalError(f"unsupported subset manifest: {manifest_path}")
    annotation_path = _annotation_path(dataset_root)
    _validate_official_annotation(annotation_path)
    index = load_coco(annotation_path)
    if manifest.get("dataset") != COCO_DATASET_NAME:
        raise DetectionEvalError("subset manifest does not identify COCO 2017 validation")
    annotations = manifest.get("annotations")
    if (
        not isinstance(annotations, dict)
        or annotations.get("instances_sha256") != COCO_INSTANCES_SHA256
    ):
        raise DetectionEvalError("subset manifest annotation identity is not pinned COCO 2017")
    expected_categories = {
        "cat": index.category_ids_by_name.get("cat"),
        "person": index.category_ids_by_name.get("person"),
    }
    if manifest.get("categories") != expected_categories:
        raise DetectionEvalError("subset manifest cat/person category mapping differs from COCO")
    expected_selection = _recompute_declared_selection(manifest, index)
    image_dir = dataset_root / "val2017"
    raw_images = manifest.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise DetectionEvalError("subset manifest images must be a non-empty array")
    seen: set[int] = set()
    for raw in raw_images:
        if not isinstance(raw, dict):
            raise DetectionEvalError("subset manifest image entries must be objects")
        image_id = raw.get("id")
        file_name = raw.get("file_name")
        identity = raw.get("verified_content")
        if (
            isinstance(image_id, bool)
            or not isinstance(image_id, int)
            or image_id in seen
            or not isinstance(file_name, str)
            or not isinstance(identity, dict)
        ):
            raise DetectionEvalError("invalid or duplicate subset manifest image entry")
        if image_id not in index.images or index.images[image_id].file_name != file_name:
            raise DetectionEvalError(f"manifest image {image_id}/{file_name} differs from COCO")
        coco_image = index.images[image_id]
        if (
            raw.get("width") != coco_image.width
            or raw.get("height") != coco_image.height
            or raw.get("license_id") != coco_image.license_id
            or raw.get("stratum") not in SELECTION_WEIGHTS
        ):
            raise DetectionEvalError(f"manifest metadata differs from COCO for image {image_id}")
        image_path = image_dir / file_name
        size, md5, sha256 = file_hashes(image_path)
        if (size, md5, sha256) != (
            identity.get("size"),
            identity.get("md5"),
            identity.get("sha256"),
        ):
            raise DetectionEvalError(f"selected image content changed: {image_path}")
        seen.add(image_id)
    actual_membership = [(int(item["id"]), str(item["stratum"])) for item in raw_images]
    expected_membership = [
        (selected.image.image_id, selected.stratum) for selected in expected_selection.images
    ]
    if actual_membership != expected_membership:
        raise DetectionEvalError(
            "subset manifest image membership does not match its deterministic seed selection"
        )
    return PreparedSubset(
        annotation_path=annotation_path,
        dataset_root=dataset_root,
        image_dir=image_dir,
        index=index,
        manifest_path=manifest_path,
        manifest=manifest,
    )
