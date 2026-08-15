from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from foliage_warden_detection_eval import dataset
from foliage_warden_detection_eval.cache import file_hashes
from foliage_warden_detection_eval.coco import CocoAnnotation, CocoImage, CocoIndex
from foliage_warden_detection_eval.constants import HARD_NEGATIVE_CATEGORY_NAMES
from foliage_warden_detection_eval.errors import DetectionEvalError
from foliage_warden_detection_eval.selection import select_images


def _index() -> CocoIndex:
    categories = {"person": 1, "cat": 17}
    categories.update(
        {name: category_id for category_id, name in enumerate(HARD_NEGATIVE_CATEGORY_NAMES, 20)}
    )
    images = {
        image_id: CocoImage(image_id, f"{image_id:012d}.jpg", 100, 80, 1)
        for image_id in range(1, 41)
    }
    annotations: list[CocoAnnotation] = []
    annotation_id = 1
    for image_id in range(1, 13):
        annotations.append(CocoAnnotation(annotation_id, image_id, 17, (1, 1, 10, 10), False))
        annotation_id += 1
    for image_id in range(13, 25):
        annotations.append(CocoAnnotation(annotation_id, image_id, 1, (1, 1, 10, 10), False))
        annotation_id += 1
    hard_id = categories[HARD_NEGATIVE_CATEGORY_NAMES[0]]
    for image_id in range(25, 33):
        annotations.append(CocoAnnotation(annotation_id, image_id, hard_id, (1, 1, 10, 10), False))
        annotation_id += 1
    return CocoIndex(
        images=images,
        annotations=tuple(annotations),
        category_ids_by_name=categories,
        category_names_by_id={value: key for key, value in categories.items()},
        licenses=(),
    )


def _identity(path: Path) -> dict[str, int | str]:
    size, md5, sha256 = file_hashes(path)
    return {
        "md5": md5,
        "sha256": sha256,
        "size": size,
        "url": f"https://example.test/{path.name}",
        "verification": "test fixture",
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    index = _index()
    selection = select_images(index, max_images=10, seed=7)
    image_dir = tmp_path / "val2017"
    image_dir.mkdir()
    identities: dict[int, dict[str, int | str]] = {}
    for selected in selection.images:
        path = image_dir / selected.image.file_name
        path.write_bytes(f"image-{selected.image.image_id}".encode())
        identities[selected.image.image_id] = _identity(path)
    manifest = dataset._manifest_for(selection, index, identities)
    manifest_path = tmp_path / "manifest.json"
    annotation_path = tmp_path / "annotations" / "instances_val2017.json"
    annotation_path.parent.mkdir()
    annotation_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "_validate_official_annotation", lambda _path: None)
    monkeypatch.setattr(dataset, "load_coco", lambda _path: index)
    return index, manifest, manifest_path, image_dir


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_load_recomputes_and_accepts_exact_declared_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, manifest_path, _ = _fixture(tmp_path, monkeypatch)
    _write_manifest(manifest_path, manifest)

    prepared = dataset.load_prepared_subset(manifest_path, tmp_path)

    assert prepared.manifest == manifest


def test_load_rejects_mutated_selection_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest, manifest_path, _ = _fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(manifest)
    changed["selection"]["actual"]["cat_positive"] += 1
    _write_manifest(manifest_path, changed)

    with pytest.raises(DetectionEvalError, match="fresh deterministic selection"):
        dataset.load_prepared_subset(manifest_path, tmp_path)


def test_load_rejects_cherry_picked_image_with_valid_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, manifest, manifest_path, image_dir = _fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(manifest)
    selected_ids = {image["id"] for image in changed["images"]}
    first = changed["images"][0]
    stratum = first["stratum"]
    all_strata = {
        selected.image.image_id: selected.stratum
        for selected in select_images(index, max_images=len(index.images), seed=7).images
    }
    replacement_id = next(
        image_id
        for image_id, candidate_stratum in all_strata.items()
        if image_id not in selected_ids and candidate_stratum == stratum
    )
    replacement = index.images[replacement_id]
    replacement_path = image_dir / replacement.file_name
    replacement_path.write_bytes(f"image-{replacement_id}".encode())
    first.update(
        {
            "file_name": replacement.file_name,
            "height": replacement.height,
            "id": replacement.image_id,
            "license_id": replacement.license_id,
            "verified_content": _identity(replacement_path),
            "width": replacement.width,
        }
    )
    _write_manifest(manifest_path, changed)

    with pytest.raises(DetectionEvalError, match="image membership"):
        dataset.load_prepared_subset(manifest_path, tmp_path)
