from __future__ import annotations

from foliage_warden_detection_eval.coco import CocoAnnotation, CocoImage, CocoIndex
from foliage_warden_detection_eval.constants import HARD_NEGATIVE_CATEGORY_NAMES
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


def test_balanced_selection_is_deterministic_and_disjoint() -> None:
    first = select_images(_index(), max_images=10, seed=7)
    second = select_images(_index(), max_images=10, seed=7)

    assert first == second
    assert len({item.image.image_id for item in first.images}) == 10
    assert first.targets == {
        "cat_positive": 4,
        "person_positive": 3,
        "hard_negative": 2,
        "background_negative": 1,
    }
    assert first.actual == first.targets


def test_seed_changes_ranked_members_not_quota() -> None:
    first = select_images(_index(), max_images=10, seed=7)
    second = select_images(_index(), max_images=10, seed=8)

    assert first.actual == second.actual
    assert first.images != second.images


def test_small_subset_still_includes_positive_and_hard_negative_strata() -> None:
    selection = select_images(_index(), max_images=3, seed=1)

    assert selection.actual == {
        "cat_positive": 1,
        "person_positive": 1,
        "hard_negative": 1,
        "background_negative": 0,
    }
