"""Pinned public-dataset identities and selection policy."""

from __future__ import annotations

COCO_DATASET_NAME = "COCO 2017 validation"
COCO_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_ANNOTATIONS_FILENAME = "annotations_trainval2017.zip"
COCO_ANNOTATIONS_SIZE = 252_907_541
COCO_ANNOTATIONS_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
COCO_ANNOTATIONS_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
COCO_INSTANCES_MEMBER = "annotations/instances_val2017.json"
COCO_INSTANCES_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
COCO_IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}"

TARGET_CATEGORY_NAMES = ("cat", "person")

# These COCO classes make useful *object-detection* negatives for this project:
# animal confusers plus plants/furniture likely to appear near a garden. They are
# not behavior negatives and do not approximate an installed camera domain.
HARD_NEGATIVE_CATEGORY_NAMES = (
    "bird",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "bench",
    "potted plant",
)

SELECTION_ALGORITHM = "balanced-sha256-v1"
SELECTION_WEIGHTS = {
    "cat_positive": 4,
    "person_positive": 3,
    "hard_negative": 2,
    "background_negative": 1,
}

PUBLIC_DATA_WARNING = (
    "COCO cat/person detection performance does not establish accuracy on the installed "
    "camera, in the garden domain, at night, under occlusion, or for EATING/DIGGING behavior. "
    "Thresholds chosen on this same subset are tuning results, not held-out estimates. It must "
    "never be used by itself to enable a physical action."
)
