from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from foliage_warden_detection_eval.constants import PUBLIC_DATA_WARNING

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "coco-val-100-seed20260814.manifest.json"
REPORT = ROOT / "baselines" / "yolox-s-opencv-coco-val-100.report.json"


def test_checked_baseline_manifest_is_self_consistent() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    images = manifest["images"]

    assert len(images) == 100
    assert len({image["id"] for image in images}) == 100
    assert dict(Counter(image["stratum"] for image in images)) == manifest["selection"]["actual"]
    assert all(len(image["verified_content"]["sha256"]) == 64 for image in images)


def test_checked_baseline_report_identifies_exact_manifest_and_scope() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    assert report["dataset"]["manifest_sha256"] == manifest_sha256
    assert report["dataset"]["image_count"] == 100
    assert report["schema_version"] == 2
    assert report["prediction_artifact"]["sha256"] == (
        "f44bcb113bc29a3a88bdf15ad23deaa6048d26a3f19d786aadf204a02aa3a828"
    )
    assert report["scope_warning"] == PUBLIC_DATA_WARNING
    assert report["metrics"]["classes"]["cat"]["score_threshold"] == 0.5
    assert report["metrics"]["classes"]["person"]["score_threshold"] == 0.5
