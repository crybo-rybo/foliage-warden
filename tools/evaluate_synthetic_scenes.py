"""Run the pinned detector over generated integration fixtures.

The resulting object-count checks are a packaging and pipeline smoke test. They
are deliberately not reported as real-world detector or behavior-model metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from foliage_warden_perception.dependencies import require_cv2
from foliage_warden_perception.geometry import load_zones
from foliage_warden_perception.pipeline import build_observation_record
from foliage_warden_perception.registry import (
    DEFAULT_MODEL_ID,
    default_registry_path,
    load_model_spec,
    resolve_and_verify_model,
)
from foliage_warden_perception.sources import ImageSource
from foliage_warden_perception.tracking import IouTracker
from foliage_warden_perception.yolox import YOLOXDetector

JsonObject = dict[str, Any]


def _object(value: Any, context: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _count(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _load_manifest(path: Path) -> list[JsonObject]:
    try:
        root = _object(json.loads(path.read_text(encoding="utf-8")), "manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fixture manifest {path}: {error}") from error
    if root.get("schema_version") != 1:
        raise ValueError("fixture manifest schema_version must equal 1")
    scenes = root.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("fixture manifest scenes must be a non-empty array")
    parsed: list[JsonObject] = []
    seen: set[str] = set()
    for index, value in enumerate(scenes):
        scene = _object(value, f"scenes[{index}]")
        scene_id = _string(scene.get("id"), f"scenes[{index}].id")
        if scene_id in seen:
            raise ValueError(f"duplicate fixture ID {scene_id!r}")
        seen.add(scene_id)
        _string(scene.get("image"), f"scenes[{index}].image")
        _string(scene.get("zones"), f"scenes[{index}].zones")
        digest = _string(scene.get("sha256"), f"scenes[{index}].sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"scenes[{index}].sha256 is not a lowercase SHA-256 digest")
        expected = _object(scene.get("expected_objects"), f"scenes[{index}].expected_objects")
        for object_class in ("cat", "person"):
            minimum = _count(expected.get(f"{object_class}_min"), f"{object_class}_min")
            maximum = _count(expected.get(f"{object_class}_max"), f"{object_class}_max")
            if minimum > maximum:
                raise ValueError(f"{scene_id}: {object_class} minimum exceeds maximum")
        parsed.append(scene)
    return parsed


def _digest(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def evaluate(
    manifest_path: Path,
    *,
    registry_path: Path,
    model_id: str,
    model_path: Path | None,
) -> JsonObject:
    fixtures = _load_manifest(manifest_path)
    base = manifest_path.parent
    spec = load_model_spec(registry_path, model_id)
    resolved_model = resolve_and_verify_model(registry_path, spec, model_path)
    cv2 = require_cv2()
    detector = YOLOXDetector(resolved_model, spec, cv2_module=cv2)

    fixture_reports: list[JsonObject] = []
    checks_total = 0
    checks_passed = 0
    for fixture in fixtures:
        fixture_id = str(fixture["id"])
        image_path = base / str(fixture["image"])
        zone_path = base / str(fixture["zones"])
        expected = _object(fixture["expected_objects"], f"{fixture_id}.expected_objects")
        actual_digest = _digest(image_path)
        digest_matches = actual_digest == fixture["sha256"]

        source = ImageSource(image_path, camera_id=f"fixture-{fixture_id}", cv2_module=cv2)
        frame = next(iter(source))
        detections, _ = detector.detect_timed(frame.bgr)
        tracks = IouTracker().update(detections, frame_index=0, timestamp_ms=0)
        observation = build_observation_record(
            frame,
            tracks,
            load_zones(zone_path),
            model_id=spec.model_id,
            model_sha256=spec.sha256,
        )
        cat_count = int(observation["cat_count"])
        person_count = sum(
            track["class"] == "PERSON" for track in observation["observation"]["tracks"]
        )
        checks = {
            "cat_count": expected["cat_min"] <= cat_count <= expected["cat_max"],
            "digest": digest_matches,
            "observe_only": observation["mode"] == "OBSERVE_ONLY",
            "person_count": expected["person_min"] <= person_count <= expected["person_max"],
            "unknown_behavior": observation["behavior"] == "UNKNOWN",
            "would_action_false": observation["would_action"] is False,
        }
        checks_total += len(checks)
        checks_passed += sum(checks.values())
        fixture_reports.append(
            {
                "cat_count": cat_count,
                "checks": checks,
                "detections": [
                    {
                        "bbox": track["bbox"],
                        "class": track["class"],
                        "confidence": track["detection_confidence"],
                        "no_fire_intersection": track.get("no_fire_intersection"),
                        "zone_id": track.get("zone_id"),
                    }
                    for track in observation["observation"]["tracks"]
                ],
                "id": fixture_id,
                "passed": all(checks.values()),
                "person_count": person_count,
            }
        )

    return {
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "fixture_count": len(fixture_reports),
        "fixtures": fixture_reports,
        "model": {"id": spec.model_id, "sha256": spec.sha256},
        "passed": checks_passed == checks_total,
        "record_type": "synthetic_perception_evaluation",
        "schema_version": 1,
        "scope": "integration smoke only; not a real-world performance claim",
    }


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "testdata" / "synthetic-scenes" / "manifest.json",
    )
    parser.add_argument("--registry", type=Path, default=default_registry_path())
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = evaluate(
            args.manifest,
            registry_path=args.registry,
            model_id=args.model_id,
            model_path=args.model,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 2
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        _atomic_write(args.output, rendered)
        print(f"wrote {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
