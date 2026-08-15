from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from foliage_warden_review.manifest import Manifest, load_manifest


class ManifestTestCase(unittest.TestCase):
    temporary_directory: tempfile.TemporaryDirectory[str]
    root: Path
    manifest_path: Path
    manifest: Manifest

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        (self.root / "clip.jpg").write_bytes(b"local-image-fixture")
        self.manifest_path = self.root / "manifest.json"
        self.write_manifest()
        self.manifest = load_manifest(self.manifest_path)

    def write_manifest(
        self, *, media_path: str = "clip.jpg", **session_overrides: object
    ) -> None:
        session: dict[str, object] = {
            "session_id": "session-001",
            "group_id": "day-001-camera-a",
            "zone_id": "pot-1-approach",
            "media": [
                {
                    "media_id": "clip-001",
                    "path": media_path,
                    "kind": "image",
                    "duration_ms": 2000,
                }
            ],
        }
        session.update(session_overrides)
        self.manifest_path.write_text(
            json.dumps({"schema_version": 1, "sessions": [session]}),
            encoding="utf-8",
        )

    def annotation(self, **overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "behavior": "SNIFFING",
            "end_ms": 900,
            "event_id": "session-001-clip-001-sniffing-100-900",
            "group_id": "day-001-camera-a",
            "media_id": "clip-001",
            "person_present": False,
            "privacy_restricted": False,
            "rationale": "Head approaches foliage without visible ingestion.",
            "session_id": "session-001",
            "staged_safe": False,
            "start_ms": 100,
            "zone_id": "pot-1-approach",
        }
        value.update(overrides)
        return value
