from __future__ import annotations

import json
import os

from foliage_warden_review.manifest import ManifestError, load_manifest
from tests.support import ManifestTestCase


class ManifestValidationTests(ManifestTestCase):
    def test_loads_media_and_only_exposes_opaque_url(self) -> None:
        session = self.manifest.client_dict()["sessions"][0]
        media = session["media"][0]
        self.assertEqual(media["display_name"], "clip.jpg")
        self.assertRegex(media["media_url"], r"^/media/[0-9a-f]{32}$")
        self.assertNotIn(str(self.root), json.dumps(self.manifest.client_dict()))

    def test_rejects_parent_absolute_and_dot_paths(self) -> None:
        outside = self.root.parent / "outside-review.jpg"
        outside.write_bytes(b"outside")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        for unsafe in (
            "../outside-review.jpg",
            str(outside),
            "./clip.jpg",
            "nested//clip.jpg",
        ):
            with self.subTest(unsafe=unsafe):
                self.write_manifest(media_path=unsafe)
                with self.assertRaisesRegex(
                    ManifestError, "manifest directory|relative path"
                ):
                    load_manifest(self.manifest_path)

    def test_rejects_symlink_that_escapes_manifest_root(self) -> None:
        outside = self.root.parent / "outside-review-symlink.jpg"
        outside.write_bytes(b"outside")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        link = self.root / "escape.jpg"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.write_manifest(media_path="escape.jpg")
        with self.assertRaisesRegex(ManifestError, "escapes"):
            load_manifest(self.manifest_path)

    def test_rejects_invalid_and_duplicate_session_identifiers(self) -> None:
        self.write_manifest(session_id="../private")
        with self.assertRaisesRegex(ManifestError, "session_id"):
            load_manifest(self.manifest_path)

        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        document["sessions"] = [
            {
                "session_id": "same",
                "group_id": "group-a",
                "media": [
                    {
                        "media_id": "one",
                        "path": "clip.jpg",
                        "kind": "image",
                        "duration_ms": 1,
                    }
                ],
            },
            {
                "session_id": "same",
                "group_id": "group-b",
                "media": [
                    {
                        "media_id": "two",
                        "path": "clip.jpg",
                        "kind": "image",
                        "duration_ms": 1,
                    }
                ],
            },
        ]
        self.manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "duplicate session_id"):
            load_manifest(self.manifest_path)

    def test_rejects_kind_extension_mismatch_and_unknown_fields(self) -> None:
        self.write_manifest()
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        document["sessions"][0]["media"][0]["kind"] = "video"
        self.manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "does not match"):
            load_manifest(self.manifest_path)

        document["sessions"][0]["media"][0]["kind"] = "image"
        document["unexpected"] = True
        self.manifest_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "unknown field"):
            load_manifest(self.manifest_path)


if __name__ == "__main__":
    import unittest

    unittest.main()
