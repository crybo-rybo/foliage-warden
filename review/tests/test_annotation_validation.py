from __future__ import annotations

from foliage_warden_review.validation import AnnotationError, validate_annotation

from tests.support import ManifestTestCase


class AnnotationValidationTests(ManifestTestCase):
    def test_accepts_all_operational_labels(self) -> None:
        for behavior in (
            "PASSING",
            "SNIFFING",
            "EATING",
            "DIGGING",
            "OTHER",
            "UNKNOWN",
        ):
            with self.subTest(behavior=behavior):
                value = validate_annotation(
                    self.annotation(behavior=behavior), self.manifest
                )
                self.assertEqual(value["behavior"], behavior)

    def test_rejects_invalid_interval_and_unknown_media(self) -> None:
        with self.assertRaisesRegex(AnnotationError, "greater than"):
            validate_annotation(
                self.annotation(start_ms=900, end_ms=900), self.manifest
            )
        with self.assertRaisesRegex(AnnotationError, "exceeds"):
            validate_annotation(self.annotation(end_ms=2001), self.manifest)
        with self.assertRaisesRegex(AnnotationError, "outside the manifest"):
            validate_annotation(self.annotation(media_id="not-listed"), self.manifest)

    def test_rejects_manifest_group_mismatch_and_bad_ids(self) -> None:
        with self.assertRaisesRegex(AnnotationError, "does not match"):
            validate_annotation(
                self.annotation(group_id="different-day"), self.manifest
            )
        with self.assertRaisesRegex(AnnotationError, "event_id"):
            validate_annotation(self.annotation(event_id="../escape"), self.manifest)

    def test_rejects_person_without_privacy_restriction(self) -> None:
        with self.assertRaisesRegex(AnnotationError, "privacy_restricted must be true"):
            validate_annotation(
                self.annotation(person_present=True, privacy_restricted=False),
                self.manifest,
            )
        value = validate_annotation(
            self.annotation(person_present=True, privacy_restricted=True), self.manifest
        )
        self.assertTrue(value["privacy_restricted"])

    def test_rejects_blank_rationale_wrong_types_and_unknown_fields(self) -> None:
        with self.assertRaisesRegex(AnnotationError, "rationale"):
            validate_annotation(self.annotation(rationale="  "), self.manifest)
        with self.assertRaisesRegex(AnnotationError, "staged_safe"):
            validate_annotation(self.annotation(staged_safe=1), self.manifest)
        with self.assertRaisesRegex(AnnotationError, "unknown field"):
            validate_annotation(self.annotation(extra=True), self.manifest)


if __name__ == "__main__":
    import unittest

    unittest.main()
