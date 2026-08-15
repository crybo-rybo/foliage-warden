from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from foliage_warden_review.storage import AnnotationStore, RevisionConflict

from tests.support import ManifestTestCase


class StorageTests(ManifestTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.store_path = self.root / "private" / "annotations.json"
        self.store = AnnotationStore(self.store_path, self.manifest)

    def test_atomic_save_preserves_other_annotations_and_prior_revision(self) -> None:
        first = self.annotation()
        second = self.annotation(
            event_id="session-001-clip-001-other-1000-1500",
            behavior="OTHER",
            start_ms=1000,
            end_ms=1500,
            rationale="Cat sits beside the pot without plant interaction.",
        )
        state = self.store.upsert(first, 0)
        state = self.store.upsert(second, state["revision"])
        updated = self.annotation(
            rationale="Head remains close; no chewing or leaf displacement."
        )
        state = self.store.upsert(updated, state["revision"])

        self.assertEqual(state["revision"], 3)
        self.assertEqual(len(state["annotations"]), 2)
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["history"][0]["annotation"], first)
        self.assertEqual(state["history"][0]["reason"], "updated")
        self.assertEqual(os.stat(self.store_path).st_mode & 0o777, 0o600)
        self.assertFalse(list(self.store_path.parent.glob("*.tmp")))

        reloaded = AnnotationStore(self.store_path, self.manifest)
        self.assertEqual(reloaded.snapshot(), state)

    def test_archive_retains_value_in_history_and_export_excludes_it(self) -> None:
        state = self.store.upsert(self.annotation(), 0)
        state = self.store.archive(self.annotation()["event_id"], state["revision"])
        self.assertEqual(state["annotations"], [])
        self.assertEqual(state["history"][0]["reason"], "archived")
        self.assertEqual(self.store.export_jsonl(), "")

    def test_rejects_stale_revision_without_changing_file(self) -> None:
        self.store.upsert(self.annotation(), 0)
        before = self.store_path.read_bytes()
        with self.assertRaisesRegex(RevisionConflict, "current revision is 1"):
            self.store.upsert(self.annotation(rationale="stale update"), 0)
        self.assertEqual(self.store_path.read_bytes(), before)

    def test_failed_atomic_write_does_not_advance_in_memory_state(self) -> None:
        before = self.store.snapshot()
        with (
            patch(
                "foliage_warden_review.storage._atomic_write",
                side_effect=OSError("disk full"),
            ),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            self.store.upsert(self.annotation(), 0)
        self.assertEqual(self.store.snapshot(), before)
        self.assertFalse(self.store_path.exists())

    def test_export_is_stable_and_ground_truth_compatible(self) -> None:
        state = self.store.upsert(
            self.annotation(
                behavior="EATING",
                staged_safe=True,
                rationale="Repeated mouth movement with visible leaf contact.",
            ),
            0,
        )
        self.store.upsert(
            self.annotation(
                event_id="session-001-clip-001-passing-10-50",
                behavior="PASSING",
                start_ms=10,
                end_ms=50,
                zone_id=None,
                rationale="Cat crosses the approach area without investigating.",
            ),
            state["revision"],
        )
        first = self.store.export_jsonl()
        second = self.store.export_jsonl()
        self.assertEqual(first, second)
        records = [json.loads(line) for line in first.splitlines()]
        self.assertEqual(
            [record["behavior"] for record in records], ["PASSING", "EATING"]
        )
        self.assertNotIn("zone_id", records[0])
        self.assertEqual(records[1]["record_type"], "ground_truth_event")
        self.assertEqual(records[1]["metadata"]["group_id"], "day-001-camera-a")

        evaluator_src = Path(__file__).resolve().parents[2] / "python" / "src"
        sys.path.insert(0, str(evaluator_src))
        self.addCleanup(lambda: sys.path.remove(str(evaluator_src)))
        from foliage_warden_eval.schemas import GroundTruthEvent

        parsed = [GroundTruthEvent.from_dict(record) for record in records]
        self.assertEqual(
            [event.behavior.value for event in parsed], ["PASSING", "EATING"]
        )

        output = self.root / "ground-truth.jsonl"
        self.store.write_export(output)
        self.assertEqual(output.read_text(encoding="utf-8"), first)
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    import unittest

    unittest.main()
