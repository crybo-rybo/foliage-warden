from __future__ import annotations

import json
from pathlib import Path

import pytest

from foliage_warden_training.manifest import ManifestError, load_manifest, summarize_records


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "clip_id": "clip-1",
        "path": "clip-1.npz",
        "label": "PASSING",
        "split": "train",
        "session_id": "session-1",
        "day": "2026-08-14",
    }
    row.update(overrides)
    return row


def test_load_manifest_and_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(staged_safe=True)])

    records = load_manifest(manifest, require_files=False)

    assert records[0].path == (tmp_path / "clip-1.npz").resolve()
    assert records[0].staged_safe is True
    assert summarize_records(records)["by_label"]["PASSING"] == 1


@pytest.mark.parametrize("leak_field", ["session_id", "day"])
def test_rejects_split_leakage(tmp_path: Path, leak_field: str) -> None:
    manifest = tmp_path / "manifest.jsonl"
    first = _row()
    second = _row(
        clip_id="clip-2",
        path="clip-2.npz",
        split="test",
        session_id="session-2",
        day="2026-08-15",
    )
    second[leak_field] = first[leak_field]
    _write_manifest(manifest, [first, second])

    with pytest.raises(ManifestError, match=f"split leakage through {leak_field}"):
        load_manifest(manifest, require_files=False)


def test_rejects_duplicate_resolved_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _row(),
            _row(
                clip_id="clip-2",
                path="./clip-1.npz",
                session_id="session-2",
            ),
        ],
    )

    with pytest.raises(ManifestError, match="duplicate path"):
        load_manifest(manifest, require_files=False)


def test_rejects_noncanonical_label(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [_row(label="eating")])

    with pytest.raises(ManifestError, match="label must be one of"):
        load_manifest(manifest, require_files=False)
