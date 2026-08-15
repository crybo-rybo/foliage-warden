from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
from foliage_warden_shadow.contracts import parse_perception_stream, read_jsonl
from foliage_warden_shadow.inference import parse_inference_requests
from support import (
    cat_track,
    default_records,
    file_sha256,
    perception_record,
    write_incident,
    write_jsonl,
    write_recorder_incident,
)

from foliage_warden_assembler import AssemblyError, assemble_incident, core
from foliage_warden_assembler.cli import main


def _read_requests(path: Path):
    return read_jsonl(path, parse_inference_requests)


def _all_output_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_real_opencv_assembly_is_causal_rgb_private_and_deterministic(tmp_path: Path) -> None:
    records = default_records()
    source_frames = []
    for ordinal in range(3):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 + ordinal * 30
        frame[:, :, 1] = 90
        frame[:, :, 2] = 220 - ordinal * 30
        source_frames.append(frame)
    incident, perceptions, _ = write_recorder_incident(tmp_path, records, frames=source_frames)

    first = assemble_incident(
        incident,
        perceptions,
        tmp_path / "first",
        window_ms=100,
        logical_latency_ms=10,
    )
    second = assemble_incident(
        incident,
        perceptions,
        tmp_path / "second",
        window_ms=100,
        logical_latency_ms=10,
    )

    assert first.request_count == 2
    assert first.skipped_target_count == 0
    assert _all_output_bytes(first.output_directory) == _all_output_bytes(second.output_directory)
    assert stat.S_IMODE(first.output_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((first.output_directory / "clips").stat().st_mode) == 0o700
    for path in first.output_directory.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

    requests = _read_requests(first.requests_path)
    assert [request.sequence for request in requests] == [0, 1]
    assert [request.frame_timestamps_ms for request in requests] == [(100, 200), (200, 300)]
    assert [request.window_end_captured_at_ms for request in requests] == [200, 300]
    assert [request.predicted_at_ms for request in requests] == [210, 310]
    selected = read_jsonl(first.perceptions_path, parse_perception_stream)
    assert [record.sequence for record in selected] == [20, 40]
    incident_records = read_jsonl(first.incident_perceptions_path, parse_perception_stream)
    assert [record.sequence for record in incident_records] == [10, 20, 40]

    capture = cv2.VideoCapture(str(incident / "clip.avi"))
    decoded_rgb = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    saved = np.load(first.output_directory / requests[0].clip_path, allow_pickle=False)
    assert saved.dtype == np.uint8
    assert saved.shape == (2, 24, 32, 3)
    np.testing.assert_array_equal(saved, np.stack(decoded_rgb[:2]))
    assert requests[0].clip_sha256 == file_sha256(first.output_directory / requests[0].clip_path)

    provenance = json.loads(first.provenance_path.read_text(encoding="utf-8"))
    assert provenance["config"]["clip_contract"] == "FULL_FRAME_RGB_UINT8_THWC_NO_CROP"
    assert provenance["inputs"]["incident"]["perception_binding"]["verified"] is True
    assert provenance["mapping"][0]["sampled_decoded_ordinals"] == [0, 1]
    assert provenance["mapping"][1]["sampled_decoded_ordinals"] == [1, 2]
    assert len(provenance["warnings"]) == 4
    for output in provenance["outputs"].values():
        output_path = first.output_directory / output["path"]
        assert output["byte_size"] == output_path.stat().st_size
        assert output["sha256"] == file_sha256(output_path)


def test_legacy_metadata_and_gapped_sequences_are_supported_structurally(tmp_path: Path) -> None:
    records = default_records()
    incident, perceptions, _ = write_incident(tmp_path, records, provenance=False)

    result = assemble_incident(
        incident,
        perceptions,
        tmp_path / "output",
        window_ms=100,
        logical_latency_ms=0,
    )

    manifest = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert manifest["inputs"]["incident"]["perception_binding"] == {
        "present": False,
        "verified": False,
    }
    assert manifest["mapping"][0]["sampled_perception_sequences"] == [10, 20]
    assert manifest["mapping"][1]["sampled_perception_sequences"] == [20, 40]


def test_full_source_stream_is_inclusively_reduced_to_bound_incident(tmp_path: Path) -> None:
    records = default_records()
    incident, perceptions, _ = write_incident(tmp_path, records)
    write_jsonl(
        perceptions,
        [perception_record(0, 0), *records, perception_record(50, 400)],
    )

    result = assemble_incident(
        incident,
        perceptions,
        tmp_path / "output",
        window_ms=100,
        logical_latency_ms=10,
    )

    manifest = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert manifest["inputs"]["perception_jsonl"]["record_count"] == 5
    assert [
        record.sequence for record in read_jsonl(result.perceptions_path, parse_perception_stream)
    ] == [
        20,
        40,
    ]


def test_suppressions_are_explicit_and_never_emit_future_frames(tmp_path: Path) -> None:
    records = [
        perception_record(0, 100),
        perception_record(1, 200, tracks=[cat_track(approach_overlap=0.0)]),
        perception_record(2, 300, tracks=[cat_track("cat-a"), cat_track("cat-b")]),
        perception_record(3, 400, tracks=[cat_track("cat-a")]),
        perception_record(4, 500, tracks=[cat_track("cat-a")]),
        perception_record(5, 600, tracks=[cat_track("cat-b")]),
    ]
    incident, perceptions, _ = write_incident(tmp_path, records)

    result = assemble_incident(
        incident,
        perceptions,
        tmp_path / "output",
        window_ms=100,
        logical_latency_ms=10,
    )

    manifest = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert [item["reason"] for item in manifest["skipped_targets"]] == [
        "NO_APPROACH_EVIDENCE",
        "MULTI_CAT_AMBIGUOUS",
        "WINDOW_CAT_CARDINALITY_AMBIGUOUS",
        "WINDOW_TRACK_IDENTITY_MISMATCH",
    ]
    assert manifest["skipped_targets"][2] == {
        "captured_at_ms": 400,
        "cat_track_ids": ["cat-a", "cat-b"],
        "offending_captured_at_ms": 300,
        "offending_sequence": 2,
        "reason": "WINDOW_CAT_CARDINALITY_AMBIGUOUS",
        "sequence": 3,
        "track_id": "cat-a",
    }
    assert manifest["skipped_targets"][3] == {
        "captured_at_ms": 600,
        "offending_captured_at_ms": 500,
        "offending_sequence": 4,
        "offending_track_id": "cat-a",
        "reason": "WINDOW_TRACK_IDENTITY_MISMATCH",
        "sequence": 5,
        "track_id": "cat-b",
    }
    requests = _read_requests(result.requests_path)
    assert len(requests) == 1
    assert requests[0].captured_at_ms == 500
    assert requests[0].frame_timestamps_ms == (400, 500)
    assert max(requests[0].frame_timestamps_ms) == requests[0].captured_at_ms


def test_missing_exact_window_start_is_recorded_when_another_target_is_eligible(
    tmp_path: Path,
) -> None:
    records = [
        perception_record(0, 100),
        perception_record(1, 250, tracks=[cat_track()]),
        perception_record(2, 350, tracks=[cat_track()]),
    ]
    incident, perceptions, _ = write_incident(tmp_path, records)

    result = assemble_incident(
        incident,
        perceptions,
        tmp_path / "output",
        window_ms=100,
        logical_latency_ms=10,
    )

    manifest = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert manifest["skipped_targets"] == [
        {
            "captured_at_ms": 250,
            "reason": "NO_EXACT_CAUSAL_WINDOW_START",
            "sequence": 1,
            "track_id": "cat-a",
            "window_start_captured_at_ms": 150,
        }
    ]
    assert _read_requests(result.requests_path)[0].frame_timestamps_ms == (250, 350)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata["privacy"].__setitem__("network", True),
            "privacy flags must all be false",
        ),
        (
            lambda metadata: metadata["timeline"].__setitem__("end_captured_at_ms", 301),
            "duration disagrees",
        ),
        (
            lambda metadata: metadata["clip"].__setitem__("width", 33),
            "dimensions disagree",
        ),
        (
            lambda metadata: metadata["trigger"]["samples"][0].__setitem__(
                "maximum_approach_overlap", 0.7
            ),
            "trigger samples disagree",
        ),
        (
            lambda metadata: metadata["clip"].__setitem__("codec", "XVID"),
            "codec disagrees",
        ),
        (
            lambda metadata: metadata["trigger"].__setitem__("minimum_approach_overlap", 0.0),
            "must be positive",
        ),
    ],
)
def test_metadata_mutations_fail_without_partial_output(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    incident, perceptions, metadata = write_incident(tmp_path, default_records())
    mutation(metadata)
    metadata_path = incident / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)
    output = tmp_path / "output"

    with pytest.raises(AssemblyError, match=message):
        assemble_incident(
            incident,
            perceptions,
            output,
            window_ms=100,
            logical_latency_ms=10,
        )
    assert not output.exists()


def test_clip_same_size_mutation_is_rejected_before_decode(tmp_path: Path) -> None:
    incident, perceptions, _ = write_incident(tmp_path, default_records())
    clip = incident / "clip.avi"
    payload = bytearray(clip.read_bytes())
    payload[len(payload) // 2] ^= 1
    clip.write_bytes(payload)
    os.chmod(clip, 0o600)

    with pytest.raises(AssemblyError, match="SHA-256 disagrees"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "output",
            window_ms=100,
            logical_latency_ms=10,
        )


def test_bound_perception_mutation_is_rejected(tmp_path: Path) -> None:
    records = default_records()
    incident, perceptions, _ = write_incident(tmp_path, records)
    mutated = deepcopy(records)
    mutated[0]["model"]["id"] = "different-detector"
    write_jsonl(perceptions, mutated)

    with pytest.raises(AssemblyError, match="frame bindings disagree"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "output",
            window_ms=100,
            logical_latency_ms=10,
        )


def test_duplicate_json_keys_are_rejected_before_mapping(tmp_path: Path) -> None:
    records = default_records()
    incident, perceptions, metadata = write_incident(tmp_path, records)
    metadata_path = incident / "metadata.json"
    metadata_path.write_text(
        '{"mode":"LIVE",' + json.dumps(metadata, sort_keys=True)[1:] + "\n",
        encoding="utf-8",
    )
    os.chmod(metadata_path, 0o600)
    with pytest.raises(AssemblyError, match="duplicate object key 'mode'"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "metadata-duplicate",
            window_ms=100,
            logical_latency_ms=10,
        )

    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)
    lines = perceptions.read_text(encoding="utf-8").splitlines()
    lines[0] = '{"sequence":999,' + lines[0][1:]
    perceptions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AssemblyError, match="duplicate object key 'sequence'"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "perception-duplicate",
            window_ms=100,
            logical_latency_ms=10,
        )


def test_decoded_frame_count_must_match_metadata(tmp_path: Path) -> None:
    incident, perceptions, metadata = write_incident(tmp_path, default_records())
    replacement = incident / "replacement.avi"
    writer = cv2.VideoWriter(
        str(replacement),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.write(np.ones((24, 32, 3), dtype=np.uint8))
    writer.release()
    clip = incident / "clip.avi"
    os.replace(replacement, clip)
    os.chmod(clip, 0o600)
    metadata["clip"]["byte_size"] = clip.stat().st_size
    metadata["clip"]["sha256"] = file_sha256(clip)
    metadata_path = incident / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)

    with pytest.raises(AssemblyError, match="frame count disagrees"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "output",
            window_ms=100,
            logical_latency_ms=10,
        )


def test_source_disagreement_is_rejected(tmp_path: Path) -> None:
    records = default_records()
    incident, perceptions, _ = write_incident(tmp_path, records, provenance=False)
    records[1]["source"]["name"] = "another.avi"
    write_jsonl(perceptions, records)

    with pytest.raises(AssemblyError, match="source disagrees"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "output",
            window_ms=100,
            logical_latency_ms=10,
        )


def test_output_is_no_overwrite_and_symlink_inputs_are_rejected(tmp_path: Path) -> None:
    incident, perceptions, _ = write_incident(tmp_path, default_records())
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(AssemblyError, match="already exists"):
        assemble_incident(
            incident,
            perceptions,
            output,
            window_ms=100,
            logical_latency_ms=10,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    linked_perception = tmp_path / "linked.jsonl"
    linked_perception.symlink_to(perceptions)
    with pytest.raises(AssemblyError, match="symbolic link"):
        assemble_incident(
            incident,
            linked_perception,
            tmp_path / "other",
            window_ms=100,
            logical_latency_ms=10,
        )


def test_output_parent_replacement_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident, perceptions, _ = write_incident(tmp_path, default_records())
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    output = output_parent / "assembled"
    displaced_parent = tmp_path / "displaced-output-parent"
    real_verify = core._verify_output_parent
    verification_count = 0

    def replace_parent_after_publication_check(
        parent_descriptor: int,
        parent_path: Path,
    ) -> None:
        nonlocal verification_count
        real_verify(parent_descriptor, parent_path)
        verification_count += 1
        if verification_count == 2:
            output_parent.rename(displaced_parent)
            output_parent.mkdir()
            sentinel = output_parent / "attacker-sentinel"
            sentinel.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(
        core,
        "_verify_output_parent",
        replace_parent_after_publication_check,
    )
    with pytest.raises(AssemblyError, match="output parent changed during assembly"):
        assemble_incident(
            incident,
            perceptions,
            output,
            window_ms=100,
            logical_latency_ms=10,
        )

    assert not output.exists()
    anchored_output = displaced_parent / "assembled"
    assert anchored_output.is_dir()
    assert (anchored_output / "provenance.json").is_file()
    assert list(displaced_parent.glob(".assembled.tmp-*")) == []
    assert (output_parent / "attacker-sentinel").read_text(encoding="utf-8") == "keep"


def test_no_eligible_target_and_excess_latency_publish_nothing(tmp_path: Path) -> None:
    records = [perception_record(1, 100, tracks=[cat_track()])]
    incident, perceptions, _ = write_incident(tmp_path, records)
    with pytest.raises(AssemblyError, match="no exactly-windowed target"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "no-target",
            window_ms=100,
            logical_latency_ms=10,
        )
    with pytest.raises(AssemblyError, match="must not exceed the shadow timeout"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "late",
            window_ms=0,
            logical_latency_ms=51,
        )
    assert not (tmp_path / "no-target").exists()
    assert not (tmp_path / "late").exists()


def test_resource_caps_fail_before_publication_and_numpy_output_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident, perceptions, _ = write_incident(tmp_path, default_records())
    raw_bytes = 3 * 24 * 32 * 3
    monkeypatch.setattr(core, "MAX_DECODED_INCIDENT_BYTES", raw_bytes - 1)
    with pytest.raises(AssemblyError, match="decoded incident"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "decoded-limit",
            window_ms=100,
            logical_latency_ms=10,
        )
    monkeypatch.setattr(core, "MAX_DECODED_INCIDENT_BYTES", raw_bytes)

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(core, "MAX_SHADOW_CLIP_BYTES", frame.nbytes - 1)
    destination = tmp_path / "too-large.npy"
    with pytest.raises(AssemblyError, match="serialized NumPy clip exceeds"):
        core._write_numpy(destination, [frame])
    assert not destination.exists()


def test_cumulative_output_cap_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incident, perceptions, _ = write_incident(tmp_path, default_records())
    monkeypatch.setattr(core, "MAX_ASSEMBLED_TARGETS", 1)
    with pytest.raises(AssemblyError, match="assembly-target limit"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "target-limit",
            window_ms=100,
            logical_latency_ms=10,
        )
    assert not (tmp_path / "target-limit").exists()
    monkeypatch.setattr(core, "MAX_ASSEMBLED_TARGETS", 1_000)
    monkeypatch.setattr(core, "MAX_SELECTED_FRAME_ENTRIES", 3)
    with pytest.raises(AssemblyError, match="selected-frame-entry limit"):
        assemble_incident(
            incident,
            perceptions,
            tmp_path / "entry-limit",
            window_ms=100,
            logical_latency_ms=10,
        )
    assert not (tmp_path / "entry-limit").exists()
    monkeypatch.setattr(core, "MAX_SELECTED_FRAME_ENTRIES", 1_000_000)
    monkeypatch.setattr(core, "MAX_ASSEMBLED_CLIP_BYTES", 1)
    output = tmp_path / "output"

    with pytest.raises(AssemblyError, match="cumulative output"):
        assemble_incident(
            incident,
            perceptions,
            output,
            window_ms=100,
            logical_latency_ms=10,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".output.tmp-*")) == []


def test_cli_is_silent_until_success_and_caps_latency(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    incident, perceptions, _ = write_incident(tmp_path, default_records())
    clip = incident / "clip.avi"
    clip.chmod(0o644)

    assert (
        main(
            [
                str(incident),
                str(perceptions),
                "--output-dir",
                str(tmp_path / "output"),
                "--window-ms",
                "100",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "permissions" in captured.err


def test_cli_rejects_overflowing_json_number_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    incident, perceptions, metadata = write_incident(tmp_path, default_records())
    metadata["clip"]["fps"] = 10**4000
    metadata_path = incident / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(metadata_path, 0o600)
    output = tmp_path / "output"

    assert (
        main(
            [
                str(incident),
                str(perceptions),
                "--output-dir",
                str(output),
                "--window-ms",
                "100",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "metadata.clip.fps must be a finite number" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
