from __future__ import annotations

import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from foliage_warden_recorder import (
    ClipEncoding,
    IncidentRecorder,
    IncidentStore,
    ObservationError,
    RecorderConfig,
    RecorderFrame,
    RecorderStateError,
    StorageError,
)


class FakeEncoder:
    suffix = ".fake"

    def __init__(self, *, byte_count: int | None = None, fail_count: int = 0) -> None:
        self.byte_count = byte_count
        self.fail_count = fail_count
        self.calls: list[list[int]] = []
        self.pixel_calls: list[list[Any]] = []

    def encode(
        self,
        frames: Sequence[RecorderFrame],
        destination: Path,
        *,
        fps: float,
    ) -> ClipEncoding:
        sequences = [frame.sequence for frame in frames]
        self.calls.append(sequences)
        self.pixel_calls.append(
            [
                bytes(frame.pixels) if isinstance(frame.pixels, bytearray) else frame.pixels
                for frame in frames
            ]
        )
        if self.fail_count:
            self.fail_count -= 1
            destination.write_bytes(b"partial")
            raise StorageError("synthetic encoder failure")
        payload = ",".join(str(value) for value in sequences).encode()
        if self.byte_count is not None:
            payload = b"x" * self.byte_count
        destination.write_bytes(payload)
        return ClipEncoding("fake", "FAKE", 16, 12, fps)


def frame(
    sequence: int,
    timestamp_ms: int | None = None,
    *,
    source_name: str = "fixture",
    pixels: Any | None = None,
) -> RecorderFrame:
    return RecorderFrame(
        sequence=sequence,
        captured_at_ms=sequence * 100 if timestamp_ms is None else timestamp_ms,
        camera_id="camera-test",
        source_kind="synthetic",
        source_name=source_name,
        pixels=f"frame-{sequence}" if pixels is None else pixels,
    )


def observation(
    item: RecorderFrame,
    *,
    cats: tuple[tuple[str, float], ...] = (),
    people: tuple[str, ...] = (),
) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = [
        {
            "class": "CAT",
            "region_evidence": {"approach_overlap": overlap},
            "track_id": track_id,
        }
        for track_id, overlap in cats
    ]
    tracks.extend({"class": "PERSON", "track_id": track_id} for track_id in people)
    return {
        "behavior": "UNKNOWN",
        "cat_count": len(cats),
        "frame": {"index": item.sequence},
        "mode": "OBSERVE_ONLY",
        "observation": {
            "camera_id": item.camera_id,
            "captured_at_ms": item.captured_at_ms,
            "tracks": tracks,
        },
        "record_type": "perception_observation",
        "schema_version": 1,
        "sequence": item.sequence,
        "source": {"kind": item.source_kind, "name": item.source_name},
        "would_action": False,
    }


def make_recorder(
    tmp_path: Path,
    *,
    config: RecorderConfig | None = None,
    encoder: FakeEncoder | None = None,
) -> tuple[IncidentRecorder, FakeEncoder]:
    selected_config = config or RecorderConfig(
        pre_event_ms=200,
        post_event_ms=200,
        max_clip_ms=2_000,
        max_buffer_frames=20,
        nominal_fps=10,
        max_incidents=10,
        max_disk_bytes=1_000_000,
    )
    selected_encoder = encoder or FakeEncoder()
    store = IncidentStore(tmp_path, selected_config)
    return (
        IncidentRecorder(store, selected_encoder, selected_config),
        selected_encoder,
    )


def test_pre_and_post_boundaries_and_deterministic_metadata(tmp_path: Path) -> None:
    recorder, encoder = make_recorder(tmp_path / "first")
    published = None
    for sequence in range(7):
        item = frame(sequence)
        cats = (("cat-1", 0.5),) if sequence == 4 else ()
        published = recorder.process(item, observation(item, cats=cats)) or published

    assert published is not None
    assert encoder.calls == [[2, 3, 4, 5, 6]]
    metadata_bytes = published.metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    assert metadata["timeline"] == {
        "duration_ms": 400,
        "end_captured_at_ms": 600,
        "end_sequence": 6,
        "first_trigger_at_ms": 400,
        "last_trigger_at_ms": 400,
        "start_captured_at_ms": 200,
        "start_sequence": 2,
    }
    assert metadata["clip"]["audio"] is False
    assert metadata["privacy"] == {"audio": False, "display": False, "network": False}

    second, _ = make_recorder(tmp_path / "second")
    second_published = None
    for sequence in range(7):
        item = frame(sequence)
        cats = (("cat-1", 0.5),) if sequence == 4 else ()
        second_published = second.process(item, observation(item, cats=cats)) or second_published
    assert second_published is not None
    assert second_published.metadata_path.read_bytes() == metadata_bytes


def test_retrigger_during_post_window_coalesces_one_incident(tmp_path: Path) -> None:
    recorder, encoder = make_recorder(tmp_path)
    published = []
    for sequence in range(6):
        item = frame(sequence)
        cats = (("cat-1", 0.6),) if sequence in {1, 3} else ()
        result = recorder.process(item, observation(item, cats=cats))
        if result is not None:
            published.append(result)

    assert len(published) == 1
    assert encoder.calls == [[0, 1, 2, 3, 4, 5]]
    metadata = json.loads(published[0].metadata_path.read_text())
    assert [sample["sequence"] for sample in metadata["trigger"]["samples"]] == [1, 3]
    assert metadata["termination"] == "post_event_elapsed"


def test_retrigger_after_post_window_starts_a_new_incident_despite_frame_gap(
    tmp_path: Path,
) -> None:
    recorder, encoder = make_recorder(tmp_path)
    first = frame(0, 0)
    recorder.process(first, observation(first, cats=(("cat-1", 0.6),)))
    early_clear = frame(1, 100)
    recorder.process(early_clear, observation(early_clear))
    late_retrigger = frame(2, 500)
    first_published = recorder.process(
        late_retrigger,
        observation(late_retrigger, cats=(("cat-1", 0.6),)),
    )
    second_published = recorder.close()

    assert first_published is not None
    assert second_published is not None
    assert encoder.calls == [[0, 1], [2]]
    assert first_published.incident_id != second_published.incident_id


def test_only_cat_in_approach_zone_triggers(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        minimum_approach_overlap=0.2,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)

    first = frame(0)
    recorder.process(first, observation(first, cats=(("outside", 0.0),)))
    second = frame(1)
    recorder.process(second, observation(second, cats=(("below", 0.19),)))
    third = frame(2)
    recorder.process(third, observation(third, people=("person-in-zone",)))
    fourth = frame(3)
    recorder.process(fourth, observation(fourth, cats=(("inside", 0.2),)))
    fifth = frame(4)
    published = recorder.process(fifth, observation(fifth))

    assert published is not None
    assert encoder.calls == [[3, 4]]


def test_continuous_presence_creates_one_max_clip_then_suppresses_until_clear(
    tmp_path: Path,
) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=100,
        max_clip_ms=200,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    published = []
    for sequence in range(7):
        item = frame(sequence)
        cats = (("cat-1", 0.8),) if sequence in {0, 1, 2, 3, 5} else ()
        result = recorder.process(item, observation(item, cats=cats))
        if result is not None:
            published.append(result)

    assert len(published) == 2
    assert encoder.calls == [[0, 1, 2], [5, 6]]
    assert json.loads(published[0].metadata_path.read_text())["termination"] == "max_clip_duration"


def test_rolling_prebuffer_is_also_bounded_by_frame_count(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=10_000,
        post_event_ms=0,
        max_clip_ms=20_000,
        max_buffer_frames=3,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    for sequence in range(6):
        item = frame(sequence)
        recorder.process(item, observation(item, cats=(("cat", 0.5),) if sequence == 5 else ()))
    final = frame(6)
    recorder.process(final, observation(final))

    assert encoder.calls == [[3, 4, 5, 6]]


def test_prebuffer_is_bounded_by_decoded_pixel_bytes(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=10_000,
        post_event_ms=0,
        max_clip_ms=20_000,
        max_buffer_frames=10,
        max_buffer_bytes=6,
        max_active_frames=20,
        max_active_bytes=100,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    for sequence in range(4):
        item = frame(sequence, pixels=bytearray(b"abc"))
        recorder.process(
            item,
            observation(item, cats=(("cat", 0.5),) if sequence == 3 else ()),
        )
        assert recorder.buffered_byte_count <= 6
    clear = frame(4, pixels=bytearray(b"abc"))
    recorder.process(clear, observation(clear))

    assert encoder.calls == [[2, 3, 4]]


def test_active_frame_limit_terminates_and_suppresses_continuous_trigger(
    tmp_path: Path,
) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=1_000,
        max_clip_ms=20_000,
        max_buffer_frames=10,
        max_buffer_bytes=100,
        max_active_frames=2,
        max_active_bytes=100,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    for sequence in range(2):
        item = frame(sequence, pixels=bytearray(b"abc"))
        assert recorder.process(item, observation(item, cats=(("cat", 0.5),))) is None
        assert recorder.active_frame_count <= 2

    limit_frame = frame(2, pixels=bytearray(b"abc"))
    published = recorder.process(
        limit_frame,
        observation(limit_frame, cats=(("cat", 0.5),)),
    )
    assert published is not None
    assert json.loads(published.metadata_path.read_text())["termination"] == "max_active_frames"
    assert encoder.calls == [[0, 1]]
    assert recorder.suppressed_until_clear

    still_active = frame(3, pixels=bytearray(b"abc"))
    assert recorder.process(still_active, observation(still_active, cats=(("cat", 0.5),))) is None
    assert not recorder.incident_active
    clear = frame(4, pixels=bytearray(b"abc"))
    assert recorder.process(clear, observation(clear)) is None
    assert not recorder.suppressed_until_clear


def test_active_byte_limit_terminates_before_exceeding_bound(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=1_000,
        max_clip_ms=20_000,
        max_buffer_frames=10,
        max_buffer_bytes=100,
        max_active_frames=10,
        max_active_bytes=6,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    for sequence in range(2):
        item = frame(sequence, pixels=bytearray(b"abc"))
        recorder.process(item, observation(item, cats=(("cat", 0.5),)))
        assert recorder.active_byte_count <= 6

    clear = frame(2, pixels=bytearray(b"abc"))
    published = recorder.process(clear, observation(clear))
    assert published is not None
    assert json.loads(published.metadata_path.read_text())["termination"] == "max_active_bytes"
    assert encoder.calls == [[0, 1]]
    assert not recorder.suppressed_until_clear
    assert recorder.buffered_byte_count == 3


def test_oversize_trigger_is_suppressed_without_storing_pixels(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=1_000,
        post_event_ms=100,
        max_clip_ms=20_000,
        max_buffer_frames=10,
        max_buffer_bytes=10,
        max_active_frames=10,
        max_active_bytes=2,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    item = frame(0, pixels=bytearray(b"abc"))
    assert recorder.process(item, observation(item, cats=(("cat", 0.5),))) is None
    assert recorder.suppressed_until_clear
    assert not recorder.incident_active
    assert recorder.buffered_frame_count == 0
    assert recorder.buffered_byte_count == 0
    assert not encoder.calls


def test_recorder_owns_buffered_pixels_when_capture_reuses_mutable_storage(
    tmp_path: Path,
) -> None:
    recorder, encoder = make_recorder(tmp_path)
    shared = bytearray(b"000")
    first = frame(0, pixels=shared)
    recorder.process(first, observation(first))
    shared[:] = b"111"
    trigger = frame(1, pixels=shared)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    shared[:] = b"222"
    clear = frame(2, pixels=shared)
    recorder.process(clear, observation(clear))
    recorder.close()

    assert encoder.pixel_calls == [[b"000", b"111", b"222"]]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(sequence=99), "sequence does not match"),
        (lambda value: value.update(schema_version=2), "schema_version must be 1"),
        (lambda value: value.update(schema_version=True), "schema_version must be 1"),
        (lambda value: value.update(schema_version=1.0), "schema_version must be 1"),
        (lambda value: value.update(mode="ARMED"), "only OBSERVE_ONLY"),
        (lambda value: value.update(would_action=True), "exactly false"),
        (lambda value: value.update(cat_count=2), "cat_count does not match"),
    ],
)
def test_invalid_observation_is_rejected_before_buffering(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    item = frame(0)
    event = observation(item)
    mutation(event)
    with pytest.raises(ObservationError, match=match):
        recorder.process(item, event)
    assert recorder.buffered_frame_count == 0


def test_sequence_and_timestamp_must_be_strictly_increasing(tmp_path: Path) -> None:
    recorder, _ = make_recorder(tmp_path)
    first = frame(1, 100)
    recorder.process(first, observation(first))

    duplicate_sequence = frame(1, 200)
    with pytest.raises(ObservationError, match="sequences must be strictly increasing"):
        recorder.process(duplicate_sequence, observation(duplicate_sequence))

    duplicate_time = frame(2, 100)
    with pytest.raises(ObservationError, match="timestamps must be strictly increasing"):
        recorder.process(duplicate_time, observation(duplicate_time))


def test_max_incident_retention_deletes_only_oldest_managed_clips(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=2,
        max_disk_bytes=100_000,
    )
    recorder, _ = make_recorder(tmp_path, config=config)
    for sequence in range(1, 7):
        item = frame(sequence)
        recorder.process(
            item,
            observation(item, cats=((f"cat-{sequence}", 0.5),) if sequence % 2 else ()),
        )

    names = sorted(path.name for path in (tmp_path / "incidents").iterdir())
    assert names == ["incident-0000000000300-0000000003", "incident-0000000000500-0000000005"]


def test_disk_retention_and_oversize_incident_are_enforced(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=10,
        max_disk_bytes=6_000,
    )
    recorder, _ = make_recorder(
        tmp_path / "retained",
        config=config,
        encoder=FakeEncoder(byte_count=4_096),
    )
    for sequence in range(1, 7):
        item = frame(sequence)
        recorder.process(
            item,
            observation(item, cats=(("cat", 0.5),) if sequence % 2 else ()),
        )
    names = list((tmp_path / "retained" / "incidents").iterdir())
    assert [path.name for path in names] == ["incident-0000000000500-0000000005"]

    oversize_config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=10,
        max_disk_bytes=100,
    )
    oversize, _ = make_recorder(
        tmp_path / "oversize",
        config=oversize_config,
        encoder=FakeEncoder(byte_count=101),
    )
    trigger = frame(0)
    oversize.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)
    with pytest.raises(StorageError, match="exceeds max_disk_bytes"):
        oversize.process(clear, observation(clear))
    assert not list((tmp_path / "oversize" / "incidents").iterdir())
    assert not list((tmp_path / "oversize" / ".staging").iterdir())


def test_encoder_failure_leaves_no_partial_and_next_incident_can_recover(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    encoder = FakeEncoder(fail_count=1)
    recorder, _ = make_recorder(tmp_path, config=config, encoder=encoder)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)
    with pytest.raises(StorageError, match="synthetic encoder failure"):
        recorder.process(clear, observation(clear))
    assert not list((tmp_path / "incidents").iterdir())
    assert not list((tmp_path / ".staging").iterdir())

    second_trigger = frame(2)
    recorder.process(second_trigger, observation(second_trigger, cats=(("cat", 0.5),)))
    second_clear = frame(3)
    recovered = recorder.process(second_clear, observation(second_clear))
    assert recovered is not None
    assert recovered.directory.is_dir()


def test_stale_staging_is_recovered_and_symlink_escape_is_rejected(tmp_path: Path) -> None:
    config = RecorderConfig(max_disk_bytes=100_000)
    root = tmp_path / "recover"
    stale = root / ".staging" / "incident-0000000000001-0000000001.tmp-stale"
    stale.mkdir(parents=True)
    (stale / "partial").write_text("partial")
    IncidentStore(root, config)
    assert not stale.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "incidents").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StorageError, match="must not be a symbolic link"):
        IncidentStore(escaped, config)
    assert not list(outside.iterdir())


def test_existing_managed_directories_are_forced_private(tmp_path: Path) -> None:
    root = tmp_path / "private"
    incidents = root / "incidents"
    staging = root / ".staging"
    incidents.mkdir(parents=True)
    staging.mkdir()
    for path in (root, incidents, staging):
        os.chmod(path, 0o777)

    IncidentStore(root, RecorderConfig(max_disk_bytes=100_000))

    for path in (root, incidents, staging):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


@pytest.mark.parametrize("directory_name", [None, "incidents", ".staging"])
def test_permissive_managed_directory_is_rejected_after_initialization(
    tmp_path: Path,
    directory_name: str | None,
) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    recorder, encoder = make_recorder(tmp_path, config=config)
    changed = tmp_path if directory_name is None else tmp_path / directory_name
    os.chmod(changed, 0o755)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)

    with pytest.raises(StorageError, match="permissions must not allow"):
        recorder.process(clear, observation(clear))
    assert not encoder.calls


def test_managed_directory_replaced_by_symlink_is_detected_before_encode(tmp_path: Path) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    encoder = FakeEncoder()
    recorder, _ = make_recorder(tmp_path / "root", config=config, encoder=encoder)
    incidents = tmp_path / "root" / "incidents"
    original = tmp_path / "root" / "incidents-original"
    incidents.rename(original)
    outside = tmp_path / "outside-after-init"
    outside.mkdir()
    incidents.symlink_to(outside, target_is_directory=True)

    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)
    with pytest.raises(StorageError, match="must remain a real directory"):
        recorder.process(clear, observation(clear))

    assert not encoder.calls
    assert not list(outside.iterdir())


def test_retention_refuses_to_delete_unrecognized_managed_looking_directory(
    tmp_path: Path,
) -> None:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=0,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=1,
        max_disk_bytes=100_000,
    )
    recorder, _ = make_recorder(tmp_path, config=config)
    unrecognized = tmp_path / "incidents" / "incident-0000000000000-0000000000"
    unrecognized.mkdir(mode=0o700)
    keep = unrecognized / "do-not-delete"
    keep.write_text("unrelated")

    trigger = frame(1)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(2)
    with pytest.raises(StorageError, match="has no safe metadata"):
        recorder.process(clear, observation(clear))

    assert keep.read_text() == "unrelated"
    assert not list((tmp_path / ".staging").iterdir())


def test_untrusted_source_names_never_become_paths(tmp_path: Path) -> None:
    recorder, _ = make_recorder(tmp_path / "root")
    trigger = frame(0, source_name="../../outside")
    recorder.process(trigger, observation(trigger, cats=(("../../../cat", 0.5),)))
    clear = frame(1, source_name="../../outside")
    recorder.process(clear, observation(clear))
    final_clear = frame(2, source_name="../../outside")
    published = recorder.process(final_clear, observation(final_clear))
    assert published is not None
    assert published.directory.resolve().is_relative_to((tmp_path / "root").resolve())
    assert not (tmp_path / "outside").exists()


def test_closed_recorder_rejects_more_frames(tmp_path: Path) -> None:
    recorder, _ = make_recorder(tmp_path)
    assert recorder.close() is None
    item = frame(0)
    with pytest.raises(RecorderStateError, match="closed"):
        recorder.process(item, observation(item))
