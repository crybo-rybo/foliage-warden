from __future__ import annotations

import hashlib
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
from foliage_warden_recorder.observation import (
    MAX_OBSERVATION_JSON_BYTES,
    MAX_OBSERVATION_JSON_DEPTH,
    MAX_OBSERVATION_JSON_NODES,
    MAX_OBSERVATION_TRACKS,
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
            "frame_id": f"{item.camera_id}:frame:{item.sequence:08d}",
            "observation_id": f"{item.camera_id}:observation:{item.sequence:08d}",
            "tracks": tracks,
        },
        "record_type": "perception_observation",
        "schema_version": 1,
        "sequence": item.sequence,
        "source": {"kind": item.source_kind, "name": item.source_name},
        "would_action": False,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def provenance(published: Any) -> dict[str, Any]:
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    return metadata["perception_provenance"]


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(metadata) + b"\n")


def recompute_binding_stream_sha256(frame_bindings: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for binding in frame_bindings:
        digest.update(canonical_json_bytes(binding))
        digest.update(b"\n")
    return digest.hexdigest()


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
    observations = []
    for sequence in range(7):
        item = frame(sequence)
        cats = (("cat-1", 0.5),) if sequence == 4 else ()
        event = observation(item, cats=cats)
        observations.append(event)
        published = recorder.process(item, event) or published

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
    assert metadata["clip"]["byte_size"] == published.clip_path.stat().st_size
    assert (
        metadata["clip"]["sha256"] == hashlib.sha256(published.clip_path.read_bytes()).hexdigest()
    )
    assert metadata["privacy"] == {"audio": False, "display": False, "network": False}
    provenance_metadata = metadata["perception_provenance"]
    bindings = provenance_metadata["frame_bindings"]
    assert provenance_metadata["record_count"] == metadata["clip"]["frame_count"] == 5
    assert [binding["encoded_frame_index"] for binding in bindings] == list(range(5))
    assert [binding["sequence"] for binding in bindings] == [2, 3, 4, 5, 6]
    for binding, event in zip(bindings, observations[2:], strict=True):
        nested = event["observation"]
        assert binding["captured_at_ms"] == nested["captured_at_ms"]
        assert binding["frame_id"] == nested["frame_id"]
        assert binding["observation_id"] == nested["observation_id"]
        assert (
            binding["perception_record_sha256"]
            == hashlib.sha256(canonical_json_bytes(event)).hexdigest()
        )
        reordered = dict(reversed(list(event.items())))
        assert (
            hashlib.sha256(canonical_json_bytes(reordered)).hexdigest()
            == binding["perception_record_sha256"]
        )
    assert provenance_metadata["stream_sha256"] == recompute_binding_stream_sha256(bindings)
    assert stat.S_IMODE(published.clip_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(published.metadata_path.stat().st_mode) == 0o600

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
    assert [
        binding["sequence"] for binding in metadata["perception_provenance"]["frame_bindings"]
    ] == [0, 1, 2, 3, 4, 5]
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
    published = recorder.process(final, observation(final))

    assert encoder.calls == [[3, 4, 5, 6]]
    assert published is not None
    assert [binding["sequence"] for binding in provenance(published)["frame_bindings"]] == [
        3,
        4,
        5,
        6,
    ]


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
    published = recorder.process(clear, observation(clear))

    assert encoder.calls == [[2, 3, 4]]
    assert published is not None
    assert [binding["sequence"] for binding in provenance(published)["frame_bindings"]] == [
        2,
        3,
        4,
    ]


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
    assert [binding["sequence"] for binding in provenance(published)["frame_bindings"]] == [0, 1]
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
    assert [binding["sequence"] for binding in provenance(published)["frame_bindings"]] == [0, 1]
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
    first_observation = observation(first)
    expected_record_sha256 = hashlib.sha256(canonical_json_bytes(first_observation)).hexdigest()
    original_frame_id = first_observation["observation"]["frame_id"]
    recorder.process(first, first_observation)
    first_observation["observation"]["frame_id"] = "mutated-after-ingestion"
    shared[:] = b"111"
    trigger = frame(1, pixels=shared)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    shared[:] = b"222"
    clear = frame(2, pixels=shared)
    recorder.process(clear, observation(clear))
    published = recorder.close()

    assert encoder.pixel_calls == [[b"000", b"111", b"222"]]
    assert published is not None
    first_binding = provenance(published)["frame_bindings"][0]
    assert first_binding["frame_id"] == original_frame_id
    assert first_binding["perception_record_sha256"] == expected_record_sha256
    assert (
        first_binding["perception_record_sha256"]
        != hashlib.sha256(canonical_json_bytes(first_observation)).hexdigest()
    )


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
        (
            lambda value: value["observation"].update(observation_id=True),
            "observation.observation_id must be a canonical identifier",
        ),
        (
            lambda value: value["observation"].update(frame_id="../frame"),
            "observation.frame_id must be a canonical identifier",
        ),
        (
            lambda value: value.update(non_json=object()),
            "contains a non-JSON value",
        ),
        (
            lambda value: value.update(non_finite=float("nan")),
            "contains a non-finite number",
        ),
        (
            lambda value: value["observation"]["tracks"].append(
                {
                    "class": "CAT",
                    "region_evidence": {"approach_overlap": 10**4000},
                    "track_id": "huge-overlap",
                }
            ),
            "integer is outside the interoperable safe range",
        ),
        (
            lambda value: value.update(non_utf8_scalar="\ud800"),
            "observation record is not canonical JSON",
        ),
        (
            lambda value: value["observation"]["tracks"].extend(
                [
                    {"class": "PERSON", "track_id": "duplicate-track"},
                    {"class": "PERSON", "track_id": "duplicate-track"},
                ]
            ),
            "track_id values must be unique",
        ),
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


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda value: value.update(nested=_nested_json_value(MAX_OBSERVATION_JSON_DEPTH + 1)),
            "depth limit",
        ),
        (
            lambda value: value.update(nodes=[0] * MAX_OBSERVATION_JSON_NODES),
            "node limit",
        ),
        (
            lambda value: value.update(payload="x" * MAX_OBSERVATION_JSON_BYTES),
            "canonical limit",
        ),
        (
            lambda value: value["observation"].update(
                tracks=[
                    {"class": "PERSON", "track_id": f"person-{index}"}
                    for index in range(MAX_OBSERVATION_TRACKS + 1)
                ]
            ),
            "track limit",
        ),
    ],
)
def test_semantically_oversize_observation_is_rejected_before_buffering(
    tmp_path: Path,
    mutation: Any,
    expected_error: str,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    item = frame(0)
    event = observation(item)
    mutation(event)

    with pytest.raises(ObservationError, match=expected_error):
        recorder.process(item, event)

    assert recorder.buffered_frame_count == 0


def _nested_json_value(depth: int) -> Any:
    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


def test_cyclic_observation_is_rejected_before_buffering(tmp_path: Path) -> None:
    recorder, _ = make_recorder(tmp_path)
    item = frame(0)
    event = observation(item)
    event["cycle"] = event

    with pytest.raises(ObservationError, match="cyclic JSON value"):
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


@pytest.mark.parametrize(
    ("identifier", "expected_error"),
    [
        ("observation_id", "observation_id must be unique across the recorder stream"),
        ("frame_id", "frame_id must be unique across the recorder stream"),
    ],
)
def test_non_adjacent_ids_must_be_unique_across_the_full_stream(
    tmp_path: Path,
    identifier: str,
    expected_error: str,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    first = frame(0)
    first_observation = observation(first)
    recorder.process(first, first_observation)
    middle = frame(1)
    recorder.process(middle, observation(middle))

    duplicate = frame(2)
    duplicate_observation = observation(duplicate)
    duplicate_observation["observation"][identifier] = first_observation["observation"][identifier]
    with pytest.raises(ObservationError, match=expected_error):
        recorder.process(duplicate, duplicate_observation)

    assert recorder.buffered_frame_count == 2
    recorder.process(duplicate, observation(duplicate))
    assert recorder.buffered_frame_count == 3


def test_accepted_observation_limit_fails_before_identifier_sets_grow(tmp_path: Path) -> None:
    config = RecorderConfig(
        max_accepted_observations=2,
        max_disk_bytes=100_000,
    )
    recorder, _ = make_recorder(tmp_path, config=config)
    for sequence in range(2):
        item = frame(sequence)
        recorder.process(item, observation(item))

    over_limit = frame(2)
    with pytest.raises(RecorderStateError, match="max_accepted_observations"):
        recorder.process(over_limit, observation(over_limit))

    assert recorder.accepted_observation_count == 2
    assert recorder.buffered_frame_count == 2


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_accepted_observation_limit_must_be_a_positive_integer(invalid_limit: Any) -> None:
    with pytest.raises(ValueError, match="max_accepted_observations must be a positive integer"):
        RecorderConfig(max_accepted_observations=invalid_limit)


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


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [("same_size", "SHA-256 mismatch"), ("append", "byte_size mismatch")],
)
def test_restart_rejects_tampered_published_clip(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
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
    recorder, _ = make_recorder(tmp_path, config=config)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)
    published = recorder.process(clear, observation(clear))
    assert published is not None

    original = published.clip_path.read_bytes()
    if mutation == "same_size":
        published.clip_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    else:
        published.clip_path.write_bytes(original + b"x")

    with pytest.raises(StorageError, match=expected_error):
        IncidentStore(tmp_path, config)

    assert published.directory.is_dir()
    assert not list((tmp_path / ".staging").iterdir())


def test_restart_accepts_legacy_schema_v1_metadata_without_provenance(tmp_path: Path) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)
    recorder.process(clear, observation(clear))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    metadata.pop("perception_provenance")
    write_metadata(published.metadata_path, metadata)

    IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


@pytest.mark.parametrize("invalid_schema_version", [True, 1.0])
def test_restart_requires_schema_version_to_be_the_actual_integer_one(
    tmp_path: Path,
    invalid_schema_version: Any,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = invalid_schema_version
    write_metadata(published.metadata_path, metadata)

    with pytest.raises(StorageError, match="mismatched metadata"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(mode="ARMED"),
        lambda value: value["privacy"].update(network=True),
        lambda value: value["privacy"].pop("display"),
        lambda value: value["clip"].update(audio=True),
    ],
)
def test_restart_revalidates_observe_only_privacy_metadata(
    tmp_path: Path,
    mutation: Any,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    mutation(metadata)
    write_metadata(published.metadata_path, metadata)

    with pytest.raises(StorageError, match="mismatched metadata"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


@pytest.mark.parametrize(
    "injected_prefix",
    [
        '"mode":"OBSERVE_ONLY",',
        '"unexpected":NaN,',
        '"unexpected":1e9999,',
        '"unexpected":"\\ud800",',
    ],
)
def test_restart_strictly_decodes_stored_metadata(
    tmp_path: Path,
    injected_prefix: str,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    original = published.metadata_path.read_text(encoding="utf-8")
    published.metadata_path.write_text(
        "{" + injected_prefix + original.removeprefix("{"),
        encoding="utf-8",
    )

    with pytest.raises(StorageError, match="invalid metadata"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


def test_restart_bounds_metadata_before_materializing_it(tmp_path: Path) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    published.metadata_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))

    with pytest.raises(StorageError, match="metadata exceeds the 4194304-byte limit"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=10_000_000))


@pytest.mark.parametrize("artifact", ["metadata_path", "clip_path"])
def test_restart_rejects_group_or_world_accessible_incident_files(
    tmp_path: Path,
    artifact: str,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    os.chmod(getattr(published, artifact), 0o640)

    with pytest.raises(StorageError, match="permissions must not allow group or other access"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


def test_new_publication_requires_perception_provenance(tmp_path: Path) -> None:
    config = RecorderConfig(max_disk_bytes=100_000)
    store = IncidentStore(tmp_path, config)
    item = frame(0)
    encoder = FakeEncoder()
    with pytest.raises(StorageError, match="has no perception provenance"):
        store.publish(
            incident_id="incident-0000000000000-0000000000",
            frames=[item],
            metadata={
                "incident_id": "incident-0000000000000-0000000000",
                "mode": "OBSERVE_ONLY",
                "privacy": {"audio": False, "display": False, "network": False},
                "record_type": "observation_clip",
                "schema_version": 1,
            },
            encoder=encoder,
            fps=10.0,
        )

    assert not encoder.calls
    assert not list((tmp_path / "incidents").iterdir())
    assert not list((tmp_path / ".staging").iterdir())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(incident_id="incident-0000000002000-0000000020"),
        lambda value: value.update(record_type="action_clip"),
        lambda value: value.update(schema_version=True),
        lambda value: value.update(mode="ARMED"),
        lambda value: value["privacy"].update(network=True),
    ],
)
def test_new_publication_rejects_poisoned_root_metadata_before_encoding(
    tmp_path: Path,
    mutation: Any,
) -> None:
    config = RecorderConfig(max_disk_bytes=1_000_000)
    recorder, _ = make_recorder(tmp_path, config=config)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    metadata.pop("clip")
    incident_id = "incident-0000000001000-0000000010"
    metadata["incident_id"] = incident_id
    mutation(metadata)
    encoder = FakeEncoder()

    with pytest.raises(StorageError, match="new incident has mismatched metadata"):
        IncidentStore(tmp_path, config).publish(
            incident_id=incident_id,
            frames=[frame(0)],
            metadata=metadata,
            encoder=encoder,
            fps=10.0,
        )

    assert not encoder.calls
    assert not (tmp_path / "incidents" / incident_id).exists()


@pytest.mark.parametrize(
    "mismatched_frame",
    [frame(1, 0), frame(0, 100)],
)
def test_new_publication_rejects_binding_that_disagrees_with_supplied_frame(
    tmp_path: Path,
    mismatched_frame: RecorderFrame,
) -> None:
    config = RecorderConfig(max_disk_bytes=1_000_000)
    recorder, _ = make_recorder(tmp_path, config=config)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    metadata.pop("clip")
    incident_id = "incident-0000000001000-0000000010"
    metadata["incident_id"] = incident_id
    encoder = FakeEncoder()

    with pytest.raises(StorageError, match="binding disagrees with its supplied frame"):
        IncidentStore(tmp_path, config).publish(
            incident_id=incident_id,
            frames=[mismatched_frame],
            metadata=metadata,
            encoder=encoder,
            fps=10.0,
        )

    assert not encoder.calls
    assert not (tmp_path / "incidents" / incident_id).exists()


def test_restart_rejects_binding_tamper_when_stream_digest_is_unchanged(
    tmp_path: Path,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    metadata["perception_provenance"]["frame_bindings"][0]["frame_id"] = "rewritten-frame"
    write_metadata(published.metadata_path, metadata)

    with pytest.raises(StorageError, match="perception stream SHA-256 mismatch"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


def test_provenance_hash_cannot_detect_coordinated_trusted_metadata_rewrite(
    tmp_path: Path,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    provenance_metadata = metadata["perception_provenance"]
    bindings = provenance_metadata["frame_bindings"]
    bindings[0]["observation_id"] = "trusted-metadata-rewrite"
    provenance_metadata["stream_sha256"] = recompute_binding_stream_sha256(bindings)
    write_metadata(published.metadata_path, metadata)

    IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


def test_restart_rejects_duplicate_ids_even_with_recomputed_stream_digest(
    tmp_path: Path,
) -> None:
    recorder, _ = make_recorder(tmp_path)
    trigger = frame(0)
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
    clear = frame(1)
    recorder.process(clear, observation(clear))
    published = recorder.close()
    assert published is not None
    metadata = json.loads(published.metadata_path.read_text(encoding="utf-8"))
    provenance_metadata = metadata["perception_provenance"]
    bindings = provenance_metadata["frame_bindings"]
    bindings[1]["observation_id"] = bindings[0]["observation_id"]
    provenance_metadata["stream_sha256"] = recompute_binding_stream_sha256(bindings)
    write_metadata(published.metadata_path, metadata)

    with pytest.raises(StorageError, match="duplicate frame binding IDs"):
        IncidentStore(tmp_path, RecorderConfig(max_disk_bytes=1_000_000))


def test_retention_fails_closed_before_deleting_tampered_incident(tmp_path: Path) -> None:
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
    first_trigger = frame(0)
    recorder.process(first_trigger, observation(first_trigger, cats=(("cat", 0.5),)))
    first_clear = frame(1)
    first = recorder.process(first_clear, observation(first_clear))
    assert first is not None

    original = first.clip_path.read_bytes()
    first.clip_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    second_trigger = frame(2)
    recorder.process(second_trigger, observation(second_trigger, cats=(("cat", 0.5),)))
    second_clear = frame(3)
    with pytest.raises(StorageError, match="SHA-256 mismatch"):
        recorder.process(second_clear, observation(second_clear))

    assert first.directory.is_dir()
    assert [path.name for path in (tmp_path / "incidents").iterdir()] == [first.incident_id]
    assert not list((tmp_path / ".staging").iterdir())


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
    recorder.process(trigger, observation(trigger, cats=(("cat", 0.5),)))
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
