from __future__ import annotations

from collections.abc import Iterator, Sequence
from io import StringIO
from pathlib import Path

import pytest

from foliage_warden_recorder import (
    ClipEncoding,
    IncidentRecorder,
    IncidentStore,
    ObservationError,
    RecorderConfig,
    RecorderFrame,
    run_paired,
)
from foliage_warden_recorder.adapters import JsonlObservations


class Source:
    def __init__(self, values: list[RecorderFrame]) -> None:
        self.values = values
        self.closed = False

    def __iter__(self) -> Iterator[RecorderFrame]:
        yield from self.values

    def close(self) -> None:
        self.closed = True


class Encoder:
    suffix = ".fake"

    def encode(
        self,
        frames: Sequence[RecorderFrame],
        destination: Path,
        *,
        fps: float,
    ) -> ClipEncoding:
        destination.write_bytes(b"clip")
        return ClipEncoding("fake", "FAKE", 1, 1, fps)


def make_frame(index: int) -> RecorderFrame:
    return RecorderFrame(index, index * 100, "cam", "synthetic", "fixture", index)


def make_observation(item: RecorderFrame, *, trigger: bool = False) -> dict:
    tracks = []
    if trigger:
        tracks.append(
            {
                "class": "CAT",
                "region_evidence": {"approach_overlap": 0.5},
                "track_id": "cat",
            }
        )
    return {
        "cat_count": len(tracks),
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


def make_recorder(tmp_path: Path) -> IncidentRecorder:
    config = RecorderConfig(
        pre_event_ms=0,
        post_event_ms=100,
        max_clip_ms=1_000,
        max_buffer_frames=5,
        nominal_fps=10,
        max_incidents=5,
        max_disk_bytes=100_000,
    )
    return IncidentRecorder(IncidentStore(tmp_path, config), Encoder(), config)


@pytest.mark.parametrize("extra_observation", [False, True])
def test_pair_count_mismatch_aborts_without_partial_publication(
    tmp_path: Path,
    extra_observation: bool,
) -> None:
    frames = [make_frame(0)]
    observations = [make_observation(frames[0], trigger=True)]
    if extra_observation:
        observations.append(make_observation(make_frame(1)))
    else:
        frames.append(make_frame(1))
    source = Source(frames)
    recorder = make_recorder(tmp_path)

    with pytest.raises(ObservationError, match="longer than"):
        run_paired(source, observations, recorder)

    assert source.closed
    assert not list((tmp_path / "incidents").iterdir())


def test_normal_source_end_publishes_active_incident(tmp_path: Path) -> None:
    item = make_frame(0)
    source = Source([item])
    published = run_paired(source, [make_observation(item, trigger=True)], make_recorder(tmp_path))

    assert source.closed
    assert len(published) == 1
    assert published[0].clip_path.read_bytes() == b"clip"


@pytest.mark.parametrize(
    ("record", "expected_error"),
    [
        ('{"value":1,"value":2}\n', "duplicate object key 'value'"),
        ('{"value":NaN}\n', "non-finite number NaN"),
        ('{"value":Infinity}\n', "non-finite number Infinity"),
        ('{"value":1e9999}\n', "non-finite number 1e9999"),
        ('{"value":"\\ud800"}\n', "string is not valid UTF-8"),
    ],
)
def test_jsonl_parser_rejects_ambiguous_non_strict_json(
    record: str,
    expected_error: str,
) -> None:
    with pytest.raises(ObservationError, match=expected_error):
        next(iter(JsonlObservations(StringIO(record))))


def test_jsonl_parser_bounds_record_bytes_before_decoding() -> None:
    record = '{"value":"' + "x" * 64 + '"}\n'
    with pytest.raises(ObservationError, match="exceeds 32 UTF-8 bytes"):
        next(iter(JsonlObservations(StringIO(record), max_record_bytes=32)))


def test_jsonl_parser_rejects_excessive_nesting_on_all_supported_pythons() -> None:
    record = '{"value":' + "[" * 1_500 + "0" + "]" * 1_500 + "}\n"
    with pytest.raises(ObservationError, match=r"decoder limit|depth limit"):
        next(iter(JsonlObservations(StringIO(record))))


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_jsonl_parser_requires_a_positive_byte_limit(invalid_limit: object) -> None:
    with pytest.raises(ValueError, match="max_record_bytes must be a positive integer"):
        JsonlObservations(StringIO("{}\n"), max_record_bytes=invalid_limit)  # type: ignore[arg-type]
