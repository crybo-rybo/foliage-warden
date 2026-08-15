from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from foliage_warden_recorder.cli import main

cv2 = pytest.importorskip("cv2")


def _observation(sequence: int, *, trigger: bool) -> dict[str, object]:
    tracks = []
    if trigger:
        tracks.append(
            {
                "class": "CAT",
                "region_evidence": {"approach_overlap": 0.5},
                "track_id": "synthetic-cat",
            }
        )
    return {
        "cat_count": len(tracks),
        "frame": {"index": sequence},
        "mode": "OBSERVE_ONLY",
        "observation": {
            "camera_id": "camera-1",
            "captured_at_ms": sequence * 100,
            "tracks": tracks,
        },
        "record_type": "perception_observation",
        "schema_version": 1,
        "sequence": sequence,
        "source": {"kind": "video", "name": "input.avi"},
        "would_action": False,
    }


def test_real_opencv_cli_writes_one_silent_bounded_clip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    video = tmp_path / "input.avi"
    observations = tmp_path / "observations.jsonl"
    output = tmp_path / "output"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for sequence in range(10):
        writer.write(np.full((48, 64, 3), sequence * 20, dtype=np.uint8))
    writer.release()
    observations.write_text(
        "".join(
            json.dumps(_observation(sequence, trigger=sequence in {4, 5})) + "\n"
            for sequence in range(10)
        ),
        encoding="utf-8",
    )

    result = main(
        [
            str(video),
            "--observations",
            str(observations),
            "--output-dir",
            str(output),
            "--pre-event-ms",
            "200",
            "--post-event-ms",
            "200",
            "--max-clip-ms",
            "2000",
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "OBSERVE_ONLY"
    assert summary["incident_count"] == 1
    incident = next((output / "incidents").iterdir())
    metadata = json.loads((incident / "metadata.json").read_text(encoding="utf-8"))
    clip = incident / metadata["clip"]["filename"]
    assert metadata["privacy"] == {"audio": False, "display": False, "network": False}
    assert metadata["timeline"]["start_sequence"] == 2
    assert metadata["timeline"]["end_sequence"] == 7
    assert metadata["clip"]["byte_size"] == clip.stat().st_size
    assert metadata["clip"]["sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()

    capture = cv2.VideoCapture(str(clip))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (48, 64, 3)
        decoded += 1
    capture.release()
    assert decoded == 6
