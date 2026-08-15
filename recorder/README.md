# Foliage Warden observe-only clip recorder

This package turns an explicit local video plus matching
`perception_observation` JSONL into short, silent incident clips for offline
model review. A clip starts only when a `CAT` track overlaps an `approach` zone.
Cat presence elsewhere, person presence, behavior labels, and policy state are
not triggers.

The recorder is deliberately not a controller. It has no action, policy,
arming, actuator, GPIO, serial, display, audio, upload, or network API. Its CLI
cannot open a camera: it decodes an existing local video path. A live or
camera-like source can be injected through the small `RecorderFrame` iterator
interface by a future observe-only process, but this package never opens one
implicitly.

## Install and test

From the repository root, using Python 3.10 or newer:

```sh
uv sync --project recorder --group dev
uv run --project recorder --group dev pytest recorder/tests
uv run --project recorder --group dev ruff check recorder
```

OpenCV is optional. Desktop offline-video use adds its headless wheel:

```sh
uv sync --project recorder --extra desktop --group dev
```

On a Jetson, do not install that wheel over JetPack. Use the system `cv2` and
the source package instead:

```sh
PYTHONPATH=recorder/src python3 -m foliage_warden_recorder.cli --help
```

## Exact offline smoke test

The following creates a ten-frame synthetic video and matching observations in
`/tmp`, runs the recorder headlessly, and inspects the one atomic incident
directory. It does not access a camera, microphone, display, or network.

```sh
uv run --project recorder --extra desktop python - <<'PY'
import json
from pathlib import Path

import cv2
import numpy as np

root = Path("/tmp/foliage-warden-recorder-smoke")
root.mkdir(exist_ok=True)
video = root / "input.avi"
events = root / "observations.jsonl"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
records = []
for sequence in range(10):
    writer.write(np.full((48, 64, 3), sequence * 20, dtype=np.uint8))
    trigger = sequence in {4, 5}
    tracks = []
    if trigger:
        tracks.append({
            "class": "CAT",
            "track_id": "synthetic-cat",
            "region_evidence": {"approach_overlap": 0.5},
        })
    records.append({
        "cat_count": len(tracks),
        "frame": {"index": sequence},
        "mode": "OBSERVE_ONLY",
        "observation": {
            "camera_id": "camera-1",
            "captured_at_ms": sequence * 100,
            "frame_id": f"camera-1:frame:{sequence:08d}",
            "observation_id": f"camera-1:observation:{sequence:08d}",
            "tracks": tracks,
        },
        "record_type": "perception_observation",
        "schema_version": 1,
        "sequence": sequence,
        "source": {"kind": "video", "name": "input.avi"},
        "would_action": False,
    })
writer.release()
events.write_text("".join(json.dumps(record) + "\n" for record in records))
PY
rm -rf /tmp/foliage-warden-recorder-smoke/output
uv run --project recorder --extra desktop foliage-warden-recorder \
  /tmp/foliage-warden-recorder-smoke/input.avi \
  --observations /tmp/foliage-warden-recorder-smoke/observations.jsonl \
  --output-dir /tmp/foliage-warden-recorder-smoke/output \
  --pre-event-ms 200 --post-event-ms 200 --max-clip-ms 2000
fd . /tmp/foliage-warden-recorder-smoke/output --type f --max-depth 3
```

The command prints one deterministic `recorder_summary`. The output contains
one `incidents/incident-.../` directory with `clip.avi` and `metadata.json`.

## Pairing and trigger contract

Every decoded frame must have exactly one observation. Before a frame enters
the rolling buffer, the recorder requires:

- schema version is `1`, `record_type` is `perception_observation`, mode is
  `OBSERVE_ONLY`, and `would_action` is exactly `false`;
- outer sequence, `frame.index`, nested capture time, camera ID, and source all
  match the paired frame;
- sequence and capture time are strictly increasing, and source identity does
  not change;
- nested `observation_id`, `frame_id`, camera ID, and track IDs are canonical
  identifiers; observation and frame IDs are unique across the full recorder
  process, and track IDs are unique within one observation;
- the complete record contains only strict JSON values with finite numbers and
  interoperable-range integers; the JSONL adapter also rejects duplicate object
  keys and non-standard `NaN`/infinity literals;
- one JSONL record is at most 1 MiB of UTF-8, and its semantic JSON is limited
  to 32 levels, 20,000 values, and 512 tracks;
- `cat_count` agrees with the `CAT` tracks; and
- every cat has finite approach overlap in `[0, 1]`.

Malformed, missing, extra, reordered, or mismatched observations abort a CLI
run without publishing the active partial clip. The default trigger is a cat
with at least 1% normalized approach-zone overlap. The threshold is explicit
and configurable, but cannot be zero.

## Privacy and resource limits

Defaults are conservative and all are configurable at construction or on the
CLI:

| Limit | Default | Behavior |
| --- | ---: | --- |
| Pre-event window | 3 s | Time-pruned rolling buffer |
| Pre-event frame cap | 300 frames | Second bound for abnormal timestamps/rates |
| Pre-event decoded-pixel cap | 256 MiB | Oldest frames are removed until both prebuffer caps hold |
| Post-event window | 3 s | Retriggers coalesce into the same incident |
| Maximum clip duration | 15 s | One clip, then suppress until the cat clears |
| Active clip frame cap | 600 frames | Clip terminates before another frame could exceed the cap |
| Active decoded-pixel cap | 512 MiB | Clip terminates before another frame could exceed the cap |
| Accepted observations | 1,000,000 | Session rejects another record before lifetime identity state can grow further |
| Retained incidents | 100 | Oldest managed incident is removed first |
| Recorder disk budget | 5 GiB | Oldest managed incidents are removed first |

The four in-memory limits are independent and apply to owned decoded-pixel
copies, not compressed output estimates. `--max-buffer-frames` and
`--max-buffer-bytes` bound the rolling history;
`--max-active-frames` and `--max-active-bytes` bound an incident being built.
If an active incident reaches either cap, it is atomically published with a
`max_active_frames` or `max_active_bytes` termination reason. A trigger that is
still active is then suppressed until a clear observation, which prevents one
continuous presence from generating repeated clips. A trigger frame too large
to fit the active byte cap is not retained and is likewise suppressed until
clear. Capture backends may reuse mutable decode buffers: the recorder copies
every retained frame so later writes cannot alter pre-event history or an
active clip.

Exact full-stream ID uniqueness requires retaining the accepted observation
and frame IDs for the recorder process lifetime. Each ID is capped at 128 ASCII
identifier characters, and `max_accepted_observations` bounds each lifetime set
to 1,000,000 entries by default. The constructor and
`--max-accepted-observations` can select a different positive finite cap. Once
the cap is reached, the recorder rejects another observation before validation
or allocation; rotate to a new recorder process/session explicitly. Decoded
camera pixels remain bounded independently by the four limits above.

If one encoded incident exceeds the disk budget, it is discarded instead of
being published. Retention only recognizes recorder-generated incident names
inside the explicit output root; unrelated files are never retention targets.
The output root and its managed directories may not be symbolic links.

Clip bytes and stable-key-order metadata are first written into a private
staging directory. A directory rename publishes both together, so consumers do
not see half an incident. Failed encodes and stale recorder staging directories
are cleaned up. Metadata records the encoded clip's byte size and SHA-256;
startup validation and retention stream the clip and fail closed on either
mismatch before removing any managed incident. The digest detects clip-byte
alteration only while metadata remains trusted; it is not a signature or proof
of origin. Files are mode `0600`. The explicit output root and managed
directories are forced to mode `0700` during initialization; later group- or
world-accessible mode changes are rejected before encoding or retention.
Stored metadata is capped at 4 MiB and read through a no-follow descriptor.
Duplicate keys, non-finite numbers, and non-interoperable integers are rejected.
Restart also revalidates schema version, `OBSERVE_ONLY`, silent clip metadata,
and the exact all-false audio, display, and network privacy flags. Provenance
remains optional only for schema-v1 incident directories written before
provenance was introduced. Metadata and clip descriptors must remain regular,
owned by the current user where the platform exposes ownership, and inaccessible
to group and other users.

Every newly published incident also has `perception_provenance`. Its
`record_count` equals `clip.frame_count`, and its `frame_bindings` are ordered by
contiguous `encoded_frame_index` from zero. Each binding records the exact
paired perception sequence, `captured_at_ms`, `observation_id`, `frame_id`, and
`perception_record_sha256`. Pre-event eviction, active limits, retrigger
coalescing, and finalization move a frame and its immutable binding together.
Before encoding, publication compares every binding's sequence and capture
timestamp to the corresponding supplied frame.

The digest bytes are defined exactly:

- `perception_record_sha256` is SHA-256 over the accepted perception record
  rendered as UTF-8 JSON with lexicographically sorted object keys, compact
  separators (`,` and `:`), JSON `false`/`true`/`null` spellings, and no trailing
  newline. The metadata labels this `JSON_SORTED_KEYS_COMPACT_UTF8_V1`.
- `stream_sha256` is SHA-256 over every ordered frame-binding object rendered
  with that same canonical JSON encoding, followed by exactly one LF byte
  (`0x0a`) after every object, including the last. The metadata labels this
  `JSONL_FRAME_BINDINGS_SORTED_KEYS_COMPACT_UTF8_V1`.

Full perception records are not copied into clip metadata. Existing schema-v1
incident directories created before provenance was added may omit this field
and remain restart/retention compatible; all new publications require and
validate it.

These hashes detect accidental record/binding mismatch only while the metadata
is trusted. They are not signatures, do not authenticate camera pixels, and do
not prove that an encoded frame visually corresponds to the named perception
record. An actor able to rewrite metadata can rewrite both bindings and hashes.
Clip content remains sensitive camera data: keep the output directory local,
restrict access, set a retention budget appropriate to the site, and obtain
consent before collecting real-world footage.
