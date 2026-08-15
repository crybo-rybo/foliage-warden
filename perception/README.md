# Foliage Warden observe-only perception

This Python 3.10+ package provides the first perception shell without claiming
to recognize harmful behavior. It loads the pinned OpenCV Zoo YOLOX-S model,
emits only `CAT` and `PERSON` detections, assigns simple IoU track IDs, and
measures normalized overlap with calibrated approach, foliage, soil, and
no-fire polygons.

Every observation is explicitly:

```json
{"behavior":"UNKNOWN","mode":"OBSERVE_ONLY","would_action":false}
```

There is no behavior classifier, policy-state transition, recorder, display,
or actuator dependency in this package. Camera mode cannot be armed and has no
code path capable of issuing an action.

## Install and test on a development machine

From the repository root:

```sh
uv sync --project perception --extra desktop --group dev
uv run --project perception --extra desktop --group dev pytest perception/tests
uv run --project perception --extra desktop --group dev ruff check perception
```

`opencv-python-headless` is deliberately an optional desktop dependency.
JetPack supplies a hardware-integrated OpenCV build on the Jetson, and a PyPI
wheel must not replace it.

Fetch and verify the exact model bytes pinned by `models/registry.json`:

```sh
uv run --project perception --extra desktop python tools/fetch_model.py yolox_s_opencv_zoo
```

Model startup checks the SHA-256 digest again and refuses missing or changed
bytes.

## Exact local smoke test

This creates a temporary blank image, runs one headless observation, prints
JSONL on stdout, and prints timing JSON on stderr:

```sh
uv run --project perception --extra desktop python - <<'PY'
import cv2
import numpy as np
cv2.imwrite("/tmp/foliage-warden-smoke.png", np.zeros((720, 1280, 3), dtype=np.uint8))
PY
uv run --project perception --extra desktop foliage-warden-perception image \
  /tmp/foliage-warden-smoke.png \
  --zones config/simulation-safe.example.json \
  --max-frames 1 \
  --benchmark
```

The expected observation has no tracks for a blank image and always has
`"behavior":"UNKNOWN"`, `"mode":"OBSERVE_ONLY"`, and
`"would_action":false`. Latencies are sent to stderr rather than embedded in
observations, so fixed media and fixed detector outputs serialize byte for
byte identically.

## Inputs

All commands default to stdout JSONL, with no GUI and no video recording:

```sh
uv run --project perception --extra desktop foliage-warden-perception image frame.jpg
uv run --project perception --extra desktop foliage-warden-perception video clip.mp4 \
  --max-frames 300 --benchmark
uv run --project perception --extra desktop foliage-warden-perception camera \
  --device 0 --width 1280 --height 720 --fps 30 --max-frames 300 --benchmark
```

Use `--output observations.jsonl` to write observation metadata explicitly.
`--max-frames` is particularly useful for repeatable smoke tests. Video time is
derived from frame index and nominal FPS. Camera time is also a nominal
observe-only timeline; it is not a safety clock and must not be used to enable
physical effects.

`--camera-id` accepts a canonical ASCII identifier beginning with a letter or
digit and containing only letters, digits, `.`, `_`, `:`, or `-`. It is capped
at 99 characters so the derived observation and frame IDs remain within the
shared 128-character contract even at the maximum interoperable frame index.

Calibration input may be the browser tool's export, a standalone `scene`
object, or the full runtime configuration:

```sh
foliage-warden-perception video clip.mp4 \
  --zones config/simulation-safe.example.json \
  --cat-confidence 0.6 --person-confidence 0.5
```

Polygons may be concave but must be simple, non-zero-area, normalized polygons.
Overlap is exact polygon/box intersection area divided by detection-box area.
`zone_evidence` includes every track and every configured zone. Cat tracks also
embed the canonical policy-facing region evidence, with `motion_score` fixed to
zero because motion/behavior inference is not implemented here.

## Jetson Orin Nano with system OpenCV

Do not create an isolated environment that shadows JetPack's `cv2`. From a
checkout on the Orin, first verify the system packages:

```sh
python3 - <<'PY'
import cv2
import numpy
print("OpenCV", cv2.__version__)
print("NumPy", numpy.__version__)
PY
```

Then use the source package through `PYTHONPATH`:

```sh
PYTHONPATH=perception/src python3 -m foliage_warden_perception camera \
  --device 0 \
  --width 1280 --height 720 --fps 30 \
  --zones config/simulation-safe.example.json \
  --backend-target opencv \
  --max-frames 300 \
  --benchmark \
  > /tmp/foliage-warden-observations.jsonl
```

The `opencv` target is the compatibility baseline for OpenCV 4.8. The package
does not call `NMSBoxesBatched` (introduced after that version). Try
`--backend-target cuda-fp16` only if `cv2.getBuildInformation()` confirms that
the installed OpenCV DNN module was compiled with CUDA; otherwise OpenCV will
fail clearly and the CPU target remains available. TensorRT/DeepStream engine
integration is intentionally outside this Python baseline.

If direct V4L2 negotiation does not select the camera's MJPEG mode, pass a
tested OpenCV GStreamer pipeline as one quoted `--device` value together with
`--gstreamer`. Example CPU-decode shape (adjust `/dev/video0` as needed):

```sh
PYTHONPATH=perception/src python3 -m foliage_warden_perception camera \
  --device 'v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false' \
  --gstreamer --max-frames 300 --benchmark \
  > /tmp/foliage-warden-observations.jsonl
```

## Output contract

Each stdout line is one stable-key-order `perception_observation` record. The
record contains source/frame metadata, pinned model identity, a nested
policy-observation object, per-track zone evidence, `cat_count`, and
`person_present`. Bounding boxes use normalized `x`, `y`, `width`, and `height`.
Floats are rounded to six decimal places to remove irrelevant platform noise.

YOLOX class confidence is `objectness * best COCO class probability`. The best
class is selected over all 80 COCO classes before filtering to cat/person, so a
dog cannot be exposed as a lower-scoring cat candidate. NMS is performed
independently for cat and person and breaks equal-score ties by original model
row, making its output independent of OpenCV NMS return conventions.

`--benchmark` emits a separate `perception_benchmark` JSON object to stderr
with count/min/mean/p50/p95/max milliseconds for capture, preprocessing,
inference, postprocessing, tracking, region geometry, JSON serialization, and
end-to-end frame time, plus effective FPS.

## Current boundary and next steps

This baseline is appropriate for collecting clips, measuring local cat/person
recall, testing calibration geometry, and profiling the Orin. It is not
evidence for `EATING` or `DIGGING`, and its `UNKNOWN` output must not be mapped
to a would-action. Production work still needs a temporal behavior model,
motion evidence, a production tracker (for example NvSORT/NvDCF), held-out
session evaluation, and integration through the separately safety-gated policy
boundary.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact algorithm
source and license attribution.
