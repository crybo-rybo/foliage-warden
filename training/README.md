# Temporal behavior training baseline

This directory contains the off-device reference pipeline for classifying short cat/plant
interaction clips as:

`PASSING`, `SNIFFING`, `EATING`, `DIGGING`, `OTHER`, or `UNKNOWN`.

It uses a small, causal CNN frame encoder plus a unidirectional GRU. The model starts from random
weights and does not download pretrained weights. Its ONNX export has a dynamic batch dimension
and fixed temporal/spatial dimensions, which keeps the eventual TensorRT build contract simple.

This is a **clip-classification baseline**, not an actuator policy. It must never authorize physical
output directly. Detection, region evidence, tracking quality, person presence, and all deterministic
safety gates remain separate runtime responsibilities.

## Environment

The package supports Python 3.10 and newer versions for which PyTorch publishes wheels. The checked-in
`.python-version` selects Python 3.10 so `uv` does not accidentally use a too-new system interpreter.

```bash
cd training
uv sync --extra dev
uv run pytest
```

No dependency downloads pretrained weights. The lockfile pins the full Python environment.

## Manifest contract

Input is JSON Lines: one object per clip, with paths resolved relative to the manifest.

```json
{"clip_id":"kitchen-20260814-001","path":"clips/001.mp4","label":"SNIFFING","split":"train","session_id":"kitchen-20260814-am","day":"2026-08-14","camera_id":"orin-usb-v1","staged_safe":false,"metadata":{"cat_id":"cat-a","plant_zone":"pot-1"}}
```

Required fields:

- `clip_id`: globally unique stable ID.
- `path`: video, `.npy`, or `.npz` clip. Arrays must be `uint8` RGB `[T,H,W,3]`; NPZ files use the
  key `frames`. Videos are decoded through OpenCV and converted to RGB.
- `label`: exactly one of the six uppercase labels above.
- `split`: `train`, `val`, or `test`.
- `session_id`: a collection-session identifier.
- `day`: an ISO date (`YYYY-MM-DD`) for the recording day.

Optional fields are `camera_id`, `staged_safe` (defaults to false), and an arbitrary JSON `metadata`
object.

Loading is intentionally strict:

- A `session_id` may occur in only one split.
- A recording `day` may occur in only one split, even when it contains several sessions.
- Clip IDs and resolved paths must be unique.
- Every referenced file must exist and every enum/date/type must be valid.

The whole-day rule is conservative. It prevents adjacent recordings with the same lighting, room,
cat, and setup from inflating held-out results. Assign splits before extracting clips; never randomly
split frames or clips from one recording session.

## Training

```bash
uv run fw-behavior-train \
  --manifest /data/foliage-warden/manifest.jsonl \
  --output-dir runs/baseline-001 \
  --epochs 30 \
  --seed 20260814
```

Training uses deterministic per-clip temporal sampling, seeded data-loader shuffling, seeded model
initialization, deterministic Torch algorithms where available, gradient clipping, and
inverse-frequency class weights by default. Evaluation samples are fixed and evenly spaced. Set
`--num-workers 0` when byte-for-byte repeatability is more important than throughput. Training fails
early if its split does not represent all six labels; silently assigning zero weight to a missing
class would produce a misleading model.

Every run writes:

- `metadata.json`: label schema, architecture/config, dependency versions, class weights, seed,
  manifest hash and summary, dedicated label/model/training-config hashes, and a content-derived
  artifact ID.
- `history.jsonl`: per-epoch loss and validation metrics.
- `best.pt` and `last.pt`: weights plus the complete metadata identity and metrics.
- `result.json`: final run pointers and best epoch.

The output directory must be new or empty; training refuses to mix or replace an existing run.

`best.pt` is selected by macro F1 over labels represented in validation. Clip metrics are useful for
model development, but product acceptance must ultimately use incident/event metrics from the replay
system, including false would-bursts per monitored hour.

## Evaluation

```bash
uv run fw-behavior-evaluate \
  --manifest /data/foliage-warden/manifest.jsonl \
  --checkpoint runs/baseline-001/best.pt \
  --split test \
  --output runs/baseline-001/test-report.json
```

The report includes six-way confusion, per-class metrics, a binary harmful-behavior view
(`EATING`/`DIGGING`), UNKNOWN rate, negative log likelihood, Brier score, 10-bin calibration error,
harmful-probability operating points, checkpoint/manifest hashes, and optional per-clip
probabilities. It also says whether the evaluation manifest exactly matches the training manifest. A
different manifest is valid for a truly external test set; the recorded hashes make that choice
auditable.

## ONNX and TensorRT handoff

```bash
uv run fw-behavior-export \
  --checkpoint runs/baseline-001/best.pt \
  --output runs/baseline-001/behavior.onnx
```

The exporter runs ONNX's structural checker, compares PyTorch and ONNX Runtime logits for batch sizes
one and two, and writes `behavior.metadata.json`. The sidecar defines the RGB float32 `N,T,C,H,W`
input, normalization, fixed clip/image dimensions, output label order, artifact identity, hashes, and
the measured parity error. Build the TensorRT engine on the target Jetson (or an identical
JetPack/TensorRT environment), then compare its logits against the PyTorch model on a frozen parity
set before accepting the engine.

## Deterministic CPU smoke test

Run the complete generate -> train -> evaluate -> export -> ONNX-check path with tiny artificial
inputs:

```bash
bash scripts/smoke.sh
```

Pass a new output directory to retain the artifacts:

```bash
bash scripts/smoke.sh /tmp/foliage-warden-training-smoke
```

The synthetic clips are obvious moving color patterns. A smoke run proves only that:

- the manifest and leakage checks accept a valid dataset;
- deterministic loading, optimization, checkpointing, evaluation, and export execute together;
- the exported file is structurally valid ONNX and its batch-1/batch-2 ONNX Runtime logits match
  PyTorch within the declared tolerance.

Synthetic accuracy, precision, recall, F1, loss, or convergence **cannot** estimate real cat behavior
performance, camera robustness, household false-positive rate, event recall, or deployment safety.
Do not compare model ideas using these fixtures and do not cite their metrics as evidence.

## Real-data evaluation protocol

Real performance requires held-out local footage from the final-ish camera position. Collect normal
household hard negatives (passing, sitting, sniffing, grooming, humans, leaf movement, lighting
changes) in addition to positive behaviors. Keep complete days and sessions together and reserve the
test days before model iteration.

Never stage chewing around a toxic plant. If positive examples must be staged, use veterinarian-
confirmed cat-safe grass and a dedicated safe digging setup, supervise the session, and record
`staged_safe: true`. Keep naturally occurring and staged-safe results separable in reports. Before any
actuator exists, run the model through long observe-only and replay evaluations and measure downstream
event precision/recall and false would-bursts per monitored hour—not just clip accuracy.
