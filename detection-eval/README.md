# Public cat/person detector baseline

This package measures the repository-pinned OpenCV Zoo YOLOX-S detector on a
small, reproducible subset of the official COCO 2017 validation split. It
selects disjoint cat-positive, person-positive, hard-negative, and background
strata; downloads only those images; and reports per-class precision, recall,
and AP50 with miss and false-positive examples.

> **Scope boundary:** COCO cat/person detection performance does not establish
> accuracy on the installed camera, in the garden domain, at night, under
> occlusion, or for `EATING`/`DIGGING` behavior. It must never be used by itself
> to enable a physical action.

The harness has no behavior classifier, policy transition, or actuator path.
The public set is useful for catching broken preprocessing, class mappings,
gross detector regressions, and confidence/recall trade-offs before collecting
held-out installed-camera sessions.

## One-command bounded run

From the repository root:

```sh
uv sync --project detection-eval --extra desktop --group dev
uv run --project perception --extra desktop python tools/fetch_model.py yolox_s_opencv_zoo
uv run --project detection-eval --extra desktop foliage-warden-detection-eval run \
  --max-images 100 \
  --seed 20260814 \
  --report artifacts/detection-eval/coco-yolox-report.json \
  --predictions artifacts/detection-eval/coco-yolox-predictions.json
```

The first run downloads and verifies the official 241 MiB train/val annotation
archive, extracts `instances_val2017.json`, deterministically selects 100 image
IDs, and downloads only those images. Dataset images, archives, model binaries,
predictions, and reports are generated below the root `artifacts/`/`models/`
paths and are not committed.

The default selection weight is 4:3:2:1 across:

- `cat_positive`: at least one COCO cat annotation (may also contain people)
- `person_positive`: person annotation and no cat annotation
- `hard_negative`: no cat/person, but an annotated animal confuser, bench, or
  potted plant
- `background_negative`: no cat/person and no configured hard-negative class

Within each stratum, IDs are ranked by SHA-256 of the algorithm version, seed,
stratum, and image ID. The manifest records the algorithm, seed, requested and
actual counts, image IDs, class IDs, and content hashes. A deficient stratum is
filled deterministically from the others rather than silently shortening the
run.

`--max-images` bounds image downloads and inference, but COCO distributes the
official instance annotations as one archive, so the annotation download is
not reduced by that flag. Start with `--max-images 20` for a quick model smoke;
use a larger sample before comparing thresholds. Treat any subset used to pick
a threshold as a development set: its resulting precision/recall is optimistic,
not a held-out estimate. Lock a different seed and manifest before reporting an
operating point, and reserve session-isolated installed-camera data for the
deployment-domain decision.

## Separate preparation and evaluation

Prepare once, then rerun inference without network access:

```sh
uv run --project detection-eval foliage-warden-detection-eval prepare \
  --max-images 100 --seed 20260814

uv run --project detection-eval --extra desktop foliage-warden-detection-eval evaluate \
  --manifest artifacts/detection-eval/coco2017/manifests/coco-val-max100-seed20260814.json \
  --dataset-root artifacts/detection-eval/coco2017
```

Add `--offline` to `prepare` or `run` to forbid network access. Offline mode
requires a verified annotation archive plus every selected image and its hash
sidecar; missing or changed bytes fail closed with a cache error.

To consume a complete existing COCO layout instead of downloading data:

```text
/datasets/coco/
├── annotations/instances_val2017.json
└── val2017/000000000139.jpg ...
```

```sh
uv run --project detection-eval --extra desktop foliage-warden-detection-eval run \
  --coco-root /datasets/coco \
  --cache-dir artifacts/detection-eval/coco2017 \
  --max-images 100 --seed 20260814
```

The existing annotation JSON must match the pinned COCO 2017 file exactly.
Selected local images are hashed into the manifest and verified again before a
later `evaluate` command.

## Metric definition

For each of `cat` and `person`, detections are sorted by descending confidence.
Each prediction matches the highest-IoU unmatched ground-truth box in the same
image and class when IoU is at least 0.50; a ground truth can be used once.
Otherwise the prediction is a false positive. `iscrowd` boxes do not enter the
recall denominator and absorb otherwise-unmatched detections that overlap them
at the threshold. AP50 is 101-point interpolated precision at IoU 0.50.

This focused value is deliberately labeled AP50, not “COCO AP.” It does not
reproduce the full `pycocotools` suite of IoU thresholds, area ranges, crowd
intersection rules, and `maxDet` settings. The JSON report states this
definition and records detector confidence floors, NMS IoU, backend, pinned
model hash/revision, subset manifest hash, counts, misses, and highest-scoring
false-positive examples. It also records the SHA-256 of the exact canonical
prediction JSON, cryptographically binding the metrics to their detector output.
It contains no timestamp, latency, or absolute source paths, so fixed inputs and
the locked Python environment produce byte-identical reports.

The detector confidence floors default to `0.001` so AP has a useful score
curve. Precision/recall default to a separate 0.50 operating threshold for
both classes. Use explicit values when testing another operating point; an
operating threshold cannot be below its detector floor:

```sh
uv run --project detection-eval --extra desktop foliage-warden-detection-eval run \
  --max-images 250 --seed 20260814 \
  --cat-operating-threshold 0.35 --person-operating-threshold 0.50
```

## Test and lint

The test suite uses only tiny synthetic annotations/predictions and fake
download responses; it never fetches COCO data:

```sh
uv run --project detection-eval --group dev pytest detection-eval/tests
uv run --project detection-eval --group dev ruff check detection-eval
```

See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for exact URLs, pinned digests,
cache verification, image-license handling, and intended use.
