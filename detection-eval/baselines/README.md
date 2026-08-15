# Checked public-data smoke baseline

The adjacent manifest and report are a reproducible, bounded integration run
of the pinned YOLOX-S model on the `balanced-sha256-v1` COCO 2017 validation
subset with seed `20260814`:

- 100 images: 40 cat-positive, 30 person-positive, 20 hard-negative, and 10
  background-negative
- OpenCV CPU backend, NMS IoU 0.50
- cat/person confidence floors 0.001 (chosen for an AP curve) and separate
  precision/recall operating thresholds of 0.50
- one-to-one AP50 using the exact definition embedded in the report

The report is an integration fixture and initial public-set reference, not a
deployment acceptance threshold. The selected sample is deliberately
class-balanced and therefore does not estimate naturally weighted COCO
performance. It is a public development/regression set: choosing a threshold on
it makes that threshold's metrics tuning results, not held-out estimates. It
also cannot estimate installed-camera or behavior accuracy.

Reproduce it from the repository root after fetching the pinned model:

```sh
uv run --python 3.10 --project detection-eval --extra desktop \
  foliage-warden-detection-eval evaluate \
  --manifest detection-eval/baselines/coco-val-100-seed20260814.manifest.json \
  --dataset-root artifacts/detection-eval/coco2017 \
  --expected-report detection-eval/baselines/yolox-s-opencv-coco-val-100.report.json \
  --report /tmp/yolox-s-opencv-coco-val-100.report.json \
  --predictions /tmp/yolox-s-opencv-coco-val-100.predictions.json
```

If the documented default cache was not prepared yet, create the same subset
first:

```sh
uv run --project detection-eval foliage-warden-detection-eval prepare \
  --cache-dir artifacts/detection-eval/coco2017 \
  --max-images 100 --seed 20260814 \
  --manifest detection-eval/baselines/coco-val-100-seed20260814.manifest.json
```

The committed manifest contains only COCO metadata, selected IDs, license IDs,
URLs, and content hashes. It contains no dataset images. Predictions stay
generated because they are readily reproduced from the manifest and pinned
model. The report records their canonical SHA-256, and `--expected-report`
requires the complete regenerated report (including that digest and every
metric) to match the checked reference byte for byte.
