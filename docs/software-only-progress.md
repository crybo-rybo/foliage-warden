# Software-only progress and handoff

Status as of 15 August 2026: the repository implements the principal components
and tests a contract-level replay from scene records through a mock `would-act`
decision. It does not implement a physical motor, air, water, or other burst
adapter, and nothing here has been armed against an animal.

## What is working

| Layer | Deliverable | Current boundary |
|---|---|---|
| Scene and runtime contracts | Draft 2020-12 schemas, safe startup-disarmed configuration, calibrated-zone model, canonical fixtures | Normalized coordinates and mock targets only |
| Policy | Deterministic C++20 finite-state machine, typed audit records, fail-closed interlocks, ESTOP, cooldown, command deduplication | `MockActuator` only |
| Simulator | Fourteen virtual-time safety scenarios with exact traces and expected outcomes | Full Python trace adapter; a smaller linked C++ conformance subset |
| Evaluator | Event matching, per-label precision/recall/F1, latency and data-quality metrics, exact false-would-action rate bound, session-safe splits, invariant checking | Meaningful performance numbers require real labeled sessions |
| Perception | Pinned OpenCV Zoo YOLOX-S preprocessing, decoding, cat/person filtering, deterministic NMS/tracking, polygon evidence, camera/video/image inputs | Always emits `UNKNOWN`, `OBSERVE_ONLY`, and `would_action=false` |
| Behavior training | Strict manifests, session/day leakage checks, causal CNN+GRU, train/evaluate/export CLIs, artifact hashes, ONNX Runtime parity | Synthetic data proves the pipeline, not behavior accuracy |
| Offline behavior inference | Strict causal clip manifests, external ONNX digest lock, sidecar/graph/embedded-metadata checks, exact preprocessing, deterministic CPU inference, and private behavior JSONL | Accepts pre-extracted full RGB clips; pixel origin and any track crop are not authenticated |
| Shadow runtime | Strict joins and hashes, prediction deadlines, capture-order handling, zone reconciliation, conservative behavior mapping, simulator plus evaluator replay | Offline/mock-only; missing or inconsistent evidence becomes `UNKNOWN` |
| Incident data | Silent, bounded, private, atomic clip recorder and local browser review/labeling workbench | Explicit local manifests; no upload or network egress |
| Detector baseline | Locked 100-image COCO 2017 development subset with cryptographically bound canonical predictions and report | Public regression set, not installed-camera acceptance data |

The canonical simulator completes all fourteen scenarios, producing the stable
trace SHA-256
`698a610a5a7e6449120c5131b20cc1e8ad737f86ccd86f686764c952b8be4e00`.
Those traces pass the evaluator's safety checks. The review workbench was also
exercised interactively in a browser, including validation, revision history,
and evaluator-compatible export.

## Measured model and target evidence

The pinned detector was evaluated on a deterministic, balanced public
development subset of 100 COCO validation images. AP50 uses every prediction
retained at the `0.001` detector confidence floor; precision and recall use an
operating score threshold of `0.5`. Cat AP50 / precision / recall were
`0.882781 / 0.969697 / 0.744186`; person results were
`0.689761 / 0.798077 / 0.535484`. This set is deliberately small and
class-balanced. It is suitable for reproducible regression and pipeline
debugging, not prevalence-weighted deployment claims or threshold selection
for the installed camera. See the
[`detection-eval` baseline report](../detection-eval/baselines/yolox-s-opencv-coco-val-100.report.json).

On the available Jetson Orin Nano Super, the YOLOX-S FP16 TensorRT engine
reached `170.19` tensor inferences/s with `6.267 ms` mean end-to-end tensor
latency. The default 52,934-parameter temporal topology reached `2,023.25`
clips/s in FP32, preserved the PyTorch argmax, and had maximum absolute logit
error `5.483e-5`. These are synthetic-tensor capacity and parity measurements,
not model-quality measurements.

The attached `1280x720@30` camera also completed ten observe-only frames
without recording. The current Python/OpenCV CPU path achieved only `2.345`
frames/s; all outputs remained `UNKNOWN`, `OBSERVE_ONLY`, and
`would_action=false`. The raw TensorRT detector has ample headroom, so the next
target-runtime engineering task is connecting that engine to decode,
pre/postprocessing, and tracking. Full details and reproduction commands are in
[`jetson-benchmark.md`](jetson-benchmark.md).

## Reproduce the software baseline

From the repository root:

```sh
just check
just verify
```

`just check` is the native build plus repository test, schema-validation, and
lint gate; its locked dependency synchronization may download packages on a
fresh machine. `just verify` adds the pinned model download/check,
generated-scene evaluation, all canonical simulations, the offline ONNX
inference bridge, and the deterministic temporal-model train/evaluate/ONNX
smoke. CI runs every non-training Python package on Python 3.10 and 3.14,
training on Python 3.10, native policy builds under GCC and sanitized Clang, and
the browser helper and calibration JavaScript tests.

## What remains before any real actuation

1. Collect consented local video from the installed view in observe-only mode,
   label whole sessions with the review workbench, and preserve household/day
   groups in train, validation, and test splits.
2. Lock a separate installed-camera detector test set, measure misses and
   person-interlock performance, and select thresholds without reusing the
   public development subset as the final test set.
3. Train and calibrate the temporal classifier on real `PASSING`, `SNIFFING`,
   `EATING`, `DIGGING`, and `OTHER/UNKNOWN` examples. Requalify ONNX/TensorRT
   parity and FP16 only after real operating thresholds exist.
4. Build the missing causal track-clip assembler that binds detector frames,
   timestamps, bounding boxes, crop policy, and pixels to the offline inference
   request. Then integrate TensorRT into the live observe-only pipeline, replace
   the simple IoU tracker if local data warrants it, and run long shadow
   sessions. Measure event recall, false would-actions per monitored hour with
   its confidence bound, person suppression, track loss, latency, and thermal
   stability.
5. Calibrate the real camera's approach, foliage, soil, and no-fire polygons.
   Review every safe aim preset against the physical room; none of the example
   normalized points is a hardware authorization.
6. Only after the software gates pass, design a separately reviewed hardware
   adapter and microcontroller protocol with a physical kill switch, watchdog,
   command deduplication, travel and energy limits, and supervised aim-only
   commissioning. Controlled burst tests and animal-safety review remain
   physical-world work.

Until those gates are complete, the correct operating mode is disarmed
observe-only collection and mock replay.
