# Jetson inference benchmarks

These are software-only capacity measurements of the pinned COCO detector and
the default temporal-model topology. They use generated input tensors; they do
not measure camera decode, preprocessing, postprocessing, tracking, detection
quality, or behavior-recognition quality.

## Target and artifact

Measured 14 August 2026 on the available Jetson Orin Nano Super in
`MAXN_SUPER` mode:

- Jetson Linux 36.4.7 (Ubuntu 22.04, kernel 5.15)
- CUDA 12.6 and TensorRT 10.3.0
- 7,619 MiB GPU/shared memory reported by TensorRT
- YOLOX-S input `1x3x640x640`, output `1x8400x85`
- pinned ONNX SHA-256
  `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`

The target generated a 20 MiB FP16 engine in 496.6 seconds. TensorRT reported
174 MiB peak GPU allocator use and 1,805 MiB peak CPU memory while building it.
The engine is deliberately not checked in because TensorRT plans are tied to
the target software and GPU.

## Inference result

`trtexec` ran one stream with CUDA graphs, a 1-second warmup, normal host/device
transfers, and a 20-second measurement window:

| Metric | Result |
|---|---:|
| Throughput | 170.19 inferences/s |
| End-to-end tensor latency, mean | 6.267 ms |
| End-to-end tensor latency, p95 | 6.318 ms |
| End-to-end tensor latency, p99 | 6.332 ms |
| GPU compute, mean | 5.871 ms |
| GPU compute, p99 | 5.896 ms |
| H2D / D2H, mean | 0.222 / 0.174 ms |
| GPU utilization during the run | 99% |
| Board power by the end of the run | 19.584 W running average |
| Peak observed GPU temperature | 58.968 C |
| RAM while loaded | 2,673 / 7,620 MiB |

The detector engine therefore has ample synthetic-inference headroom for a
30 FPS camera stream on this power profile. That conclusion is limited to
capacity: the complete observe-only pipeline still needs camera-path timing,
pre/postprocessing timing, and local-video accuracy evaluation.

## Reproduction

Fetch the model using `tools/fetch_model.py`, copy it to a temporary target
path, verify the digest there, and build on the target:

```sh
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/foliage-warden-yolox.onnx \
  --saveEngine=/tmp/foliage-warden-yolox-fp16.engine \
  --fp16 \
  --skipInference \
  --profilingVerbosity=none
```

Then benchmark the generated engine:

```sh
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=/tmp/foliage-warden-yolox-fp16.engine \
  --warmUp=1000 \
  --duration=20 \
  --useCudaGraph \
  --exportTimes=/tmp/foliage-warden-yolox-fp16-times.json
```

Run `tegrastats --interval 1000` alongside the benchmark to capture power,
memory, utilization, and temperatures. Rebuild and remeasure after any JetPack,
TensorRT, model, clock, or power-profile change.

## Temporal behavior model

The default causal CNN+GRU topology from `training/` was also exported to ONNX
and exercised on the target. This is a deployment and numerical-parity test
only. The weights came from a one-epoch synthetic pipeline smoke run, so neither
the logits nor the winning class say anything about real behavior accuracy.

- input: `1x16x3x96x96` float32 frames; output: six logits
- parameters: 52,934
- ONNX SHA-256:
  `9c91d06cf96a238cc3fce0f730f958bda4ec263f65314ce0c354475333777209`
- fixed parity input SHA-256:
  `273aa517c5b963aadce3da2ba741f969213a78fb7ce562ba3741f2e0be1e1c78`

TensorRT accepted the exported recurrent graph and built both precision modes.
The FP16 plan was 798,516 bytes and took 70.5 seconds to build; the FP32 plan
was 759,588 bytes and took 30.7 seconds. Both builds reported 75 MiB peak GPU
allocator use. Peak CPU build memory was 1,661 MiB for FP16 and 1,554 MiB for
FP32.

One fixed tensor was evaluated by the exporting PyTorch process and each target
engine. Both TensorRT modes preserved the PyTorch argmax. Maximum absolute logit
error was `3.986e-3` for FP16 and `5.483e-5` for FP32. Because a real model's
decision thresholds have not been calibrated, FP32 is the conservative initial
choice for shadow evaluation; FP16 must be requalified against a labeled corpus
before its outputs can influence policy.

Sustained one-stream measurements used CUDA graphs, a 1-second warmup, and a
20-second measurement window:

| Metric | FP16 | FP32 |
|---|---:|---:|
| Throughput | 3,097.71 clips/s | 2,023.25 clips/s |
| End-to-end tensor latency, mean | 0.424 ms | 0.616 ms |
| End-to-end tensor latency, p95 | 0.431 ms | 0.631 ms |
| End-to-end tensor latency, p99 | 0.434 ms | 0.639 ms |
| GPU compute, mean | 0.320 ms | 0.492 ms |
| GPU compute, p99 | 0.326 ms | 0.499 ms |

These figures show that the default temporal topology is computationally small
relative to the detector. They exclude crop extraction, clip buffering, image
normalization, scheduling, camera decode, and detector work, and are not an
end-to-end frame-rate claim.

Reproduce the FP32 build and benchmark after exporting a model from
`training/`:

```sh
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/behavior.onnx \
  --saveEngine=/tmp/behavior-fp32.engine \
  --minShapes=frames:1x16x3x96x96 \
  --optShapes=frames:1x16x3x96x96 \
  --maxShapes=frames:2x16x3x96x96 \
  --skipInference \
  --profilingVerbosity=none

/usr/src/tensorrt/bin/trtexec \
  --loadEngine=/tmp/behavior-fp32.engine \
  --shapes=frames:1x16x3x96x96 \
  --warmUp=1000 \
  --duration=20 \
  --useCudaGraph \
  --profilingVerbosity=none
```
