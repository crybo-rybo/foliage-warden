# Jetson YOLOX-S benchmark

This is a software-only capacity measurement of the pinned COCO detector. It
uses generated input tensors; it does not measure camera decode, preprocessing,
postprocessing, tracking, detection quality, or behavior recognition.

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
