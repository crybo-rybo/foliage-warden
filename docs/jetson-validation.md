# Jetson software-only validation

The Jetson is a compute target during observe-only development. Nothing in this
workflow arms or communicates with a physical actuator.

## Inventory

Copy `tools/jetson_probe.py` to the target or stream it over SSH:

```sh
ssh rybo@rybo-desktop.local python3 - < tools/jetson_probe.py
```

The JSON output records the BSP, CUDA/TensorRT tools, camera devices, GStreamer
plugins, and available Python inference modules. Save it as a CI artifact when
comparing target images; do not commit hostnames or other deployment-specific
details without review.

## Target build

Use a temporary checkout or source copy and build the pure C++ policy library:

```sh
cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --parallel
ctest --test-dir build/cpp --output-on-failure
```

TensorRT engines are target-version-specific. Keep ONNX as the portable model
artifact and regenerate `.engine`/`.plan` files only after the deployed JetPack
version is pinned.

The first reproducible target-side engine build and synthetic inference results
are recorded in [`jetson-benchmark.md`](jetson-benchmark.md).

## Boundary

Permitted here: compilation, unit tests, prerecorded replay, mock action logs,
camera negotiation without recording, and inference benchmarks. A live armed
run, motor movement, burst output, or unattended household deployment requires
the later supervised hardware-validation plan.
