# Deterministic policy core

This directory contains the hardware-independent C++20 safety policy and a deterministic mock
actuator. It intentionally contains no serial, GPIO, motor, or burst-device adapter.

The policy starts in `DISARMED`, requires an explicit acknowledged `ARM`, and follows the state
sequence described in the project brief. Each call to `PolicyEngine::step` receives caller-owned
monotonic milliseconds plus optional timestamped perception and safety snapshots. There is no
wall-clock access, sleeping, background thread, random source, or implicit retry.

The policy requires both elapsed persistence and multiple distinct frame IDs for tracking and
harmful-behavior confirmation. A safe aim command must be acknowledged and followed by a fresh,
safe frame before a burst can be attempted. The burst latch is set before the actuator is called;
therefore a timeout is treated as an unknown physical outcome and can never cause a retry.
Occupancy must clear for multiple frames over a configured interval before a new event is created.

Every step returns a `DecisionOutput` with typed interlocks, transitions, semantic commands,
actuator results, the continuous-event ID, and the burst-attempt latch. Command IDs are monotonic
and the `MockActuator` caches results by ID so repeated delivery is deterministic and deduplicated.
The engine is deliberately non-copyable: one serialized caller owns one engine/actuator pair. A
deployment must persist or allocate `initial_command_id` so IDs remain unique across process
restarts, and a real adapter must independently clamp the physical burst duration.

Build and test independently from the repository root:

```sh
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Debug
cmake --build build/cpp -j
ctest --test-dir build/cpp --output-on-failure
```

The public integration surface is under `include/foliage/`:

- `policy_types.hpp`: timestamps, perception/safety snapshots, configuration, and audit enums
- `actuator.hpp`: semantic commands/results and the transport-neutral `IActuator` interface
- `mock_actuator.hpp`: scripted, stateful, deduplicating actuator for simulation and fault injection
- `policy_engine.hpp`: deterministic state machine
