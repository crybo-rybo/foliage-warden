# Foliage Warden deterministic simulator

This package is the executable Python 3.10+ reference implementation of the
version 1 scenario contracts in [`../scenarios`](../scenarios). It expands
scripted observations, advances a virtual monotonic clock, runs policy
persistence and safety interlocks, and sends transport-independent commands to
an in-memory scripted mock.

It contains no serial, GPIO, socket, or network actuator implementation. It
cannot create a physical effect. Passing these scenarios tests policy and mock
transport behavior only; it says nothing about detector or behavior-model
accuracy in real video.

## Run it

From this directory:

```sh
uv run --extra dev foliage-warden-sim --all
uv run foliage-warden-sim ../scenarios/03-eating-persistence.json
uv run --extra dev pytest
```

The CLI writes one deterministic aggregate JSON summary to stdout and exits
nonzero when a contract, expectation, invariant, or trace comparison fails.
Every scenario is executed twice by default and its deterministic signature is
compared before it can pass.

Useful artifacts:

```sh
uv run foliage-warden-sim --all \
  --event-jsonl artifacts/events.jsonl \
  --audit-jsonl artifacts/actions.jsonl \
  --evaluator-jsonl artifacts/replay.jsonl \
  --trace artifacts/reference-trace.json

uv run foliage-warden-sim --all \
  --compare-trace artifacts/reference-trace.json
```

- `--event-jsonl` conforms to `schemas/event-record.schema.json`.
- `--audit-jsonl` conforms to `schemas/action-audit.schema.json`.
  Every fail-closed observation decision is represented by a `SUPPRESS` record
  with its evidence, interlock snapshot, and reason codes.
  Policy commands are flushed first with `outcome: PENDING`, before the mock is
  invoked, and receive a second append-only audit record when ACK, denial,
  transport error, or timeout is known. This preserves the authorization record
  across an injected process restart.
- `--evaluator-jsonl` contains `session`, `prediction_event`, and accepted
  unique `action` records that the existing `foliage-warden-eval` parser can
  consume directly. Duplicate transport attempts remain in the canonical audit
  stream but are omitted from evaluator actions because they were never
  accepted for dispatch; including them would intentionally trigger the
  evaluator's duplicate-command safety violation.
- `--trace` includes the canonical event, action-audit, and evaluator records in
  stable key order. `--compare-trace` is an exact byte comparison suitable for
  CI and cross-implementation adapters.

Generated `OBSERVATION_SERIES` IDs use `id_prefix-NNNNNN`, with the same value
for `event_id`, `observation_id`, and `frame_id`. Internal work is ordered by
due time and insertion order. At a time shared by external input and internal
work, all external inputs run first, as required by the architecture contract.

## Scope and C++ conformance

The Python simulator implements the repository's JSON scenario interface. The
current C++ core exposes a different lower-level input and timing contract, so
the Python runner does not call it and a canonical pass is not presented as a
C++ conformance result. `tests/test_cpp_conformance.py` builds a small
simulator-owned probe against `../cpp` when a C++ toolchain is available. It
checks the overlap that is expressible today—startup disarmed, manual arm,
fail-closed person handling, and one-shot burst behavior—separately from the 14
JSON fixture passes. Full trace equivalence requires a C++ adapter for the
version 1 observation/config/scenario schemas. In particular, that adapter must:

- translate string track IDs, behavior score vectors, region evidence, and
  camera/actuator status events into the C++ core's compact numeric inputs;
- provide the same delayed ACK, DROP, timeout, restart, and external-at-deadline
  ordering around the C++ core's currently synchronous actuator interface;
- normalize the C++ core's actuator-level ARM action (the scenario contract
  treats manual ARM as a control input, not an action count); and
- emit the repository event/audit schemas and compare the normalized policy
  trace byte-for-byte with `--trace` output.
