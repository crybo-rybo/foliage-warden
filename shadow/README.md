# Observe-only shadow integration

`shadow/` connects the current cat/person perception stream to six-way temporal behavior
predictions and the existing deterministic policy simulator. It is an integration and evaluation
surface, not a deployment runtime:

- every input and output is file/stdio JSON or an in-memory Python value;
- behavior and perception records must both say `OBSERVE_ONLY` and `would_action: false`;
- the only policy transport is the simulator's in-memory `MOCK` actuator;
- runtime configuration is rejected unless it uses a virtual clock, scripted camera, mock backend,
  and `allow_physical_effects: false`;
- evaluator safety checks must report zero violations or the run fails.

There is deliberately no serial, GPIO, network, motor, or burst-device adapter in this package.
`BURST` in its output means a replayed **would-burst decision** only. The simulator audit records
also state `physical_effect_possible: false`.

## Behavior prediction JSONL

One record is required per cat track/frame when a prediction arrives. Identity matching is exact on
`observation_id`, `frame_id`, and `track_id`; the capture timestamp must also agree.

```json
{"captured_at_ms":100,"config":{"id":"behavior-config-v1","sha256":"2222222222222222222222222222222222222222222222222222222222222222"},"frame_id":"camera-1:frame:00000000","mode":"OBSERVE_ONLY","model":{"id":"behavior-model-v1","sha256":"1111111111111111111111111111111111111111111111111111111111111111"},"observation_id":"camera-1:observation:00000000","predicted_at_ms":110,"predicted_label":"EATING","probabilities":{"DIGGING":0.01,"EATING":0.95,"OTHER":0.01,"PASSING":0.01,"SNIFFING":0.01,"UNKNOWN":0.01},"record_type":"behavior_prediction","schema_version":1,"sequence":0,"track_id":"cat-a","would_action":false}
```

The parser rejects unknown/missing fields, non-finite or out-of-range values, non-integer schema
versions, an incorrect six-way sum, a label that differs from deterministic argmax, invalid hashes,
duplicate identities, mixed model/config identities, and non-deterministic stream order. The
behavior stream must be ordered by `(predicted_at_ms, sequence)`; perception must be ordered by
`(captured_at_ms, sequence)`.

Six-way probabilities map to the four-way policy contract as follows:

| Behavior output | Policy label | Policy score contribution |
| --- | --- | --- |
| `PASSING` | `CLEAR` | `CLEAR` |
| `SNIFFING` | `CLEAR` | `CLEAR` |
| `EATING` | `EATING` | `EATING` |
| `DIGGING` | `DIGGING` | `DIGGING` |
| `OTHER` | `UNKNOWN` | `UNKNOWN` |
| `UNKNOWN` | `UNKNOWN` | `UNKNOWN` |

`OTHER` is intentionally not treated as proof of a clear scene. The canonical `CLEAR` score is
`P(PASSING) + P(SNIFFING)` and canonical `UNKNOWN` is `P(OTHER) + P(UNKNOWN)`.

Missing, timed-out, capture-mismatched, and over-latency predictions replace behavior evidence with
`UNKNOWN=1.0`. A well-formed prediction whose three-part identity matches no cat is retained as an
`UNMATCHED_PREDICTION` diagnostic and is never joined approximately. Crossed prediction completions
are buffered so observations always reach policy in capture order. Person tracks, multiple cats,
frame capture time, tracking quality/age, ambiguity, no-fire intersection, and region evidence pass
through only after the duplicated per-zone and policy views agree exactly. Zone IDs and types must
also match the runtime scene. A simulation-safe preset is attached only when the runtime config has
exactly one preset for the detected approach zone.

## Offline ONNX inference bridge

The optional `foliage-warden-shadow-infer` command turns **pre-extracted** RGB NumPy clips into the
strict behavior JSONL above. It is deliberately a file-only adapter: it has no camera, video
decoder, recorder, network, policy, actuator, or wall-clock API. Install the inference extra on a
Python version supported by ONNX Runtime (the integration suite currently runs on Python 3.10):

```bash
uv sync --project shadow --extra inference
```

The command requires three independent locks for every run:

- a perception JSONL whose exact cat `observation_id` / `frame_id` / `track_id` / capture time is
  copied into the request manifest;
- the training export's `.metadata.json`, checked against the graph's input/output tensors and all
  embedded `foliage_warden.*` metadata; and
- an externally supplied `--expected-onnx-sha256`, which must equal both the actual ONNX bytes and
  the digest in the sidecar. The training `artifact_id` is not used as a substitute for this byte
  digest.

There must be exactly one request for every CAT track in the supplied perception stream, and no
request for a person or unknown track. The request schema is
`schemas/behavior-inference-request.schema.json`; one record looks like this:

```json
{"captured_at_ms":100,"clip":{"format":"NUMPY_RGB_UINT8_THWC","frame_timestamps_ms":[0,50,100],"path":"clips/cat-a.npy","sha256":"3333333333333333333333333333333333333333333333333333333333333333","window_end_captured_at_ms":100,"window_start_captured_at_ms":0},"frame_id":"camera-1:frame:00000000","observation_id":"camera-1:observation:00000000","predicted_at_ms":110,"record_type":"behavior_inference_request","schema_version":1,"sequence":0,"track_id":"cat-a"}
```

Clip paths are normalized relative paths beneath the manifest directory. Only `.npy` and `.npz`
are accepted; an NPZ must contain exactly one `frames` array. The decoded value must be exact
`uint8` RGB `[T,H,W,3]`, its byte SHA and size are checked, and its frame count must equal the
strictly increasing timestamp list. Every declared timestamp must be at or before the target
capture, the final timestamp and window end must equal that capture, and `--window-ms` must match
the declared start. These checks prevent a manifest from *declaring* future-frame input. They do
not authenticate that the pixels really came from those times or from the named track.

```bash
uv run --project shadow --extra inference foliage-warden-shadow-infer \
  observations.jsonl clip-requests.jsonl \
  --model behavior.onnx \
  --metadata behavior.metadata.json \
  --expected-onnx-sha256 <sha256-of-behavior.onnx> \
  --window-ms 1000 \
  --logical-latency-ms 25 \
  --output behavior-predictions.jsonl
```

`predicted_at_ms` is a replay timestamp, not measured inference latency. It must equal
`captured_at_ms + --logical-latency-ms`; no wall clock is read. The adapter reproduces training's
evaluation preprocessing (endpoint-inclusive `numpy.rint(linspace)` sampling, OpenCV `INTER_AREA`
resize, RGB scaling, and export-declared normalization), runs the verified graph twice with a
single-threaded CPU provider, and requires bitwise-identical finite logits. Stable float64 softmax
and fixed label-order argmax produce the output. The canonical config SHA covers logical latency,
window, sampling, preprocessing, normalization, label/export identities, and CPU runtime settings.
It also binds the operating-system name/release, machine architecture, libc, Python implementation,
compiler/version, NumPy/OpenCV/ONNX Runtime versions, and SHA-256 digests of the native build
metadata reported by those packages. This distinguishes the runtime environments we can inspect; it
is not a signed wheel, compiler, or supply-chain attestation. An output file is written atomically
with mode `0600`; stdout remains empty if any request fails.

This is an offline **clip-to-prediction-to-shadow** path, not perception-to-crop end to end. The
repository still needs a causal assembler that extracts and hashes per-track RGB clips from
recorder/perception artifacts, carries authenticated source-frame provenance, and proves that its
crop/window mapping matches the live detector. Until that exists, clip origin, crop correctness,
and timestamp truth remain unvalidated inputs and must not be presented as deployment evidence.

## Run a replay

```bash
uv run --project shadow foliage-warden-shadow \
  observations.jsonl behavior-predictions.jsonl \
  --config config/simulation-safe.example.json \
  --scenario-out /tmp/shadow.scenario.json \
  --evaluator-jsonl /tmp/shadow.evaluator.jsonl \
  --event-jsonl /tmp/shadow.events.jsonl \
  --audit-jsonl /tmp/shadow.audit.jsonl \
  --fusion-jsonl /tmp/shadow.fusion.jsonl \
  --summary /tmp/shadow.summary.json \
  --pretty
```

When `--config` is omitted, the CLI uses the simulator package's bundled,
simulation-safe example configuration. Editable checkouts use the byte-identical
repository copy; installed wheels use the packaged copy.

The default behavior timeout is 50 ms and the maximum accepted prediction latency is 250 ms. A
prediction beyond either bound remains visible in diagnostics but contributes UNKNOWN behavior. A
timeout releases the UNKNOWN frame at its deadline; a result that arrives before the timeout but
exceeds the independent latency bound is released as UNKNOWN when it arrives. If completions cross,
later captures wait behind earlier frames rather than allowing old evidence to replay after new.
Override the bounds with `--prediction-timeout-ms` and `--max-prediction-latency-ms` for controlled
experiments.

The CLI requires the in-memory runtime configuration to match `--config` exactly, snapshots that
validated content for both simulator passes, and refuses a split-brain replay. It then assembles a
schema-v1 scenario and probes it once to record replay-derived exact output expectations. The
finalized scenario executes twice through `foliage_warden_sim`; the runner compares deterministic
signatures, validates every evaluator record with `foliage_warden_eval`, and runs the evaluator's
non-negotiable safety checks. Replay-derived expectations prove reproducibility; they are not an
acceptance oracle for model quality.

## Test suite

```bash
cd shadow
just check
```

Deterministic synthetic fixtures exercise eating, digging, sniffing, missing predictions, timeout
and latency bounds, crossed arrivals, mismatched identities, configuration split-brain, contradictory
zone/no-fire evidence, person presence, multiple cats, stale frames, no-fire overlap, weak tracks,
and the one-burst-per-continuous-incident latch. Every case asserts zero evaluator safety violations,
zero physical bursts, zero retries, and byte-stable CLI output.

The optional inference integration uses a real generated ONNX graph and ONNX Runtime to check
numerical preprocessing/softmax results, byte-stable output, independent model locks, causal
manifest checks, and fail-closed clip/model/metadata errors. It also sends the actual inferred
prediction through fusion, the deterministic simulator, and evaluator safety checks:

```bash
cd shadow
just test-inference
```

The normal test suite skips that module when optional inference dependencies are unavailable, so
the base mock-only shadow package remains testable on newer Python versions before ONNX Runtime
publishes matching wheels.

All probabilities and scene evidence in those fixtures are deliberately injected. They test contract
plumbing, fail-closed gates, temporal policy behavior, and serialization only. They do **not** measure
classifier accuracy, calibration, real cat behavior, household false positives, camera robustness,
or deployment safety.
