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

All probabilities and scene evidence in those fixtures are deliberately injected. They test contract
plumbing, fail-closed gates, temporal policy behavior, and serialization only. They do **not** measure
classifier accuracy, calibration, real cat behavior, household false positives, camera robustness,
or deployment safety.
