# Foliage Warden offline incident assembler

This package turns one published recorder incident plus the original strict
perception JSONL into causal full-frame RGB NumPy clips and shadow behavior
inference requests. It is an offline evidence adapter: it has no camera, network,
policy, action, or actuator API.

The assembler fails closed before publication unless all of these agree:

- the incident is a private, current-user-owned, non-symlink directory containing
  only private regular `metadata.json` and encoded clip files;
- recorder schema, observe-only/privacy fields, resource limits, source, timeline,
  trigger samples, clip size, and encoded SHA-256 are internally valid;
- the clip is a RIFF AVI whose OpenCV codec, frame count, dimensions, and FPS
  exactly match metadata, and decoding yields exactly the declared frames;
- the original JSONL passes shadow's strict perception parser, and inclusive
  selection between the declared start/end sequences has the exact count,
  endpoints, source identity, frame dimensions, and recomputed trigger samples;
- recorder frame bindings, when present, cryptographically match the canonical
  supplied perception records and their exact decoded ordinals;
- every emitted target contains exactly one CAT with positive approach overlap,
  has a real frame exactly at `captured_at_ms - window_ms`, and its causal window
  contains only zero-CAT frames or one CAT with that target's track identity.

Gapped source sequences are valid. Ordinals come from the validated ordered
incident selection, never from `sequence - start_sequence`, and timestamps always
come from perception rather than encoded FPS.

## Use

```bash
cd assembler
uv run --locked foliage-warden-assemble \
  ../recordings/incidents/incident-0000000010000-0000000010 \
  ../perception.jsonl \
  --output-dir ../artifacts/assembled-incident \
  --window-ms 500 \
  --logical-latency-ms 10
```

The output contains decoded household pixels and detailed perception records. Keep
it beneath the repository's recursively ignored `artifacts/` root (or another
private, excluded location); do not commit or share it as ordinary test output.

The output directory must not exist. The assembler holds an
`O_DIRECTORY|O_NOFOLLOW` descriptor for its verified parent, prepares the result
through that descriptor, and publishes with a dirfd-relative atomic Linux
`renameat2(RENAME_NOREPLACE)`. Parent replacement cannot redirect publication,
prepublication validation failures expose no partial result, and an existing path
is never replaced. Directories use mode `0700`; files use `0600`.

```text
artifacts/assembled-incident/
├── behavior-inference-requests.jsonl
├── incident-perceptions.jsonl
├── selected-perceptions.jsonl
├── provenance.json
└── clips/
    ├── clip-000000.npy
    └── ...
```

`selected-perceptions.jsonl` contains the unchanged strict JSON objects for the
emitted target observations. Pass this derived stream—not the entire source
JSONL—to shadow inference: its binder requires exactly one request for every CAT
in the supplied perception stream.

```bash
uv run --locked --project ../shadow --extra inference foliage-warden-shadow-infer \
  ../artifacts/assembled-incident/selected-perceptions.jsonl \
  ../artifacts/assembled-incident/behavior-inference-requests.jsonl \
  --model /pinned/export/behavior.onnx \
  --metadata /pinned/export/behavior.metadata.json \
  --expected-onnx-sha256 <lowercase-sha256> \
  --logical-latency-ms 10 \
  --window-ms 500 \
  --output ../artifacts/behavior-predictions.jsonl
```

`incident-perceptions.jsonl` is a separate canonical copy of every reconciled,
provenance-bound observation in the published incident. Pass that full stream,
plus the sparse predictions produced above, to policy replay:

```bash
uv run --locked --project ../shadow foliage-warden-shadow \
  ../artifacts/assembled-incident/incident-perceptions.jsonl \
  ../artifacts/behavior-predictions.jsonl \
  --config ../config/simulation-safe.example.json
```

Do not replay `selected-perceptions.jsonl` as the policy input. Full replay keeps
clear/no-CAT, PERSON, and ambiguous multi-CAT observations in temporal order;
CATs omitted from inference have no prediction and therefore fail closed to
shadow's `MISSING` diagnostic and `UNKNOWN` behavior.

The logical latency is replay metadata, not measured runtime. It is capped at
50 ms so emitted predictions remain within shadow's default fusion timeout.

## Target and pixel contract

Each `.npy` is deterministic `uint8 [T,H,W,3]` RGB decoded from the recorder's
BGR OpenCV stream. The causal array starts at the exact configured window start,
ends at its target, and contains all intervening incident frames. The assembler
does not crop, resize, normalize, or model-sample. Shadow inference owns the fixed
temporal sampling and full-frame preprocessing used by the current training
contract.

Multi-CAT targets and any causal window containing a multi-CAT frame are suppressed
as ambiguous because the full-frame model is not track-conditioned. A window may
contain zero-CAT history, but any sole CAT in the window must have the target's
track identity. Single-CAT targets without approach evidence are also recorded as
suppressed. A target lacking an exact causal start frame is skipped without
fabricating a timestamp. All suppressions appear in `provenance.json`, and the
assembler fails without output if no eligible target remains.

Encoded source clips are capped at 1 GiB, perception JSONL at 64 MiB, decoded
incident memory at 512 MiB, each serialized shadow clip at 256 MiB, and cumulative
duplicated clip output at 1 GiB. Source streams are also capped at 100,000 records,
incidents at 2,000 frames, and eligible targets at 1,000 so tiny-frame inputs cannot
turn overlapping provenance windows into unbounded quadratic memory. Across all
targets, at most 1,000,000 frame entries may be materialized. Exact serialized NPY
sizes are checked against both per-clip and cumulative disk limits before a file is
created. These bounds are part of the adapter's denial-of-service surface, not
statements about useful model input.

## What provenance does not prove

`provenance.json` binds input metadata, encoded clip bytes, source perception
JSONL, every emitted `.npy`, all three derived JSONL streams, and the exact selected
ordinals/sequences/timestamps. New recorder frame bindings additionally bind the
canonical perception record bytes to recorder ordinals. The manifest records the
Python, NumPy, OpenCV, and OpenCV-build identities used for lossy decoding, and its
assembly ID incorporates those identities plus all derived mapping/output hashes.

Those hashes are not signatures or trusted camera attestation. Legacy recorder
artifacts can only re-establish alignment structurally, and even new bindings do
not prove that a named camera produced the encoded pixels or that detections
describe the same exposure. MJPEG decoding is lossy. No track-crop correctness,
model quality, calibration, domain-transfer, deployment-safety, or physical-action
claim is made.

## Verification

```bash
just check
just test-inference
uv run --locked --python 3.14 pytest
```

`test-inference` publishes synthetic pixels and strict records through the actual
recorder and OpenCV encoder, assembles the resulting incident, runs controlled real
ONNX Runtime inference on the sparse target stream, and executes mock-only shadow
replay over the complete incident stream. No camera or network is used.
The Python 3.10 suite also builds clean assembler, shadow, simulator, and evaluator
wheels, installs them outside the repository, verifies their import paths, and runs
the installed assembler CLI help entry point.
