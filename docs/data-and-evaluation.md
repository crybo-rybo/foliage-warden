# Data and evaluation protocol

## Decision this dataset supports

The dataset must answer whether an observe-only system can distinguish a
continuous harmful plant interaction from ordinary household activity with
enough precision to justify a later supervised hardware pilot. It is not a
general cat-behavior corpus and it must not be optimized for attractive
frame-level demo results.

The primary unit is an **incident**, not a frame. One incident begins when a
cat starts interacting with a protected plant zone and ends only after the cat
and behavior remain clear for the configured incident-clear duration.

## Safety and privacy boundary

- Never stage contact with a toxic plant. Use veterinarian-confirmed cat-safe
  grass for eating examples and a dedicated clean, safe substrate for digging.
- Physical removal or isolation remains the protection for dangerous plants.
- Collect all footage in observe-only mode. No motor, burst, or deterrent output
  belongs in data collection.
- Point cameras away from private areas where possible. Mark clips containing a
  person and restrict their retention and review.
- Store raw recordings and derived clips under ignored directories such as
  `recordings/`; never add household video to Git.
- Obtain consent from every person who may be recorded before collecting a
  shareable dataset.

## Operational label taxonomy

| Label | Observable definition | Common confusion |
|---|---|---|
| `PASSING` | Cat traverses or briefly occupies the approach zone without investigating the plant | Short pauses near the pot |
| `SNIFFING` | Head approaches foliage or soil for investigation without visible ingestion or repeated soil displacement | Eating from an occluded angle |
| `EATING` | Repeated mouth/head motion with visible foliage contact and evidence consistent with ingestion | Sniffing, licking, leaf movement |
| `DIGGING` | Repeated paw motion directed into the soil region with local substrate motion | Stepping, toy pawing, grooming |
| `OTHER` | A known non-target behavior that does not fit the named hard negatives | Grooming, sitting, play |
| `UNKNOWN` | Evidence is insufficient, occluded, contradictory, or outside the annotation policy | Never force an uncertain harmful/clear label |

Annotators should label what is visibly supported, not infer intent. An
`UNKNOWN` label is preferable to optimistic ground truth.

## Event boundaries

Use integer milliseconds relative to the original recording. Start at the
first frame with sustained evidence of the labeled interaction; end at the
last supported frame before a clear gap. Preserve a surrounding context window
when extracting model clips, but keep ground-truth boundaries on the incident
itself.

Each ground-truth JSONL record follows the evaluator's
`GroundTruthEvent` contract:

```json
{"record_type":"ground_truth_event","schema_version":1,"event_id":"session-014-eating-01","session_id":"session-014","behavior":"EATING","start_ms":12840,"end_ms":16420,"zone_id":"pot-1-approach","staged_safe":true,"metadata":{"lighting":"evening","annotator":"reviewer-a"}}
```

## Collection order

1. Record empty-room and ordinary household operation to measure leaf motion,
   lighting changes, camera noise, and human activity.
2. Record natural cat approaches without encouraging plant contact.
3. Review uncertainty and false-positive candidates from the detector/zone
   pipeline.
4. If more positives are required, run short supervised sessions using only the
   safe setups described above.
5. Re-record after any camera, lens, lighting, plant-layout, or calibration
   change; those changes define a new data domain.

Prioritize hard negatives: sniffing, head occlusion, grooming beside the pot,
human watering, moving foliage, pawing a toy near soil, and entering only the
edge of the approach polygon.

## Split hygiene

Never randomly split frames or overlapping clips. All material from the same
recording session stays in one split. Prefer a broader `group_id` for the day,
camera configuration, or staged bout when adjacent sessions could leak nearly
identical backgrounds and behavior.

Use the checked-in splitter:

```sh
cd python
uv run foliage-warden-split manifest.jsonl --output assignments.json
```

Keep the final test assignments frozen. Active-learning selections return to
training or validation; they do not migrate into the held-out test set.

## Annotation quality

- Write a short rationale for every `EATING`, `DIGGING`, and `UNKNOWN` incident.
- Double-label all harmful events and a representative sample of hard
  negatives during the first iteration.
- Resolve disagreements without showing model predictions to the adjudicator.
- Track agreement by label and confusion pair; overall agreement can hide the
  exact eating-versus-sniffing problem this project must solve.
- Version label corrections rather than silently rewriting a published eval.

## Evaluation gates

Run the event evaluator on untouched sessions:

```sh
cd python
uv run foliage-warden-eval evaluate \
  --ground-truth tests/fixtures/ground_truth.jsonl \
  --replay tests/fixtures/replay.jsonl
```

Report at least:

- event precision, recall, and F1 overall and for eating/digging separately;
- observed false would-actions per monitored hour and its one-sided confidence
  bound;
- onset-to-READY and READY-to-command latency distributions;
- track-loss and `UNKNOWN` duration rates;
- every safety-invariant violation, even when model metrics improve;
- results sliced by session, lighting, zone, camera configuration, and whether
  the event was staged safely.

The README's targets are release gates, not claims. Do not tune a threshold on
the held-out test set to make a gate pass. Zero observed false actions must be
reported with its exposure and confidence bound; zero in a short run is not a
zero underlying rate.

## Public data and synthetic data

Broad animal-action datasets may initialize temporal features, and licensed cat
videos may expand pose/viewpoint diversity. They are not substitutes for the
installed camera domain and must not enter the local held-out test set.

Synthetic observations and composited imagery are useful for policy, geometry,
logging, and fault-injection tests. They cannot establish eating-versus-sniffing
or digging precision. Every performance claim must identify whether it came
from scripted events, synthetic video, public video, staged-safe local video,
or natural local operation.
