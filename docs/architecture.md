# Simulation and safety architecture

## Purpose and boundary

Foliage Warden is split at a narrow policy-observation and action interface so the risky logic can be exercised without a camera model, motors, or a burst device:

```text
scripted observations / video replay / live perception
                         |
                         v
                policy observation v1
                         |
                         v
             deterministic safety policy
                         |
                         v
            transport-independent commands
                  /              \
        mock/replay adapter     hardware adapter
          no physical path     unavailable in simulation
```

The canonical scripted scenarios provide fused detections, track quality, behavior scores, region evidence, and aim-safety results. Consequently, passing them is evidence about policy and transport behavior only. It is not evidence that a trained model recognizes eating, digging, a cat, or a person in real video. Model evaluation must use held-out, session-separated clips and feed its outputs through the same observation interface.

No configuration or scenario in this repository authorizes a physical effect. The example configuration uses a virtual clock, a scripted camera, a mock actuator, `allow_physical_effects: false`, and unconditional startup `DISARMED`.

## Canonical contracts

All contracts use JSON Schema draft 2020-12 and have `schema_version: 1`:

- `schemas/runtime-config.schema.json` defines bounded runtime, perception, policy, calibration, actuator, and audit settings.
- `schemas/scenario.schema.json` defines deterministic external inputs and assertions.
- `schemas/event-record.schema.json` defines the general append-only JSONL event stream.
- `schemas/action-audit.schema.json` defines the evidence and every interlock value for an action decision.
- `schemas/common.schema.json` owns shared state, reason, geometry, observation, and command types.

Unknown object fields are rejected. A producer that needs a breaking field or meaning change must publish a new schema version rather than silently changing version 1.

The configuration identity recorded in logs is SHA-256 over its RFC 8785 JSON Canonicalization Scheme representation. The runtime records both `config_id` and the derived digest. File paths and formatting are not part of identity.

## Scene calibration

Scene coordinates are normalized image coordinates: top-left is `(0, 0)` and bottom-right is `(1, 1)`. `scene.zones` is a flat list of polygons:

- `approach` is the protected cat-entry region;
- `foliage` and `soil` provide behavior-specific spatial evidence;
- `no_fire` is a hard exclusion region.

An aim preset refers to its protected `zone_id` and has a normalized image `point`. That image point documents intent; it is not itself a motor calibration. A hardware adapter will eventually require a separately measured, bounded mapping from preset ID to hardware coordinates. It must never derive a free-form target from a cat bounding box.

JSON Schema enforces coordinate bounds and polygon vertex counts. A semantic configuration validator must additionally reject:

- duplicate zone or preset IDs;
- self-intersecting or zero-area polygons;
- `x + width > 1` or `y + height > 1` for a bounding box;
- a preset whose `zone_id` does not name an `approach` zone;
- a plant approach zone with no foliage/soil evidence region as required by its intended behavior;
- a requested burst duration greater than `hardware_max_duration_ms`;
- a LIVE/physical configuration without an externally validated calibration artifact.

The checked-in scene is deliberately labeled as an unvalidated example.

## Virtual time and event ordering

Policy decisions use a monotonic clock only. Wall time is metadata and cannot change an outcome. Scripted runs start at virtual millisecond zero.

Before execution, an `OBSERVATION_SERIES` is expanded into `count` observations. For zero-based index `i`:

- delivery time is `at_ms + i * interval_ms`;
- capture time is delivery time plus `capture_offset_ms`;
- observation and frame IDs are `id_prefix` plus a zero-padded index;
- every template track's age is `initial_track_age_ms + i * interval_ms`.

The loader rejects a generated capture time below zero or an integer overflow. Expanded inputs are ordered by `(at_ms, sequence, event_id)`, and duplicate event IDs or duplicate `(at_ms, sequence)` keys are invalid.

When advancing from time `A` to `B`, the engine handles internal acknowledgements and deadlines strictly before `B` in due-time and insertion order. It then applies all external inputs at `B` in declared sequence order. External health and control changes at an exact deadline therefore take effect before an acknowledgement due at that same instant—a conservative tie break. Finally, it drains internal work due exactly at `B`. Cascaded work retains its true due time rather than being stamped with the next external event's time.

A `TICK` has no payload. It advances the clock so pending timeouts or acknowledgements can settle. A run ends after processing internal work due no later than its final external event. It never silently advances beyond the declared timeline.

`captured_at_ms` may be earlier than delivery time to inject latency. It may never be later. Frame age is `delivery_at_ms - captured_at_ms`; an age above `camera.max_frame_age_ms` rejects the entire observation before any persistence accumulator changes.

## Observation and interlock semantics

An observation is the immutable result of one frame's perception and fusion work. Person presence is global: any accepted `PERSON` track above the configured detection threshold blocks action, regardless of zone. Likewise, more than one accepted cat track anywhere in the frame blocks the MVP.

An armed policy evaluates an observation in fail-closed order:

1. camera and actuator health;
2. capture freshness;
3. person, multiple-cat, and ambiguous-track interlocks;
4. cat confidence, track age, track quality, and gap continuity;
5. protected-zone membership and behavior probability;
6. behavior-specific foliage or soil/motion evidence;
7. safe-preset availability and no-fire intersection;
8. incident latch and cooldown availability.

A hard interlock records its specific reason and clears that observation from confirmation. `UNKNOWN` is never harmful evidence. `CLEAR`, an out-of-zone track, a missing track beyond `max_track_gap_ms`, or a blocking interlock breaks persistence rather than contributing a low score.

EATING and DIGGING accumulate independently for the same `(track_id, zone_id, behavior)` key. Confirmation requires both:

- at least `min_supporting_observations` qualifying observations inside `confirmation_window_ms`; and
- elapsed time from the first to latest qualifying observation of at least `harmful_persistence_ms`.

No interpolation occurs between observations. A single frame can never satisfy confirmation.

## State and incident semantics

The normal action path is:

```text
DISARMED -> MONITORING -> TRACKING -> CONFIRMING
          -> AIMING -> READY -> BURST -> COOLDOWN
```

`ARM` is a manual control input that moves a healthy runtime from `DISARMED` to `MONITORING`; it is not an actuator command in scenario action counts. A rejected or non-harmful observation may visit `TRACKING` for auditability and returns to `MONITORING`. Safety holds are reasoned decisions, not a separate policy state.

Health timeouts, ESTOP, transport ambiguity, and latched component faults enter `FAULT`. A process restart is a new startup boundary: volatile confirmation, incident, cooldown, and armed state are discarded, the reason `PROCESS_RESTART` is recorded, and policy state becomes `DISARMED`. Reconnection never arms the policy.

An incident begins with the first qualifying harmful observation for a track and protected zone. The incident action latch is set as soon as a unique BURST command is accepted for dispatch—not when its ACK arrives—because a lost ACK leaves the physical outcome unknown. The latch resets only after the relevant cat/behavior is continuously clear for `incident_clear_ms`. Merely reaching the end of `cooldown_ms` cannot reset it. Thus continuous harmful behavior beyond cooldown still permits exactly one burst.

`COOLDOWN` ends only when both the cooldown duration and incident-clear condition are satisfied. This deliberately makes the state conservative for an animal that remains at the plant.

## Mock actuator and command semantics

The action sequence is intentionally two-stage:

1. enter `AIMING`, issue `GOTO_PRESET`, and wait for ACK;
2. after a successful preset ACK and a fresh re-check of all interlocks, enter `READY`, record one would-burst decision, and issue `BURST`.

Every command ID is unique within a run. The actuator keeps a deduplication ledger for at least the run lifetime. Reuse of an ID returns `DUPLICATE` and never repeats an effect. `INJECT_ACTION` is a simulator-only transport test hook; production input code must not expose it.

`BURST` is non-idempotent. A denial, transport error, or missing ACK never causes an automatic retry. A missing ACK at `ack_timeout_ms` enters `FAULT`, with the incident action latch still set. The mock adapter can respond with ACK, DENIED, DROP, or TRANSPORT_ERROR according to `actuator_script`; none can cause a physical effect.

Scenario count terms are precise:

- `ready_transitions` counts entries into READY;
- `would_burst_decisions` counts policy authorizations, regardless of adapter type;
- `burst_commands_issued` counts unique BURST commands accepted for dispatch, excluding duplicates;
- `burst_commands_acked` counts those commands with ACK outcomes;
- `physical_bursts` requires an external hardware effect confirmation and is always zero in this corpus;
- `automatic_retries` counts policy-generated resend attempts, which must be zero for BURST;
- `duplicate_commands_suppressed` counts deduplication-ledger hits.

## Audit and replay

General events and action audits are append-only JSON Lines. `sequence` is the authoritative total order within a run. An action decision records its input evidence, before/after policy state, reason codes, full interlock snapshot, exact command, dispatch mode, retry policy, and outcome. A mock or replay record must set `physical_effect_possible` to false.

Given identical versioned configuration, scenario, and executable/model identities, two runs must produce the same ordered state, reason, command, and outcome fields. Run IDs, wall timestamps, and audit IDs may differ and are excluded from the deterministic comparison. Logs should be flushed for every action decision so a crash cannot erase the explanation for a would-act or did-act event.

Additional semantic scenario validation must reject:

- non-monotonic or colliding timeline order keys;
- future capture timestamps;
- behavior score totals outside `[0.99, 1.01]`;
- an expected positive `physical_bursts` count in a simulation fixture;
- an expected BURST retry;
- a command-ID seed that can overflow after expansion;
- an expected referenced command ID inconsistent with deterministic allocation.

## Crossing into real hardware

Simulation and replay adapters must be built without opening serial devices, GPIO, or network actuator endpoints. The hardware adapter is a separate dependency selected only by a validated LIVE configuration with `backend: SERIAL`, `enabled: true`, and `allow_physical_effects: true`. Those fields are necessary but never sufficient permission.

Before a hardware adapter can be exercised, work outside this simulated phase must provide all of the following:

- mechanically bounded and measured preset calibration;
- an independent physical kill switch and fail-closed microcontroller watchdog;
- hardware-side duration clamps and command deduplication;
- verified no-fire geometry and occupied-path clearance;
- supervised aim-only tests with burst disabled;
- an explicit operator arm action after every boot or restart;
- a reviewed, animal-safe deterrent mechanism and controlled pilot plan.

No simulated pass rate can waive those requirements.
