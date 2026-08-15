# Canonical deterministic scenarios

This directory is the version 1 safety and policy contract for the simulator. The fixtures inject already-fused policy observations; they test timing, state, interlock, and mock-transport behavior. They do **not** measure detector, tracker, or behavior-model accuracy.

Every scenario:

- references [`../config/simulation-safe.example.json`](../config/simulation-safe.example.json);
- begins at virtual time zero in `DISARMED` with no physical-effect path;
- declares exact safety-critical counts and an ordered action result sequence;
- is deterministic under the ordering rules in [`../docs/architecture.md`](../docs/architecture.md);
- must validate against [`../schemas/scenario.schema.json`](../schemas/scenario.schema.json).

## Coverage

| Fixture | What it proves | Expected burst commands |
|---|---|---:|
| `01-clear-pass.json` | Cat presence and approach-zone overlap are insufficient | 0 |
| `02-sniffing-hard-negative.json` | Sniffing with foliage overlap remains a hard negative | 0 |
| `03-eating-persistence.json` | Persistent eating plus foliage evidence reaches one mock action | 1 |
| `04-digging-persistence.json` | Persistent digging requires soil overlap and motion | 1 |
| `05-person-interlock.json` | Any visible person globally blocks action | 0 |
| `06-multiple-cats.json` | Multiple visible cats block the MVP | 0 |
| `07-stale-frame.json` | Stale inference is rejected before accumulation | 0 |
| `08-poor-track.json` | Classifier confidence cannot overcome poor tracking | 0 |
| `09-no-fire-intersection.json` | Calibrated no-fire geometry is a hard gate | 0 |
| `10-hardware-not-ready.json` | An unavailable actuator blocks readiness | 0 |
| `11-missing-burst-ack.json` | Ambiguous burst outcome faults and is never retried | 1 |
| `12-duplicate-command-id.json` | Transport deduplication suppresses a replayed burst ID | 1 unique |
| `13-continuous-incident-cooldown.json` | Cooldown expiry alone cannot create a second incident | 1 |
| `14-camera-loss-restart.json` | Camera loss faults; restart returns to disarmed | 0 |

`INJECT_ACTION` exists only for deterministic transport tests such as duplicate-ID handling. A production runtime must not expose that input. `OBSERVATION_SERIES` is compact fixture syntax; the runner expands it into ordinary observations before replay.

## Adding a fixture

Use a new stable `scenario_id` and requirement ID. Keep time in integer milliseconds, use explicit `sequence` values, and state why each expected hold or action is correct. Positive fixtures must still expect `physical_bursts: 0`; this corpus is never a hardware test suite.

Do not weaken an existing fixture in place after it has been used as an acceptance gate. Add a versioned replacement and retain the old fixture so historical run results remain reproducible.
