# Foliage Warden evaluation tools

This package evaluates complete harmful-behavior incidents and simulated
would-actions. It intentionally does not score randomly sampled video frames.

Run the included fixture:

```sh
uv run --extra dev foliage-warden-eval evaluate \
  --ground-truth tests/fixtures/ground_truth.jsonl \
  --replay tests/fixtures/replay.jsonl
```

The replay file may contain `session`, `prediction_event`, and `action` records.
Session records provide monitored exposure and optional duration counters for
track-loss and `UNKNOWN` rates. See `foliage_warden_eval.schemas` for the typed
wire contract.

Split a dataset manifest without leaking sessions (or an optional broader
`group_id`, such as a recording day) between splits:

```sh
uv run foliage-warden-split manifest.jsonl --output assignments.json
```

Both commands serialize JSON with sorted keys and do not include wall-clock
timestamps, so identical inputs and options produce byte-identical reports.

