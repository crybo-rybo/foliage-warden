# Observe-only review tool

This package serves an explicit local media manifest in a small browser UI and
exports deterministic JSONL accepted by the repository's `GroundTruthEvent`
parser. It supports `PASSING`, `SNIFFING`, `EATING`, `DIGGING`, `OTHER`, and
`UNKNOWN` intervals with rationale, staged-safe, person, privacy, group, media,
and optional zone metadata.

This is an annotation workbench, not an inference system. It makes no claim
about behavioral accuracy and has no motor, burst, serial, GPIO, network upload,
or actuator command path.

## Safety and privacy boundary

- Review only prerecorded footage obtained with consent. Point cameras away
  from private areas and follow the retention rules in `docs/data-and-evaluation.md`.
- A person-present label is always privacy restricted. The server and browser
  both reject a person-present annotation without that restriction.
- Never stage contact with a toxic plant. A staged-safe label means a supervised
  setup using veterinarian-confirmed safe plant material or clean safe substrate.
- Keep manifests, recordings, annotation stores, and exports out of Git. The
  repository's ignored `recordings/` directory is the intended local location.
- `UNKNOWN` is the correct label when visible evidence is insufficient. The tool
  does not expose model predictions, which reduces annotation bias.

The HTTP server always binds `127.0.0.1`; there is no command-line option to
bind a LAN interface. It serves only four packaged UI assets and media paths
resolved from the manifest. Requests cannot name arbitrary files. Mutating API
calls require a same-origin loopback `Origin`, and all responses carry a strict
Content Security Policy.

## Manifest

Paths are relative to the manifest's directory, must point to existing files,
and may not traverse or follow a symlink outside that directory. Session,
group, media, zone, and event IDs use letters, numbers, `.`, `_`, and `-`, must
start with a letter or number, and are limited to 128 characters.

```json
{
  "schema_version": 1,
  "sessions": [
    {
      "session_id": "session-001",
      "group_id": "day-001-camera-a",
      "zone_id": "pot-1-approach",
      "media": [
        {
          "media_id": "clip-001",
          "path": "clips/clip-001.mp4",
          "kind": "video",
          "duration_ms": 18420,
          "description": "Unprompted approach, evening light"
        },
        {
          "media_id": "still-001",
          "path": "clips/still-001.jpg",
          "kind": "image",
          "duration_ms": 1
        }
      ]
    }
  ]
}
```

`duration_ms` is authoritative for interval validation. For a still image, use
an artificial duration such as `1` and label `0`–`1` ms.

## Run

From this directory:

```sh
uv run foliage-warden-review serve \
  --manifest ../recordings/review-manifest.json \
  --annotations ../recordings/annotations.json \
  --port 8765
```

Open the printed `http://127.0.0.1:8765` URL. Media never leaves the local
process. The annotation store is atomically replaced with mode `0600`; saves
use revision checks to reject stale browser updates. Editing or archiving an
event retains the superseded value in `history`.

Download JSONL from the UI or write it atomically from the command line:

```sh
uv run foliage-warden-review export \
  --manifest ../recordings/review-manifest.json \
  --annotations ../recordings/annotations.json \
  --output ../recordings/ground-truth.jsonl
```

The export contains current annotations only and uses stable record ordering,
key ordering, and compact JSON. Review-specific fields are nested under
`metadata`, keeping every line compatible with `GroundTruthEvent` version 1.

## Verify

The runtime has no third-party dependencies. Development checks use Python's
standard library and Node's built-in test runner:

```sh
uv run python -m unittest discover -s tests -v
node --check src/foliage_warden_review/web/core.js
node --check src/foliage_warden_review/web/app.js
node --test tests-js/core.test.js
```
