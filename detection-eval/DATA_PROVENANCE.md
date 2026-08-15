# COCO 2017 provenance and data-use notes

## Pinned source

The harness uses the official COCO image bucket through its TLS-valid Amazon S3
path:

- Annotation archive:
  `https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`
- Selected image template:
  `https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}`
- COCO project: <https://cocodataset.org/>
- Official API repository: <https://github.com/cocodataset/cocoapi>

The annotation identity is pinned in source rather than trusting a filename:

| Property | Expected value |
|---|---|
| byte size | `252907541` |
| S3 ETag / MD5 | `f4bbac642086de4f52a3fdda2de5fa2c` |
| SHA-256 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| extracted `annotations/instances_val2017.json` SHA-256 | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |

The byte size and simple (non-multipart) MD5 ETag are metadata returned by the
official S3 object. The SHA-256 values were independently computed over those
verified bytes and are pinned here for stronger future cache checks.

For each selected image, the downloader requires `Content-Length` and a simple
MD5 ETag from the TLS-authenticated official S3 response, checks both against
the downloaded bytes, computes SHA-256, and writes a sidecar. Subsequent online
or offline runs re-hash the file. An existing user-provided COCO root has no
fresh remote metadata, so the harness records and subsequently checks local
MD5, SHA-256, and byte size in the subset manifest.

Downloads are written to a temporary file in the destination directory and
renamed into place only after verification. A partial response therefore never
becomes a valid cache entry. The source tree ignores archives and common image
extensions, and the documented cache lives under the repository's ignored
`artifacts/` directory. Do not add dataset images to Git.

## Licenses and attribution

COCO does not apply one blanket image license. `instances_val2017.json`
contains a `license` ID for each image and a corresponding top-level `licenses`
table. The subset manifest preserves each selected image's `license_id`; users
must inspect the pinned annotation file and comply with the corresponding
source-image terms when retaining, sharing, or publishing examples. The
Simplified BSD license in the COCO API repository covers its API code and
should not be treated as a license for the photographs themselves.

This package does not copy COCO API code and does not require `pycocotools`.
The invoked detector implementation and model provenance are documented in
`perception/THIRD_PARTY_NOTICES.md` and `models/registry.json` at the repository
root.

## Intended use and limitations

COCO was not captured from this project's installed camera and does not label
cat eating, digging, harmless sniffing, plant contact, or garden safety zones.
Its class mix, viewpoints, illumination, distances, compression, and
occlusions differ from deployment. Hard-negative here means “no COCO cat or
person annotation, but contains an explicitly listed detection confuser”; it
does not mean a behavior hard-negative.

Therefore this baseline can expose detector plumbing regressions and public-set
cat/person trade-offs, but cannot validate the behavior model, installed-camera
domain performance, policy interlocks, or a physical-action decision. Those
claims require separately consented, session-isolated local data and end-to-end
hardware safety testing.

The selected public subset is a development/regression fixture. If it is used
to choose a confidence threshold, the resulting metric is a tuning result and
must not be described as held out. Report an operating point only on a separately
locked seed/manifest, followed by the installed-camera evaluation required for
the actual domain.
