# Synthetic perception scenes

These generated images are small integration fixtures for the pinned cat/person
detector, tracker wire format, and normalized zone geometry. They are not a
behavior-recognition dataset and must never be included in a real-world model
quality claim.

Each scene has manually approximated polygons for the generated composition and
an object-count expectation in `manifest.json`. The `behavior_hint` field only
describes the prompt; it is not ground truth that eating or digging occurred.
Generated still images also cannot test temporal persistence.

Run the observe-only detector from the repository root after fetching the
pinned model:

```sh
uv run --project perception --extra desktop python tools/fetch_model.py \
  yolox_s_opencv_zoo
uv run --project perception --extra desktop python \
  tools/evaluate_synthetic_scenes.py
```

The four project-bound assets were created with the built-in image-generation
tool on 14 August 2026. The normalized prompt set is retained in
`prompts.json`; SHA-256 digests in `manifest.json` make accidental changes
visible.
