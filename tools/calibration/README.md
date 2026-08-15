# Local scene calibration

This dependency-free browser tool draws normalized scene zones and safe aim
presets over a reference image. It does not upload the image or talk to an
actuator.

Serve it locally so browser module imports work:

```sh
uv run python -m http.server --bind 127.0.0.1 8080 --directory tools/calibration
```

Then open `http://127.0.0.1:8080`. Exported coordinates are normalized to
`[0, 1]`. The output contains a schema-valid `scene` fragment; importing a full
runtime configuration preserves its other fields when exporting. Keep
reference images containing private household scenes outside Git.

Run the geometry tests with:

```sh
node --test tools/calibration/geometry.test.mjs
```
