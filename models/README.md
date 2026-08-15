# Model artifacts

Model binaries are fetched into this directory but are intentionally excluded
from Git. `registry.json` pins the source revision and SHA-256 digest for every
approved artifact.

Fetch the default observe-only detector with:

```sh
uv run python tools/fetch_model.py yolox_s_opencv_zoo
```

The initial YOLOX-S model is a COCO baseline for detecting `person` and `cat`.
It is not a behavior classifier and its published COCO metrics are not evidence
that it meets the project acceptance criteria in the installed camera view.
Those criteria must be evaluated on session-separated local footage.

