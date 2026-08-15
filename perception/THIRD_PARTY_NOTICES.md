# Third-party notices

## OpenCV Zoo YOLOX

The preprocessing and raw YOLOX grid-decode semantics in
`src/foliage_warden_perception/yolox.py` were adapted from these files:

- Project: OpenCV Zoo
- Files: `models/object_detection_yolox/yolox.py` and `demo.py`
- Revision: `47534e27c9851bb1128ccc0102f1145e27f23f98`
- Source: <https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_yolox>
- License: Apache License 2.0, <https://www.apache.org/licenses/LICENSE-2.0>

The implementation here was rewritten around typed normalized detections and
uses an original deterministic NumPy class-aware NMS implementation in place
of OpenCV Zoo's `cv2.dnn.NMSBoxesBatched` call, because the target Jetson's
OpenCV 4.8 does not provide that API.

The model artifact identified as `yolox_s_opencv_zoo` in
`../models/registry.json` is also distributed by OpenCV Zoo under Apache-2.0.
It is downloaded separately, SHA-256 verified, and excluded from Git.
