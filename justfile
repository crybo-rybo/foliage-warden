set shell := ["bash", "-euo", "pipefail", "-c"]

project_root := justfile_directory()
build_dir := project_root / "build" / "cpp"
python_dir := project_root / "python"
simulator_dir := project_root / "simulator"
perception_dir := project_root / "perception"
training_dir := project_root / "training"
review_dir := project_root / "review"
recorder_dir := project_root / "recorder"
shadow_dir := project_root / "shadow"
detection_eval_dir := project_root / "detection-eval"

default: check

configure build_type="Debug":
    cmake -S "{{project_root}}" -B "{{build_dir}}" -DCMAKE_BUILD_TYPE="{{build_type}}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

build build_type="Debug": (configure build_type)
    cmake --build "{{build_dir}}" --parallel

test-cpp: (build "Debug")
    ctest --test-dir "{{build_dir}}" --output-on-failure

sync-python:
    uv sync --project "{{python_dir}}" --extra dev --locked

test-python: sync-python
    cd "{{python_dir}}" && uv run --extra dev pytest

sync-simulator:
    uv sync --project "{{simulator_dir}}" --extra dev --locked

test-simulator: sync-simulator
    uv run --project "{{simulator_dir}}" --extra dev pytest "{{simulator_dir}}/tests"

sync-perception:
    uv sync --project "{{perception_dir}}" --extra desktop --group dev --locked

test-perception: sync-perception
    uv run --project "{{perception_dir}}" --extra desktop --group dev pytest "{{perception_dir}}/tests"

sync-training:
    uv sync --project "{{training_dir}}" --extra dev --locked

test-training: sync-training
    uv run --project "{{training_dir}}" --extra dev pytest "{{training_dir}}/tests"

sync-review:
    uv sync --project "{{review_dir}}" --group dev --locked

test-review: sync-review
    cd "{{review_dir}}" && uv run --group dev python -m unittest discover -s tests
    node --check "{{review_dir}}/src/foliage_warden_review/web/core.js"
    node --check "{{review_dir}}/src/foliage_warden_review/web/app.js"
    node --test "{{review_dir}}/tests-js/core.test.js"

sync-recorder:
    uv sync --project "{{recorder_dir}}" --extra desktop --group dev --locked

test-recorder: sync-recorder
    uv run --project "{{recorder_dir}}" --extra desktop --group dev pytest "{{recorder_dir}}/tests"

sync-shadow:
    uv sync --project "{{shadow_dir}}" --group dev --locked

test-shadow: sync-shadow
    uv run --project "{{shadow_dir}}" --group dev pytest "{{shadow_dir}}/tests"

sync-shadow-inference:
    uv sync --project "{{shadow_dir}}" --group dev --group inference-test --locked

test-shadow-inference: sync-shadow-inference
    uv run --project "{{shadow_dir}}" --group dev --group inference-test pytest "{{shadow_dir}}/tests/test_inference.py"

sync-detection-eval:
    uv sync --project "{{detection_eval_dir}}" --group dev --locked

test-detection-eval: sync-detection-eval
    uv run --project "{{detection_eval_dir}}" --group dev pytest "{{detection_eval_dir}}/tests"

test-calibration:
    node --test "{{project_root}}/tools/calibration/geometry.test.mjs"

validate-contracts: sync-python
    uv run --project "{{python_dir}}" --extra dev python "{{project_root}}/tools/validate_contracts.py"

lint: sync-python sync-simulator sync-perception sync-training sync-review sync-recorder sync-shadow sync-detection-eval
    cd "{{python_dir}}" && uv run --extra dev ruff check src tests ../tools/validate_contracts.py ../tools/jetson_probe.py
    cd "{{simulator_dir}}" && uv run --extra dev ruff check src tests
    cd "{{perception_dir}}" && uv run --extra desktop --group dev ruff check . ../tools/fetch_model.py ../tools/evaluate_synthetic_scenes.py
    cd "{{training_dir}}" && uv run --extra dev ruff check .
    cd "{{review_dir}}" && uv run --group dev ruff check src tests
    cd "{{recorder_dir}}" && uv run --group dev ruff check .
    cd "{{shadow_dir}}" && uv run --group dev ruff check src tests
    cd "{{detection_eval_dir}}" && uv run --group dev ruff check .
    shellcheck "{{training_dir}}/scripts/smoke.sh" "{{project_root}}/tools/verify_behavior_bridge.sh"

test: test-cpp test-python test-simulator test-perception test-training test-review test-recorder test-shadow test-detection-eval test-calibration validate-contracts

verify-simulator: test-simulator
    uv run --project "{{simulator_dir}}" --extra dev foliage-warden-sim --all

verify-perception: test-perception
    uv run --project "{{perception_dir}}" --extra desktop python "{{project_root}}/tools/fetch_model.py" yolox_s_opencv_zoo
    uv run --project "{{perception_dir}}" --extra desktop python "{{project_root}}/tools/evaluate_synthetic_scenes.py"

verify-training: test-training
    bash "{{training_dir}}/scripts/smoke.sh"

verify-behavior-bridge: test-training test-shadow-inference
    bash "{{project_root}}/tools/verify_behavior_bridge.sh"

verify: check verify-simulator verify-perception verify-behavior-bridge

check: lint test
