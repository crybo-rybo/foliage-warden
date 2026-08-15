set shell := ["bash", "-euo", "pipefail", "-c"]

project_root := justfile_directory()
build_dir := project_root / "build" / "cpp"
python_dir := project_root / "python"

default: check

configure build_type="Debug":
    cmake -S "{{project_root}}" -B "{{build_dir}}" -DCMAKE_BUILD_TYPE="{{build_type}}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

build build_type="Debug": (configure build_type)
    cmake --build "{{build_dir}}" --parallel

test-cpp: (build "Debug")
    ctest --test-dir "{{build_dir}}" --output-on-failure

sync-python:
    uv sync --project "{{python_dir}}" --extra dev

test-python: sync-python
    uv run --project "{{python_dir}}" pytest

test: test-cpp test-python

check: test
