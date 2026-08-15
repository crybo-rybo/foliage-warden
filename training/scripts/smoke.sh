#!/usr/bin/env bash
set -euo pipefail

training_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cleanup_smoke_dir=false
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [new-output-directory]" >&2
  exit 2
elif [[ $# -eq 1 ]]; then
  smoke_dir="$1"
else
  smoke_dir="$(mktemp -d -t foliage-warden-training-smoke.XXXXXX)"
  cleanup_smoke_dir=true
fi

cleanup() {
  if [[ "$cleanup_smoke_dir" == true ]]; then
    rm -rf -- "$smoke_dir"
  fi
}
trap cleanup EXIT

uv run --project "$training_dir" fw-behavior-generate-synthetic \
  --output-dir "$smoke_dir/data" \
  --clips-per-label 1 \
  --frames 4 \
  --image-size 24 \
  --seed 17

uv run --project "$training_dir" fw-behavior-train \
  --manifest "$smoke_dir/data/manifest.jsonl" \
  --output-dir "$smoke_dir/run" \
  --epochs 1 \
  --batch-size 6 \
  --num-frames 2 \
  --image-size 16 \
  --feature-dim 8 \
  --hidden-dim 8 \
  --dropout 0 \
  --num-workers 0 \
  --device cpu \
  --seed 17

uv run --project "$training_dir" fw-behavior-evaluate \
  --manifest "$smoke_dir/data/manifest.jsonl" \
  --checkpoint "$smoke_dir/run/best.pt" \
  --split test \
  --output "$smoke_dir/run/test-report.json" \
  --batch-size 6 \
  --num-workers 0 \
  --device cpu \
  --seed 17

uv run --project "$training_dir" fw-behavior-export \
  --checkpoint "$smoke_dir/run/best.pt" \
  --output "$smoke_dir/run/behavior.onnx" \
  --seed 17

uv run --project "$training_dir" python - "$smoke_dir/run/behavior.onnx" <<'PY'
import sys

import onnx

model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)
print(f"ONNX check passed: {sys.argv[1]}")
PY

if [[ "$cleanup_smoke_dir" == false ]]; then
  echo "Smoke artifacts retained in: $smoke_dir"
fi
