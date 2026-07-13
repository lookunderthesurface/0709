#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/gsm8k_eval.py --backend ar \
  --model "${MODEL_8B:-/home/hwc/models/Meta-Llama-3.1-8B-Instruct}" \
  --data-file "${GSM8K_FILE:-/home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl}" \
  --output-dir "${OUTPUT_DIR:-../0709_outputs/gsm8k_ar_8b}" \
  --gpu-id "${GPU_ID:-0}" --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --context-length "${CONTEXT_LENGTH:-4096}" --warmup-runs "${WARMUP_RUNS:-1}" "$@"
