#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/gsm8k_eval.py --backend atlas_serial \
  --model "${DRAFTER_MODEL:-/home/hwc/models/Llama-3.2-1B-Instruct}" \
  --target-model "${TARGET_MODEL:-/home/hwc/models/Meta-Llama-3.1-8B-Instruct}" \
  --data-file "${GSM8K_FILE:-/home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl}" \
  --output-dir "${OUTPUT_DIR:-../0709_outputs/gsm8k_atlas_serial}" \
  --gpu-id "${GPU_ID:-0}" --k "${K:-3}" --d "${D:-4}" \
  --path-score-weights "${PATH_SCORE_WEIGHTS:-0.45,0.30,0.17,0.08}" \
  --fallback-threshold "${FALLBACK_THRESHOLD:--0.50}" \
  --first-token-threshold "${FIRST_TOKEN_THRESHOLD:--0.70}" \
  --fallback-ar-tokens "${FALLBACK_AR_TOKENS:-4}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" --context-length "${CONTEXT_LENGTH:-4096}" \
  --warmup-runs "${WARMUP_RUNS:-1}" "$@"
