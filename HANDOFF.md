# ATLAS 0709 Handoff

This file is the short continuation note for a new Codex window. The detailed
commands remain in `README.md` and `TESTING_METHODS.md`.

## Fixed Environment

Remote machine and project:

```text
/home/hwc/workspace/0701/0709
conda environment: atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Models:

```text
Drafter: /home/hwc/models/Llama-3.2-1B-Instruct
Target:  /home/hwc/models/Meta-Llama-3.1-8B-Instruct
```

Results are outside the project so copying `0709` does not copy large
artifacts:

```text
/home/hwc/workspace/0709_outputs
```

Use `../0709_outputs/...` for relative output paths when the working directory
is the project root.

The previously recorded Target service PID is `277743`. Before restarting a
Target service, run:

```bash
kill 277743
```

## Current Runtime

The production path is asynchronous:

```text
Drafter prefill + Target prefill
-> Drafter build tree (k routes, d depths)
-> Target verify stage-1 while Drafter builds forest (k^2 routes)
-> Target selects/prunes or returns fallback AR
-> preserve selected physical KV and continue the next round
```

The implementation uses SGLang model execution and paged KV pools, with
FlashInfer attention for the real model path. ATLAS owns route state, tree /
forest scheduling, commit, prune, fallback, and cross-round handoff.

The current async path keeps selected KV across rounds. It should not be
described as an HF/SDPA benchmark.

## Quality Controls

These are Target-server flags and must be set when starting Target:

```text
--path-score-weights 0.45,0.30,0.17,0.08
--fallback-threshold -0.50
--first-token-threshold -0.70
--fallback-ar-tokens 4
```

For `d=4`, earlier Target token log probabilities have larger weights. Fallback
is selected when the first token or the weighted complete-path score is below
its threshold. Target then greedily generates up to four AR tokens, commits
them, and the Edge discards all draft routes and rebuilds stage-1 from the new
committed prefix.

Thresholds are empirical starting points, not universal values:

```text
omit thresholds       old best-of-k, fallback disabled
-4.5 / -6.0           very loose, usually almost no fallback
-0.50 / -0.70         strong fallback starting point
closer to zero        more aggressive fallback and more Target AR work
```

Always inspect `rounds_detail`, `best_selection_score`,
`best_first_token_logprob`, `fallback_reason`, and `fallback_token_ids`.

## New Serial Benchmark Baseline

Added for the single-GPU contention question. It intentionally does not build
forest and does not overlap work:

```text
Drafter build tree -> Target verify -> select/fallback -> next round
```

The serial path uses the same weighted scoring and fallback settings as async,
but does not reuse stage-2 Drafter KV across rounds. It is a benchmark baseline,
not a replacement for the production async algorithm.

The GSM8K serial launcher is now a single-process path: Drafter and Target are
loaded by `gsm8k_eval.py`, Target verification is an in-process method call,
and no Target HTTP server or `TARGET_URL` is involved.

Relevant files:

```text
src/atlas_0709/serial_system.py
src/atlas_0709/__init__.py
scripts/gsm8k_eval.py
scripts/run_gsm8k_atlas_serial.sh
scripts/mtbench_deepseek_eval.py
```

Use:

```bash
bash scripts/run_gsm8k_atlas_serial.sh \
  --protocol llama-8shot \
  --data-file /home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl \
  --strict-marker \
  --fallback-ar-tokens 4
```

The path weights and score thresholds in the previous section belong on the
Target server command, not on `gsm8k_eval.py`.

For MT-Bench, add:

```text
--execution-mode serial
```

to `scripts/mtbench_relaxed_spec_deepseek_eval.py`. The default remains
`async`; serial artifacts record `execution_mode=serial` and
`build_forest=false`.

## Evaluation Preferences

- Use the FastChat MT-Bench file:
  `/home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl`.
- Full MT-Bench means all 80 questions and both turns, 160 answers per
  candidate. Save generations before judging.
- Keep AR 1B, AR 8B, and ATLAS answers in separate, immutable artifacts. For a
  fair three-way judgment, load the saved answers and send the same question,
  rubric, and candidate set to the judge; do not regenerate AR answers during
  the ATLAS run.
- DeepSeek judging currently has no usable seed. Record the model, thinking,
  reasoning effort, max tokens, prompt hash, system fingerprint, and failed
  request policy. Exact API replay is not guaranteed.
- GSM8K uses the `llama-8shot` protocol and strict answer extraction. Compare
  ATLAS against separate 1B and 8B AR runs using the same dataset and token
  limits.
- Do not compare one build-tree step with AR four-token generation. Compare a
  complete `d`-depth tree or forest component with the matched AR workload.
- For speed claims, prefer the real SGLang + FlashInfer paged path and include
  backend metadata. Do not call an HF/SDPA result FlashInfer or shared-KV
  performance.
- Preserve the standard algorithmic flow. Avoid adding unrelated optimization
  variants to the quality benchmark.

## Validation After Sync

The local Windows workspace has no usable Python interpreter. Run validation on
Herta after syncing:

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -m py_compile \
  src/atlas_0709/serial_system.py \
  src/atlas_0709/__init__.py \
  scripts/gsm8k_eval.py \
  scripts/mtbench_deepseek_eval.py

python scripts/gsm8k_eval.py --help | grep atlas_serial
python scripts/mtbench_relaxed_spec_deepseek_eval.py --help | grep execution-mode
```

If a new window needs the full command history, read this file first, then
`TESTING_METHODS.md`, then `README.md`.
