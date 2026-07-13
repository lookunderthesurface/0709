# ATLAS 0709 Handoff

Read this file first when opening a new work window. It is intentionally
action-oriented. Detailed historical notes are in `README.md` and
`TESTING_METHODS.md`.

## 0. Start Here: Every New Terminal

Never assume the shell was opened in the project directory. Define the root,
enter it, and print the resolved working directory before any project command:

```bash
PROJECT_ROOT=/home/hwc/workspace/0701/0709
cd "$PROJECT_ROOT"
pwd
conda activate atlas
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
```

When giving the user a server command, invoke newly added scripts through
`$PROJECT_ROOT/scripts/<name>.py`, not through an assumed current directory.
Before a long GPU run, fail fast if the entry point is missing:

```bash
test -f "$PROJECT_ROOT/scripts/<name>.py"
```

Optional GPU selection:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Normal operation does not require `pip install -e .`. This project is run from
the copied source tree through `PYTHONPATH`.

## 1. Fixed Paths

```text
Project:  /home/hwc/workspace/0701/0709
Conda:    atlas
Drafter:  /home/hwc/models/Llama-3.2-1B-Instruct
Target:   /home/hwc/models/Meta-Llama-3.1-8B-Instruct
MTBench:  /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
GSM8K:    /home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl
Results:  /home/hwc/workspace/0709_outputs
```

The result directory is outside `0709`, so copying the project does not copy
large benchmark artifacts. From the project root, use `../0709_outputs/...`.

The previously recorded Target service PID is `277743`. If the Target service
must be restarted, clean up the old process first:

```bash
kill 277743
```

Do not kill it merely for a serial benchmark: serial mode does not use the
network Target server, but the old process may still occupy GPU memory.

## 2. What the Project Actually Uses

```text
Drafter: SGLang ModelRunner + FlashInfer paged decode + paged KV pool
Target:  direct full-model FlashInfer masked tree verify + paged KV
Transport: HTTP JSON RPC between Edge/Drafter and Target in async mode
Package: atlas_0709
```

Important source areas:

```text
src/atlas_0709/distributed_system.py       async tree/forest controller
src/atlas_0709/serial_system.py            serial tree -> verify baseline
src/atlas_0709/flashinfer_paged/           Drafter SGLang/FlashInfer route KV
src/atlas_0709/flashinfer_full_verify.py   Target full-model FlashInfer verify
src/atlas_0709/target_runtime.py           Target persistent KV and scoring
src/atlas_0709/rpc.py                      Target HTTP protocol
src/atlas_0709/edge_timeline.py            Edge-observed black-box SVG/JSON
```

The validated environment recorded in the project is approximately:

```text
PyTorch 2.11.0 + CUDA 13.0
FlashInfer 0.6.12
SGLang 0.5.14
Transformers 5.8.1
```

## 3. Production Async Flow

The default algorithm is:

```text
Drafter prefill and Target prefill
-> Drafter build tree: k routes, d depths
-> Target verifies stage-1 tree while Drafter builds forest: k^2 routes
-> Target selects one stage-1 route, or returns fallback AR tokens
-> commit/prune and preserve selected physical KV
-> resume retained routes or start the next tree/forest round
```

The Target verify result may arrive before the last forest step. In that case
forest work is stopped between decode steps, unselected routes are released,
and the selected route's KV is handed off. No CUDA kernel is preempted in the
middle of a step.

The async path is the production system. It overlaps two processes and is
expected to be tested on different GPUs or different machines when measuring
overlap performance. On one GPU, Drafter and Target kernels compete for the
same device; use the serial baseline below to isolate that scheduler effect.

Start Target in terminal 1:

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -m atlas_0709.target_server \
  --model /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 18090 \
  --k 3 \
  --d 4 \
  --prefix-len 16384 \
  --page-size 16 \
  --dtype float16 \
  --device cuda \
  --path-score-weights 0.45,0.30,0.17,0.08 \
  --fallback-threshold -0.50 \
  --first-token-threshold -0.70 \
  --fallback-ar-tokens 4
```

For a different machine, bind Target with `--host 0.0.0.0` and use the Target
machine's reachable IP in the Edge URL. The Edge client must never connect to
`0.0.0.0`; that address only means "listen on all interfaces".

Run the Edge/Drafter in terminal 2:

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -m atlas_0709.distributed_system \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-url http://127.0.0.1:18090 \
  --prompt "ATLAS asynchronous tree speculation." \
  --k 3 \
  --d 4 \
  --max-new-tokens 64 \
  --context-length 16384 \
  --dtype float16 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --json-out ../0709_outputs/atlas_0709_generation.json \
  --timeline-svg ../0709_outputs/atlas_0709_edge_blackbox.svg \
  --timeline-json ../0709_outputs/atlas_0709_edge_blackbox.json
```

Health check:

```bash
curl http://127.0.0.1:18090/health | python -m json.tool
```

The SVG is measured by the Edge as a Cloud black box. It includes network,
queue, Target verify, Target KV commit, and response wait. It does not use a
Target-internal duration as the Cloud interval. Green blocks are individual
forest depths; blue blocks are post-prune build-tree depths; a visible gap
between blocks means separate token steps.

## 4. Quality Controls

These flags belong to the Target server, not the async Edge command:

```text
--path-score-weights 0.45,0.30,0.17,0.08
--fallback-threshold -0.50
--first-token-threshold -0.70
--fallback-ar-tokens 4
```

For `d=4`, Target selects using:

```text
0.45 * logprob(token_1)
+ 0.30 * logprob(token_2)
+ 0.17 * logprob(token_3)
+ 0.08 * logprob(token_4)
```

Earlier positions have more influence. This is a Target path-position weight;
it is not a Drafter/Target score mixture. Drafter cumulative log probability
is only a tie-breaker in the current Target selection implementation.

Fallback is triggered when either condition holds:

```text
best_first_token_logprob < first_token_threshold
best_selection_score < fallback_threshold
```

Because log probabilities are negative, a threshold closer to zero is more
aggressive. Starting references for `k=3, d=4`:

```text
omit thresholds       original best-of-k, fallback disabled
-4.5 / -6.0           very loose, usually almost no fallback
-0.50 / -0.70         strong fallback starting point
closer to zero        more Target AR work, potentially higher quality
```

On fallback, Target greedily generates up to four causally dependent AR tokens,
commits them to Target KV, and returns them. Edge discards speculative routes,
appends all known fallback tokens to Drafter with one multi-token EXTEND, and
starts the next stage-1 tree from the resulting prefix. Historical KV is not
re-prefilled. Always inspect actual `fallback_rate`; thresholds do not imply a
fixed fallback percentage.

## 5. Serial Quality/Speed Baseline

This is the requested one-GPU comparison mode. It is deliberately not async
and never builds forest:

```text
Drafter build tree (d depths)
-> Target masked tree verify
-> select or fallback AR
-> commit and continue
```

It uses the same 1B Drafter, 8B Target, weighted path scoring, fallback, and
persistent KV policy as the quality flow. It removes process overlap and does
not reuse stage-2 forest KV. It is a benchmark baseline, not the production
algorithm.

Do not start `target_server` for this mode. The launcher loads Target in the
same process:

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 \
DRAFTER_MODEL=/home/hwc/models/Llama-3.2-1B-Instruct \
TARGET_MODEL=/home/hwc/models/Meta-Llama-3.1-8B-Instruct \
PATH_SCORE_WEIGHTS=0.45,0.30,0.17,0.08 \
FALLBACK_THRESHOLD=-0.50 \
FIRST_TOKEN_THRESHOLD=-0.70 \
FALLBACK_AR_TOKENS=4 \
OUTPUT_DIR=../0709_outputs/gsm8k_atlas_serial_weighted_fallback \
bash scripts/run_gsm8k_atlas_serial.sh \
  --protocol llama-8shot \
  --data-file /home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl \
  --strict-marker \
  --max-new-tokens 512 \
  --context-length 8192 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1
```

The direct Python equivalent is `python scripts/gsm8k_eval.py --backend
atlas_serial`; the normal async backend is `--backend atlas`.

## 6. MT-Bench Quality Evaluation

Dataset:

```text
/home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
```

DeepSeek judging is post-generation. Save generations before judging. The API
does not provide a usable seed, so exact replay is not guaranteed. Keep the
same judge model, thinking mode, reasoning effort, max tokens, rubric, and
output artifacts across candidates. The scripts record hashes, system
fingerprints, request metadata, and failed requests.

Run all 80 questions and both turns for async ATLAS after starting Target:

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export DEEPSEEK_API_KEY='YOUR_DEEPSEEK_API_KEY'

python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-url http://127.0.0.1:18090 \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --k 3 --d 4 \
  --max-new-tokens 512 \
  --context-length 16384 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --dtype float16 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --judge-model deepseek-v4-pro \
  --judge-thinking enabled \
  --judge-reasoning-effort high \
  --judge-max-tokens 4096 \
  --output-dir ../0709_outputs/mtbench_atlas_async_full
```

For the MT-Bench tree-only serial scheduler, use the same evaluator and add:

```text
--execution-mode serial
```

Use a separate output directory. Important implementation detail: the current
MT-Bench evaluator still constructs `RemoteTargetClient` in both modes, so
`--execution-mode serial` still requires the network Target server and does
not remove cross-process GPU contention. It disables forest/overlap in the
ATLAS scheduler, but it is not the same-process baseline. Use
`scripts/gsm8k_eval.py --backend atlas_serial` for the true same-process
serial quality/speed baseline.

For fair AR baselines, stop a same-GPU Target server first, then run:

```bash
python scripts/mtbench_ar_deepseek_eval.py \
  --model-1b /home/hwc/models/Llama-3.2-1B-Instruct \
  --model-8b /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --candidate both \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --max-new-tokens 512 \
  --context-length 16384 \
  --dtype float16 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 32 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --judge-model deepseek-v4-pro \
  --judge-thinking enabled \
  --judge-reasoning-effort high \
  --judge-max-tokens 4096 \
  --output-dir ../0709_outputs/mtbench_ar_1b_8b
```

The AR path also uses SGLang + FlashInfer paged greedy batch-1 decode. It is
not an HF/SDPA baseline.

For a three-candidate comparison, first preserve AR generations, then pass
the saved AR `generations.jsonl` and its run id to the relaxed-spec evaluator.
The script checks matching questions, turns, references, token budgets, and
answer counts. It judges candidates pointwise with the same blind rubric; it
does not regenerate the AR answers during the ATLAS run. See the full command
in `README.md` under "Three-way quality comparison".

## 7. GSM8K and Quality Sweeps

Protocol for published-style comparison:

```text
llama-8shot + greedy decoding + --strict-marker
```

AR launchers:

```bash
bash scripts/run_gsm8k_1b_ar.sh
bash scripts/run_gsm8k_8b_ar.sh
```

Async ATLAS requires the network Target server:

```bash
bash scripts/run_gsm8k_atlas.sh
```

Quality ablation scripts run the serial system and write consolidated CSV/JSON
under `../0709_outputs`:

```bash
python scripts/run_gsm8k_serial_fallback_sweep.py \
  --max-examples 100 \
  --thresholds none -1.0 -0.75 -0.50 -0.25 \
  --path-alpha 0.50 \
  --output-dir ../0709_outputs/gsm8k_serial_fallback_sweep

python scripts/run_gsm8k_serial_path_weight_sweep.py \
  --max-examples 100 \
  --alphas 0.0 0.25 0.50 0.75 1.0 \
  --output-dir ../0709_outputs/gsm8k_serial_path_weight_sweep
```

Use 100 examples only for screening. Re-run shortlisted settings on all 1319
test examples before making a quality claim. Never mix settings with `--resume`
or reuse the same output directory.

## 8. Speed Tests and Fairness Rules

For component timing, use:

```bash
python benchmarks/bench_atlas_0709_isolated_components.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-model /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --k 3 --d 4 \
  --prefix-len 8192 \
  --repeat-token-id 42 \
  --shared-path-tokens 2 \
  --dtype float16 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup 3 --iters 10 \
  --json-out ../0709_outputs/atlas_0709_isolated_components.json
```

This uses synthetic token ids, runs tree, forest, and verify sequentially, and
excludes prefill from timed regions. It reports the complete `d`-depth tree and
forest totals plus each depth. It must not compare one build-tree step with
AR four-token generation. Compare a complete `d`-step tree/forest workload
against a matched AR workload. For performance claims, use real SGLang +
FlashInfer metadata and never call HF/SDPA an ATLAS shared-KV result.

The real route-KV smoke test is:

```bash
python tools/smoke_flashinfer_paged_tree_forest.py \
  --model /home/hwc/models/Llama-3.2-1B-Instruct \
  --prompt "ATLAS route KV alignment." \
  --k 3 --d 4 --page-size 16 \
  --context-length 8192 \
  --max-running-requests 128 \
  --max-total-tokens 65536 \
  --check-hf-logits
```

## 9. Current Scope and Caveats

- Do not describe HF/SDPA execution as FlashInfer performance.
- Drafter uses the working SGLang FlashInfer paged batch path.
- Target uses direct full-model FlashInfer masked tree verify with RoPE and
  logit alignment checks available in the full-model benchmark.
- The production async flow retains selected KV and does not re-prefill the
  historical prefix between rounds.
- The serial path is intentionally tree-only and sequential.
- Three-level FlashInfer Cascade is not the current production tree/forest path.
- One Target server owns one active generation session at a time.
- DeepSeek judging has no guaranteed seed; preserve generation/judgment files.
- The local Windows workspace has no usable Python/GPU runtime. Validate on
  Herta, not by claiming local tests passed.

## 10. First Validation in a New Window

After the setup commands in section 0:

```bash
python -m py_compile \
  src/atlas_0709/serial_system.py \
  src/atlas_0709/__init__.py \
  scripts/gsm8k_eval.py \
  scripts/mtbench_deepseek_eval.py \
  scripts/run_gsm8k_serial_fallback_sweep.py \
  scripts/run_gsm8k_serial_path_weight_sweep.py

python scripts/gsm8k_eval.py --help | grep atlas_serial
python scripts/mtbench_relaxed_spec_deepseek_eval.py --help | grep execution-mode
```

Then inspect the Target health endpoint before any async quality run. Read
`TESTING_METHODS.md` for artifact inspection and fallback-rate analysis.

## 11. User Workflow, Git, and Server Delivery

The user develops through the local Windows workspace but runs Python/GPU
experiments on Herta. Source-code work is not delivered merely because a file
was edited locally. For changes intended for an experiment, the expected flow
is:

```text
inspect existing dirty worktree
-> edit only the requested files
-> run available local static/control-flow checks
-> commit only the files changed for this task
-> push the commit to origin/main
-> verify origin/main contains the commit and every new entry point
-> on Herta: enter the fixed project root, pull, and test that files exist
-> only then provide or run the GPU experiment command
```

Preserve unrelated user changes. Never use `git add .`, `git reset --hard`, or
`git checkout --` in a dirty worktree. Stage explicit paths only:

```bash
git status --short
git diff -- <changed-file-1> <changed-file-2>
git add -- <changed-file-1> <changed-file-2>
git diff --cached --check
git diff --cached --stat
git commit -m "<specific summary>"
```

Before pushing, check whether the remote moved. Do not silently merge unrelated
work or rewrite shared history:

```bash
git fetch origin main
git rev-list --left-right --count origin/main...HEAD
git log --oneline --decorate -5
git push origin main
```

If `origin/main` is ahead or the worktree contains changes that make rebase or
pull unsafe, stop and explain the exact state rather than discarding anything.
If the environment requires explicit permission to upload workspace code,
request that permission early. Never say a script is available on the server
until push succeeded and the remote was verified.

After push, verify the exact commit and files rather than relying only on a
successful-looking message:

```bash
git fetch origin main
git rev-parse HEAD
git rev-parse origin/main
git ls-tree -r --name-only origin/main -- scripts/
```

Herta update and entry-point check:

```bash
PROJECT_ROOT=/home/hwc/workspace/0701/0709
cd "$PROJECT_ROOT"
pwd
git status --short
git pull --ff-only
git rev-parse --short HEAD
test -f "$PROJECT_ROOT/scripts/<new-entry-point>.py"
```

If Herta has local modifications, do not pull over them or stash them without
permission. Report the dirty paths first. A Python error containing an absolute
missing path is a file-availability/path problem. Before blaming the Python
environment, run these checks:

```bash
pwd
git rev-parse --show-toplevel
git rev-parse --short HEAD
test -f "$PROJECT_ROOT/scripts/<new-entry-point>.py"
```

The user's experiment preferences are evidence-first and controlled:

- Read `HANDOFF.md` first, then `TESTING_METHODS.md`, then relevant README
  sections before changing commands or experiment design.
- Begin with one-variable controlled sweeps. Do not jump to interaction grids,
  combined ablations, or scheduler changes unless explicitly requested.
- Keep protocol, example order, extraction rule, token budget, models, `k`,
  `d`, warmup, and all non-tested parameters fixed.
- Use a new immutable output directory for every setting. Never combine
  different settings with `--resume`.
- Use 100 GSM8K examples only for screening; use all 1319 before making a
  quality claim, and preserve `predictions.jsonl` as well as `summary.json`.
- Server commands must be complete and copy-pasteable: fixed `PROJECT_ROOT`,
  `cd`, `pwd`, Conda activation, `PYTHONPATH`, absolute model/data/output paths,
  entry-point existence checks, and the full Python invocation.
- The local Windows workspace can support static checks but not genuine
  SGLang/FlashInfer GPU validation. State that limitation precisely.
