# ATLAS 0709

运行结果默认保存在项目同级目录 `../0709_outputs/`，不再放入项目目录，便于复制或同步 `0709` 源码。显式传入 `--output-dir`、`--json-out` 或 `--timeline-*` 时，以命令行路径为准。

For a compact continuation note when switching work windows, read
[`HANDOFF.md`](HANDOFF.md) first. Detailed test commands and historical results
are in [`TESTING_METHODS.md`](TESTING_METHODS.md).

ATLAS 0709 combines the strongest parts of the earlier prototypes:

- the real SGLang + FlashInfer paged tree/forest decode path from `0708`;
- the complete asynchronous tree → forest → prune → resume flow from `0701`;
- persistent physical KV on both Drafter and Target;
- one masked Target tree forward with shared-node deduplication;
- selected Target tree KV committed in place, with no full-prefix prefill between rounds.

## Runtime flow

1. Drafter and Target prefill the prompt concurrently.
2. Drafter keeps `k` routes and builds a depth-`d` stage-1 tree.
3. Target verifies the stage-1 tree asynchronously while Drafter advances `k²` forest routes.
4. Target deduplicates shared route prefixes, runs one ancestor-masked FlashInfer forward, selects a complete route by Target path probability, and compacts that route's KV into its committed prefix.
5. Drafter keeps only the selected stage-1 root's `k` stage-2 routes and promotes their existing physical KV:
   - a partial forest resumes as the remaining build-tree depths;
   - a completed forest becomes the next stage-1 tree directly.
6. No Drafter or Target prefix is rebuilt between rounds.

GPU kernels are not preempted mid-step. Forest cancellation is checked between decode steps.

The Drafter route pool supports normal multi-token pages. When a route forks
inside a partially filled page, ATLAS allocates a private destination page,
copies the valid tail-page KV through SGLang's `move_kv_cache`, and then appends
the child's pending token. Full prefix pages remain shared. Pruning keeps only
pages reachable from the selected prefix and retained routes, then returns
unreachable speculative pages to SGLang's allocator.

## Quality controls: weighted paths and Target fallback

The default behavior is the original best-of-`k` selection: Target scores all
stage-1 paths with the sum of their Target token log probabilities and commits
the highest-scoring path. The quality controls below are optional and preserve
that behavior when their flags are omitted.

### Front-loaded path scoring

For a depth-4 path with Target token log probabilities `l1, l2, l3, l4`,
`--path-score-weights` selects by:

```text
selection_score = w1*l1 + w2*l2 + w3*l3 + w4*l4
```

The weights are normalized internally, must be non-negative, and their count
must equal `--d`. The recommended depth-4 setting is:

```text
0.45, 0.30, 0.17, 0.08
```

This gives earlier tokens more influence because an error near the beginning
changes the context for every later token. It does not change the generated
tokens by itself; it changes which verified route is committed.

### Low-confidence Target fallback

When fallback is enabled, Target first scores the complete stage-1 paths. It
returns `fallback_ar` instead of selecting a draft path when either condition
is true:

```text
best_first_token_logprob < first_token_threshold
or
best_selection_score < fallback_threshold
```

The first-token check is evaluated first. Log probabilities are negative, so a
threshold closer to zero is stricter: `-0.50` rejects more paths than `-1.00`.
If fallback triggers, Target greedily generates at most
`--fallback-ar-tokens` tokens, commits them to its persistent KV, and returns
them to the Edge. The Edge then discards all current draft routes, appends the
AR tokens to `generated_text`, and rebuilds the Drafter stage-1 routes from the
new committed prefix. Unselected forest routes are never reused after a
fallback.

The current recommended configuration for `k=3, d=4` is:

```text
--path-score-weights 0.45,0.30,0.17,0.08
--fallback-threshold -0.50
--first-token-threshold -0.70
--fallback-ar-tokens 4
```

Thresholds are workload-dependent. Use these as starting points, not as
universal quality guarantees:

| Configuration | Expected behavior |
| --- | --- |
| omit both thresholds | Baseline best-of-`k`; no fallback |
| `-4.5`, `-6.0` | Very loose; usually almost no fallback |
| `-0.50`, `-0.70` | Strong fallback; roughly 20% of rounds fell back in the current full MT-Bench run |
| thresholds closer to `0` | More aggressive fallback; higher Target AR work and potentially higher quality |

Always record the actual fallback rate from `rounds_detail`; the rate can vary
by category and prompt. A fallback is reported with one of these reasons:
`first_token_below_threshold` or `path_score_below_threshold`.

### Target command with the recommended controls

```bash
python -m atlas_0709.target_server \
  --model /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
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

Confirm that the running server really received the controls:

```bash
curl http://127.0.0.1:18090/health | python -m json.tool
```

The response should include `score_weights`, `fallback_threshold`,
`first_token_threshold`, and `fallback_ar_tokens`. If those fields are
missing, restart the Target process; changing the Edge command alone does not
change Target-side scoring.

## Environment

Use the existing `atlas` Conda environment:

```bash
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate atlas
cd /workspace/projects/0709
export PYTHONPATH="$PWD/src:$PYTHONPATH"
pip install -e . --no-deps
```

The validated local environment contains PyTorch 2.11/CUDA 13.0,
FlashInfer 0.6.12, SGLang 0.5.14, and Transformers 5.8.1.

## Run the two-process system

Target process:

```bash
python -m atlas_0709.target_server \
  --model /workspace/models/Meta-Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 18090 \
  --k 3 \
  --d 4 \
  --page-size 16 \
  --dtype float16 \
  --device cuda
```

Drafter/coordinator process:

```bash
python -m atlas_0709.distributed_system \
  --drafter-model /workspace/models/Llama-3.2-1B \
  --target-url http://TARGET_HOST:18090 \
  --prompt "ATLAS asynchronous tree speculation." \
  --k 3 \
  --d 4 \
  --max-new-tokens 64 \
  --dtype float16 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --json-out atlas_0709_generation.json \
  --timeline-svg ../0709_outputs/atlas_0709_edge_blackbox.svg \
  --timeline-json ../0709_outputs/atlas_0709_edge_blackbox.json
```

For real overlap, run Drafter and Target on different GPUs or machines.

The timeline is measured entirely by the Edge coordinator. The Cloud is a
black box:

```text
Cloud black-box interval =
  Edge request submission
  -> network / queue / Target masked verify / Target KV commit
  -> Edge response receipt
```

No Target-reported internal duration is used in the SVG. Each round shows the
Cloud black-box interval against the real Edge forest steps, exposed Cloud wait,
and the local commit/prune/resume handoff. The SVG header reports how much of
the observed Cloud latency was hidden by Edge work.

Edge and Target prompt prefill start together, but they do not join at a
barrier. As soon as Edge prefill finishes, Edge builds the initial stage-1 tree.
If Target prompt prefill is still running after that tree is ready, Edge starts
the first stage-2 forest before the prompt acknowledgement. The first verify is
submitted after the Target prompt prefill completes and reuses whatever forest
depth Edge has already produced.

Forest and post-prune build-tree work is drawn one depth at a time. Each green
block is one `k^2` forest decode step; each blue block is one `k` build-tree
step after Target selection and pruning. Adjacent token steps are separated by
a visible white cut instead of an in-bar label. Each round label reports
`forest=<count>` and `post-prune tree=<count>`.

`--warmup-runs 1` is the default. The warmup runs in the same Drafter process
with the same `k`, `d`, page size, model runner, and Target connection. It
generates at least `page_size + 2*d` unmeasured tokens. This exercises prefill,
tree, forest, Target verify/commit, cross-round handoff, and the first physical
KV page recycle before measurement. The measured run then resets both KV
sessions with a fresh prompt prefill while retaining loaded kernels, allocator
kernels, memory pools, and CUDA context. Warmup time is excluded from both
timeline outputs. Use `--warmup-runs 0` only to measure first-forward cold-start
latency.

With the paths used on Herta:

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
  --dtype float16 \
  --context-length 9000 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --json-out atlas_0709_generation.json \
  --timeline-svg ../0709_outputs/atlas_0709_edge_blackbox.svg \
  --timeline-json ../0709_outputs/atlas_0709_edge_blackbox.json
```

## MT-Bench quality evaluation with DeepSeek

This evaluator runs the real two-process ATLAS 0709 system. It uses the
Drafter tokenizer's chat template, feeds the first assistant answer back into
the second-turn conversation, and stops on Llama's `<|eot_id|>` when available.
DeepSeek only judges saved answers after Drafter generation has finished.

The judge uses the current official `deepseek-v4-pro` Chat Completions API,
thinking mode, and JSON Output. MT-Bench reference answers embedded in the
FastChat question file are supplied to the judge when present. Failed API
requests are recorded separately and are never converted into zero scores.

Start the Target in terminal 1:

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
  --device cuda
```

For different machines, bind the Target with `--host 0.0.0.0` and replace
`127.0.0.1` below with the Target machine's reachable IP address.

Run a five-question end-to-end check in terminal 2:

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export DEEPSEEK_API_KEY='YOUR_DEEPSEEK_API_KEY'

python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-url http://127.0.0.1:18090 \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --k 3 \
  --d 4 \
  --max-new-tokens 1024 \
  --context-length 16384 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --dtype float16 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --max-questions 5 \
  --judge-model deepseek-v4-pro \
  --output-dir ../0709_outputs/mtbench_atlas_0709_smoke
```

Remove `--max-questions 5` and use a new output directory for the complete
80-question, two-turn run:

```bash
python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-url http://127.0.0.1:18090 \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --k 3 \
  --d 4 \
  --max-new-tokens 1024 \
  --context-length 16384 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --dtype float16 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --judge-model deepseek-v4-pro \
  --judge-thinking enabled \
  --judge-reasoning-effort high \
  --output-dir ../0709_outputs/mtbench_atlas_0709_full
```

Resume both phases after interruption by adding:

```bash
--resume-generate --resume-judge
```

Generate without spending API credit by adding `--skip-judge`. To judge those
saved answers later, rerun the same command with `--skip-generate
--resume-judge`. If the machine cannot download FastChat data, pass:

```bash
--questions-file /path/to/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
```

The output directory contains `generations.jsonl`, `judgments.jsonl`,
`generation_failures.jsonl`, `judge_failures.jsonl`, and `summary.json`.
`summary.json` reports overall, per-category, and per-turn scores plus
generation throughput. This is a DeepSeek-judged MT-Bench measurement and is
not directly comparable with the official GPT-4-judged leaderboard.

### Fair 1B and 8B AR quality baselines

Stop the ATLAS Target server first when it shares the same GPU. The AR script
loads the 1B and 8B models in separate sequential worker processes. This gives
each SGLang runner a fresh model-parallel/NCCL lifecycle and releases the first
model before loading the second. Both use SGLang ModelRunner, FlashInfer
attention, paged physical KV, strict batch-1 greedy decoding, their own
official chat template, the same questions, conversation history, stop policy,
and `max_new_tokens`.

```bash
python scripts/mtbench_ar_deepseek_eval.py \
  --model-1b /home/hwc/models/Llama-3.2-1B-Instruct \
  --model-8b /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --max-new-tokens 1024 \
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
  --output-dir ../0709_outputs/mtbench_ar_1b_8b
```

The AR and relaxed-spec scripts call the same blind pointwise judge function.
The judge sees no candidate name or decoding method. Both use the same rubric,
DeepSeek model/settings, MT-Bench references, and failed-request policy.
For lower judge variance, use `--judge-repeats 3` in both runs; this triples
judge API usage. Compare `judge.by_candidate` in each `summary.json`.

To get one result sooner, run one model at a time with `--candidate 1b` or
`--candidate 8b`. Keep the same high-quality judge settings for every
candidate. Separate output directories keep the two runs independent. Running
both processes concurrently is recommended only when they use different GPUs.

### Three-way quality comparison

The relaxed-spec evaluator can import the saved FlashInfer AR generations and
judge all three candidates in one experiment. Each answer is still judged
pointwise in an independent stateless API request to avoid A/B/C position and
comparison bias. Candidate identity is omitted from the judge prompt.

```bash
python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-url http://127.0.0.1:18090 \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --k 3 \
  --d 4 \
  --max-new-tokens 512 \
  --context-length 16384 \
  --dtype float16 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --max-questions 2 \
  --ar-generations ../0709_outputs/mtbench_flashinfer_ar_smoke/generations.jsonl \
  --ar-run-id c780caa89cd6d345 \
  --judge-model deepseek-v4-pro \
  --judge-thinking enabled \
  --judge-reasoning-effort high \
  --judge-max-tokens 4096 \
  --judge-repeats 3 \
  --output-dir ../0709_outputs/mtbench_three_way_smoke
```

The import fails fast unless all three candidates have exactly the same
question ids, turn text, reference answers, answer count, and
`max_new_tokens`. Outputs are written to `three_way_generations.jsonl`,
`three_way_judgments.jsonl`, and `three_way_summary.json`. The summary records
SHA-256 hashes for both artifacts. Every judgment records the system/user
prompt hashes, response id/model, backend `system_fingerprint`, finish reason,
usage, and exact request parameters. DeepSeek currently exposes no seed, so
exact API replay is not guaranteed; preserve these artifacts as the canonical
result and use repeated scores for a variance estimate.

## GSM8K exact-match evaluation

The GSM8K evaluator uses the same real FlashInfer paged AR and two-process
ATLAS paths as the MT-Bench scripts. Its default `llama-8shot` protocol matches
lm-evaluation-harness `gsm8k_cot_llama`: Meta's eight CoT demonstrations are
represented as multi-turn chat messages, decoding is greedy, and responses end
with `The final answer is [answer]`. It reports exact-match accuracy, extraction failures,
per-example latency, and generation throughput. Outputs are `predictions.jsonl`,
`failures.jsonl`, and `summary.json`.

Run the two AR baselines (stop a same-GPU Target server first):

```bash
bash scripts/run_gsm8k_1b_ar.sh
bash scripts/run_gsm8k_8b_ar.sh
```

For ATLAS, first start the 8B Target exactly as in the MT-Bench section, then:

```bash
bash scripts/run_gsm8k_atlas.sh
```

For a single-GPU scheduling baseline, use the serial ATLAS launcher:

```bash
bash scripts/run_gsm8k_atlas_serial.sh
```

The serial launcher keeps the same 1B Drafter, 8B Target, weighted path
scoring, and Target fallback configuration, but enforces this order for every
round:

```text
Drafter build tree (d steps)
    -> Target masked tree verify
    -> select or fallback AR
    -> rebuild Drafter prefix
```

It never builds forest, never overlaps Drafter and Target work, and does not
reuse stage-2 Drafter KV across rounds. This is a quality-matched scheduling
baseline, not the production asynchronous algorithm. Its throughput is the
appropriate comparison for measuring whether the asynchronous implementation
is slowed by single-GPU contention.

The same mode is available for MT-Bench:

```bash
python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-url http://127.0.0.1:18090 \
  --questions-file /home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl \
  --execution-mode serial \
  --k 3 --d 4 \
  --max-new-tokens 512 \
  --output-dir ../0709_outputs/mtbench_atlas_serial_tree
```

The generated artifact records `execution_mode=serial` and
`build_forest=false`. The default `--execution-mode async` remains unchanged.

Serial generation also records three throughput measurements with the same
numerator, the number of tokens committed to the generated response:

```text
drafter_tokens_per_second = generated_tokens / drafter_elapsed_s
target_tokens_per_second  = generated_tokens / target_elapsed_s
overall_tokens_per_second = generated_tokens / total_elapsed_s
```

`drafter_elapsed_s` includes each round's Drafter prefix rebuild, route
initialization, and complete tree build. `target_elapsed_s` is the accumulated
verify RPC interval, including optional Target fallback AR and RPC latency.
`total_elapsed_s` spans the serial generation loop and exposes controller
overhead in addition to both phases. Target prompt prefill is reported as
`target_prefill_elapsed_s` and deliberately excluded from all three rates.
GSM8K `summary.json` and MT-Bench candidate summaries contain a weighted
`serial_speed` aggregate computed as total generated tokens divided by summed
phase time, rather than an unweighted mean of per-turn rates.

The launchers default to
`/home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl`.
Use environment variables to override paths/settings, and append evaluator
arguments directly. For example:

```bash
MODEL_1B=/path/to/1b OUTPUT_DIR=../0709_outputs/gsm8k_1b_smoke \
  bash scripts/run_gsm8k_1b_ar.sh --max-examples 10

TARGET_URL=http://TARGET_HOST:18090 OUTPUT_DIR=../0709_outputs/gsm8k_atlas_full \
  bash scripts/run_gsm8k_atlas.sh --resume
```

The published Meta reference scores under this protocol are 44.4 for Llama
3.2 1B Instruct and 84.5 for Llama 3.1 8B Instruct (`em_maj1@1`). A missing
final-answer phrase falls back to the last number unless `--strict-marker` is
set. Use `--protocol zero-shot` only for a separate zero-shot experiment; its
score is not directly comparable with those published numbers.

## Validation

CPU/control-flow checks:

```bash
pytest -q
python tools/smoke_mock.py
python tools/smoke_reuse_handoff.py
```

Real Drafter route-KV alignment check with normal 16-token pages:

```bash
python tools/smoke_flashinfer_paged_tree_forest.py \
  --model /home/hwc/models/Llama-3.2-1B-Instruct \
  --prompt "ATLAS route KV alignment." \
  --k 3 \
  --d 4 \
  --page-size 16 \
  --context-length 8192 \
  --max-running-requests 128 \
  --max-total-tokens 65536 \
  --check-hf-logits
```

The check covers every initial tree and forest depth plus one forest step after
committing a stage-1 route and promoting its retained second-stage KV. It fails
if any FlashInfer frontier falls outside the configured HF logit tolerance.
The report also records copied COW pages/tokens.

Two-round real Target KV commit check:

```bash
python tools/smoke_target_incremental_commit.py \
  --model /workspace/models/Llama-3.2-1B \
  --prompt-token-ids 1,2,3,4,5,6,7,8 \
  --d 4 \
  --page-size 16 \
  --dtype float16
```

The smoke compares logits from the committed masked-tree KV against a fresh HF
prefill, then performs a second verify using only the persisted Target paged KV.

## Isolated component timing

Stop the network Target server first if it shares the benchmark GPU. This
benchmark uses synthetic token ids, excludes all prefills from timing, and runs
tree, forest, and verify sequentially so they cannot interfere with each other:

```bash
python benchmarks/bench_atlas_0709_isolated_components.py \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-model /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --k 3 \
  --d 4 \
  --prefix-len 8192 \
  --repeat-token-id 42 \
  --shared-path-tokens 2 \
  --dtype float16 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup 3 \
  --iters 10 \
  --json-out ../0709_outputs/atlas_0709_isolated_components.json
```

The report uses arithmetic mean latency as the primary number and also retains
the median and raw samples. It includes:

```text
Drafter FlashInfer paged AR decode, batch 1, one token
complete d-step build tree plus every individual depth
complete d-step build forest plus every individual depth
Target Direct FlashInfer AR decode, batch 1, one token
Target masked tree verify
```

Target verify includes selected-path Target KV commit but excludes Target
prefill and network RTT. Tree/forest setup and all prefills are outside the
timed regions. Semantic quality and logit alignment are intentionally not
checked.

## Current scope

- Drafter tree/forest execution remains ordinary SGLang FlashInfer paged batch,
  preserving the fastest working `0708` path.
- Target tree attention is shared-node masked verify; Target prefix and selected
  suffix KV persist across rounds.
- One Target server currently owns one active generation session at a time.
- Forest cancellation occurs between full-model decode steps, not inside a
  running CUDA kernel.
- Three-level Cascade for the forest is not enabled yet; it should be added only
  after a same-model, same-layout benchmark beats the current paged batch path.
