# ATLAS 0709 Handoff

Read this file first when opening a new work window. It is intentionally
action-oriented. Detailed historical notes are in `README.md` and
`TESTING_METHODS.md`.

## Current Research Decision (2026-07-14)

Read this section before changing the implementation or launching a long GPU
run. The active project is again **0709**. The 0713 fixed route-matrix/cascade
experiment is retained as a negative-result branch, not the main direction.

### Research priority

The next goal is to study how tree/forest policy changes the end-to-end
speed--quality frontier. Do not make "enable CUDA Graph" the research claim and
do not spend the next window optimizing 0713 before the policy opportunity is
established.

The intended paper question is narrower than generic adaptive tree decoding:

```text
Under a stochastic Edge--Cloud verification deadline, which root/node should
the weak Edge Drafter expand next, how wide/deep should it expand, and when
should it stop, so that useful candidate quality is maximized while latency,
wasted forest work, and persistent paged-KV cost stay within budget?
```

The promising 0709-specific mechanism is an interruptible/anytime forest plus
the physical paged-KV lifecycle: fork/COW, partial completion, cancellation at
a decode-depth boundary, selected-subtree promotion, and reuse across rounds.
Plain confidence-based dynamic `k,d`, optimal tree shape, asynchronous
draft/verify, and a generic quality--runtime knob all have close prior work.
In particular, the next literature comparison must include EAGLE-2, Sequoia,
OPT-Tree, Fuzzy Speculative Decoding, PEARL, and 2026 Saguaro/Speculative
Speculative Decoding. Two especially close 2026 Edge--Cloud papers are
PicoSpec (arXiv:2603.19133), which already presents an asynchronous pipelined
collaborative SD system, and *Speculation at a Distance*
(arXiv:2606.25091), which argues that WAN single-request latency gains occupy a
narrow regime and that multi-tenant capacity may be the stronger distributed
claim. Do not claim that 0709 is the first dynamic, asynchronous, or
Edge--Cloud speculative tree. Differentiate it through deadline-aware
interruptible forest construction, the persistent physical KV lifecycle, and
measured SLO/capacity tradeoffs.

CUDA Graph, synchronization removal, route-row/page-table cleanup, and cheaper
tail-page COW remain important **engineering enablers and fairness controls**.
The Drafter currently uses eager forwards because
`SGLangRunnerConfig.disable_cuda_graph=True`. Merely changing this flag is not
a paper-level contribution. A later graph-friendly policy implementation may
bucket logical decisions into a small set of static shapes, but policy research
comes first. Final performance conclusions must eventually use comparable
optimization levels because eager overhead can change which policy looks best.

### Current hardware constraint and the single-GPU proxy

There is currently only **one H800** available. Do not run the network Target
and Drafter concurrently on that GPU and interpret their contended wall time as
Edge--Cloud overlap. For the current research phase, load and measure them
sequentially in isolation on the same GPU. This is an analytical proxy, not a
claim about a deployed Edge--Cloud system.

The production ordering matters:

```text
exposed Stage-1 tree
-> send verification request
-> Stage-2 forest overlaps Cloud request/verify/response
-> select/promote/commit
```

Define the isolated medians for one round as:

```text
T_tree       = complete Stage-1 build-tree time (exposed before the request)
T_forest     = complete d-depth Stage-2 forest time (potential overlap window)
T_verify     = Target masked verify + Target KV commit
T_cloud(RTT) = T_verify + total request/response RTT and any RPC overhead
```

Use the following first-order model:

```text
target_fully_hidden = T_cloud(RTT) <= T_forest
exposed_cloud_wait  = max(0, T_cloud(RTT) - T_forest)
round_critical_path ~= T_tree + max(T_forest, T_cloud(RTT))
                       + Edge selection/promotion/control
```

The user expects a real weak Edge to take longer to perform four Drafter decode
depths than the Cloud round trip plus Target verify. Under that condition the
Target path is treated as hidden while researching tree policy. Test and report
the inequality; do not silently assume it. The isolated H800 proxy currently
omits real network, queueing, and heterogeneous-device effects. Sweep assumed
total RTT values now, then validate the model on the real Edge--Cloud platform
only after the tree strategy is stable.

`T_forest`, rather than `T_tree + T_forest`, is the correct hiding window.
Full-depth `T_forest` is an upper-bound window when the forest can be cancelled
early. Also retain each per-depth timing so an anytime policy can reason about
partial forests.

### Evidence already obtained

These H800 results are preliminary evidence and hypothesis generators, not
publication claims. The current primary AR comparison is the direct eager
SGLang baseline; the older ATLAS-control-path matched AR remains a diagnostic:

```text
k=3, d=4, prefix=8192, FP16, FlashInfer 0.6.12

native eager batch-3 AR x4:  25.096 ms mean
0709 Stage-1 tree:           28.978 ms mean   (+15.47%, paired)
native eager batch-9 AR x4:  26.806 ms mean
0709 Stage-2 forest:         31.686 ms mean   (+18.22%, paired)
Target masked tree verify:   24.224 ms
Target FI AR1 / AR2:         18.402 / 34.734 ms
Target linear verify:        28.484 ms
```

Thus the measured same-H800 `T_verify` is below the full four-depth
`T_forest`; before adding RPC overhead the decode-ready margin is about 7.46
ms. This only
predicts full hiding for total added RTT/overhead within that margin on this
proxy. A weak real Edge may provide a larger forest window.

A real two-process, one-H800 async regression test also confirms that the
Drafter forest path became faster, while exposing the expected same-GPU
contention. Against the pre-hot-path source, matched forest depth 1 and depth 2
steps were 16.24% and 24.97% faster. The Edge completed three rather than two
forest depths per round, and the fixed 128-token timeline was 7.78% shorter.
On 87 strictly matched GSM8K examples, default async E2E time was 5.75% lower
per round. Target black-box time was nevertheless 5.92% higher, consistent
with the faster Edge occupying the same GPU more aggressively. This is an
implementation regression result, not a deployed Edge--Cloud overlap claim;
see Section 3.

The serial GSM8K 100-example screen showed a real candidate quality--speed
signal, mostly controlled by fallback:

```text
fallback disabled:   69% EM, 77.80 output tok/s
threshold -1.00:     81% EM, 74.53 output tok/s
threshold -0.75:     84% EM, 71.10 output tok/s
threshold -0.50:     87% EM, 65.05 output tok/s
threshold -0.25:     88% EM, 54.77 output tok/s
```

The path-weight-alpha screen produced only 67--71% EM at about 77.5--77.9
tok/s. With 100 examples that spread is not enough to identify a winner.
Treat fallback as the stronger current hypothesis and alpha as unresolved.
Never tune on the GSM8K test subset and then present the same subset as final
evidence; create a development split, freeze policy, and use the full test set
for the final comparison.

The historical alpha=0.25, no-fallback serial profile attributed 59.63% of the
core time to the Drafter and 40.32% to the Target. A 2026-07-14 recheck at
`d4d54ec` did not show an end-to-end gain: on the 84 examples whose old and new
responses and work counts match exactly, Drafter time increased 3.20%, Target
time was effectively unchanged (+0.28%), and serial core time increased 2.02%.
See the detailed comparison and accounting limits in Section 5. This still
supports studying overlap, but it does not establish a serial-system speedup.

The 0713 route-matrix branch correctly implements shared-prefix/private-suffix
attention with LSE merge, but it did not beat the fair ordinary shared-page-
table baseline at the relevant route count. At `R=16`, fair ordinary versus
current 0713 was 115.200 versus 123.040 us with cache eviction, and 130.176
versus 155.728 us with a hot cache. Route-matrix attention became favorable
only at much higher route counts (for example `R=27` in the synthetic sweep).
Keep 0713 for reference/high-route-count experiments; do not replace 0709 with
it now.

### Exact and relaxed modes must remain separate

Standard lossless speculative decoding preserves the Target distribution; in
that mode the policy should change speed/acceptance, not output quality. The
current 0709 weighted Target path selection and fallback implement a relaxed
best-path mode and can change the output distribution. Its speed--quality
Pareto is scientifically meaningful, but it must not be described as lossless
or distribution-equivalent.

For future experiments, report two explicitly named modes if both are kept:

```text
exact/lossless: standard acceptance or sampling; quality should be invariant
relaxed:        best-path/fallback policy; report quality and distribution drift
```

### Immediate next-window plan: policy feasibility gate

Before implementing a learned/online policy, establish that per-request or
per-round adaptation has useful headroom:

1. Add/verify per-round logging for draft entropy, top-1/top-2 margin,
   cumulative route probability, Target-selected root/rank, accepted/committed
   tokens, useful and discarded forest nodes, completed forest depths, Target
   latency, route/KV/COW cost, and task result.
2. On a development set, run controlled fixed-policy sweeps over `k`, `d`,
   forest width/budget/depth cap, path scoring, and fallback threshold. Change
   one variable at a time first and preserve an immutable output directory per
   setting.
3. Combine isolated `T_tree`, per-depth `T_forest`, and `T_verify` with assumed
   RTT values to estimate the ideal Edge--Cloud critical path. Report raw
   component times as well as the estimate.
4. Construct a hindsight oracle that chooses the best policy per prompt/round
   under a latency or GPU-work budget. Compare its Pareto frontier with the best
   single fixed policy.
5. Continue to an online deadline-aware controller only if the oracle provides
   a material improvement beyond measurement noise and its decisions are
   predictable from features available before Target returns.
6. After the strategy is stable, implement the necessary eager/CUDA-Graph and
   KV/runtime optimizations, then repeat final measurements on a real weak Edge
   plus Cloud Target with controlled RTT.

A candidate policy objective is:

```text
maximize E[task/output quality]
         - lambda * E[end-to-end latency]
         - mu * E[wasted forest GPU work / KV pages]
```

or, more cleanly for evaluation, maximize quality subject to fixed p95 latency
and GPU-seconds/token budgets. Compare Pareto frontiers at equal quality, equal
latency, and equal compute instead of presenting only one operating point.

## Drafter Hot-Path Completion (2026-07-14)

The four requested Drafter engineering items and the fair native batch-AR
baseline are implemented, pushed, and validated on Herta at
`3caf628149c67fdf349532d1e0ac0dae41911087`.
Do not restart these changes from scratch. The relevant commit sequence is:

```text
0f9f8eb  Add Drafter hot-path counters and paged metadata builder
095b2ab  Fix paged metadata test fixture
f2e36a5  Use physical page metadata in Drafter attention
802bf99  Add page-metadata logits traces
e51953a  Avoid redundant partial-tail COW copies
732b68e  Update Drafter request rows incrementally
6a581ef  Batch Drafter frontier selection transfers
c3cb670  Add matched Drafter hot-path benchmarks
dab52c3  Remove Drafter bridge scalar synchronizations
f71275d  Make Drafter transfer accounting explicit
4cace1c  Fix Target scoring test fixture
434e088  Add paired native SGLang batch AR baseline
654cfe9  Keep benchmark tests dependency free
24d0b83  Match native AR req table dtype
bd25f83  Aggregate paired speed ratios geometrically
3caf628  Clarify native AR comparison scope
```

### What is now true

- Page-size-16 attention metadata is built from validated physical slot paths.
  `kv_indptr` counts physical pages, `kv_page_indices` contains physical page
  ids, and `kv_last_page_len` records each route's valid final-page length.
  CUDA layout checks use `torch._assert_async`; a runtime without that API now
  fails explicitly instead of silently introducing a host synchronization.
- GPU frontier selection preserves Global/per-root top-k and the deterministic
  `(score desc, parent route asc, rank asc, token asc)` tie order. All selected
  candidate fields are packed into one D2H transfer per depth. Standard
  production decode has zero additional bridge transfers per depth.
- Partial-tail COW is keyed by a CPU writer lease that follows prefix slices,
  fork/reconcile, private COW pages, page transitions, promotion, retain, and
  clear. Prefix slices ending at different offsets of the same physical page
  share one lease, so the first fork performs N-1 copies and private writers do
  not copy again until they enter a different shared partial page.
- Req rows track `written_length` and a retry-safe dirty start. An inherited
  ordinary decode writes one new slot. Fresh sibling rows require a full
  initialization; a reused row after COW patches only the changed tail plus the
  appended slot. Successful-only counters separate full-init, append, and COW
  patch elements.
- Newly allocated page ids stay on GPU during decode. Ownership ids are
  materialized in one batch only at prune/report boundaries. Prefix metadata
  and page-reference materialization are also phase-boundary transfers, not
  per-depth decode transfers.

The remaining visible transfer inventory must be described precisely:

```text
per tree/forest depth:
  frontier.py packed candidate materialization       1 batched D2H
  SGLangRoutePoolBridge production hot path           0 D2H

tree/forest initialization:
  initial top-k token/score materialization           1 batched D2H

commit/prune/report boundaries:
  prefix/page ownership/reference metadata            batched D2H as needed
```

`SGLANG_DEBUG_MEMORY_POOL` must remain disabled for the no-extra-sync result;
upstream allocator debug assertions may synchronize. The repository counters
are not a CUPTI/Nsight interceptor for third-party internals. Do not claim that
the whole round has only one synchronization: the proven statement is one
candidate D2H batch per decode depth in the standard production path.

### Correctness and regression evidence

Herta ran all test functions in the seven current test modules with the atlas
environment: **46/46 passed**. `compileall` and `git diff --check` passed.
Ruff is not installed in that environment and was not installed because this
task forbids adding dependencies.

The current real 1B smoke used `prefix=15, k=3, d=4, page=16`, normal physical
page metadata, and an independent HF full-history comparison. Tree, forest,
and the post-commit forest step all passed:

```text
tree:        max abs diff 0.0546875, every step top-1 match rate 1.0
forest:      max abs diff 0.0517578125, every step top-1 match rate 1.0
post-commit: max abs diff 0.0312500, top-1 match rate 1.0
promotion:   committed 4 KV tokens, retained 3 routes,
             released 12 req rows and 17 KV pages
bridge transfers at every tree/forest depth: 0 batches / 0 elements
```

Raw output:

```text
/home/hwc/workspace/0701/0709_outputs/sync_4cace1c/prefix15_hf.json
```

Earlier page-size A/B traces remain at:

```text
/home/hwc/workspace/0701/0709_outputs/page_metadata_802bf99_boundaries
/home/hwc/workspace/0701/0709_outputs/page_metadata_802bf99_long8192
```

For the 8192-token trace, page and legacy token metadata retain the same top-1
and route/token/rank selection structure. Float logits/scores differ within
the recorded tolerance. Use `summary_structural.json`; the older `summary.json`
also compared cumulative float scores and must not be described as a route
structure mismatch.

### Formal matched benchmark

The fixed condition was `k=3, d=4, prefix=8192, FP16, page=16`, warmup 1,
three measured iterations in each of **three independent processes**. Prefill
and setup are excluded from timing and counters. Tree is compared with batch-3
AR for four steps; forest is compared with batch-9 AR for four steps. Across
the nine raw samples:

```text
workload                         mean ms   median ms   ratio to matched AR
matched batch-3 AR x4              27.533      27.558       1.000x
Stage-1 tree x4                    28.868      28.954       1.048x
matched batch-9 AR x4              29.742      29.724       1.000x
Stage-2 forest x4                  31.671      31.671       1.065x
```

Independent-process means were:

```text
run 1: matched-tree 27.914, tree 29.331, matched-forest 30.221, forest 32.188
run 2: matched-tree 27.569, tree 28.940, matched-forest 29.746, forest 31.664
run 3: matched-tree 27.116, tree 28.332, matched-forest 29.258, forest 31.162
```

Aggregate per-depth means/medians were:

```text
depth                  1              2              3              4
matched tree AR   7.426/7.473    6.698/6.671    6.723/6.708    6.641/6.643
tree              7.661/7.717    7.065/7.069    7.078/7.085    7.018/7.027
matched forest AR 7.691/7.719    7.332/7.351    7.340/7.320    7.334/7.358
forest            8.004/8.027    7.866/7.862    7.903/7.886    7.850/7.869
```

The JSON files contain all raw total and per-depth samples, plus mean/median
counter samples:

```text
/home/hwc/workspace/0701/0709_outputs/matched_4cace1c/run1.json
/home/hwc/workspace/0701/0709_outputs/matched_4cace1c/run2.json
/home/hwc/workspace/0701/0709_outputs/matched_4cace1c/run3.json
```

For one tree/forest iteration, counters are deterministic across the three
processes:

```text
                              tree x4       forest x4
legacy-equivalent token indices   98,334        295,146
physical page indices              6,156         18,468
bridge D2H batches                      0              0
candidate D2H batches                   4              4
COW pages / tokens                   4 / 8        17 / 90
req-row elements                    57,364        163,978
  full-init elements                57,359        163,962
  append elements                        5             16
  COW-patch elements                     0              0
```

Thus physical metadata uses 93.74% fewer indices than token metadata. The
combined tree+forest req-row writes are 221,342 elements versus the old
full-rewrite estimate of 393,480, a 43.75% reduction. Necessary fresh-fork row
initialization remains O(sequence length); inherited ordinary appends are O(1).
COW patch count is zero in this exact route ordering because the reused row is
the source writer and fresh siblings are fully initialized, but dedicated
failure/retry and dirty-tail tests cover that path.

The old single-process `981f5c2` harness reported tree 28.290 ms and forest
33.281 ms. The current nine-sample means are respectively 2.04% slower and
4.84% faster, but that older run lacked the new fair matched-AR workloads and
multi-process repetition. Treat it only as a noisy historical before/after,
not a causal performance claim.

### Primary native eager batch-AR baseline

The old `matched_drafter_ar_*` controls still execute ATLAS log-softmax/top-k,
packed candidate D2H, Python route/node materialization, and route-row
lifecycle. They isolate policy differences inside the ATLAS control plane but
are **not** a native AR speed baseline. Keep them as sequential diagnostics;
do not use their 1.048x/1.065x ratios as the paper-facing AR comparison.

The primary baseline at `3caf628` is
`native_eager_sglang_batch_ar_page16`:

- same initialized SGLang 0.5.14 ModelRunner, weights, FP16, FlashInfer
  page-size-16 path, eager execution, allocator, prefix, batch, depth,
  sequence lengths, initial pending tokens, physical KV, and metadata counts;
- tree starts from the same `P` frontier with `B=3`; forest starts from the
  actual completed Stage-1 state at `P+d` with `B=9`;
- both arms exclude model load, prefill, logical frontier construction,
  initial req-row initialization, and initial partial-tail COW;
- native timed work is decode allocation, one batched req-row scatter,
  page-table construction, ModelRunner forward/KV write, and GPU argmax; it
  has no RouteState/KVTree/candidate materialization and keeps the next input
  on GPU;
- ATLAS still constructs frontier input ids from Python routes and performs
  subsequent branch/COW/top-k/materialization inside timing. Therefore the
  result is a decode-critical-path comparison including each algorithm's
  control plane, not a pure model-kernel comparison;
- one CUDA synchronization occurs before the four dependent steps and one at
  completion. There are no benchmark-injected barriers between depths;
- fresh setups are interleaved in ABBA blocks. Each process has 3 warmups and
  10 measured samples per arm; seeds 0, 1, and 2 were run as three independent
  processes. Ratios are paired within each ABBA block and aggregated in log
  space.

Across 30 samples per arm and 15 ABBA blocks:

```text
workload                         mean ms   median ms   paired ATLAS/native
native eager batch-3 AR x4         25.096      25.082       1.000x
decode-ready Stage-1 tree x4       28.978      28.992       1.155x
native eager batch-9 AR x4         26.806      26.674       1.000x
decode-ready Stage-2 forest x4     31.686      31.690       1.182x
```

The paired mean deltas are 3.882 ms for tree and 4.880 ms for forest. The
independent-process ratios were:

```text
run       tree/native    forest/native
1             1.1516           1.1746
2             1.1522           1.1839
3             1.1603           1.1882
```

With only three independent processes, process-level t intervals are wide and
must not be over-interpreted: tree `[1.1427, 1.1668]`, forest
`[1.1651, 1.1996]`. The raw ABBA blocks, process values, and aggregation are in
the summary file.

Correctness/audit evidence:

```text
real tree native vs matched AR, two depths:    max abs diff 0, top-1 100%
real forest native vs matched AR, two depths:  max abs diff 0, top-1 100%
native host transfers in every formal sample:  0 batches / 0 elements
native GPU argmax calls per formal sample:      4

tree metadata totals:    native=ATLAS=98,334 token / 6,156 page indices
forest metadata totals:  native=ATLAS=295,146 token / 18,468 page indices
```

Artifacts:

```text
/home/hwc/workspace/0701/0709_outputs/native_ar_alignment_bd25f83.json
/home/hwc/workspace/0701/0709_outputs/native_ar_baseline_3caf628/run1.json
/home/hwc/workspace/0701/0709_outputs/native_ar_baseline_3caf628/run2.json
/home/hwc/workspace/0701/0709_outputs/native_ar_baseline_3caf628/run3.json
/home/hwc/workspace/0701/0709_outputs/native_ar_baseline_3caf628/summary.json
```

This is a strong low-level eager GPU-resident AR baseline, not a complete
SGLang server/scheduler/radix-cache/CUDA-Graph benchmark. A later full-server
baseline must be reported separately rather than conflated with this result.

### Reproducible Herta commands and operational notes

For every command, keep writes in `/home/hwc/workspace/0701`. Herta Git needs
the user-provided `http_proxy` and `https_proxy` exports in that shell. Do not
commit the credentials, persist them in Git config, or modify another user's
directory.

The smoke invocation was:

```bash
PROJECT_ROOT=/home/hwc/workspace/0701/0709
cd "$PROJECT_ROOT"
pwd
source /home/hwc/miniconda3/etc/profile.d/conda.sh
conda activate atlas
export PYTHONPATH="$PROJECT_ROOT/src"
test -f "$PROJECT_ROOT/tools/smoke_flashinfer_paged_tree_forest.py"
python "$PROJECT_ROOT/tools/smoke_flashinfer_paged_tree_forest.py" \
  --model /home/hwc/models/Llama-3.2-1B-Instruct \
  --prompt-token-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --k 3 --d 4 --page-size 16 --context-length 8192 \
  --mem-fraction-static 0.65 --max-running-requests 128 \
  --max-total-tokens 65536 --check-hf-logits \
  --json-out /home/hwc/workspace/0701/0709_outputs/sync_4cace1c/prefix15_hf.json
```

The matched benchmark command is the same for `run1.json`, `run2.json`, and
`run3.json`, each launched as a new process:

```bash
PROJECT_ROOT=/home/hwc/workspace/0701/0709
cd "$PROJECT_ROOT"
pwd
source /home/hwc/miniconda3/etc/profile.d/conda.sh
conda activate atlas
export PYTHONPATH="$PROJECT_ROOT/src"
test -f "$PROJECT_ROOT/benchmarks/bench_atlas_0709_isolated_components.py"
python "$PROJECT_ROOT/benchmarks/bench_atlas_0709_isolated_components.py" \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-model /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --k 3 --d 4 --prefix-len 8192 --page-size 16 --dtype float16 \
  --prefill-chunk-size 8192 --mem-fraction-static 0.65 \
  --max-running-requests 256 --max-total-tokens 65536 \
  --warmup 1 --iters 3 --skip-native-batch-ar --skip-verify \
  --json-out /home/hwc/workspace/0701/0709_outputs/matched_4cace1c/run1.json
```

The primary native baseline command uses an even iteration count so every
sample belongs to a complete ABBA block. Repeat it in fresh processes with
`--pair-order-seed 0`, `1`, and `2`, changing only the output filename:

```bash
PROJECT_ROOT=/home/hwc/workspace/0701/0709
cd "$PROJECT_ROOT"
source /home/hwc/miniconda3/etc/profile.d/conda.sh
conda activate atlas
export PYTHONPATH="$PROJECT_ROOT/src"
export SGLANG_DEBUG_MEMORY_POOL=0

python "$PROJECT_ROOT/benchmarks/bench_atlas_0709_isolated_components.py" \
  --drafter-model /home/hwc/models/Llama-3.2-1B-Instruct \
  --target-model /home/hwc/models/Meta-Llama-3.1-8B-Instruct \
  --k 3 --d 4 --prefix-len 8192 --page-size 16 --dtype float16 \
  --prefill-chunk-size 8192 --context-length 8200 \
  --mem-fraction-static 0.75 --max-running-requests 256 \
  --max-total-tokens 65536 --warmup 3 --iters 10 \
  --pair-order-seed 0 --skip-verify \
  --json-out \
    /home/hwc/workspace/0701/0709_outputs/native_ar_baseline_3caf628/run1.json
```

Remaining engineering caveats: CUDA Graph is still intentionally disabled;
phase-boundary ownership materialization remains; generic/non-SGLang allocators
and arbitrary caller-provided output slots would need an explicit new-page
uniqueness contract; and an Nsight/CUPTI run is still required before making an
absolute claim about all third-party internal synchronizations.

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
Results:  /home/hwc/workspace/0701/0709_outputs
```

The result directory is outside `0709`, so copying the project does not copy
large benchmark artifacts. From the project root, use `../0709_outputs/...`.
Herta is shared by multiple users. Source and result writes for this work must
stay under `/home/hwc/workspace/0701`; do not inspect, alter, or clean another
user's workspace. The two explicitly listed read-only model paths are outside
that workspace by design.

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

**Current single-GPU warning:** keep the commands in this section for
functional testing and future multi-device/real-platform validation. They are
not the current performance protocol. With only one H800, use the isolated
sequential procedure in Section 8 and the timing model at the top of this file.
Same-GPU process overlap measures resource contention, not the intended weak
Edge plus Cloud deployment.

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

### 2026-07-14 single-H800 async regression test

This test deliberately ran the real two-process async implementation on one
H800: an 8B Target HTTP server and a 1B Drafter/coordinator both used GPU 0.
It is useful for implementation and scheduling regression detection, but the
two models contend for the same GPU and it is not a proxy for independent Edge
and Cloud accelerators.

The controlled forest A/B kept one current no-fallback Target server fixed and
alternated three fresh Edge processes per arm in A/B/B/A/A/B order. Arm A was
`981f5c2`; its `src/atlas_0709` tree is identical to the historical async
artifact commit `e7eeefd`. Arm B was current `a920912`. Every process used a
4,096-token repeated prefix, 128 generated tokens, exactly 32 rounds, no EOS,
`k=3`, `d=4`, FP16, page size 16, context 8192, one warmup, and the same
localhost Target. Raw `timeline.json` forest spans are valid step wall times:
the candidate D2H at each step boundary synchronizes Drafter CUDA work.

The fair comparison is per depth. Total forest time is not comparable because
the faster arm completes more depths before the Target response arrives.

| Actual async forest step | Pre-hot-path mean +/- SD, n=3 | Current mean +/- SD, n=3 | Latency change |
|---|---:|---:|---:|
| Depth 1 | 13.519610 +/- 0.974974 ms | 11.323960 +/- 0.030652 ms | -16.2405% |
| Depth 2 | 27.374190 +/- 0.146920 ms | 20.540105 +/- 0.147599 ms | -24.9654% |

There is no matched old depth-3/depth-4 sample: the old Target response almost
always stopped the forest after depth 2. The realized scheduling result was:

| Fixed 128-token async workload | Pre-hot-path mean +/- SD, n=3 | Current mean +/- SD, n=3 | Change |
|---|---:|---:|---:|
| Forest depths completed per round | 2.0000 +/- 0 | 3.0000 +/- 0 | +1 depth |
| Post-prune tree steps per round | 1.9375 +/- 0 | 0.96875 +/- 0 | approximately halved |
| Target verify black-box | 34.651784 +/- 0.748543 ms/round | 38.636421 +/- 0.039999 ms/round | +11.4991% |
| Edge timeline | 63.503402 +/- 0.203521 ms/round | 58.560619 +/- 0.040492 ms/round | -7.7835% |
| Edge timeline throughput | 62.989182 +/- 0.201504 tok/s | 68.305310 +/- 0.047241 tok/s | +8.4397% |

All old rounds returned before a full forest; current completed a full forest
in one of 32 rounds per process and otherwise usually reached depth 3. The
post-prune tree step itself was essentially unchanged (7.043994 versus
7.171059 ms), but only half as many were needed. Initial tree time is omitted:
it overlaps same-GPU Target prompt prefill and had large order-dependent
variance. Generated length, round count, and physical shapes were fixed, but
token outputs were not identical across arms; async control is wall-clock
dependent because the completed forest depth changes the handoff state.

The operational default was also rerun on GSM8K examples 0--99 with the
historical settings: weights `0.45,0.30,0.17,0.08`, fallback -0.5,
first-token -0.7, AR4, max-new 512, context/prefill 4096, page 16, FP16,
memory fraction 0.75, 32 requests, 32768 tokens, and one warmup. The old
single run is confirmed as `e7eeefd`; current is the mean of two `a920912`
runs, whose observable responses, token counts, and round counts reproduced
exactly.

Full trajectories drifted from 11,368 tokens / 2,880 rounds / 86% EM to
11,632 / 2,946 / 85% EM. The primary comparison therefore uses the 87 examples
whose response strings, generated-token counts, and round counts match old and
both current runs exactly: 9,737 tokens and 2,467 rounds on each side.

| Strictly matched default async, 87 examples | Old single run | Current mean +/- sample SD, n=2 | Change |
|---|---:|---:|---:|
| E2E time | 72.356745 ms/round | 68.193085 +/- 0.316992 ms/round | -5.7541% |
| E2E throughput | 54.547770 tok/s | 57.878914 +/- 0.269047 tok/s | +6.1069% |
| Edge timeline | 72.291850 ms/round | 68.128032 +/- 0.315522 ms/round | -5.7598% |
| Target verify black-box | 44.056417 ms/round | 46.665391 +/- 0.022246 ms/round | +5.9224% |
| Cloud/forest overlap | 43.098149 ms/round | 41.195650 +/- 0.025827 ms/round | -4.4143% |
| Exposed cloud wait | 0.000681 ms/round | 4.703310 +/- 0.015731 ms/round | +4.702628 ms |

On all 100 examples, current E2E was 68.249555 +/- 0.316807 ms/round and
57.853083 +/- 0.268548 tok/s, versus old 72.691091 ms/round and 54.301320
tok/s. Do not use that full result as a same-work model-only comparison because
13 response strings changed. The matched and full results agree that forest
acceleration reaches system E2E, while same-GPU Target slowdown and exposed
wait consume part of the gain. On separate Edge/Cloud devices the contention
term should differ and must be measured rather than inferred.

The Cloud interval is an Edge-observed black box containing thread scheduling,
JSON, localhost HTTP, Target queueing, verify, and response. Target scoring
synchronizes its main forward, but the final Target KV commit/fallback forward
still lacks an explicit tail synchronization. Treat the direct forest spans as
the clean Drafter evidence and the total timeline as the one-GPU system result;
do not turn the split into a publication-grade hardware attribution.

Artifacts:

```text
controlled A/B root: /home/hwc/workspace/0701/0709_outputs/async_ab_981f5c2_a920912/controlled
  old_run1 .. old_run3/{result.json,timeline.json,console.log}
  current_run1 .. current_run3/{result.json,timeline.json,console.log}
old GSM8K: /home/hwc/workspace/0701/0709_outputs/gsm8k_async_wrapper_reuse
current GSM8K: /home/hwc/workspace/0701/0709_outputs/async_ab_981f5c2_a920912/gsm8k_default
  current_run1, current_run2
```

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

### 2026-07-14 Drafter/Target speed recheck

This recheck used the same H800 and the historical serial accounting, with
GSM8K examples 0--99, `llama-8shot`, strict-marker extraction, FP16, `k=3`,
`d=4`, max-new-tokens 512, context/prefill 4096, page size 16, static memory
fraction 0.75, max-running-requests 32, max-total-tokens 32768, one warmup, and
a 128 MiB Target workspace. Current code was clean at `d4d54ec`.

The primary metric is milliseconds per serial round. Component tok/s uses the
same committed output-token count as numerator for both models; it is useful
for system accounting but is not either model's actual input-token throughput.
`+` in the change column means slower.

The clearest pure verify comparison is alpha=0.25 with both fallback triggers
disabled. Old and current full runs differ by 9 output tokens and 2 rounds, so
the table uses the 84 examples with byte-identical responses and exactly
matching work: 8,467 output tokens and 2,148 rounds in every run.

| Phase | Old single run (ms/round) | Current mean +/- sample SD, n=3 | Change |
|---|---:|---:|---:|
| Drafter tree + handoff | 28.310811 | 29.218059 +/- 0.352267 | +3.2046% |
| Target masked verify | 19.142821 | 19.196739 +/- 0.170154 | +0.2817% |
| Serial core total | 47.474217 | 48.434887 +/- 0.501142 | +2.0236% |

The corresponding committed-output rates were 139.233252 -> 134.922918
tok/s for Drafter, 205.915642 -> 205.348061 tok/s for Target, and 83.030466
-> 81.389397 tok/s overall. On all 100 examples, end-to-end rate including
prompt prefills was 77.894548 tok/s old versus 76.487470 +/- 0.732169 tok/s
current.

The operational default uses weights `0.45,0.30,0.17,0.08`, fallback threshold
-0.5, first-token threshold -0.7, and four fallback AR tokens. Its full
trajectory changed from 11,353 tokens / 2,877 rounds / 441 fallbacks to 11,627
/ 2,945 / 440; one current response reached the 512-token cap. The table
therefore uses the stricter 85-example intersection with byte-identical
responses and identical work: 9,562 tokens, 2,423 rounds, 362 fallbacks, and
1,448 fallback-appended tokens in every run.

| Phase | Old single run (ms/round) | Current mean +/- sample SD, n=2 | Change |
|---|---:|---:|---:|
| Drafter tree + handoff | 29.476867 | 30.255666 +/- 0.026156 | +2.6421% |
| Target verify + fallback | 28.981839 | 28.999736 +/- 0.015939 | +0.0618% |
| Serial core total | 58.478294 | 59.274950 +/- 0.042245 | +1.3623% |

The corresponding committed-output rates were 133.879478 -> 130.433386
tok/s for Drafter, 136.166222 -> 136.082207 tok/s for Target, and 67.483971
-> 66.577003 tok/s overall. The all-100 end-to-end rate was 64.026452 tok/s
old versus 63.569604 +/- 0.052380 tok/s current, but that aggregate includes
the changed generation trajectory and is not a model-only speed comparison.

Current repeats reproduced their observable trajectories exactly. The old
artifacts, however, contain predictions and summaries but no exact Git hash or
complete command manifest. Their serialized settings and reconstructed launcher
defaults match this recheck, so use them as historical references rather than a
same-binary controlled A/B. The old run is also a single observation; the
current sample deviations above describe repeatability and are not a
significance test.

The phase timers intentionally preserve the historical boundary. In-process
Target scoring synchronizes the main verify through a CPU copy, but there is no
explicit CUDA synchronization immediately after `target_client.verify()`.
Target KV-commit tail work, and the final fallback forward when applicable, can
therefore leak into the following Drafter handoff synchronization. Treat the
Drafter/Target split as a regression diagnostic. Serial total and end-to-end
time are the more robust system measurements; add a separately named,
explicitly synchronized timer before making publication-grade phase-attribution
claims, and do not mix that future timer with these historical numbers.

Artifacts:

```text
old no-fallback:  /home/hwc/workspace/0701/0709_outputs/gsm8k_serial_path_weight_sweep/alpha_0p25
old default:      /home/hwc/workspace/0701/0709_outputs/gsm8k_serial_wrapper_reuse
current root:     /home/hwc/workspace/0701/0709_outputs/serial_speed_d4d54ec
no-fallback runs: alpha0p25_nofallback_run1 .. run3
default runs:     current_default_run1 .. run2
```

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

For the current one-H800 research phase, component timing is the preferred
performance protocol. Stop any same-GPU Target server, then use:

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

The benchmark deliberately destroys the Drafter runtime and clears CUDA state
before loading the Target. Its `target_verify` timing includes Target KV commit
but not network RTT. Extract the hiding model without rerunning either model:

```bash
python - <<'PY'
import json

path = "/home/hwc/workspace/0701/0709_outputs/atlas_0709_isolated_components.json"
with open(path) as f:
    report = json.load(f)

tree_ms = report["build_tree"]["total"]["median_ms"]
forest_ms = report["build_forest"]["total"]["median_ms"]
verify_ms = report["target_verify"]["median_ms"]

print(f"T_tree   = {tree_ms:.3f} ms (exposed)")
print(f"T_forest = {forest_ms:.3f} ms (overlap window)")
print(f"T_verify = {verify_ms:.3f} ms (includes Target KV commit)")
for total_rtt_ms in (0, 1, 5, 10, 20):
    cloud_ms = verify_ms + total_rtt_ms
    wait_ms = max(0.0, cloud_ms - forest_ms)
    print(
        f"RTT={total_rtt_ms:>2} ms  cloud={cloud_ms:7.3f} ms  "
        f"hidden={str(cloud_ms <= forest_ms):5s}  exposed_wait={wait_ms:7.3f} ms"
    )
PY
```

This produces an analytical overlap estimate only. It does not include real
Edge slowdown, RPC serialization, Cloud queueing, interference, cancellation
latency, or network variance. Preserve the raw per-depth samples so the later
deadline-aware policy can be evaluated without pretending that a full forest
always completes.

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

## 12. 2026-07-15 KV Semantics and Greedy-Equivalence Audit

Commits `98e41fb` and `4e59f20` add correctness-only controls without changing
the default production selection or scheduler:

- Target `--route-selection-policy first_route` scores every input path but
  commits the first payload route. It rejects enabled fallback thresholds.
- `DistributedAtlasConfig.fixed_forest_depth` removes the Target wall-clock
  boundary from replay experiments; depths `0..d` can be tested separately.
- `--validate-state-alignment` checks coordinator, Drafter logical prefix,
  physical prefix slots, route node-to-slot mappings, and available req rows
  after every active handoff.
- The Edge now sends `selected_path_max_tokens`; Target applies the same output
  budget/EOS truncation before committing KV. Previously the terminal Target
  cache could contain the unused tail of a full `d`-token route.
- Target verifies route count/depth, unique route IDs, and exposes tokenizer
  vocabulary/special-token metadata through health.

Important definition: first route is not Drafter greedy AR when `k>1`. The
frontier is global beam search by cumulative draft log probability. Exact
greedy equivalence is therefore tested with `k=1`; k=3 correctness needs the
separate all-frontier HF check or a future protected greedy-spine oracle.

Herta unit/control result at `4e59f20`: all 51 test functions passed. The
`atlas` environment did not contain pytest, so a dependency-free collector
executed the same plain-assert tests and supplied the three monkeypatch fixtures.

Real H800 results (FP16, page size 16, deterministic-algorithms requested,
TF32 off, seed 0, no algorithmic sampling):

1. `k=1`, prefix length 15, 32 generated tokens, fixed forest depths
   `0,1,2,3,4`, two repeats per depth: all 10 ATLAS trajectories exactly
   matched independent Drafter paged greedy AR token-for-token. Every AR and
   ATLAS token hash was
   `cba8b40d8bd3cf4970845429ca32ca95beb1c5aace2ac15c15cc75eb9aa52d2c`.
2. Chat-template prompt `Hello`, 64 generated tokens, fixed depths `0,2,4`,
   two repeats per depth: all six trajectories and both AR repeats had the same
   token hash
   `fec14f9e97f45fd0577fd16ba0b5e59e8756d4e4fe39626431ed740d1d9dbc46`.
3. Terminal budget 17 with `d=4`, fixed depth 2, two repeats: exact AR match.
   The last round committed one token and Target prefix length changed
   `52 -> 53`, confirming the terminal over-commit fix.
4. Target first-route persistent-KV versus fresh full-prefix HF, 16 rounds,
   prefix `15 -> 79`: 16/16 next-token top-1 matched. Across all rounds,
   canonical-KV max abs diff was `0.044921875`; the final mean abs diff was
   `0.0009161857`; next-logit max abs diff was `0.03515625`.
5. Drafter `k=3,d=4` real tree/forest/post-commit HF alignment passed at the
   prefix-15 partial-page boundary. Maximum absolute logit differences were
   `0.0546875` (tree), `0.0517578125` (forest), and `0.03125`
   (post-commit); every checked frontier row had top-1 match rate `1.0`.

Artifacts:

```text
/home/hwc/workspace/0701/0709_outputs/kv_semantics_98e41fb/k1_async_vs_ar_short.json
/home/hwc/workspace/0701/0709_outputs/kv_semantics_98e41fb/k1_async_vs_ar_chat.json
/home/hwc/workspace/0701/0709_outputs/kv_semantics_98e41fb/k1_terminal_budget17.json
/home/hwc/workspace/0701/0709_outputs/kv_semantics_98e41fb/target_incremental_vs_fresh_4round.json
/home/hwc/workspace/0701/0709_outputs/kv_semantics_98e41fb/target_incremental_vs_fresh_16round.json
/home/hwc/workspace/0701/0709_outputs/kv_semantics_98e41fb/drafter_k3_hf_prefix15.json
```

Interpretation: the tested persistent KV/history paths are structurally and
semantically consistent. Small FP16 differences against fresh HF recomputation
remain and can flip a near-tied beam cutoff; these results do not prove bitwise
equivalence or all k=3 route-selection cases. Still missing for a publication
correctness appendix: forced selection of every k=3 route/rank, long prefixes
and all page boundaries, hundreds of allocation/recycle rounds, fixed-depth
fresh-process replay, stable exact-tie handling, and per-node Target margins.

Paper readiness after this audit: runtime prototype about 70%, correctness
evidence about 55%, core deadline-aware anytime policy about 15%, paper-level
evaluation about 20%, and overall main-conference readiness about 30%. The next
research gate is not another micro-optimization: first run a fixed-policy grid
plus hindsight deadline oracle. Only implement an online policy if the oracle
shows a meaningful quality/latency/GPU-work Pareto gap over the best fixed
policy. A main-paper result also needs multiple model pairs, real separated
Edge/Cloud hardware, RTT/jitter/queue sweeps, full tasks, strong sync/async tree
baselines, ablations, and paired confidence intervals.
