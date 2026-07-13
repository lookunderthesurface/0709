# ATLAS 0709 Test Methods

This file summarizes the current ATLAS 0709 testing workflow for continuing work.
For a compact cross-window handoff covering runtime state and experiment
preferences, read [`HANDOFF.md`](HANDOFF.md) first.

## Environment

```bash
cd /home/hwc/workspace/0701/0709
conda activate atlas
export PYTHONPATH="$PWD/src:$PYTHONPATH"

export DRAFTER=/home/hwc/models/Llama-3.2-1B-Instruct
export TARGET=/home/hwc/models/Meta-Llama-3.1-8B-Instruct
export QUESTIONS=/home/hwc/workspace/thirdparty/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl
```

## Current System

- Drafter: 1B, SGLang + FlashInfer paged decode.
- Target: 8B, Direct FlashInfer full-model masked tree verify.
- Main flow: build tree -> build forest + target verify -> commit selected stage-1 path.
- Optional quality controls:
  - Weighted path scoring: `--path-score-weights 0.45,0.30,0.17,0.08`
  - Low-confidence fallback: Target generates 4 greedy AR tokens.
  - First-token gate.
- If thresholds are omitted, the old best-of-N path selection still runs.

The quality-control flags are Target-side settings. They must be passed when
starting `atlas_0709.target_server`; adding them only to the Edge command has
no effect.

For `d=4`, the weighted score is:

```text
0.45 * logprob(token_1)
+ 0.30 * logprob(token_2)
+ 0.17 * logprob(token_3)
+ 0.08 * logprob(token_4)
```

Weights are normalized internally and must contain exactly `d` non-negative
values. A threshold compares against this weighted score when weights are
enabled, or against the unweighted path-logprob sum when they are omitted.

Fallback is triggered in this order:

1. `best_first_token_logprob < first_token_threshold`
2. `best_selection_score < fallback_threshold`

Because log probabilities are negative, a value closer to zero is stricter.
For example, `-0.50` triggers more often than `-1.00`. On fallback, Target
greedily generates up to `fallback_ar_tokens` tokens and commits them in place.
The Edge drops speculative routes but preserves committed Drafter prefix KV,
appends the known suffix with one multi-token EXTEND, and builds the next
stage-1 tree from the final logits without historical re-prefill.

## Start Target Server

Original selection behavior:

```bash
python -m atlas_0709.target_server \
  --model $TARGET \
  --host 0.0.0.0 \
  --port 18090 \
  --k 3 \
  --d 4 \
  --prefix-len 16384 \
  --page-size 16 \
  --dtype float16 \
  --device cuda
```

Strong fallback test configuration:

```bash
python -m atlas_0709.target_server \
  --model $TARGET \
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

The serial quality baseline is launched with:

```bash
bash scripts/run_gsm8k_atlas_serial.sh
```

It uses the same Target scoring and fallback flags, but its round order is
strictly `build_tree -> verify -> commit/fallback -> reuse_prefix_kv`. It does
not build forest, overlap Target work with Drafter work, or reuse stage-2
Drafter KV. The committed Drafter prefix KV persists across rounds.
Use `--backend atlas_serial` directly with `scripts/gsm8k_eval.py` if custom
environment variables are not convenient.

For MT-Bench, add:

```text
--execution-mode serial
```

to `scripts/mtbench_relaxed_spec_deepseek_eval.py`. The default mode remains
`async`.

Threshold reference for `k=3, d=4`:

| Thresholds | Use |
| --- | --- |
| omitted | Original best-of-N behavior; fallback disabled |
| `--fallback-threshold -4.5 --first-token-threshold -6.0` | Very loose; normally produces almost no fallback |
| `--fallback-threshold -0.50 --first-token-threshold -0.70` | Recommended strong fallback starting point; approximately 20% fallback was observed in the full 80-question run |
| closer to `0` | More aggressive fallback; more Target AR work |

These are empirical starting points, not model-independent guarantees. Tune
them only after collecting `best_selection_score` and
`best_first_token_logprob` from a representative prompt set.

Confirm the running Target configuration:

```bash
curl http://127.0.0.1:18090/health | python -m json.tool
```

Expected metadata fields:

```json
"score_weights": [0.45, 0.3, 0.17, 0.08],
"fallback_threshold": -0.5,
"first_token_threshold": -0.7,
"fallback_ar_tokens": 4
```

## Single Prompt Trace

Use this to inspect round-level fallback decisions, path scores, and commit behavior.

```bash
python -m atlas_0709.distributed_system \
  --drafter-model $DRAFTER \
  --target-url http://127.0.0.1:18090 \
  --prompt "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions." \
  --k 3 \
  --d 4 \
  --max-new-tokens 128 \
  --eos-token-id 128009 \
  --dtype float16 \
  --context-length 16384 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --fallback-ar-tokens 4 \
  --json-out ../0709_outputs/fallback_single_trace.json
```

Fallback statistics:

```bash
python - <<'PY'
import json
data = json.load(open("../0709_outputs/fallback_single_trace.json", encoding="utf-8"))
rounds = data["rounds"]
fallback = [r for r in rounds if r.get("decision") == "fallback_ar"]
print("rounds:", len(rounds))
print("fallback:", len(fallback))
print("fallback_rate:", len(fallback) / len(rounds) if rounds else 0)
print("first 10 decisions:", [r.get("decision") for r in rounds[:10]])
for r in fallback[:20]:
    print(r.get("fallback_reason"), r.get("fallback_token_ids"))
PY
```

## MT-Bench Smoke Test

Use this for a fast two-question quality and fallback sanity check.

```bash
export DEEPSEEK_API_KEY='YOUR_KEY'

python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model $DRAFTER \
  --target-url http://127.0.0.1:18090 \
  --questions-file $QUESTIONS \
  --k 3 \
  --d 4 \
  --max-new-tokens 512 \
  --context-length 16384 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --dtype float16 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --fallback-ar-tokens 4 \
  --max-questions 2 \
  --judge-model deepseek-v4-pro \
  --judge-thinking enabled \
  --judge-reasoning-effort high \
  --judge-max-tokens 4096 \
  --output-dir ../0709_outputs/mtbench_fallback_trace_smoke_k3d4
```

## Full MT-Bench 80 Questions

```bash
python scripts/mtbench_relaxed_spec_deepseek_eval.py \
  --drafter-model $DRAFTER \
  --target-url http://127.0.0.1:18090 \
  --questions-file $QUESTIONS \
  --k 3 \
  --d 4 \
  --max-new-tokens 512 \
  --context-length 16384 \
  --page-size 16 \
  --prefill-chunk-size 8192 \
  --dtype float16 \
  --mem-fraction-static 0.75 \
  --max-running-requests 256 \
  --max-total-tokens 65536 \
  --warmup-runs 1 \
  --fallback-ar-tokens 4 \
  --judge-model deepseek-v4-pro \
  --judge-thinking enabled \
  --judge-reasoning-effort high \
  --judge-max-tokens 4096 \
  --output-dir ../0709_outputs/mtbench_fallback_strong_full_k3d4_t050_ft070
```

## Fallback And Path Probability Statistics

Newer `generations.jsonl` files store per-turn `generation_stats[*].rounds_detail` with:

- `decision`
- `fallback_reason`
- `fallback_token_ids`
- `best_selection_score`
- `best_target_logprob`
- `best_first_token_logprob`
- `target_scores`

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
from collections import Counter

path = Path("../0709_outputs/mtbench_fallback_strong_full_k3d4_t050_ft070/generations.jsonl")
rounds = []
for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    for stat in row.get("generation_stats", []):
        rounds.extend(stat.get("rounds_detail", []))

fallback = [r for r in rounds if r.get("decision") == "fallback_ar"]
scores = [r["best_selection_score"] for r in rounds if r.get("best_selection_score") is not None]
firsts = [r["best_first_token_logprob"] for r in rounds if r.get("best_first_token_logprob") is not None]

def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs)-1, int((len(xs)-1)*p))]

print("rounds:", len(rounds))
print("fallback:", len(fallback))
print("fallback_rate:", len(fallback) / len(rounds) if rounds else None)
print("fallback_reasons:", dict(Counter(r.get("fallback_reason") for r in fallback)))
print("best_selection_score p10/p50/p90:", pct(scores, .1), pct(scores, .5), pct(scores, .9))
print("best_first_token_logprob p10/p50/p90:", pct(firsts, .1), pct(firsts, .5), pct(firsts, .9))
PY
```

## Score Summary

```bash
python - <<'PY'
import json
p = "../0709_outputs/mtbench_fallback_strong_full_k3d4_t050_ft070/summary.json"
s = json.load(open(p, encoding="utf-8"))
c = s["generation"]["by_candidate"]["atlas_0709_relaxed_spec"]
j = s["judge"]["by_candidate"]["atlas_0709_relaxed_spec"]
print("questions_selected:", s["questions_selected"])
print("generation_failures:", s["generation_failures"])
print("judge_failures:", s["judge"]["failures"])
print("overall_mean:", j["mean"])
print("tokens_per_second:", c["mean_tokens_per_second"])
print("mean_elapsed_s_per_turn:", c["mean_elapsed_s_per_turn"])
print("category_scores:")
for k, v in j["by_category"].items():
    print(k, v["mean"])
PY
```

## Known Results

Previous full 80-question three-model comparison:

```text
ar_1b mean: 4.328
atlas_0709_relaxed_spec mean: 5.409
ar_8b mean: 6.256
```

Later two-question fallback smoke:

```text
weak fallback config -4.5/-6.0:
score mean ~= 7.125

new trace two questions:
score mean ~= 8.0
rounds: 432
fallback: 0
best_selection_score p10/p50/p90 ~= -0.856 / -0.173 / -0.001
best_first_token_logprob p10/p50/p90 ~= -0.948 / -0.020 / 0
```

`-4.5/-6.0` is too loose and usually does not trigger fallback. Strong fallback test values:

```text
fallback_threshold = -0.50
first_token_threshold = -0.70
```

## Notes

- If `rounds_detail` is empty, the remote script is not synced to the newer `scripts/mtbench_deepseek_eval.py`.
- If `/health` does not show threshold fields, restart the Target server with the new flags.
- DeepSeek API judging is not exactly reproducible. The summary reports `exact_api_replay_guaranteed: false`.
- A DeepSeek API key was exposed during testing. Rotate the key before formal experiments.
