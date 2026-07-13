from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TimedResult:
    median_ms: float
    mean_ms: float
    samples_ms: list[float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark clean ATLAS component shapes with ordinary Hugging Face KV decode. "
            "This is not a shared-KV or Cascade benchmark."
        )
    )
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--prompt", default="ATLAS clean component speed benchmark.")
    parser.add_argument("--prompt-token-ids", default=None)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--repeat-token-id", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--stateful-cache",
        action="store_true",
        help=(
            "Reuse and mutate one cache across samples. Faster to run but not suitable "
            "for fair component comparisons. By default each measured sample starts "
            "from a fresh prefilled cache and excludes prefill time."
        ),
    )
    parser.add_argument("--json-out", default=None)
    return parser


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def parse_prompt_token_ids(
    *,
    tokenizer,
    prompt: str,
    prompt_token_ids: str | None,
    prefix_len: int,
    repeat_token_id: int,
) -> list[int]:
    if prompt_token_ids:
        base = [int(part.strip()) for part in prompt_token_ids.split(",") if part.strip()]
    elif repeat_token_id is not None:
        base = [int(repeat_token_id)]
    else:
        base = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
    if not base:
        raise RuntimeError("empty prompt token list")
    repeats = (int(prefix_len) + len(base) - 1) // len(base)
    return (base * repeats)[: int(prefix_len)]


def load_model(model_path: str, *, dtype: torch.dtype, device: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return tokenizer, model


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_stateful(fn: Callable[[], None], *, warmup: int, iters: int) -> TimedResult:
    for _ in range(warmup):
        fn()
        sync_cuda()
    samples: list[float] = []
    for _ in range(iters):
        sync_cuda()
        start = time.perf_counter()
        fn()
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
    return TimedResult(
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.fmean(samples)),
        samples_ms=samples,
    )


def measure_with_setup(
    setup_fn: Callable[[], object],
    run_fn: Callable[[object], None],
    *,
    warmup: int,
    iters: int,
) -> TimedResult:
    for _ in range(warmup):
        state = setup_fn()
        sync_cuda()
        run_fn(state)
        sync_cuda()
        del state
    samples: list[float] = []
    for _ in range(iters):
        state = setup_fn()
        sync_cuda()
        start = time.perf_counter()
        run_fn(state)
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
        del state
    return TimedResult(
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.fmean(samples)),
        samples_ms=samples,
    )


@torch.inference_mode()
def model_forward(model, input_ids: torch.Tensor, past_key_values=None):
    kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    try:
        return model(**kwargs, return_legacy_cache=True)
    except TypeError:
        return model(**kwargs)


@torch.inference_mode()
def prefill_cache(model, prompt_token_ids: Sequence[int], *, batch_size: int, device: str):
    row = torch.tensor([int(token_id) for token_id in prompt_token_ids], dtype=torch.long, device=device)
    input_ids = row.unsqueeze(0).repeat(int(batch_size), 1)
    output = model_forward(model, input_ids)
    return output.past_key_values, output.logits[:, -1, :].detach()


@torch.inference_mode()
def decode_one(model, past_key_values, input_ids: torch.Tensor):
    output = model_forward(model, input_ids[:, None], past_key_values=past_key_values)
    return output.past_key_values, output.logits[:, -1, :].detach()


def cache_select_batch(cache, indices: torch.Tensor, *, old_batch_size: int):
    if hasattr(cache, "batch_select_indices"):
        result = cache.batch_select_indices(indices)
        return cache if result is None else result
    return _select_legacy_cache(cache, indices, old_batch_size=old_batch_size)


def _select_legacy_cache(value, indices: torch.Tensor, *, old_batch_size: int):
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == int(old_batch_size):
            return value.index_select(0, indices)
        return value
    if isinstance(value, tuple):
        return tuple(_select_legacy_cache(item, indices, old_batch_size=old_batch_size) for item in value)
    if isinstance(value, list):
        return [_select_legacy_cache(item, indices, old_batch_size=old_batch_size) for item in value]
    raise TypeError(
        f"unsupported past_key_values type {type(value).__name__}; "
        "try a transformers version that supports return_legacy_cache=True"
    )


def next_argmax_inputs(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1).to(dtype=torch.long)


def init_topk_inputs(logits: torch.Tensor, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs, token_ids = torch.topk(torch.log_softmax(logits[0], dim=-1), k=int(k), dim=-1)
    return token_ids.to(dtype=torch.long), logprobs.detach()


def select_global_tree_candidates(
    logits: torch.Tensor,
    route_scores: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logprobs, token_ids = torch.topk(torch.log_softmax(logits, dim=-1), k=int(k), dim=-1)
    scores = route_scores[:, None] + logprobs
    flat_scores = scores.flatten()
    selected_scores, flat_indices = torch.topk(flat_scores, k=int(k), dim=-1)
    parent_indices = torch.div(flat_indices, int(k), rounding_mode="floor").to(dtype=torch.long)
    rank_indices = torch.remainder(flat_indices, int(k)).to(dtype=torch.long)
    selected_tokens = token_ids[parent_indices, rank_indices].to(dtype=torch.long)
    return parent_indices, selected_tokens, selected_scores.detach()


def select_forest_candidates_by_root(
    logits: torch.Tensor,
    route_scores: torch.Tensor,
    root_ids: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logprobs, token_ids = torch.topk(torch.log_softmax(logits, dim=-1), k=int(k), dim=-1)
    selected_parent_chunks: list[torch.Tensor] = []
    selected_token_chunks: list[torch.Tensor] = []
    selected_score_chunks: list[torch.Tensor] = []
    selected_root_chunks: list[torch.Tensor] = []
    for root_id in torch.unique(root_ids, sorted=True):
        route_indices = (root_ids == root_id).nonzero(as_tuple=False).flatten()
        scores = route_scores[route_indices, None] + logprobs[route_indices]
        flat_scores = scores.flatten()
        selected_scores, flat_indices = torch.topk(flat_scores, k=int(k), dim=-1)
        local_parent = torch.div(flat_indices, int(k), rounding_mode="floor").to(dtype=torch.long)
        rank_indices = torch.remainder(flat_indices, int(k)).to(dtype=torch.long)
        parent_indices = route_indices[local_parent]
        selected_parent_chunks.append(parent_indices)
        selected_token_chunks.append(token_ids[parent_indices, rank_indices].to(dtype=torch.long))
        selected_score_chunks.append(selected_scores.detach())
        selected_root_chunks.append(torch.full((int(k),), int(root_id), dtype=torch.long, device=root_ids.device))
    return (
        torch.cat(selected_parent_chunks, dim=0),
        torch.cat(selected_token_chunks, dim=0),
        torch.cat(selected_score_chunks, dim=0),
        torch.cat(selected_root_chunks, dim=0),
    )


def bench_ar_decode(
    model,
    prompt_token_ids: Sequence[int],
    *,
    batch_size: int,
    steps: int,
    device: str,
    warmup: int,
    iters: int,
    stateful_cache: bool,
) -> TimedResult:
    def setup():
        past, logits = prefill_cache(model, prompt_token_ids, batch_size=batch_size, device=device)
        return past, next_argmax_inputs(logits)

    def run_state(state) -> None:
        past, input_ids = state
        for _ in range(int(steps)):
            past, logits_i = decode_one(model, past, input_ids)
            input_ids = next_argmax_inputs(logits_i)

    if not stateful_cache:
        return measure_with_setup(setup, run_state, warmup=warmup, iters=iters)

    past, input_ids = setup()

    def run() -> None:
        nonlocal past, input_ids
        for _ in range(int(steps)):
            past, logits_i = decode_one(model, past, input_ids)
            input_ids = next_argmax_inputs(logits_i)

    return measure_stateful(run, warmup=warmup, iters=iters)


def bench_stage1_tree(
    model,
    prompt_token_ids: Sequence[int],
    *,
    k: int,
    depth: int,
    device: str,
    warmup: int,
    iters: int,
    stateful_cache: bool,
) -> TimedResult:
    def setup():
        past, logits = prefill_cache(model, prompt_token_ids, batch_size=k, device=device)
        input_ids, route_scores = init_topk_inputs(logits, k=k)
        return past, input_ids, route_scores

    def run_state(state) -> None:
        past, input_ids, route_scores = state
        for _ in range(int(depth)):
            old_batch = int(input_ids.numel())
            past, logits_i = decode_one(model, past, input_ids)
            parent_indices, input_ids, route_scores = select_global_tree_candidates(logits_i, route_scores, k=k)
            past = cache_select_batch(past, parent_indices, old_batch_size=old_batch)

    if not stateful_cache:
        return measure_with_setup(setup, run_state, warmup=warmup, iters=iters)

    past, input_ids, route_scores = setup()

    def run() -> None:
        nonlocal past, input_ids, route_scores
        for _ in range(int(depth)):
            old_batch = int(input_ids.numel())
            past, logits_i = decode_one(model, past, input_ids)
            parent_indices, input_ids, route_scores = select_global_tree_candidates(logits_i, route_scores, k=k)
            past = cache_select_batch(past, parent_indices, old_batch_size=old_batch)

    return measure_stateful(run, warmup=warmup, iters=iters)


@torch.inference_mode()
def prepare_forest_state(model, prompt_token_ids: Sequence[int], *, k: int, depth: int, device: str):
    past, logits = prefill_cache(model, prompt_token_ids, batch_size=k, device=device)
    input_ids, route_scores = init_topk_inputs(logits, k=k)
    for _ in range(int(depth)):
        old_batch = int(input_ids.numel())
        past, logits_i = decode_one(model, past, input_ids)
        parent_indices, input_ids, route_scores = select_global_tree_candidates(logits_i, route_scores, k=k)
        past = cache_select_batch(past, parent_indices, old_batch_size=old_batch)

    logprobs, token_ids = torch.topk(torch.log_softmax(logits_i, dim=-1), k=int(k), dim=-1)
    parent_indices = torch.arange(int(k), device=logits_i.device, dtype=torch.long).repeat_interleave(int(k))
    rank_indices = torch.arange(int(k), device=logits_i.device, dtype=torch.long).repeat(int(k))
    forest_input_ids = token_ids[parent_indices, rank_indices].to(dtype=torch.long)
    forest_scores = route_scores[parent_indices] + logprobs[parent_indices, rank_indices]
    root_ids = torch.arange(int(k), device=logits_i.device, dtype=torch.long).repeat_interleave(int(k))
    forest_past = cache_select_batch(past, parent_indices, old_batch_size=int(k))
    return forest_past, forest_input_ids, forest_scores.detach(), root_ids


def bench_stage2_forest(
    model,
    prompt_token_ids: Sequence[int],
    *,
    k: int,
    depth: int,
    device: str,
    warmup: int,
    iters: int,
    stateful_cache: bool,
) -> TimedResult:
    def setup():
        return prepare_forest_state(
            model,
            prompt_token_ids,
            k=k,
            depth=depth,
            device=device,
        )

    def run_state(state) -> None:
        past, input_ids, route_scores, root_ids = state
        for _ in range(int(depth)):
            old_batch = int(input_ids.numel())
            past, logits_i = decode_one(model, past, input_ids)
            parent_indices, input_ids, route_scores, root_ids = select_forest_candidates_by_root(
                logits_i,
                route_scores,
                root_ids,
                k=k,
            )
            past = cache_select_batch(past, parent_indices, old_batch_size=old_batch)

    if not stateful_cache:
        return measure_with_setup(setup, run_state, warmup=warmup, iters=iters)

    past, input_ids, route_scores, root_ids = setup()

    def run() -> None:
        nonlocal past, input_ids, route_scores, root_ids
        for _ in range(int(depth)):
            old_batch = int(input_ids.numel())
            past, logits_i = decode_one(model, past, input_ids)
            parent_indices, input_ids, route_scores, root_ids = select_forest_candidates_by_root(
                logits_i,
                route_scores,
                root_ids,
                k=k,
            )
            past = cache_select_batch(past, parent_indices, old_batch_size=old_batch)

    return measure_stateful(run, warmup=warmup, iters=iters)


def bench_verify_append(
    model,
    prompt_token_ids: Sequence[int],
    *,
    k: int,
    depth: int,
    device: str,
    warmup: int,
    iters: int,
    stateful_cache: bool,
) -> TimedResult:
    def setup():
        past, logits = prefill_cache(model, prompt_token_ids, batch_size=k, device=device)
        _, token_ids = torch.topk(torch.log_softmax(logits, dim=-1), k=int(depth), dim=-1)
        verify_tokens = token_ids[:, : int(depth)].to(dtype=torch.long)
        return past, verify_tokens

    def run_state(state) -> None:
        past, verify_tokens = state
        model_forward(model, verify_tokens, past_key_values=past)

    if not stateful_cache:
        return measure_with_setup(setup, run_state, warmup=warmup, iters=iters)

    past, verify_tokens = setup()

    def run() -> None:
        nonlocal past, verify_tokens
        run_state((past, verify_tokens))

    return measure_stateful(run, warmup=warmup, iters=iters)


def result_dict(result: TimedResult) -> dict[str, object]:
    return {
        "median_ms": result.median_ms,
        "mean_ms": result.mean_ms,
        "samples_ms": result.samples_ms,
    }


def print_result(name: str, result: TimedResult) -> None:
    print(f"{name:<36} median={result.median_ms:9.3f} ms mean={result.mean_ms:9.3f} ms")


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> int:
    args = build_parser().parse_args()
    if args.k <= 0 or args.d <= 0:
        raise ValueError("k and d must be positive")
    dtype = dtype_from_name(args.dtype)
    target_model_path = args.target_model or args.drafter_model

    drafter_tokenizer, drafter = load_model(
        args.drafter_model,
        dtype=dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_token_ids = parse_prompt_token_ids(
        tokenizer=drafter_tokenizer,
        prompt=args.prompt,
        prompt_token_ids=args.prompt_token_ids,
        prefix_len=args.prefix_len,
        repeat_token_id=args.repeat_token_id,
    )

    metadata = {
        "backend": "hf_kv_ordinary_batch",
        "shared_kv_optimized": False,
        "cascade": False,
        "masked_verify": False,
        "drafter_model": args.drafter_model,
        "target_model": target_model_path,
        "k": args.k,
        "d": args.d,
        "prefix_len": len(prompt_token_ids),
        "dtype": args.dtype,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "stateful_cache": bool(args.stateful_cache),
        "cache_reset_each_sample": not bool(args.stateful_cache),
        "prefill_excluded_from_timing": True,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))

    print("\n=== Drafter component speed ===")
    drafter_batch1_ar = bench_ar_decode(
        drafter,
        prompt_token_ids,
        batch_size=1,
        steps=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    drafter_batchk_ar = bench_ar_decode(
        drafter,
        prompt_token_ids,
        batch_size=args.k,
        steps=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    stage1_tree = bench_stage1_tree(
        drafter,
        prompt_token_ids,
        k=args.k,
        depth=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    drafter_batchk2_ar = bench_ar_decode(
        drafter,
        prompt_token_ids,
        batch_size=args.k * args.k,
        steps=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    stage2_forest = bench_stage2_forest(
        drafter,
        prompt_token_ids,
        k=args.k,
        depth=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    print_result(f"batch1_ar_{args.d}_tokens", drafter_batch1_ar)
    print_result(f"batch{args.k}_ar_{args.d}_tokens", drafter_batchk_ar)
    print_result(f"build_tree_stage1_k{args.k}_d{args.d}", stage1_tree)
    print_result(f"batch{args.k * args.k}_ar_{args.d}_tokens", drafter_batchk2_ar)
    print_result(f"build_forest_stage2_k2_{args.k * args.k}_d{args.d}", stage2_forest)
    print("-- ratios, median --")
    print(f"tree / batch1_ar_{args.d:<3}          {stage1_tree.median_ms / drafter_batch1_ar.median_ms:9.3f}x")
    print(f"tree / batch{args.k}_ar_{args.d:<3}           {stage1_tree.median_ms / drafter_batchk_ar.median_ms:9.3f}x")
    print(f"forest / batch1_ar_{args.d:<3}        {stage2_forest.median_ms / drafter_batch1_ar.median_ms:9.3f}x")
    print(f"forest / batch{args.k * args.k}_ar_{args.d:<3}         {stage2_forest.median_ms / drafter_batchk2_ar.median_ms:9.3f}x")

    drafter_results = {
        f"batch1_ar_{args.d}_tokens": result_dict(drafter_batch1_ar),
        f"batch{args.k}_ar_{args.d}_tokens": result_dict(drafter_batchk_ar),
        f"build_tree_stage1_k{args.k}_d{args.d}": result_dict(stage1_tree),
        f"batch{args.k * args.k}_ar_{args.d}_tokens": result_dict(drafter_batchk2_ar),
        f"build_forest_stage2_k2_{args.k * args.k}_d{args.d}": result_dict(stage2_forest),
        "ratios_median": {
            "tree_over_batch1_ar": stage1_tree.median_ms / drafter_batch1_ar.median_ms,
            "tree_over_batchk_ar": stage1_tree.median_ms / drafter_batchk_ar.median_ms,
            "forest_over_batch1_ar": stage2_forest.median_ms / drafter_batch1_ar.median_ms,
            "forest_over_batchk2_ar": stage2_forest.median_ms / drafter_batchk2_ar.median_ms,
        },
    }

    free_model(drafter)

    print("\n=== Target verify component speed ===")
    _, target = load_model(
        target_model_path,
        dtype=dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    target_decode1 = bench_ar_decode(
        target,
        prompt_token_ids,
        batch_size=1,
        steps=1,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    target_batch1_ar = bench_ar_decode(
        target,
        prompt_token_ids,
        batch_size=1,
        steps=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    target_batchk_ar = bench_ar_decode(
        target,
        prompt_token_ids,
        batch_size=args.k,
        steps=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    target_verify = bench_verify_append(
        target,
        prompt_token_ids,
        k=args.k,
        depth=args.d,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        stateful_cache=args.stateful_cache,
    )
    print_result("target_batch1_decode_1_token", target_decode1)
    print_result(f"target_batch1_ar_{args.d}_tokens", target_batch1_ar)
    print_result(f"target_batch{args.k}_ar_{args.d}_tokens", target_batchk_ar)
    print_result(f"target_verify_k{args.k}_paths_d{args.d}", target_verify)
    print("-- ratios, median --")
    print(f"verify / decode1              {target_verify.median_ms / target_decode1.median_ms:9.3f}x")
    print(f"verify / batch1_ar_{args.d:<3}        {target_verify.median_ms / target_batch1_ar.median_ms:9.3f}x")
    print(f"verify / batch{args.k}_ar_{args.d:<3}         {target_verify.median_ms / target_batchk_ar.median_ms:9.3f}x")

    report = {
        **metadata,
        "drafter": drafter_results,
        "target": {
            "target_batch1_decode_1_token": result_dict(target_decode1),
            f"target_batch1_ar_{args.d}_tokens": result_dict(target_batch1_ar),
            f"target_batch{args.k}_ar_{args.d}_tokens": result_dict(target_batchk_ar),
            f"target_verify_k{args.k}_paths_d{args.d}": result_dict(target_verify),
            "ratios_median": {
                "verify_over_decode1": target_verify.median_ms / target_decode1.median_ms,
                "verify_over_batch1_ar": target_verify.median_ms / target_batch1_ar.median_ms,
                "verify_over_batchk_ar": target_verify.median_ms / target_batchk_ar.median_ms,
            },
        },
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
