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
from transformers import AutoTokenizer

from atlas_0709.flashinfer_full_verify import (
    FlashInferFullVerifyConfig,
    FlashInferFullVerifyRunner,
    dtype_from_name,
    load_model,
    print_benchmark_report,
)
from atlas_0709.flashinfer_paged.builders import (
    build_forest_depths,
    build_tree_depths,
    initialize_forest_routes,
    initialize_stage1_routes,
)
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.runtime_metadata import package_version
from atlas_0709.flashinfer_paged.sglang_runtime import (
    SGLangFlashInferFrontierModelBackend,
    SGLangRunnerConfig,
    create_sglang_model_runner,
    prefill_sglang_prefix,
    sglang_runner_component_report,
)
from atlas_0709.flashinfer_paged.types import DecodePhase, DraftPrefixState, PendingCandidate, RouteState
from atlas_0709.flashinfer_paged.utils import topk_logprobs


@dataclass(frozen=True)
class TimedResult:
    median_ms: float
    mean_ms: float
    samples_ms: list[float]


@dataclass
class PreparedContext:
    store: KVTreeStore
    backend: SGLangFlashInferFrontierModelBackend
    prefix: DraftPrefixState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark real SGLang/FlashInfer paged decode baselines and ATLAS tree/forest controllers."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--target-model",
        default=None,
        help="Optional target model path. When set, also benchmarks Direct FlashInfer full-model verify.",
    )
    parser.add_argument("--prompt", default="ATLAS staged decode benchmark.")
    parser.add_argument("--prompt-token-ids", default=None, help="Comma-separated token ids. Overrides --prompt.")
    parser.add_argument("--prefix-len", type=int, default=None, help="Repeat/truncate the prompt to this token length.")
    parser.add_argument("--repeat-token-id", type=int, default=None, help="Use this token id repeated --prefix-len times.")
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--pairs", nargs=2, type=int, action="append", metavar=("K", "D"))
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--verify-warmup", type=int, default=None)
    parser.add_argument("--verify-iters", type=int, default=None)
    parser.add_argument("--verify-workspace-mb", type=int, default=128)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--use-packed-custom-mask", action="store_true")
    parser.add_argument("--skip-logit-alignment-check", action="store_true")
    parser.add_argument("--fail-on-logit-mismatch", action="store_true")
    parser.add_argument("--alignment-atol", type=float, default=1.0)
    parser.add_argument("--alignment-rtol", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser


def parse_token_ids(
    raw: str | None,
    *,
    model: str,
    prompt: str,
    prefix_len: int | None = None,
    repeat_token_id: int | None = None,
) -> list[int]:
    if repeat_token_id is not None:
        if prefix_len is None or prefix_len <= 0:
            raise RuntimeError("--repeat-token-id requires a positive --prefix-len")
        return [int(repeat_token_id)] * int(prefix_len)
    if raw is not None:
        token_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if prefix_len is not None:
            return _resize_token_ids(token_ids, int(prefix_len))
        return token_ids
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer produced an empty prompt")
    token_ids = [int(token_id) for token_id in token_ids]
    if prefix_len is not None:
        return _resize_token_ids(token_ids, int(prefix_len))
    return token_ids


def _resize_token_ids(token_ids: Sequence[int], length: int) -> list[int]:
    if length <= 0:
        raise RuntimeError("--prefix-len must be positive")
    if not token_ids:
        raise RuntimeError("cannot resize an empty token list")
    repeated = (list(token_ids) * ((length + len(token_ids) - 1) // len(token_ids)))[:length]
    return [int(token_id) for token_id in repeated]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def required_context_length(prompt_len: int, pairs: Sequence[Sequence[int]]) -> int:
    """Minimum SGLang req-pool row capacity for this benchmark.

    Stage-2 forest workloads start after a stage-1 tree of depth d and then
    decode another d tokens, so the longest route needs prompt_len + 2*d slots.
    """

    max_depth = max(int(depth) for _, depth in pairs)
    return int(prompt_len) + 2 * max_depth


def resolve_context_length(requested: int | None, *, prompt_len: int, pairs: Sequence[Sequence[int]]) -> int:
    required = required_context_length(prompt_len, pairs)
    if requested is None:
        return required
    if int(requested) < required:
        print(
            "[warn] --context-length is too small for this workload; "
            f"using {required} instead of {int(requested)} "
            f"(prompt_len={int(prompt_len)}, max_stage_depth={max(int(depth) for _, depth in pairs)})."
        )
        return required
    return int(requested)


def measure_workload(
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
    samples: list[float] = []
    for _ in range(iters):
        state = setup_fn()
        sync_cuda()
        start = time.perf_counter()
        run_fn(state)
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
    return TimedResult(
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.fmean(samples)),
        samples_ms=samples,
    )


def prepare_context(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
) -> PreparedContext:
    store = KVTreeStore()
    backend = SGLangFlashInferFrontierModelBackend.from_runner(
        route_store=store,
        model_runner=runner,
        page_size=page_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    backend.route_pool.clear_physical_state()
    prefill = prefill_sglang_prefix(
        model_runner=runner,
        route_pool=backend.route_pool,
        prompt_token_ids=prompt_token_ids,
        chunk_size=prefill_chunk_size,
    )
    prefix = backend.attach_prefilled_prefix(
        prompt_token_ids=prefill.prompt_token_ids,
        prefix_slot_ids=prefill.prefix_slot_ids,
        next_token_logits=prefill.next_token_logits,
    )
    return PreparedContext(store=store, backend=backend, prefix=prefix)


def make_initial_routes(prefix: DraftPrefixState, *, batch: int, store: KVTreeStore) -> list[RouteState]:
    token_logprobs, token_ids = topk_logprobs(prefix.next_token_logits, k=batch)
    routes: list[RouteState] = []
    for token_id, token_logprob in zip(token_ids, token_logprobs):
        route_id = store.allocate_route_id()
        route = RouteState(
            route_id=route_id,
            stage1_root_id=route_id,
            parent_route_id=None,
            materialized_leaf_node_id=None,
            pending_token_id=int(token_id),
            cumulative_logprob=float(token_logprob.detach().cpu()),
            stage1_depth=0,
            stage2_depth=0,
            kv_view=prefix.prefix_kv_view.fork(),
        )
        routes.append(store.register_route(route))
    return routes


def advance_top1_per_route(
    routes: Sequence[RouteState],
    *,
    store: KVTreeStore,
    backend: SGLangFlashInferFrontierModelBackend,
) -> list[RouteState]:
    decode_output = backend.decode_frontier_one_token(routes)
    if decode_output.new_node_ids_cpu is not None:
        new_node_ids = [int(value) for value in decode_output.new_node_ids_cpu]
    elif isinstance(decode_output.new_node_ids, torch.Tensor):
        new_node_ids = [
            int(value)
            for value in decode_output.new_node_ids.detach().cpu().tolist()
        ]
    else:
        new_node_ids = [int(value) for value in decode_output.new_node_ids]
    materialized = store.mark_routes_materialized(
        list(routes),
        new_node_ids,
        phase=DecodePhase.STAGE1,
    )
    candidates: list[PendingCandidate] = []
    token_logprobs, token_ids = torch.topk(
        torch.log_softmax(decode_output.next_token_logits, dim=-1),
        k=1,
        dim=-1,
    )
    for row, route in enumerate(materialized):
        candidates.append(
            PendingCandidate(
                parent_route_id=route.route_id,
                stage1_root_id=route.stage1_root_id,
                pending_token_id=int(token_ids[row, 0]),
                cumulative_logprob=route.cumulative_logprob + float(token_logprobs[row, 0].detach().cpu()),
                rank_in_parent=0,
                parent_logprob=route.cumulative_logprob,
            )
        )
    next_routes = store.materialize_route_descriptors(candidates, parent_routes=materialized)
    store.release_routes_without_descendants(materialized, next_routes)
    return next_routes


def setup_batch_ar(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
    batch: int,
) -> tuple[PreparedContext, list[RouteState]]:
    ctx = prepare_context(
        runner=runner,
        prompt_token_ids=prompt_token_ids,
        page_size=page_size,
        prefill_chunk_size=prefill_chunk_size,
    )
    routes = make_initial_routes(ctx.prefix, batch=batch, store=ctx.store)
    return ctx, routes


def run_batch_ar(state: object, *, depth: int) -> None:
    ctx, routes = state
    for _ in range(depth):
        routes = advance_top1_per_route(routes, store=ctx.store, backend=ctx.backend)


def setup_atlas_tree(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
    k: int,
) -> tuple[PreparedContext, list[RouteState], int]:
    ctx = prepare_context(
        runner=runner,
        prompt_token_ids=prompt_token_ids,
        page_size=page_size,
        prefill_chunk_size=prefill_chunk_size,
    )
    routes = initialize_stage1_routes(ctx.prefix, k=k, route_store=ctx.store)
    return ctx, routes, k


def run_atlas_tree(state: object, *, depth: int) -> None:
    ctx, routes, k = state
    build_tree_depths(
        routes,
        depth=depth,
        k=k,
        route_store=ctx.store,
        model_backend=ctx.backend,
        phase=DecodePhase.STAGE1,
    )


def setup_atlas_forest(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
    k: int,
    depth: int,
) -> tuple[PreparedContext, list[RouteState], int]:
    ctx = prepare_context(
        runner=runner,
        prompt_token_ids=prompt_token_ids,
        page_size=page_size,
        prefill_chunk_size=prefill_chunk_size,
    )
    routes = initialize_stage1_routes(ctx.prefix, k=k, route_store=ctx.store)
    tree = build_tree_depths(
        routes,
        depth=depth,
        k=k,
        route_store=ctx.store,
        model_backend=ctx.backend,
        phase=DecodePhase.STAGE1,
    )
    forest_routes = initialize_forest_routes(
        tree.completed_routes,
        tree.last_logits,
        k=k,
        route_store=ctx.store,
    )
    return ctx, forest_routes, k


def setup_batch_ar_forest_context(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
    k: int,
    depth: int,
) -> tuple[PreparedContext, list[RouteState]]:
    ctx, forest_routes, _ = setup_atlas_forest(
        runner=runner,
        prompt_token_ids=prompt_token_ids,
        page_size=page_size,
        prefill_chunk_size=prefill_chunk_size,
        k=k,
        depth=depth,
    )
    return ctx, forest_routes


def run_atlas_forest(state: object, *, depth: int) -> None:
    ctx, forest_routes, k = state
    build_forest_depths(
        forest_routes,
        depth=depth,
        k=k,
        route_store=ctx.store,
        model_backend=ctx.backend,
    )


def result_dict(result: TimedResult) -> dict[str, float | list[float]]:
    return {
        "median_ms": result.median_ms,
        "mean_ms": result.mean_ms,
        "samples_ms": result.samples_ms,
    }


def print_result(name: str, result: TimedResult) -> None:
    print(f"{name:<30} median={result.median_ms:9.3f} ms mean={result.mean_ms:9.3f} ms")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_target_verify_benchmarks(
    *,
    args: argparse.Namespace,
    pairs: Sequence[Sequence[int]],
    prompt_token_ids: Sequence[int],
) -> dict[str, object]:
    if args.target_model is None:
        print("\n[info] --target-model not set; skipping Target verify benchmark.")
        return {
            "skipped": True,
            "reason": "--target-model not set",
            "verify_backend": "none",
        }

    verify_warmup = args.verify_warmup if args.verify_warmup is not None else args.warmup
    verify_iters = args.verify_iters if args.verify_iters is not None else args.iters
    print("\n=== Loading target model for Direct FlashInfer verify ===")
    _tokenizer, target_model = load_model(
        args.target_model,
        dtype=dtype_from_name(args.dtype),
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    verify_results: list[dict[str, object]] = []
    runner = None
    try:
        for k, depth in pairs:
            config = FlashInferFullVerifyConfig(
                k=int(k),
                d=int(depth),
                prefix_len=len(prompt_token_ids),
                repeat_token_id=int(args.repeat_token_id) if args.repeat_token_id is not None else 0,
                page_size=args.page_size,
                dtype=args.dtype,
                device="cuda",
                warmup=verify_warmup,
                iters=verify_iters,
                workspace_mb=args.verify_workspace_mb,
                flashinfer_backend=args.flashinfer_backend,
                trust_remote_code=args.trust_remote_code,
                use_packed_custom_mask=args.use_packed_custom_mask,
                check_logit_alignment=not args.skip_logit_alignment_check,
                fail_on_logit_mismatch=args.fail_on_logit_mismatch,
                alignment_atol=args.alignment_atol,
                alignment_rtol=args.alignment_rtol,
            )
            runner = FlashInferFullVerifyRunner(
                model=target_model,
                config=config,
                model_name=args.target_model,
            )
            result = runner.benchmark(prompt_token_ids)
            print_benchmark_report(result)
            verify_results.append(result.to_dict())
    finally:
        runner = None
        del target_model
        cleanup_cuda()

    logits_aligned = bool(verify_results) and all(bool(item.get("logits_aligned")) for item in verify_results)
    return {
        "skipped": False,
        "verify_backend": "direct_flashinfer_full_llama_masked_verify",
        "semantic_correctness_required": not args.skip_logit_alignment_check,
        "logits_aligned": logits_aligned,
        "rope_applied": True,
        "benchmarks": verify_results,
    }


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    pairs = args.pairs or [[3, 4]]
    prompt_token_ids = parse_token_ids(
        args.prompt_token_ids,
        model=args.model,
        prompt=args.prompt,
        prefix_len=args.prefix_len,
        repeat_token_id=args.repeat_token_id,
    )
    context_length = resolve_context_length(
        args.context_length,
        prompt_len=len(prompt_token_ids),
        pairs=pairs,
    )
    config = SGLangRunnerConfig(
        model_path=args.model,
        dtype=args.dtype,
        context_length=context_length,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        gpu_id=args.gpu_id,
        nccl_port=args.nccl_port,
    )

    report: dict[str, object] = {}
    runner = None
    try:
        runner = create_sglang_model_runner(config, initialize=True)
        metadata = {
            **sglang_runner_component_report(runner),
            "backend": "sglang_flashinfer_paged_decode",
            "reference_only": False,
            "paged_kv": True,
            "paged_kv_enabled": True,
            "cascade": False,
            "cascade_level": 0,
            "verify_backend": (
                "direct_flashinfer_full_llama_masked_verify" if args.target_model else "none"
            ),
            "flashinfer_version": package_version("flashinfer-python") or package_version("flashinfer"),
            "page_size": args.page_size,
            "requested_context_length": args.context_length,
            "effective_context_length": context_length,
            "required_context_length": required_context_length(len(prompt_token_ids), pairs),
            "prompt_len": len(prompt_token_ids),
            "prefill_chunk_size": args.prefill_chunk_size,
            "warmup": args.warmup,
            "iters": args.iters,
        }
        metadata["kv_cache_class"] = metadata.get("token_to_kv_pool_class")
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        report["metadata"] = metadata
        pair_results = []

        for k, depth in pairs:
            print(f"\n=== SGLang FlashInfer paged decode k={k}, d={depth} ===")
            batch1 = measure_workload(
                lambda: setup_batch_ar(
                    runner=runner,
                    prompt_token_ids=prompt_token_ids,
                    page_size=args.page_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    batch=1,
                ),
                lambda state, depth=depth: run_batch_ar(state, depth=depth),
                warmup=args.warmup,
                iters=args.iters,
            )
            batchk = measure_workload(
                lambda k=k: setup_batch_ar(
                    runner=runner,
                    prompt_token_ids=prompt_token_ids,
                    page_size=args.page_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    batch=k,
                ),
                lambda state, depth=depth: run_batch_ar(state, depth=depth),
                warmup=args.warmup,
                iters=args.iters,
            )
            tree = measure_workload(
                lambda k=k: setup_atlas_tree(
                    runner=runner,
                    prompt_token_ids=prompt_token_ids,
                    page_size=args.page_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    k=k,
                ),
                lambda state, depth=depth: run_atlas_tree(state, depth=depth),
                warmup=args.warmup,
                iters=args.iters,
            )
            batchk2 = measure_workload(
                lambda k=k, depth=depth: setup_batch_ar_forest_context(
                    runner=runner,
                    prompt_token_ids=prompt_token_ids,
                    page_size=args.page_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    k=k,
                    depth=depth,
                ),
                lambda state, depth=depth: run_batch_ar(state, depth=depth),
                warmup=args.warmup,
                iters=args.iters,
            )
            forest = measure_workload(
                lambda k=k: setup_atlas_forest(
                    runner=runner,
                    prompt_token_ids=prompt_token_ids,
                    page_size=args.page_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    k=k,
                    depth=depth,
                ),
                lambda state, depth=depth: run_atlas_forest(state, depth=depth),
                warmup=args.warmup,
                iters=args.iters,
            )

            print_result(f"batch1_ar_{depth}_tokens", batch1)
            print_result(f"batch{k}_ar_{depth}_tokens", batchk)
            print_result(f"atlas_tree_{depth}_frontiers", tree)
            print_result(f"batch{k * k}_forest_ctx_ar_{depth}", batchk2)
            print_result(f"atlas_forest_{depth}_frontiers", forest)
            print("-- ratios, median --")
            print(f"batch{k}_ar/tree              {batchk.median_ms / tree.median_ms:9.3f}x")
            print(f"tree/batch1_ar               {tree.median_ms / batch1.median_ms:9.3f}x")
            print(f"batch{k * k}_forest_ctx/forest {batchk2.median_ms / forest.median_ms:7.3f}x")
            print(f"forest/batch1_ar             {forest.median_ms / batch1.median_ms:9.3f}x")

            pair_results.append(
                {
                    "k": k,
                    "d": depth,
                    "batch1_ar": result_dict(batch1),
                    "batchk_ar": result_dict(batchk),
                    "atlas_tree": result_dict(tree),
                    "batchk2_forest_context_ar": result_dict(batchk2),
                    "atlas_forest": result_dict(forest),
                    "ratios_median": {
                        "batchk_ar_over_tree": batchk.median_ms / tree.median_ms,
                        "tree_over_batch1_ar": tree.median_ms / batch1.median_ms,
                        "batchk2_forest_context_ar_over_forest": batchk2.median_ms / forest.median_ms,
                        "forest_over_batch1_ar": forest.median_ms / batch1.median_ms,
                    },
                }
            )

        report["pairs"] = pair_results
    finally:
        runner = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        cleanup_cuda()

    report["target_verify"] = run_target_verify_benchmarks(
        args=args,
        pairs=pairs,
        prompt_token_ids=prompt_token_ids,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
