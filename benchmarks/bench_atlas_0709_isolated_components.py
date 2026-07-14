from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from atlas_0709.flashinfer_full_verify import (
    FlashInferFullVerifyConfig,
    bench_flashinfer_full_ar_decode,
)
from atlas_0709.flashinfer_paged.builders import (
    build_forest_one_depth,
    build_tree_depths,
    build_tree_one_depth,
    initialize_forest_routes,
    initialize_stage1_routes,
)
from atlas_0709.flashinfer_paged.frontier import advance_frontier_one_token
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.sglang_runtime import (
    SGLangFlashInferFrontierModelBackend,
    SGLangRunnerConfig,
    create_sglang_model_runner,
    prefill_sglang_prefix,
    sglang_runner_component_report,
)
from atlas_0709.flashinfer_paged.types import (
    DecodePhase,
    DraftPrefixState,
    PendingCandidate,
    PendingCandidateBatch,
    RouteState,
)
from atlas_0709.flashinfer_paged.utils import global_topk_candidates
from atlas_0709.target_runtime import (
    DirectFlashInferMaskedTreeVerifyBackend,
    VerifyRoutePayload,
)


@dataclass
class PreparedDraftContext:
    store: KVTreeStore
    backend: SGLangFlashInferFrontierModelBackend
    prefix: DraftPrefixState


@dataclass
class FrontierState:
    context: PreparedDraftContext
    routes: list[RouteState]
    k: int


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mean_ms: float
    samples_ms: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "samples_ms": self.samples_ms,
        }


@dataclass(frozen=True)
class SteppedTiming:
    total: Timing
    steps: list[Timing]
    counter_delta_samples: list[list[dict[str, int]]] = field(default_factory=list)
    counter_cumulative_samples: list[list[dict[str, int]]] = field(default_factory=list)
    counter_total_delta_samples: list[dict[str, int]] = field(default_factory=list)
    measurement_syncs: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        step_reports: list[dict[str, object]] = []
        for index, step_timing in enumerate(self.steps):
            step_report: dict[str, object] = {
                "depth": index + 1,
                **step_timing.to_dict(),
            }
            if index < len(self.counter_delta_samples):
                step_report["hot_path_counter_delta"] = summarize_counter_samples(
                    self.counter_delta_samples[index]
                )
            if index < len(self.counter_cumulative_samples):
                step_report["hot_path_counter_cumulative"] = summarize_counter_samples(
                    self.counter_cumulative_samples[index]
                )
            step_reports.append(step_report)

        report: dict[str, object] = {
            "total": self.total.to_dict(),
            "steps": step_reports,
        }
        if self.counter_total_delta_samples:
            report["hot_path_counter_total_delta"] = summarize_counter_samples(
                self.counter_total_delta_samples
            )
        if self.measurement_syncs:
            report["measurement_syncs"] = dict(self.measurement_syncs)
        return report


@dataclass(frozen=True)
class PerRouteTop1Selection:
    """Keep one greedy continuation for each active route.

    This is the matched ordinary-AR baseline: it uses the same frontier batch,
    physical KV, and model forward as tree/forest construction without global
    or per-root competition across routes.
    """

    phase: DecodePhase

    def select(
        self,
        candidates: Sequence[PendingCandidate],
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> list[PendingCandidate]:
        del k
        by_parent: dict[int, list[PendingCandidate]] = {}
        for candidate in candidates:
            by_parent.setdefault(int(candidate.parent_route_id), []).append(candidate)

        selected: list[PendingCandidate] = []
        for route in active_routes:
            route_candidates = by_parent.get(int(route.route_id), [])
            if not route_candidates:
                raise ValueError(f"route {int(route.route_id)} produced no AR candidate")
            selected.extend(global_topk_candidates(route_candidates, num_selected=1))
        return selected

    def select_indices(
        self,
        candidates: PendingCandidateBatch,
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> torch.Tensor:
        del k
        if int(candidates.candidates_per_parent) != 1:
            raise ValueError("matched AR requires exactly one candidate per parent")
        if int(candidates.candidate_count) != len(active_routes):
            raise ValueError(
                "matched AR candidate count must equal the active route batch size"
            )
        return torch.arange(
            candidates.candidate_count,
            device=candidates.pending_token_ids.device,
            dtype=torch.long,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ATLAS 0709 tree, forest, and Target verify in isolation "
            "with synthetic token ids and no semantic-quality requirement."
        )
    )
    parser.add_argument(
        "--drafter-model",
        default="/home/hwc/models/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--target-model",
        default="/home/hwc/models/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--repeat-token-id", type=int, default=42)
    parser.add_argument(
        "--shared-path-tokens",
        type=int,
        default=2,
        help="Number of common leading tokens in the synthetic Target paths.",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-tree", action="store_true")
    parser.add_argument("--skip-forest", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--legacy-token-attention-metadata",
        action="store_true",
        help="Use SGLang's page_size=1 token metadata for a matched A/B baseline.",
    )
    parser.add_argument(
        "--skip-page-layout-validation",
        action="store_true",
        help="Skip ATLAS physical page-layout validation during timed diagnostics.",
    )
    parser.add_argument("--json-out", default=None)
    return parser


def sync_cuda() -> None:
    torch.cuda.synchronize()


def timing(samples_ms: Sequence[float]) -> Timing:
    values = [float(value) for value in samples_ms]
    return Timing(
        median_ms=float(statistics.median(values)),
        mean_ms=float(statistics.fmean(values)),
        samples_ms=values,
    )


def summarize_counter_samples(
    samples: Sequence[Mapping[str, int]],
) -> dict[str, dict[str, object]]:
    keys = sorted({str(key) for sample in samples for key in sample})
    summary: dict[str, dict[str, object]] = {}
    for key in keys:
        values = [int(sample.get(key, 0)) for sample in samples]
        summary[key] = {
            "median": float(statistics.median(values)),
            "mean": float(statistics.fmean(values)),
            "samples": values,
        }
    return summary


def subtract_counters(
    after: Mapping[str, int],
    before: Mapping[str, int],
) -> dict[str, int]:
    keys = {str(key) for key in after} | {str(key) for key in before}
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
    }


def state_hot_path_counters(state: FrontierState) -> dict[str, int]:
    counter_fn = getattr(state.context.backend.route_pool, "hot_path_counters", None)
    if not callable(counter_fn):
        return {}
    return {
        str(key): int(value)
        for key, value in dict(counter_fn()).items()
    }


def step_hot_path_counters(result: Any, state: FrontierState) -> dict[str, int]:
    decode_output = getattr(result, "decode_output", None)
    metadata = getattr(decode_output, "metadata", None)
    if isinstance(metadata, Mapping):
        counters = metadata.get("hot_path_counters_total")
        if isinstance(counters, Mapping):
            return {
                str(key): int(value)
                for key, value in counters.items()
            }
    return state_hot_path_counters(state)


def step_selection_counter_delta(result: Any) -> dict[str, int]:
    stats = getattr(result, "selection_stats", None)
    if stats is None:
        return {}
    names = (
        "candidate_count",
        "selected_count",
        "host_materialization_batches",
        "host_materialization_elements",
        "host_transfer_batches",
        "host_transfer_elements",
    )
    return {
        f"selection_{name}": int(getattr(stats, name, 0))
        for name in names
    }


def add_counters(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> dict[str, int]:
    keys = {str(key) for key in left} | {str(key) for key in right}
    return {
        key: int(left.get(key, 0)) + int(right.get(key, 0))
        for key in sorted(keys)
    }


def add_total_host_transfer_counters(
    counters: Mapping[str, int],
) -> dict[str, int]:
    result = {str(key): int(value) for key, value in counters.items()}
    result["total_host_transfer_batches"] = int(
        result.get("hot_path_host_transfer_batches", 0)
        + result.get("selection_host_transfer_batches", 0)
    )
    result["total_host_transfer_elements"] = int(
        result.get("hot_path_host_transfer_elements", 0)
        + result.get("selection_host_transfer_elements", 0)
    )
    return result


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def prepare_draft_context(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
) -> PreparedDraftContext:
    store = KVTreeStore()
    backend = SGLangFlashInferFrontierModelBackend.from_runner(
        route_store=store,
        model_runner=runner,
        page_size=page_size,
        device="cuda",
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
    store.committed_token_ids = [int(token_id) for token_id in prompt_token_ids]
    return PreparedDraftContext(store=store, backend=backend, prefix=prefix)


def setup_tree_state(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
    k: int,
) -> FrontierState:
    context = prepare_draft_context(
        runner=runner,
        prompt_token_ids=prompt_token_ids,
        page_size=page_size,
        prefill_chunk_size=prefill_chunk_size,
    )
    routes = initialize_stage1_routes(
        context.prefix,
        k=k,
        route_store=context.store,
    )
    return FrontierState(context=context, routes=routes, k=k)


def setup_forest_state(
    *,
    runner,
    prompt_token_ids: Sequence[int],
    page_size: int,
    prefill_chunk_size: int,
    k: int,
    d: int,
) -> FrontierState:
    context = prepare_draft_context(
        runner=runner,
        prompt_token_ids=prompt_token_ids,
        page_size=page_size,
        prefill_chunk_size=prefill_chunk_size,
    )
    routes = initialize_stage1_routes(
        context.prefix,
        k=k,
        route_store=context.store,
    )
    stage1 = build_tree_depths(
        routes,
        depth=d,
        k=k,
        route_store=context.store,
        model_backend=context.backend,
    )
    forest_routes = initialize_forest_routes(
        stage1.completed_routes,
        stage1.last_logits,
        k=k,
        route_store=context.store,
    )
    return FrontierState(context=context, routes=forest_routes, k=k)


def run_tree_step(state: FrontierState) -> Any:
    result = build_tree_one_depth(
        state.routes,
        k=state.k,
        route_store=state.context.store,
        model_backend=state.context.backend,
    )
    state.routes = result.next_routes
    return result


def run_forest_step(state: FrontierState) -> Any:
    result = build_forest_one_depth(
        state.routes,
        k=state.k,
        route_store=state.context.store,
        model_backend=state.context.backend,
    )
    state.routes = result.next_routes
    return result


def run_matched_ar_step(
    state: FrontierState,
    *,
    phase: DecodePhase,
) -> Any:
    result = advance_frontier_one_token(
        state.routes,
        k=1,
        route_store=state.context.store,
        model_backend=state.context.backend,
        selection_policy=PerRouteTop1Selection(phase=phase),
    )
    state.routes = result.next_routes
    return result


def measure_stepped_component(
    *,
    setup_fn: Callable[[], FrontierState],
    step_fn: Callable[[FrontierState], Any],
    depth: int,
    warmup: int,
    iters: int,
) -> SteppedTiming:
    for _ in range(warmup):
        state = setup_fn()
        for _ in range(depth):
            step_fn(state)
        sync_cuda()

    step_samples: list[list[float]] = [[] for _ in range(depth)]
    total_samples: list[float] = []
    counter_delta_samples: list[list[dict[str, int]]] = [
        [] for _ in range(depth)
    ]
    counter_cumulative_samples: list[list[dict[str, int]]] = [
        [] for _ in range(depth)
    ]
    counter_total_delta_samples: list[dict[str, int]] = []
    for _ in range(iters):
        state = setup_fn()
        setup_counters = state_hot_path_counters(state)
        sync_cuda()
        total_start = time.perf_counter()
        step_results: list[Any] = []
        for step_index in range(depth):
            sync_cuda()
            step_start = time.perf_counter()
            result = step_fn(state)
            sync_cuda()
            step_samples[step_index].append(
                (time.perf_counter() - step_start) * 1000.0
            )
            step_results.append(result)
        total_samples.append((time.perf_counter() - total_start) * 1000.0)

        # Extract Python counter dictionaries only after the timed region.  The
        # production backend stores an immutable cumulative snapshot in every
        # step result, so per-depth evidence does not perturb total latency.
        step_counter_totals = [
            step_hot_path_counters(result, state)
            for result in step_results
        ]
        step_selection_deltas = [
            step_selection_counter_delta(result)
            for result in step_results
        ]
        previous_counters = setup_counters
        cumulative_selection_counters: dict[str, int] = {}
        for step_index, counters_after_step in enumerate(step_counter_totals):
            bridge_delta = subtract_counters(
                counters_after_step,
                previous_counters,
            )
            cumulative_selection_counters = add_counters(
                cumulative_selection_counters,
                step_selection_deltas[step_index],
            )
            counter_delta_samples[step_index].append(
                add_total_host_transfer_counters(
                    add_counters(bridge_delta, step_selection_deltas[step_index])
                )
            )
            counter_cumulative_samples[step_index].append(
                add_total_host_transfer_counters(
                    add_counters(
                        subtract_counters(counters_after_step, setup_counters),
                        cumulative_selection_counters,
                    )
                )
            )
            previous_counters = counters_after_step
        counter_total_delta_samples.append(
            add_total_host_transfer_counters(
                add_counters(
                    subtract_counters(previous_counters, setup_counters),
                    cumulative_selection_counters,
                )
            )
        )

    measured_syncs_per_iteration = 1 + 2 * int(depth)
    measurement_syncs = {
        "kind": "benchmark_explicit_cuda_synchronize",
        "warmup_after_workload_total": int(warmup),
        "before_timed_region_per_iteration": 1,
        "inside_total_timing_per_iteration": 2 * int(depth),
        "inside_each_step_sample": 1,
        "measured_per_iteration": measured_syncs_per_iteration,
        "measured_total": int(iters) * measured_syncs_per_iteration,
        "overall_total": int(warmup) + int(iters) * measured_syncs_per_iteration,
        "excluded_from_hot_path_counters": True,
        "counter_summary_extraction_in_timed_region": False,
    }

    return SteppedTiming(
        total=timing(total_samples),
        steps=[timing(samples) for samples in step_samples],
        counter_delta_samples=counter_delta_samples,
        counter_cumulative_samples=counter_cumulative_samples,
        counter_total_delta_samples=counter_total_delta_samples,
        measurement_syncs=measurement_syncs,
    )


def synthetic_verify_paths(
    *,
    k: int,
    d: int,
    shared_path_tokens: int,
    base_token_id: int,
    vocab_size: int,
) -> list[VerifyRoutePayload]:
    shared = max(0, min(int(shared_path_tokens), int(d)))
    common = [
        int((base_token_id + offset + 1) % vocab_size)
        for offset in range(shared)
    ]
    routes: list[VerifyRoutePayload] = []
    for route_index in range(k):
        private = [
            int(
                (
                    base_token_id
                    + 17
                    + route_index * max(d, 1)
                    + offset
                )
                % vocab_size
            )
            for offset in range(d - shared)
        ]
        routes.append(
            VerifyRoutePayload(
                route_id=route_index + 1,
                token_ids=tuple([*common, *private]),
                draft_logprob=-float(route_index),
            )
        )
    return routes


def measure_target_verify(
    *,
    backend: DirectFlashInferMaskedTreeVerifyBackend,
    prompt_token_ids: Sequence[int],
    routes: Sequence[VerifyRoutePayload],
    warmup: int,
    iters: int,
) -> tuple[Timing, dict[str, object]]:
    last_metadata: dict[str, object] = {}
    for _ in range(warmup):
        backend.prefill(prompt_token_ids)
        backend.verify_payloads(
            prefix_token_ids=prompt_token_ids,
            routes=routes,
        )
        sync_cuda()

    samples: list[float] = []
    for _ in range(iters):
        backend.prefill(prompt_token_ids)
        sync_cuda()
        start = time.perf_counter()
        result = backend.verify_payloads(
            prefix_token_ids=prompt_token_ids,
            routes=routes,
        )
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
        last_metadata = dict(result.metadata)
    return timing(samples), last_metadata


def print_stepped(name: str, result: SteppedTiming) -> None:
    print(
        f"{name}_total{'':<17} mean={result.total.mean_ms:9.3f} ms "
        f"median={result.total.median_ms:9.3f} ms "
        f"mean/depth={result.total.mean_ms / len(result.steps):8.3f} ms"
    )
    for index, step in enumerate(result.steps, start=1):
        print(
            f"  depth_{index:<2}{'':<18} mean={step.mean_ms:9.3f} ms "
            f"median={step.median_ms:9.3f} ms"
        )


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.k <= 0 or args.d <= 0:
        raise SystemExit("--k and --d must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")

    print(
        "[isolation] Stop the network Target server before this benchmark when it "
        "shares the same GPU. No tree/forest/verify components run concurrently.",
        flush=True,
    )
    prompt_token_ids = [int(args.repeat_token_id)] * int(args.prefix_len)
    required_context = len(prompt_token_ids) + 2 * int(args.d)
    context_length = max(
        required_context,
        required_context if args.context_length is None else int(args.context_length),
    )
    report: dict[str, object] = {
        "metadata": {
            "benchmark": "atlas_0709_isolated_components",
            "semantic_correctness_required": False,
            "synthetic_token_ids": True,
            "concurrent_target_drafter": False,
            "prefill_included": False,
            "k": int(args.k),
            "d": int(args.d),
            "prefix_len": len(prompt_token_ids),
            "page_size": int(args.page_size),
            "dtype": args.dtype,
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "attention_metadata_mode": (
                "legacy_token_page_size_1"
                if args.legacy_token_attention_metadata
                else f"physical_page_size_{int(args.page_size)}"
            ),
            "page_layout_validation": not args.skip_page_layout_validation,
        }
    }

    runner = None
    try:
        if not (args.skip_tree and args.skip_forest):
            runner_config = SGLangRunnerConfig(
                model_path=args.drafter_model,
                dtype=args.dtype,
                context_length=context_length,
                page_size=args.page_size,
                mem_fraction_static=args.mem_fraction_static,
                max_running_requests=args.max_running_requests,
                max_total_tokens=args.max_total_tokens,
                gpu_id=args.gpu_id,
                nccl_port=args.nccl_port,
                trust_remote_code=args.trust_remote_code,
                use_page_attention_metadata=not args.legacy_token_attention_metadata,
                validate_page_layout=not args.skip_page_layout_validation,
            )
            runner = create_sglang_model_runner(runner_config, initialize=True)
            report["drafter_runtime"] = sglang_runner_component_report(runner)

            drafter_ar = measure_stepped_component(
                setup_fn=lambda: setup_tree_state(
                    runner=runner,
                    prompt_token_ids=prompt_token_ids,
                    page_size=args.page_size,
                    prefill_chunk_size=args.prefill_chunk_size,
                    k=1,
                ),
                step_fn=run_tree_step,
                depth=1,
                warmup=args.warmup,
                iters=args.iters,
            ).total
            report["drafter_ar_decode_1_token"] = drafter_ar.to_dict()
            print("\n=== Isolated Drafter AR decode ===")
            print(
                "drafter_ar_decode_1_token       "
                f"mean={drafter_ar.mean_ms:9.3f} ms "
                f"median={drafter_ar.median_ms:9.3f} ms "
                f"rate={1000.0 / drafter_ar.mean_ms:8.2f} tok/s"
            )

            if not args.skip_tree:
                matched_tree_ar = measure_stepped_component(
                    setup_fn=lambda: setup_tree_state(
                        runner=runner,
                        prompt_token_ids=prompt_token_ids,
                        page_size=args.page_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                        k=args.k,
                    ),
                    step_fn=lambda state: run_matched_ar_step(
                        state,
                        phase=DecodePhase.STAGE1,
                    ),
                    depth=args.d,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                report["matched_drafter_ar_tree_workload"] = {
                    "batch_size": int(args.k),
                    "decode_steps": int(args.d),
                    "selection": "independent_per_route_greedy_top1",
                    "setup_and_prefill_included": False,
                    **matched_tree_ar.to_dict(),
                }
                print("\n=== Matched Drafter AR for tree workload ===")
                print_stepped("matched_tree_ar", matched_tree_ar)

                tree = measure_stepped_component(
                    setup_fn=lambda: setup_tree_state(
                        runner=runner,
                        prompt_token_ids=prompt_token_ids,
                        page_size=args.page_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                        k=args.k,
                    ),
                    step_fn=run_tree_step,
                    depth=args.d,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                report["build_tree"] = tree.to_dict()
                report["build_tree"]["mean_over_drafter_ar1"] = (
                    tree.total.mean_ms / drafter_ar.mean_ms
                )
                report["build_tree"]["mean_over_matched_drafter_ar"] = (
                    tree.total.mean_ms / matched_tree_ar.total.mean_ms
                )
                report["build_tree"]["matched_drafter_ar_report_key"] = (
                    "matched_drafter_ar_tree_workload"
                )
                print("\n=== Isolated Drafter build tree ===")
                print_stepped("build_tree", tree)
                print(
                    f"  mean tree/ar1             "
                    f"{tree.total.mean_ms / drafter_ar.mean_ms:9.3f}x"
                )
                print(
                    f"  mean tree/matched-ar      "
                    f"{tree.total.mean_ms / matched_tree_ar.total.mean_ms:9.3f}x"
                )

            if not args.skip_forest:
                matched_forest_ar = measure_stepped_component(
                    setup_fn=lambda: setup_forest_state(
                        runner=runner,
                        prompt_token_ids=prompt_token_ids,
                        page_size=args.page_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                        k=args.k,
                        d=args.d,
                    ),
                    step_fn=lambda state: run_matched_ar_step(
                        state,
                        phase=DecodePhase.STAGE2,
                    ),
                    depth=args.d,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                report["matched_drafter_ar_forest_workload"] = {
                    "batch_size": int(args.k * args.k),
                    "decode_steps": int(args.d),
                    "selection": "independent_per_route_greedy_top1",
                    "setup_and_prefill_included": False,
                    **matched_forest_ar.to_dict(),
                }
                print("\n=== Matched Drafter AR for forest workload ===")
                print_stepped("matched_forest_ar", matched_forest_ar)

                forest = measure_stepped_component(
                    setup_fn=lambda: setup_forest_state(
                        runner=runner,
                        prompt_token_ids=prompt_token_ids,
                        page_size=args.page_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                        k=args.k,
                        d=args.d,
                    ),
                    step_fn=run_forest_step,
                    depth=args.d,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                report["build_forest"] = forest.to_dict()
                report["build_forest"]["mean_over_drafter_ar1"] = (
                    forest.total.mean_ms / drafter_ar.mean_ms
                )
                report["build_forest"]["mean_over_matched_drafter_ar"] = (
                    forest.total.mean_ms / matched_forest_ar.total.mean_ms
                )
                report["build_forest"]["matched_drafter_ar_report_key"] = (
                    "matched_drafter_ar_forest_workload"
                )
                print("\n=== Isolated Drafter build forest ===")
                print_stepped("build_forest", forest)
                print(
                    f"  mean forest/ar1           "
                    f"{forest.total.mean_ms / drafter_ar.mean_ms:9.3f}x"
                )
                print(
                    f"  mean forest/matched-ar    "
                    f"{forest.total.mean_ms / matched_forest_ar.total.mean_ms:9.3f}x"
                )
    finally:
        runner = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        cleanup_cuda()

    if not args.skip_verify:
        verify_config = FlashInferFullVerifyConfig(
            k=args.k,
            d=args.d,
            prefix_len=len(prompt_token_ids),
            repeat_token_id=args.repeat_token_id,
            page_size=args.page_size,
            dtype=args.dtype,
            device="cuda",
            warmup=args.warmup,
            iters=args.iters,
            workspace_mb=args.workspace_mb,
            trust_remote_code=args.trust_remote_code,
            use_packed_custom_mask=False,
            check_logit_alignment=False,
            fail_on_logit_mismatch=False,
        )
        target = DirectFlashInferMaskedTreeVerifyBackend(
            model_path=args.target_model,
            config=verify_config,
        )
        raw_target_ar = bench_flashinfer_full_ar_decode(
            model=target.model,
            prompt_token_ids=prompt_token_ids,
            steps=1,
            args=verify_config,
            dtype=target.dtype,
            flashinfer=target.flashinfer,
        )
        target_ar = Timing(
            median_ms=float(raw_target_ar.median_ms),
            mean_ms=float(raw_target_ar.mean_ms),
            samples_ms=[float(value) for value in raw_target_ar.samples_ms],
        )
        report["target_ar_decode_1_token"] = target_ar.to_dict()
        print("\n=== Isolated Target AR decode ===")
        print(
            "target_ar_decode_1_token        "
            f"mean={target_ar.mean_ms:9.3f} ms "
            f"median={target_ar.median_ms:9.3f} ms "
            f"rate={1000.0 / target_ar.mean_ms:8.2f} tok/s"
        )

        vocab_size = int(target.model.config.vocab_size)
        verify_routes = synthetic_verify_paths(
            k=args.k,
            d=args.d,
            shared_path_tokens=args.shared_path_tokens,
            base_token_id=args.repeat_token_id,
            vocab_size=vocab_size,
        )
        verify, verify_metadata = measure_target_verify(
            backend=target,
            prompt_token_ids=prompt_token_ids,
            routes=verify_routes,
            warmup=args.warmup,
            iters=args.iters,
        )
        report["target_verify"] = {
            **verify.to_dict(),
            "route_count": len(verify_routes),
            "path_depth": args.d,
            "shared_path_tokens": args.shared_path_tokens,
            "includes_target_kv_commit": True,
            "mean_over_target_ar1": verify.mean_ms / target_ar.mean_ms,
            "runtime_metadata": verify_metadata,
        }
        print("\n=== Isolated Target masked tree verify ===")
        print(
            f"target_verify_k{args.k}_d{args.d}      "
            f"mean={verify.mean_ms:9.3f} ms "
            f"median={verify.median_ms:9.3f} ms"
        )
        print(
            f"  mean verify/target_ar1      "
            f"{verify.mean_ms / target_ar.mean_ms:9.3f}x"
        )
        del target
        cleanup_cuda()

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n[json] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
