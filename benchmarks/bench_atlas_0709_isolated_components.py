from __future__ import annotations

import argparse
import gc
import json
import math
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
from atlas_0709.flashinfer_paged.flashinfer_backends import SGLangRouteKVMetadata
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.paged_metadata import (
    build_flashinfer_paged_kv_metadata,
)
from atlas_0709.flashinfer_paged.sglang_page_attention import AtlasPagedDecodeSpec
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


@dataclass
class NativeBatchARState:
    """Direct SGLang batch AR state without ATLAS route materialization."""

    context: PreparedDraftContext
    route_ids: list[int]
    req_pool_indices: torch.Tensor
    slot_paths: list[torch.Tensor]
    input_ids: torch.Tensor
    sequence_length: int
    counter_totals: dict[str, int]


@dataclass(frozen=True)
class NativeBatchARDecodeOutput:
    next_token_logits: torch.Tensor
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class NativeBatchARStepResult:
    decode_output: NativeBatchARDecodeOutput
    selection_stats: None = None


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
class _MeasuredSteppedIteration:
    total_ms: float
    step_ms: list[float]
    counter_deltas: list[dict[str, int]]
    counter_cumulative: list[dict[str, int]]
    counter_total_delta: dict[str, int]


@dataclass(frozen=True)
class _MeasuredCriticalPathIteration:
    total_ms: float
    counter_total_delta: dict[str, int]


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
class CriticalPathTiming:
    """End-to-end stepped workload timing with CUDA syncs only at boundaries."""

    total: Timing
    counter_total_delta_samples: list[dict[str, int]] = field(default_factory=list)
    measurement_syncs: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        report: dict[str, object] = {"total": self.total.to_dict()}
        if self.counter_total_delta_samples:
            report["hot_path_counter_total_delta"] = summarize_counter_samples(
                self.counter_total_delta_samples
            )
        if self.measurement_syncs:
            report["measurement_syncs"] = dict(self.measurement_syncs)
        return report


@dataclass(frozen=True)
class PairedCriticalPathTiming:
    arm_a: CriticalPathTiming
    arm_b: CriticalPathTiming
    measurement_protocol: Mapping[str, object]
    paired_comparison: Mapping[str, object]


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
    parser.add_argument(
        "--skip-native-batch-ar",
        action="store_true",
        help="Skip the direct GPU-resident SGLang batch-AR control.",
    )
    parser.add_argument(
        "--pair-order-seed",
        type=int,
        default=0,
        help="Rotate the balanced ABBA order for native-AR versus tree/forest pairs.",
    )
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


def state_hot_path_counters(state: Any) -> dict[str, int]:
    if isinstance(state, NativeBatchARState):
        return {
            str(key): int(value)
            for key, value in state.counter_totals.items()
        }
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


def prepare_frontier_decode_ready(state: FrontierState) -> FrontierState:
    """Move lazy row initialization and writer COW outside decode timing.

    ATLAS normally performs these operations lazily in the first frontier
    step.  The native batch-AR baseline starts from already initialized
    requests, so the primary scientific comparison uses this common
    decode-ready boundary for both arms.  The original component benchmark is
    still reported separately and keeps the production lazy costs.
    """

    if not state.routes:
        raise ValueError("decode-ready frontier cannot be empty")
    bridge = state.context.backend.route_pool
    bridge.ensure_route_rows(state.routes)
    slot_paths = [bridge._slot_path_for_route(route) for route in state.routes]
    slot_paths, _ = bridge._copy_partial_tail_pages_with_dirty_starts(
        state.routes,
        slot_paths,
    )
    lengths = {int(slot_path.numel()) for slot_path in slot_paths}
    if len(lengths) != 1:
        raise RuntimeError("decode-ready frontier routes have unequal KV lengths")

    for route, slot_path in zip(state.routes, slot_paths):
        route_id = int(route.route_id)
        row = bridge.route_rows[route_id]
        bridge._write_req_token_row(int(row.req_pool_index), slot_path)
        row.written_length = int(slot_path.numel())
        row.pending_dirty_start = None
        bridge.route_slot_paths[route_id] = slot_path
    return state


def setup_native_batch_ar_state(
    *,
    frontier_state: FrontierState,
    decode_steps: int,
) -> NativeBatchARState:
    """Convert an exactly matched ATLAS frontier into direct native batch AR.

    The logical histories, initial pending token ids, physical page-16 KV,
    allocator state, sequence order, and request-row initialization all come
    from the same tree/forest setup used by the comparison arm.  Timed decode
    then bypasses RouteState/KVTree/candidate materialization and retains the
    greedy continuation on GPU.
    """

    frontier_state = prepare_frontier_decode_ready(frontier_state)
    bridge = frontier_state.context.backend.route_pool
    slot_paths = [
        bridge._slot_path_for_route(route) for route in frontier_state.routes
    ]
    lengths = {int(slot_path.numel()) for slot_path in slot_paths}
    if len(lengths) != 1:
        raise RuntimeError("native batch AR requires equal-length histories")
    sequence_length = next(iter(lengths))
    if int(decode_steps) <= 0:
        raise ValueError("native batch AR decode_steps must be positive")
    if (
        bridge.max_context_len is not None
        and sequence_length + int(decode_steps) > int(bridge.max_context_len)
    ):
        raise RuntimeError(
            "native batch AR request rows are too short for the measured decode"
        )
    route_ids = [int(route.route_id) for route in frontier_state.routes]
    req_pool_index_list = [
        int(bridge.route_rows[route_id].req_pool_index) for route_id in route_ids
    ]
    if len(set(req_pool_index_list)) != len(req_pool_index_list):
        raise RuntimeError("native batch AR request rows must be unique")
    req_pool_indices = torch.tensor(
        req_pool_index_list,
        device=bridge.device,
        dtype=torch.long,
    )
    input_ids = torch.tensor(
        [int(route.pending_token_id) for route in frontier_state.routes],
        device=bridge.device,
        dtype=torch.long,
    )
    counters = bridge.hot_path_counters()
    counters.update(
        {
            "native_gpu_argmax_calls": 0,
            "native_req_row_batch_write_calls": 0,
        }
    )
    return NativeBatchARState(
        context=frontier_state.context,
        route_ids=route_ids,
        req_pool_indices=req_pool_indices,
        slot_paths=slot_paths,
        input_ids=input_ids,
        sequence_length=sequence_length,
        counter_totals=counters,
    )


def _assert_unique_native_output_slots(output_slot_ids: torch.Tensor) -> None:
    if int(output_slot_ids.numel()) <= 1:
        return
    duplicates = output_slot_ids[:, None].eq(output_slot_ids[None, :])
    duplicates.fill_diagonal_(False)
    condition = duplicates.logical_not().all()
    if condition.device.type == "cpu":
        if not bool(condition.item()):
            raise RuntimeError("native batch AR allocator returned duplicate KV slots")
        return
    assert_async = getattr(torch, "_assert_async", None)
    if not callable(assert_async):
        raise RuntimeError("native CUDA validation requires torch._assert_async")
    assert_async(condition, "native batch AR allocator returned duplicate KV slots")


@torch.inference_mode()
def run_native_batch_ar_step(state: NativeBatchARState) -> NativeBatchARStepResult:
    bridge = state.context.backend.route_pool
    batch_size = int(state.req_pool_indices.numel())
    previous_len = int(state.sequence_length)
    seq_len = previous_len + 1
    last_locs = torch.cat(
        [slot_path[-1:].to(dtype=torch.long) for slot_path in state.slot_paths],
        dim=0,
    )
    output_slot_ids = bridge.allocate_decode_slots(
        seq_lens=[seq_len] * batch_size,
        last_locs=last_locs,
    )
    if int(output_slot_ids.numel()) != batch_size:
        raise RuntimeError("native batch AR allocator returned the wrong slot count")
    _assert_unique_native_output_slots(output_slot_ids)
    bridge._record_pending_owned_slots(output_slot_ids)

    append_positions = torch.full(
        (batch_size,),
        previous_len,
        device=output_slot_ids.device,
        dtype=torch.long,
    )
    req_to_token = getattr(bridge.req_to_token_pool, "req_to_token", None)
    req_row_values = (
        output_slot_ids
        if not isinstance(req_to_token, torch.Tensor)
        else output_slot_ids.to(dtype=req_to_token.dtype)
    )
    bridge.req_to_token_pool.write(
        (state.req_pool_indices, append_positions),
        req_row_values,
    )
    full_slot_paths = [
        torch.cat((slot_path, output_slot_ids[index : index + 1]), dim=0)
        for index, slot_path in enumerate(state.slot_paths)
    ]

    seq_lens_cpu = torch.full((batch_size,), seq_len, dtype=torch.long)
    seq_lens = seq_lens_cpu.to(device=output_slot_ids.device)
    paged_decode_spec = None
    attention_page_size = 1
    page_index_count = batch_size * seq_len
    if bridge.use_page_attention_metadata:
        paged = build_flashinfer_paged_kv_metadata(
            full_slot_paths,
            page_size=bridge.page_size,
            validate_layout=bridge.validate_page_layout,
        )
        paged_decode_spec = AtlasPagedDecodeSpec(
            kv_indptr=paged.kv_indptr,
            kv_indices=paged.kv_page_indices,
            kv_last_page_len=paged.kv_last_page_len,
            page_size=int(paged.page_size),
        )
        attention_page_size = int(paged.page_size)
        page_index_count = int(paged.page_index_count)
    metadata = SGLangRouteKVMetadata(
        req_pool_indices=state.req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=output_slot_ids,
        positions=append_positions,
        seq_lens_sum=batch_size * seq_len,
        seq_lens_cpu=seq_lens_cpu,
        orig_seq_lens=seq_lens,
        attention_page_size=attention_page_size,
        token_index_count=batch_size * seq_len,
        page_index_count=page_index_count,
        paged_decode_spec=paged_decode_spec,
    )
    logits = state.context.backend.executor.forward_frontier(
        input_ids=state.input_ids,
        route_kv_metadata=metadata,
    )
    next_input_ids = torch.argmax(logits, dim=-1).to(dtype=torch.long)

    state.slot_paths = full_slot_paths
    for route_id, slot_path in zip(state.route_ids, full_slot_paths):
        bridge.route_slot_paths[int(route_id)] = slot_path
        bridge.route_rows[int(route_id)].written_length = int(seq_len)
        bridge.route_rows[int(route_id)].pending_dirty_start = None
        if previous_len % int(bridge.page_size) == 0:
            bridge.route_tail_page_keys[int(route_id)] = bridge._new_tail_page_key()
    state.input_ids = next_input_ids
    state.sequence_length = seq_len
    increments = {
        "attention_metadata_builds": 1,
        "attention_metadata_token_indices": batch_size * seq_len,
        "attention_metadata_page_indices": page_index_count,
        "req_row_write_calls": 1,
        "req_row_elements_written": batch_size,
        "req_row_append_calls": 1,
        "req_row_append_elements": batch_size,
        "native_gpu_argmax_calls": 1,
        "native_req_row_batch_write_calls": 1,
    }
    for name, value in increments.items():
        state.counter_totals[name] = int(state.counter_totals.get(name, 0)) + int(value)

    return NativeBatchARStepResult(
        decode_output=NativeBatchARDecodeOutput(
            next_token_logits=logits,
            metadata={
                "backend": "native_sglang_gpu_batch_ar",
                "reference_only": False,
                "paged_kv": bool(bridge.use_page_attention_metadata),
                "attention_page_size": int(attention_page_size),
                "gpu_resident_greedy": True,
                "route_materialization": False,
                "candidate_host_transfer": False,
            },
        )
    )


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


def _run_stepped_warmup(
    *,
    setup_fn: Callable[[], Any],
    step_fn: Callable[[Any], Any],
    depth: int,
) -> None:
    state = setup_fn()
    for _ in range(depth):
        step_fn(state)
    sync_cuda()


def _measure_stepped_iteration(
    *,
    setup_fn: Callable[[], Any],
    step_fn: Callable[[Any], Any],
    depth: int,
) -> _MeasuredSteppedIteration:
    state = setup_fn()
    setup_counters = state_hot_path_counters(state)
    sync_cuda()
    total_start = time.perf_counter()
    step_results: list[Any] = []
    step_ms: list[float] = []
    for _ in range(depth):
        sync_cuda()
        step_start = time.perf_counter()
        result = step_fn(state)
        sync_cuda()
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
        step_results.append(result)
    total_ms = (time.perf_counter() - total_start) * 1000.0

    # Extract Python counter dictionaries only after the timed region.  The
    # backend stores an immutable cumulative snapshot in every step result, so
    # per-depth evidence does not perturb total latency.
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
    counter_deltas: list[dict[str, int]] = []
    counter_cumulative: list[dict[str, int]] = []
    for step_index, counters_after_step in enumerate(step_counter_totals):
        bridge_delta = subtract_counters(counters_after_step, previous_counters)
        cumulative_selection_counters = add_counters(
            cumulative_selection_counters,
            step_selection_deltas[step_index],
        )
        counter_deltas.append(
            add_total_host_transfer_counters(
                add_counters(bridge_delta, step_selection_deltas[step_index])
            )
        )
        counter_cumulative.append(
            add_total_host_transfer_counters(
                add_counters(
                    subtract_counters(counters_after_step, setup_counters),
                    cumulative_selection_counters,
                )
            )
        )
        previous_counters = counters_after_step
    counter_total_delta = add_total_host_transfer_counters(
        add_counters(
            subtract_counters(previous_counters, setup_counters),
            cumulative_selection_counters,
        )
    )
    return _MeasuredSteppedIteration(
        total_ms=total_ms,
        step_ms=step_ms,
        counter_deltas=counter_deltas,
        counter_cumulative=counter_cumulative,
        counter_total_delta=counter_total_delta,
    )


def _measurement_sync_report(*, depth: int, warmup: int, iters: int) -> dict[str, object]:
    measured_syncs_per_iteration = 1 + 2 * int(depth)
    return {
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


def _stepped_timing_from_iterations(
    iterations: Sequence[_MeasuredSteppedIteration],
    *,
    depth: int,
    warmup: int,
) -> SteppedTiming:
    if not iterations:
        raise ValueError("at least one measured iteration is required")
    return SteppedTiming(
        total=timing([iteration.total_ms for iteration in iterations]),
        steps=[
            timing([iteration.step_ms[index] for iteration in iterations])
            for index in range(depth)
        ],
        counter_delta_samples=[
            [iteration.counter_deltas[index] for iteration in iterations]
            for index in range(depth)
        ],
        counter_cumulative_samples=[
            [iteration.counter_cumulative[index] for iteration in iterations]
            for index in range(depth)
        ],
        counter_total_delta_samples=[
            iteration.counter_total_delta for iteration in iterations
        ],
        measurement_syncs=_measurement_sync_report(
            depth=depth,
            warmup=warmup,
            iters=len(iterations),
        ),
    )


def measure_stepped_component(
    *,
    setup_fn: Callable[[], Any],
    step_fn: Callable[[Any], Any],
    depth: int,
    warmup: int,
    iters: int,
) -> SteppedTiming:
    for _ in range(warmup):
        _run_stepped_warmup(setup_fn=setup_fn, step_fn=step_fn, depth=depth)
    iterations = [
        _measure_stepped_iteration(
            setup_fn=setup_fn,
            step_fn=step_fn,
            depth=depth,
        )
        for _ in range(iters)
    ]
    return _stepped_timing_from_iterations(
        iterations,
        depth=depth,
        warmup=warmup,
    )


def _run_critical_path_warmup(
    *,
    setup_fn: Callable[[], Any],
    step_fn: Callable[[Any], Any],
    depth: int,
) -> None:
    state = setup_fn()
    for _ in range(depth):
        step_fn(state)
    sync_cuda()


def _measure_critical_path_iteration(
    *,
    setup_fn: Callable[[], Any],
    step_fn: Callable[[Any], Any],
    depth: int,
) -> _MeasuredCriticalPathIteration:
    """Measure d dependent decode steps without host barriers between them."""

    state = setup_fn()
    setup_counters = state_hot_path_counters(state)
    sync_cuda()
    start = time.perf_counter()
    step_results: list[Any] = []
    for _ in range(depth):
        result = step_fn(state)
        step_results.append(result)
    sync_cuda()
    total_ms = (time.perf_counter() - start) * 1000.0
    selection_counters: dict[str, int] = {}
    for result in step_results:
        selection_counters = add_counters(
            selection_counters,
            step_selection_counter_delta(result),
        )
    counter_total_delta = add_total_host_transfer_counters(
        add_counters(
            subtract_counters(state_hot_path_counters(state), setup_counters),
            selection_counters,
        )
    )
    return _MeasuredCriticalPathIteration(
        total_ms=total_ms,
        counter_total_delta=counter_total_delta,
    )


def _critical_path_timing_from_iterations(
    iterations: Sequence[_MeasuredCriticalPathIteration],
    *,
    warmup: int,
) -> CriticalPathTiming:
    if not iterations:
        raise ValueError("at least one critical-path iteration is required")
    iters = len(iterations)
    return CriticalPathTiming(
        total=timing([iteration.total_ms for iteration in iterations]),
        counter_total_delta_samples=[
            iteration.counter_total_delta for iteration in iterations
        ],
        measurement_syncs={
            "kind": "benchmark_boundary_cuda_synchronize",
            "warmup_after_workload_total": int(warmup),
            "before_timed_region_per_iteration": 1,
            "workload_completion_inside_timing_per_iteration": 1,
            "host_barriers_between_depths": 0,
            "measured_per_iteration": 2,
            "measured_total": 2 * int(iters),
            "overall_total": int(warmup) + 2 * int(iters),
            "counter_summary_extraction_in_timed_region": False,
        },
    )


def _paired_numeric_summary(values: Sequence[float]) -> dict[str, object]:
    samples = [float(value) for value in values]
    return {
        "median": float(statistics.median(samples)),
        "mean": float(statistics.fmean(samples)),
        "samples": samples,
    }


def _paired_ratio_summary(values: Sequence[float]) -> dict[str, object]:
    samples = [float(value) for value in values]
    if not samples or min(samples) <= 0.0:
        raise ValueError("paired ratios must be positive")
    return {
        "median": float(statistics.median(samples)),
        "geometric_mean": float(
            math.exp(statistics.fmean(math.log(value) for value in samples))
        ),
        "arithmetic_mean_diagnostic": float(statistics.fmean(samples)),
        "samples": samples,
    }


def measure_paired_critical_path_components(
    *,
    setup_a: Callable[[], Any],
    step_a: Callable[[Any], Any],
    arm_a_name: str,
    setup_b: Callable[[], Any],
    step_b: Callable[[Any], Any],
    arm_b_name: str,
    depth: int,
    warmup: int,
    iters: int,
    order_seed: int,
) -> PairedCriticalPathTiming:
    """Pair two d-step critical paths in ABBA blocks with fresh setup."""

    if int(depth) <= 0:
        raise ValueError("paired critical-path depth must be positive")
    if int(iters) < 2 or int(iters) % 2:
        raise ValueError("paired critical-path iters must be a positive even number")
    workloads = {
        "a": (setup_a, step_a),
        "b": (setup_b, step_b),
    }

    def round_order(round_index: int) -> tuple[str, str]:
        return (
            ("a", "b")
            if (int(order_seed) + int(round_index)) % 2 == 0
            else ("b", "a")
        )

    warmup_orders: list[str] = []
    for warmup_index in range(int(warmup)):
        order = round_order(warmup_index)
        warmup_orders.append("".join(order).upper())
        for label in order:
            setup_fn, step_fn = workloads[label]
            _run_critical_path_warmup(
                setup_fn=setup_fn,
                step_fn=step_fn,
                depth=depth,
            )

    iterations: dict[str, list[_MeasuredCriticalPathIteration]] = {"a": [], "b": []}
    measured_orders: list[str] = []
    for iteration_index in range(int(iters)):
        order = round_order(iteration_index)
        measured_orders.append("".join(order).upper())
        for label in order:
            setup_fn, step_fn = workloads[label]
            iterations[label].append(
                _measure_critical_path_iteration(
                    setup_fn=setup_fn,
                    step_fn=step_fn,
                    depth=depth,
                )
            )

    arm_a = _critical_path_timing_from_iterations(
        iterations["a"],
        warmup=warmup,
    )
    arm_b = _critical_path_timing_from_iterations(
        iterations["b"],
        warmup=warmup,
    )
    blocks: list[dict[str, object]] = []
    block_deltas: list[float] = []
    block_ratios: list[float] = []
    for block_start in range(0, int(iters), 2):
        indices = [block_start, block_start + 1]
        a_samples = [arm_a.total.samples_ms[index] for index in indices]
        b_samples = [arm_b.total.samples_ms[index] for index in indices]
        if min(*a_samples, *b_samples) <= 0.0:
            raise RuntimeError("paired timing samples must be positive")
        delta_ms = float(statistics.fmean(b_samples) - statistics.fmean(a_samples))
        ratio = float(
            math.sqrt(
                (b_samples[0] * b_samples[1])
                / (a_samples[0] * a_samples[1])
            )
        )
        block_deltas.append(delta_ms)
        block_ratios.append(ratio)
        blocks.append(
            {
                "round_indices": indices,
                "a_sample_indices": indices,
                "b_sample_indices": indices,
                "delta_b_minus_a_ms": delta_ms,
                "geometric_ratio_b_over_a": ratio,
            }
        )

    protocol = {
        "name": "paired_abba_fresh_setup_boundary_sync",
        "arm_a": str(arm_a_name),
        "arm_b": str(arm_b_name),
        "order_seed": int(order_seed),
        "warmup_round_orders": warmup_orders,
        "measured_round_orders": measured_orders,
        "iters_per_arm": int(iters),
        "fresh_setup_per_sample": True,
        "setup_and_prefill_timed": False,
        "host_barriers_between_depths": 0,
        "complete_abba_blocks": int(iters) // 2,
    }
    comparison = {
        "blocks": blocks,
        "total": {
            "delta_b_minus_a_ms": _paired_numeric_summary(block_deltas),
            "geometric_ratio_b_over_a": _paired_ratio_summary(block_ratios),
            "unpaired_mean_ratio_b_over_a_diagnostic": (
                arm_b.total.mean_ms / arm_a.total.mean_ms
            ),
        },
    }
    return PairedCriticalPathTiming(
        arm_a=arm_a,
        arm_b=arm_b,
        measurement_protocol=protocol,
        paired_comparison=comparison,
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
    if (
        not args.skip_native_batch_ar
        and not (args.skip_tree and args.skip_forest)
        and (args.iters < 2 or args.iters % 2)
    ):
        raise SystemExit(
            "native batch-AR paired comparison requires an even --iters >= 2"
        )

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
            "pair_order_seed": int(args.pair_order_seed),
            "native_batch_ar_enabled": not args.skip_native_batch_ar,
            "native_batch_ar_definition": (
                "direct_sglang_forward_gpu_argmax_batched_req_row_append"
            ),
            "primary_speed_protocol": "paired_abba_boundary_sync_decode_ready",
            "primary_speed_setup_boundary": (
                "prefill_route_initialization_req_rows_and_initial_tail_cow_excluded_"
                "for_both_arms"
            ),
            "cuda_graph_enabled": False,
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
                    "comparison_role": "atlas_control_path_diagnostic_not_native_ar",
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
                report["build_tree"]["comparison_role"] = (
                    "sequential_unpaired_component_diagnostic"
                )
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

                if not args.skip_native_batch_ar:
                    paired_tree = measure_paired_critical_path_components(
                        setup_a=lambda: setup_native_batch_ar_state(
                            frontier_state=setup_tree_state(
                                runner=runner,
                                prompt_token_ids=prompt_token_ids,
                                page_size=args.page_size,
                                prefill_chunk_size=args.prefill_chunk_size,
                                k=args.k,
                            ),
                            decode_steps=args.d,
                        ),
                        step_a=run_native_batch_ar_step,
                        arm_a_name="native_eager_sglang_batch_ar_page16",
                        setup_b=lambda: prepare_frontier_decode_ready(
                            setup_tree_state(
                                runner=runner,
                                prompt_token_ids=prompt_token_ids,
                                page_size=args.page_size,
                                prefill_chunk_size=args.prefill_chunk_size,
                                k=args.k,
                            )
                        ),
                        step_b=run_tree_step,
                        arm_b_name="atlas_tree_global_topk_page16",
                        depth=args.d,
                        warmup=args.warmup,
                        iters=args.iters,
                        order_seed=args.pair_order_seed,
                    )
                    tree_native_key = "native_eager_sglang_batch_ar_tree_workload"
                    tree_ready_key = "build_tree_decode_ready_critical_path"
                    tree_pair_key = "native_batch_ar_vs_build_tree_paired"
                    report[tree_native_key] = {
                        "baseline_scope": "low_level_eager_sglang_model_runner",
                        "full_scheduler_or_server_baseline": False,
                        "batch_size": int(args.k),
                        "decode_steps": int(args.d),
                        "initial_sequence_length": len(prompt_token_ids),
                        "final_sequence_length": len(prompt_token_ids) + int(args.d),
                        "model_input_tokens": int(args.k * args.d),
                        "initial_state_source": "same_atlas_tree_frontier",
                        "gpu_resident_greedy": True,
                        "initial_input_ids_staged_on_gpu_before_timing": True,
                        "route_materialization": False,
                        "candidate_host_transfer": False,
                        "setup_and_prefill_included": False,
                        "initial_req_rows_and_initial_tail_cow_excluded": True,
                        **paired_tree.arm_a.to_dict(),
                    }
                    report[tree_ready_key] = {
                        "batch_size": int(args.k),
                        "decode_steps": int(args.d),
                        "initial_sequence_length": len(prompt_token_ids),
                        "final_sequence_length": len(prompt_token_ids) + int(args.d),
                        "model_input_tokens": int(args.k * args.d),
                        "setup_and_prefill_included": False,
                        "initial_req_rows_and_initial_tail_cow_excluded": True,
                        "frontier_input_ids_built_from_routes_inside_timing": True,
                        **paired_tree.arm_b.to_dict(),
                    }
                    report[tree_pair_key] = {
                        "measurement_protocol": dict(
                            paired_tree.measurement_protocol
                        ),
                        "paired_comparison": dict(paired_tree.paired_comparison),
                        "arm_a_report_key": tree_native_key,
                        "arm_b_report_key": tree_ready_key,
                        "comparison_scope": (
                            "decode_critical_path_including_algorithm_control_plane"
                        ),
                        "pure_model_kernel_comparison": False,
                    }
                    tree_pair_total = paired_tree.paired_comparison["total"]
                    tree_pair_ratio = float(
                        tree_pair_total["geometric_ratio_b_over_a"][
                            "geometric_mean"
                        ]
                    )
                    tree_pair_delta = float(
                        tree_pair_total["delta_b_minus_a_ms"]["mean"]
                    )
                    report["build_tree"][
                        "paired_decode_ready_geomean_over_native_batch_ar"
                    ] = tree_pair_ratio
                    report["build_tree"]["native_batch_ar_paired_report_key"] = (
                        tree_pair_key
                    )
                    print("\n=== Primary paired native batch-AR vs tree ===")
                    print(
                        "native_batch_ar_tree_workload  "
                        f"mean={paired_tree.arm_a.total.mean_ms:9.3f} ms "
                        f"median={paired_tree.arm_a.total.median_ms:9.3f} ms"
                    )
                    print(
                        "atlas_tree_decode_ready        "
                        f"mean={paired_tree.arm_b.total.mean_ms:9.3f} ms "
                        f"median={paired_tree.arm_b.total.median_ms:9.3f} ms"
                    )
                    print(
                        "  paired tree/native-ar       "
                        f"{tree_pair_ratio:9.3f}x  delta={tree_pair_delta:9.3f} ms"
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
                    "comparison_role": "atlas_control_path_diagnostic_not_native_ar",
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
                report["build_forest"]["comparison_role"] = (
                    "sequential_unpaired_component_diagnostic"
                )
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

                if not args.skip_native_batch_ar:
                    paired_forest = measure_paired_critical_path_components(
                        setup_a=lambda: setup_native_batch_ar_state(
                            frontier_state=setup_forest_state(
                                runner=runner,
                                prompt_token_ids=prompt_token_ids,
                                page_size=args.page_size,
                                prefill_chunk_size=args.prefill_chunk_size,
                                k=args.k,
                                d=args.d,
                            ),
                            decode_steps=args.d,
                        ),
                        step_a=run_native_batch_ar_step,
                        arm_a_name="native_eager_sglang_batch_ar_page16",
                        setup_b=lambda: prepare_frontier_decode_ready(
                            setup_forest_state(
                                runner=runner,
                                prompt_token_ids=prompt_token_ids,
                                page_size=args.page_size,
                                prefill_chunk_size=args.prefill_chunk_size,
                                k=args.k,
                                d=args.d,
                            )
                        ),
                        step_b=run_forest_step,
                        arm_b_name="atlas_forest_per_root_topk_page16",
                        depth=args.d,
                        warmup=args.warmup,
                        iters=args.iters,
                        order_seed=args.pair_order_seed,
                    )
                    forest_initial_length = len(prompt_token_ids) + int(args.d)
                    forest_native_key = (
                        "native_eager_sglang_batch_ar_forest_workload"
                    )
                    forest_ready_key = "build_forest_decode_ready_critical_path"
                    forest_pair_key = "native_batch_ar_vs_build_forest_paired"
                    report[forest_native_key] = {
                        "baseline_scope": "low_level_eager_sglang_model_runner",
                        "full_scheduler_or_server_baseline": False,
                        "batch_size": int(args.k * args.k),
                        "decode_steps": int(args.d),
                        "initial_sequence_length": forest_initial_length,
                        "final_sequence_length": forest_initial_length + int(args.d),
                        "model_input_tokens": int(args.k * args.k * args.d),
                        "initial_state_source": "same_atlas_forest_frontier",
                        "gpu_resident_greedy": True,
                        "initial_input_ids_staged_on_gpu_before_timing": True,
                        "route_materialization": False,
                        "candidate_host_transfer": False,
                        "setup_and_prefill_included": False,
                        "initial_req_rows_and_initial_tail_cow_excluded": True,
                        **paired_forest.arm_a.to_dict(),
                    }
                    report[forest_ready_key] = {
                        "batch_size": int(args.k * args.k),
                        "decode_steps": int(args.d),
                        "initial_sequence_length": forest_initial_length,
                        "final_sequence_length": forest_initial_length + int(args.d),
                        "model_input_tokens": int(args.k * args.k * args.d),
                        "setup_and_prefill_included": False,
                        "initial_req_rows_and_initial_tail_cow_excluded": True,
                        "frontier_input_ids_built_from_routes_inside_timing": True,
                        **paired_forest.arm_b.to_dict(),
                    }
                    report[forest_pair_key] = {
                        "measurement_protocol": dict(
                            paired_forest.measurement_protocol
                        ),
                        "paired_comparison": dict(paired_forest.paired_comparison),
                        "arm_a_report_key": forest_native_key,
                        "arm_b_report_key": forest_ready_key,
                        "comparison_scope": (
                            "decode_critical_path_including_algorithm_control_plane"
                        ),
                        "pure_model_kernel_comparison": False,
                    }
                    forest_pair_total = paired_forest.paired_comparison["total"]
                    forest_pair_ratio = float(
                        forest_pair_total["geometric_ratio_b_over_a"][
                            "geometric_mean"
                        ]
                    )
                    forest_pair_delta = float(
                        forest_pair_total["delta_b_minus_a_ms"]["mean"]
                    )
                    report["build_forest"][
                        "paired_decode_ready_geomean_over_native_batch_ar"
                    ] = forest_pair_ratio
                    report["build_forest"]["native_batch_ar_paired_report_key"] = (
                        forest_pair_key
                    )
                    print("\n=== Primary paired native batch-AR vs forest ===")
                    print(
                        "native_batch_ar_forest_workload "
                        f"mean={paired_forest.arm_a.total.mean_ms:9.3f} ms "
                        f"median={paired_forest.arm_a.total.median_ms:9.3f} ms"
                    )
                    print(
                        "atlas_forest_decode_ready       "
                        f"mean={paired_forest.arm_b.total.mean_ms:9.3f} ms "
                        f"median={paired_forest.arm_b.total.median_ms:9.3f} ms"
                    )
                    print(
                        "  paired forest/native-ar     "
                        f"{forest_pair_ratio:9.3f}x  "
                        f"delta={forest_pair_delta:9.3f} ms"
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
