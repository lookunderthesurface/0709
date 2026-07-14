from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from .kv import KVTreeStore
from .sampling import DrafterSamplingContext, batch_parent_token_candidates
from .types import (
    DecodePhase,
    FrontierDecodeOutput,
    FrontierSelectionStats,
    FrontierStepOutput,
    PendingCandidate,
    PendingCandidateBatch,
    RouteState,
)
from .utils import (
    global_topk_candidate_indices,
    global_topk_candidates,
    group_candidates_by_stage1_root,
    per_root_topk_candidate_indices,
)


class FrontierModelBackend(Protocol):
    def decode_frontier_one_token(
        self,
        active_routes: Sequence[RouteState],
        attention_backend: Any = None,
    ) -> FrontierDecodeOutput: ...


class SelectionPolicy(Protocol):
    phase: DecodePhase

    def select(
        self,
        candidates: Sequence[PendingCandidate],
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> list[PendingCandidate]: ...

    def select_indices(
        self,
        candidates: PendingCandidateBatch,
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class GlobalTopKSelection:
    phase: DecodePhase = DecodePhase.STAGE1

    def select(
        self,
        candidates: Sequence[PendingCandidate],
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> list[PendingCandidate]:
        return global_topk_candidates(candidates, num_selected=k)

    def select_indices(
        self,
        candidates: PendingCandidateBatch,
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> torch.Tensor:
        return global_topk_candidate_indices(candidates, num_selected=k)


@dataclass(frozen=True)
class PerStage1RootTopKSelection:
    phase: DecodePhase = DecodePhase.STAGE2

    def select(
        self,
        candidates: Sequence[PendingCandidate],
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> list[PendingCandidate]:
        active_roots = {route.stage1_root_id for route in active_routes}
        grouped = group_candidates_by_stage1_root(candidates)

        selected: list[PendingCandidate] = []
        for root_id in sorted(active_roots):
            root_candidates = grouped.get(root_id, [])
            if not root_candidates:
                raise ValueError(f"stage1 root {root_id} produced no candidates")
            selected.extend(global_topk_candidates(root_candidates, num_selected=k))
        return selected

    def select_indices(
        self,
        candidates: PendingCandidateBatch,
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> torch.Tensor:
        route_counts = Counter(int(route.stage1_root_id) for route in active_routes)
        root_candidate_counts = [
            (root_id, route_count * candidates.candidates_per_parent)
            for root_id, route_count in route_counts.items()
        ]
        empty_roots = [
            root_id
            for root_id, candidate_count in root_candidate_counts
            if candidate_count == 0
        ]
        if empty_roots:
            raise ValueError(f"stage1 root {min(empty_roots)} produced no candidates")
        return per_root_topk_candidate_indices(
            candidates,
            num_selected=k,
            root_candidate_counts=root_candidate_counts,
        )


def build_pending_candidate_batch(
    decoded_routes: Sequence[RouteState],
    next_token_logits: torch.Tensor,
    *,
    k: int,
    sampling: DrafterSamplingContext | None = None,
    relative_parent_paths: Sequence[Sequence[int]] | None = None,
) -> PendingCandidateBatch:
    if len(decoded_routes) != int(next_token_logits.shape[0]):
        raise ValueError(
            "decoded_routes and next_token_logits must have the same batch size"
        )
    if k <= 0:
        raise ValueError("k must be positive")

    if relative_parent_paths is None:
        relative_parent_paths = tuple(() for _ in decoded_routes)
    token_logprobs, token_ids = batch_parent_token_candidates(
        next_token_logits,
        k=k,
        sampling=sampling,
        relative_parent_paths=relative_parent_paths,
    )
    actual_k = int(token_ids.shape[-1])
    device = next_token_logits.device
    batch_size = len(decoded_routes)
    parent_route_ids = torch.tensor(
        [int(route.route_id) for route in decoded_routes],
        device=device,
        dtype=torch.long,
    ).repeat_interleave(actual_k)
    stage1_root_ids = torch.tensor(
        [int(route.stage1_root_id) for route in decoded_routes],
        device=device,
        dtype=torch.long,
    ).repeat_interleave(actual_k)
    parent_row_indices = torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    ).repeat_interleave(actual_k)
    ranks_in_parent = torch.arange(
        actual_k,
        device=device,
        dtype=torch.long,
    ).repeat(batch_size)
    parent_logprobs = torch.tensor(
        [float(route.cumulative_logprob) for route in decoded_routes],
        device=device,
        dtype=torch.float64,
    ).repeat_interleave(actual_k)
    cumulative_logprobs = parent_logprobs + token_logprobs.reshape(-1).to(torch.float64)
    return PendingCandidateBatch(
        parent_route_ids=parent_route_ids,
        stage1_root_ids=stage1_root_ids,
        parent_row_indices=parent_row_indices,
        pending_token_ids=token_ids.reshape(-1),
        cumulative_logprobs=cumulative_logprobs,
        parent_logprobs=parent_logprobs,
        ranks_in_parent=ranks_in_parent,
        candidates_per_parent=actual_k,
    )


def materialize_pending_candidate_batch(
    candidates: PendingCandidateBatch,
    selected_indices: torch.Tensor,
    *,
    new_node_ids_cpu: Sequence[int] | None = None,
    new_node_ids_to_transfer: torch.Tensor | None = None,
    new_node_ids_source: str = "not_applicable",
) -> tuple[
    list[PendingCandidate],
    list[PendingCandidate],
    list[int],
    FrontierSelectionStats,
]:
    """Materialize all Python descriptors through exactly one packed host batch.

    Integer fields and float64 score bit patterns share one int64 record tensor.
    The selected GPU indices are encoded as an order column, so they do not
    require a second device-to-host copy.  Backends that cannot expose CPU node
    ids can append those ids to the same transfer.
    """

    candidate_count = candidates.candidate_count
    selected_count = int(selected_indices.numel())
    device = candidates.pending_token_ids.device
    selected_indices = selected_indices.to(device=device, dtype=torch.long)
    selection_order = torch.full(
        (candidate_count,),
        -1,
        device=device,
        dtype=torch.long,
    )
    if selected_count:
        selection_order[selected_indices] = torch.arange(
            selected_count,
            device=device,
            dtype=torch.long,
        )

    record_kind = torch.zeros(candidate_count, device=device, dtype=torch.long)
    candidate_records = torch.stack(
        (
            record_kind,
            candidates.parent_route_ids.to(dtype=torch.long),
            candidates.stage1_root_ids.to(dtype=torch.long),
            candidates.pending_token_ids.to(dtype=torch.long),
            candidates.ranks_in_parent.to(dtype=torch.long),
            candidates.cumulative_logprobs.to(dtype=torch.float64)
            .contiguous()
            .view(torch.int64),
            candidates.parent_logprobs.to(dtype=torch.float64)
            .contiguous()
            .view(torch.int64),
            selection_order,
        ),
        dim=1,
    )

    records = [candidate_records]
    transferred_node_count = 0
    if new_node_ids_to_transfer is not None:
        node_ids = new_node_ids_to_transfer.to(device=device, dtype=torch.long).reshape(
            -1
        )
        transferred_node_count = int(node_ids.numel())
        node_records = torch.zeros(
            (transferred_node_count, candidate_records.shape[1]),
            device=device,
            dtype=torch.long,
        )
        node_records[:, 0] = 1
        node_records[:, 1] = node_ids
        records.append(node_records)

    packed = torch.cat(records, dim=0)
    packed_host = packed.detach().cpu()
    packed_rows = packed_host.tolist()
    materialized: list[PendingCandidate] = []
    selected_by_order: list[PendingCandidate | None] = [None] * selected_count
    for row in packed_rows[:candidate_count]:
        candidate = PendingCandidate(
            parent_route_id=int(row[1]),
            stage1_root_id=int(row[2]),
            pending_token_id=int(row[3]),
            cumulative_logprob=_float64_from_int64_bits(int(row[5])),
            rank_in_parent=int(row[4]),
            parent_logprob=_float64_from_int64_bits(int(row[6])),
        )
        materialized.append(candidate)
        selected_order = int(row[7])
        if selected_order >= 0:
            selected_by_order[selected_order] = candidate

    if any(candidate is None for candidate in selected_by_order):
        raise RuntimeError("selected candidate order was not fully materialized")
    selected = [candidate for candidate in selected_by_order if candidate is not None]

    resolved_node_ids = (
        [int(node_id) for node_id in new_node_ids_cpu]
        if new_node_ids_cpu is not None
        else [int(row[1]) for row in packed_rows[candidate_count:]]
    )
    if new_node_ids_cpu is None and len(resolved_node_ids) != transferred_node_count:
        raise RuntimeError("new node id transfer produced the wrong number of ids")

    is_host_transfer = packed.device.type != "cpu"
    stats = FrontierSelectionStats(
        candidate_count=candidate_count,
        selected_count=selected_count,
        host_materialization_batches=1,
        host_materialization_elements=int(packed.numel()),
        host_transfer_batches=int(is_host_transfer),
        host_transfer_elements=int(packed.numel()) if is_host_transfer else 0,
        selection_device=str(device),
        new_node_ids_source=new_node_ids_source,
    )
    return materialized, selected, resolved_node_ids, stats


def collect_pending_candidates(
    decoded_routes: Sequence[RouteState],
    next_token_logits: torch.Tensor,
    *,
    k: int,
) -> list[PendingCandidate]:
    candidate_batch = build_pending_candidate_batch(
        decoded_routes,
        next_token_logits,
        k=k,
    )
    empty_selection = torch.empty(
        (0,),
        device=next_token_logits.device,
        dtype=torch.long,
    )
    candidates, _, _, _ = materialize_pending_candidate_batch(
        candidate_batch,
        empty_selection,
    )
    return candidates


def advance_frontier_one_token(
    active_routes: Sequence[RouteState],
    *,
    k: int,
    route_store: KVTreeStore,
    model_backend: FrontierModelBackend,
    selection_policy: SelectionPolicy,
    attention_backend: Any = None,
    sampling: DrafterSamplingContext | None = None,
) -> FrontierStepOutput:
    """Decode one frontier depth and materialize its GPU selection once."""

    if not active_routes:
        raise ValueError("active_routes cannot be empty")

    node_counter_before = _node_counter_state(route_store)
    decode_output = model_backend.decode_frontier_one_token(
        active_routes,
        attention_backend=attention_backend,
    )
    node_counter_after = _node_counter_state(route_store)

    candidate_batch = build_pending_candidate_batch(
        active_routes,
        decode_output.next_token_logits,
        k=k,
        sampling=sampling,
        relative_parent_paths=[
            route_store.pending_token_path(route) for route in active_routes
        ],
    )
    selected_indices = selection_policy.select_indices(
        candidate_batch,
        k=k,
        active_routes=active_routes,
    )
    cpu_node_ids, node_ids_to_transfer, node_ids_source = _resolve_new_node_ids(
        decode_output,
        expected_count=len(active_routes),
        counter_before=node_counter_before,
        counter_after=node_counter_after,
    )
    candidates, selected_candidates, new_node_ids, selection_stats = (
        materialize_pending_candidate_batch(
            candidate_batch,
            selected_indices,
            new_node_ids_cpu=cpu_node_ids,
            new_node_ids_to_transfer=node_ids_to_transfer,
            new_node_ids_source=node_ids_source,
        )
    )

    decoded_routes = route_store.mark_routes_materialized(
        list(active_routes),
        new_node_ids,
        phase=selection_policy.phase,
    )
    next_routes = route_store.materialize_route_descriptors(
        selected_candidates,
        parent_routes=decoded_routes,
    )
    route_store.release_routes_without_descendants(decoded_routes, next_routes)

    return FrontierStepOutput(
        decoded_routes=decoded_routes,
        next_routes=next_routes,
        candidates=candidates,
        selected_candidates=selected_candidates,
        decode_output=decode_output,
        selection_stats=selection_stats,
    )


def _resolve_new_node_ids(
    decode_output: FrontierDecodeOutput,
    *,
    expected_count: int,
    counter_before: tuple[int, int] | None,
    counter_after: tuple[int, int] | None,
) -> tuple[list[int] | None, torch.Tensor | None, str]:
    if decode_output.new_node_ids_cpu is not None:
        node_ids = [int(node_id) for node_id in decode_output.new_node_ids_cpu]
        if len(node_ids) != expected_count:
            raise ValueError("new_node_ids_cpu must have one id per active route")
        return node_ids, None, "decode_output_cpu"

    if not isinstance(decode_output.new_node_ids, torch.Tensor):
        node_ids = [int(node_id) for node_id in decode_output.new_node_ids]
        if len(node_ids) != expected_count:
            raise ValueError("new_node_ids must have one id per active route")
        return node_ids, None, "decode_output_sequence"

    if int(decode_output.new_node_ids.numel()) != expected_count:
        raise ValueError("new_node_ids must have one id per active route")
    if (
        counter_before is not None
        and counter_after is not None
        and counter_before[1] == counter_after[1]
        and counter_after[0] == counter_before[0] + expected_count * counter_before[1]
    ):
        node_ids = [
            counter_before[0] + index * counter_before[1]
            for index in range(expected_count)
        ]
        return node_ids, None, "route_store_counter"
    return None, decode_output.new_node_ids, "packed_transfer"


def _node_counter_state(route_store: KVTreeStore) -> tuple[int, int] | None:
    """Read itertools.count state without consuming an id or touching CUDA."""

    counter = getattr(route_store, "_node_ids", None)
    reduce_method = getattr(counter, "__reduce__", None)
    if not callable(reduce_method):
        return None
    try:
        reduced = reduce_method()
        args = reduced[1]
        next_value = int(args[0])
        step = int(args[1]) if len(args) > 1 else 1
    except (IndexError, TypeError, ValueError):
        return None
    return next_value, step


def _float64_from_int64_bits(value: int) -> float:
    return struct.unpack("=d", struct.pack("=q", int(value)))[0]
