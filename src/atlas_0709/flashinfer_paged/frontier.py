from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from .kv import KVTreeStore
from .types import DecodePhase, FrontierDecodeOutput, FrontierStepOutput, PendingCandidate, RouteState
from .utils import global_topk_candidates, group_candidates_by_stage1_root, tensor_to_int_list


class FrontierModelBackend(Protocol):
    def decode_frontier_one_token(
        self,
        active_routes: Sequence[RouteState],
        attention_backend: Any = None,
    ) -> FrontierDecodeOutput:
        ...


class SelectionPolicy(Protocol):
    phase: DecodePhase

    def select(
        self,
        candidates: Sequence[PendingCandidate],
        *,
        k: int,
        active_routes: Sequence[RouteState],
    ) -> list[PendingCandidate]:
        ...


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


def collect_pending_candidates(
    decoded_routes: Sequence[RouteState],
    next_token_logits: torch.Tensor,
    *,
    k: int,
) -> list[PendingCandidate]:
    if len(decoded_routes) != int(next_token_logits.shape[0]):
        raise ValueError("decoded_routes and next_token_logits must have the same batch size")

    candidates: list[PendingCandidate] = []
    actual_k = min(int(k), int(next_token_logits.shape[-1]))
    token_logprobs, token_ids = torch.topk(
        torch.log_softmax(next_token_logits, dim=-1),
        k=actual_k,
        dim=-1,
    )
    for row, route in enumerate(decoded_routes):
        for rank in range(actual_k):
            token_id = token_ids[row, rank]
            token_logprob = token_logprobs[row, rank]
            candidates.append(
                PendingCandidate(
                    parent_route_id=route.route_id,
                    stage1_root_id=route.stage1_root_id,
                    pending_token_id=int(token_id),
                    cumulative_logprob=route.cumulative_logprob + float(token_logprob.detach().cpu()),
                    rank_in_parent=rank,
                    parent_logprob=route.cumulative_logprob,
                )
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
) -> FrontierStepOutput:
    """Decode exactly one pending token for every active route.

    Tree and forest construction both call this function. They differ only in
    the provided selection policy and backend descriptor.
    """

    if not active_routes:
        raise ValueError("active_routes cannot be empty")

    decode_output = model_backend.decode_frontier_one_token(
        active_routes,
        attention_backend=attention_backend,
    )
    decoded_routes = route_store.mark_routes_materialized(
        list(active_routes),
        tensor_to_int_list(decode_output.new_node_ids),
        phase=selection_policy.phase,
    )
    candidates = collect_pending_candidates(
        decoded_routes,
        decode_output.next_token_logits,
        k=k,
    )
    selected_candidates = selection_policy.select(
        candidates,
        k=k,
        active_routes=decoded_routes,
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
    )
