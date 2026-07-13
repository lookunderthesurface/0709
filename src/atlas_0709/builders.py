from __future__ import annotations

from typing import Sequence

import torch

from .backends import BatchDecodeBackend
from .route_store import RouteStore
from .types import BuildResult, DecodePhase, FrontierStepResult, PendingCandidate, PrefixState, RouteState
from .utils import batch_topk_logprobs, select_global_topk, select_topk_per_stage1_root, topk_logprobs


def initialize_stage1_routes(
    prefix_state: PrefixState,
    *,
    k: int,
    route_store: RouteStore,
) -> list[RouteState]:
    token_logprobs, token_ids = topk_logprobs(prefix_state.next_token_logits, k=k)
    routes: list[RouteState] = []
    for token_id, token_logprob in zip(token_ids, token_logprobs):
        route_id = route_store.allocate_route_id()
        route = RouteState(
            route_id=route_id,
            stage1_root_id=route_id,
            parent_route_id=None,
            materialized_tokens=(),
            pending_token_id=int(token_id),
            cumulative_logprob=float(token_logprob.detach().cpu()),
            stage1_depth=0,
            stage2_depth=0,
        )
        routes.append(route_store.register(route))
    return routes


def build_tree_step(
    active_routes: Sequence[RouteState],
    *,
    prefix_token_ids: Sequence[int],
    k: int,
    route_store: RouteStore,
    drafter: BatchDecodeBackend,
    phase: DecodePhase = DecodePhase.STAGE1,
) -> FrontierStepResult:
    if len(active_routes) != int(k):
        raise ValueError(f"tree step expects {k} active routes, got {len(active_routes)}")
    return _advance_one_step(
        active_routes,
        prefix_token_ids=prefix_token_ids,
        k=k,
        route_store=route_store,
        drafter=drafter,
        phase=phase,
        per_stage1_root=False,
    )


def build_tree(
    active_routes: Sequence[RouteState],
    *,
    prefix_token_ids: Sequence[int],
    depth: int,
    k: int,
    route_store: RouteStore,
    drafter: BatchDecodeBackend,
) -> BuildResult:
    if depth <= 0:
        raise ValueError("depth must be positive")
    steps: list[FrontierStepResult] = []
    frontier = list(active_routes)
    for _ in range(depth):
        step = build_tree_step(
            frontier,
            prefix_token_ids=prefix_token_ids,
            k=k,
            route_store=route_store,
            drafter=drafter,
            phase=DecodePhase.STAGE1,
        )
        steps.append(step)
        frontier = step.next_routes
    last = steps[-1]
    return BuildResult(
        completed_routes=last.decoded_routes,
        next_frontier_routes=last.next_routes,
        last_logits=last.decode_output.logits,
        steps=steps,
    )


def initialize_forest_routes(
    stage1_routes: Sequence[RouteState],
    stage1_last_logits: torch.Tensor,
    *,
    k: int,
    route_store: RouteStore,
) -> list[RouteState]:
    if len(stage1_routes) != int(stage1_last_logits.shape[0]):
        raise ValueError("stage1 routes and logits must have matching rows")

    token_logprobs, token_ids = batch_topk_logprobs(stage1_last_logits, k=k)
    routes: list[RouteState] = []
    for row, stage1_route in enumerate(stage1_routes):
        for rank in range(token_ids.shape[1]):
            route = RouteState(
                route_id=route_store.allocate_route_id(),
                stage1_root_id=stage1_route.route_id,
                parent_route_id=stage1_route.route_id,
                materialized_tokens=stage1_route.materialized_tokens,
                pending_token_id=int(token_ids[row, rank]),
                cumulative_logprob=stage1_route.cumulative_logprob + float(token_logprobs[row, rank].detach().cpu()),
                stage1_depth=stage1_route.stage1_depth,
                stage2_depth=0,
            )
            routes.append(route_store.register(route))
    return routes


def build_forest_step(
    active_routes: Sequence[RouteState],
    *,
    prefix_token_ids: Sequence[int],
    k: int,
    route_store: RouteStore,
    drafter: BatchDecodeBackend,
) -> FrontierStepResult:
    expected = int(k) * int(k)
    if len(active_routes) != expected:
        raise ValueError(f"forest step expects {expected} active routes, got {len(active_routes)}")
    return _advance_one_step(
        active_routes,
        prefix_token_ids=prefix_token_ids,
        k=k,
        route_store=route_store,
        drafter=drafter,
        phase=DecodePhase.STAGE2,
        per_stage1_root=True,
    )


def _advance_one_step(
    active_routes: Sequence[RouteState],
    *,
    prefix_token_ids: Sequence[int],
    k: int,
    route_store: RouteStore,
    drafter: BatchDecodeBackend,
    phase: DecodePhase,
    per_stage1_root: bool,
) -> FrontierStepResult:
    decode_output = drafter.decode_batch(prefix_token_ids=prefix_token_ids, routes=active_routes)
    decoded_routes = route_store.materialize_pending(active_routes, phase=phase)
    candidates = _collect_candidates(decoded_routes, decode_output.logits, k=k)
    if per_stage1_root:
        selected = select_topk_per_stage1_root(
            candidates,
            active_root_ids=[route.stage1_root_id for route in active_routes],
            k=k,
        )
    else:
        selected = select_global_topk(candidates, k=k)
    next_routes = route_store.make_next_routes(selected, parent_routes=decoded_routes)
    return FrontierStepResult(
        decoded_routes=decoded_routes,
        candidates=candidates,
        selected_candidates=selected,
        next_routes=next_routes,
        decode_output=decode_output,
    )


def _collect_candidates(
    decoded_routes: Sequence[RouteState],
    logits: torch.Tensor,
    *,
    k: int,
) -> list[PendingCandidate]:
    if len(decoded_routes) != int(logits.shape[0]):
        raise ValueError("decoded routes and logits must have matching rows")
    token_logprobs, token_ids = batch_topk_logprobs(logits, k=k)
    candidates: list[PendingCandidate] = []
    for row, route in enumerate(decoded_routes):
        for rank in range(token_ids.shape[1]):
            candidates.append(
                PendingCandidate(
                    parent_route_id=route.route_id,
                    stage1_root_id=route.stage1_root_id,
                    pending_token_id=int(token_ids[row, rank]),
                    cumulative_logprob=route.cumulative_logprob + float(token_logprobs[row, rank].detach().cpu()),
                    rank_in_parent=rank,
                )
            )
    return candidates

