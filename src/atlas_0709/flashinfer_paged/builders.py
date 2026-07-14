from __future__ import annotations

from typing import Any, Sequence

import torch

from .frontier import (
    FrontierModelBackend,
    GlobalTopKSelection,
    PerStage1RootTopKSelection,
    advance_frontier_one_token,
    collect_pending_candidates,
)
from .kv import KVTreeStore
from .sampling import (
    DrafterSamplingContext,
    batch_parent_token_candidates,
    single_parent_token_candidates,
)
from .types import (
    BuildDepthsOutput,
    DecodePhase,
    DraftPrefixState,
    FrontierStepOutput,
    PendingCandidate,
    RouteState,
)
from .utils import topk_pairs_to_python


def initialize_stage1_routes(
    draft_prefix_state: DraftPrefixState,
    *,
    k: int,
    route_store: KVTreeStore,
    sampling: DrafterSamplingContext | None = None,
) -> list[RouteState]:
    token_logprobs, token_ids = single_parent_token_candidates(
        draft_prefix_state.next_token_logits,
        k=k,
        sampling=sampling,
    )
    routes: list[RouteState] = []

    for token_id, token_logprob in topk_pairs_to_python(token_ids, token_logprobs):
        route_id = route_store.allocate_route_id()
        route = RouteState(
            route_id=route_id,
            stage1_root_id=route_id,
            parent_route_id=None,
            materialized_leaf_node_id=None,
            pending_token_id=token_id,
            cumulative_logprob=token_logprob,
            stage1_depth=0,
            stage2_depth=0,
            kv_view=draft_prefix_state.prefix_kv_view.fork(),
            token_logprobs=(float(token_logprob),),
        )
        routes.append(route_store.register_route(route))
    return routes


def build_tree_one_depth(
    active_routes: Sequence[RouteState],
    *,
    k: int,
    route_store: KVTreeStore,
    model_backend: FrontierModelBackend,
    attention_backend: Any = None,
    phase: DecodePhase = DecodePhase.STAGE1,
    sampling: DrafterSamplingContext | None = None,
) -> FrontierStepOutput:
    if len(active_routes) != k:
        raise ValueError(
            f"build_tree_one_depth expects {k} active routes, got {len(active_routes)}"
        )
    return advance_frontier_one_token(
        active_routes,
        k=k,
        route_store=route_store,
        model_backend=model_backend,
        selection_policy=GlobalTopKSelection(phase=phase),
        attention_backend=attention_backend,
        sampling=sampling,
    )


def build_tree_depths(
    active_routes: Sequence[RouteState],
    *,
    depth: int,
    k: int,
    route_store: KVTreeStore,
    model_backend: FrontierModelBackend,
    attention_backend: Any = None,
    phase: DecodePhase = DecodePhase.STAGE1,
    sampling: DrafterSamplingContext | None = None,
) -> BuildDepthsOutput:
    if depth <= 0:
        raise ValueError("depth must be positive")

    steps: list[FrontierStepOutput] = []
    frontier = list(active_routes)
    for _ in range(depth):
        step = build_tree_one_depth(
            frontier,
            k=k,
            route_store=route_store,
            model_backend=model_backend,
            attention_backend=attention_backend,
            phase=phase,
            sampling=sampling,
        )
        steps.append(step)
        frontier = step.next_routes

    last_step = steps[-1]
    return BuildDepthsOutput(
        completed_routes=last_step.decoded_routes,
        next_frontier_routes=last_step.next_routes,
        last_logits=last_step.decode_output.next_token_logits,
        steps=steps,
    )


def initialize_forest_routes(
    stage1_routes: Sequence[RouteState],
    stage1_last_logits: torch.Tensor,
    *,
    k: int,
    route_store: KVTreeStore,
    sampling: DrafterSamplingContext | None = None,
) -> list[RouteState]:
    if len(stage1_routes) != int(stage1_last_logits.shape[0]):
        raise ValueError(
            "stage1_routes and stage1_last_logits must have the same batch size"
        )

    forest_routes: list[RouteState] = []
    relative_parent_paths = [
        route_store.materialized_token_path(route) for route in stage1_routes
    ]
    token_logprobs, token_ids = batch_parent_token_candidates(
        stage1_last_logits,
        k=k,
        sampling=sampling,
        relative_parent_paths=relative_parent_paths,
    )
    actual_k = int(token_ids.shape[-1])
    topk_pairs = topk_pairs_to_python(token_ids, token_logprobs)
    for row, stage1_route in enumerate(stage1_routes):
        for rank in range(actual_k):
            token_id, token_logprob = topk_pairs[row * actual_k + rank]
            route_id = route_store.allocate_route_id()
            route = RouteState(
                route_id=route_id,
                stage1_root_id=stage1_route.route_id,
                parent_route_id=stage1_route.route_id,
                materialized_leaf_node_id=stage1_route.materialized_leaf_node_id,
                pending_token_id=token_id,
                cumulative_logprob=stage1_route.cumulative_logprob + token_logprob,
                stage1_depth=stage1_route.stage1_depth,
                stage2_depth=0,
                kv_view=stage1_route.kv_view.fork(),
                token_logprobs=(
                    *stage1_route.token_logprobs,
                    float(token_logprob),
                ),
            )
            forest_routes.append(route_store.register_route(route))
    return forest_routes


def build_forest_one_depth(
    active_routes: Sequence[RouteState],
    *,
    k: int,
    route_store: KVTreeStore,
    model_backend: FrontierModelBackend,
    attention_backend: Any = None,
    sampling: DrafterSamplingContext | None = None,
) -> FrontierStepOutput:
    expected = k * k
    if len(active_routes) != expected:
        raise ValueError(
            f"build_forest_one_depth expects {expected} active routes, got {len(active_routes)}"
        )
    return advance_frontier_one_token(
        active_routes,
        k=k,
        route_store=route_store,
        model_backend=model_backend,
        selection_policy=PerStage1RootTopKSelection(phase=DecodePhase.STAGE2),
        attention_backend=attention_backend,
        sampling=sampling,
    )


def build_forest_depths(
    active_routes: Sequence[RouteState],
    *,
    depth: int,
    k: int,
    route_store: KVTreeStore,
    model_backend: FrontierModelBackend,
    attention_backend: Any = None,
    sampling: DrafterSamplingContext | None = None,
) -> BuildDepthsOutput:
    if depth <= 0:
        raise ValueError("depth must be positive")

    steps: list[FrontierStepOutput] = []
    frontier = list(active_routes)
    for _ in range(depth):
        step = build_forest_one_depth(
            frontier,
            k=k,
            route_store=route_store,
            model_backend=model_backend,
            attention_backend=attention_backend,
            sampling=sampling,
        )
        steps.append(step)
        frontier = step.next_routes

    last_step = steps[-1]
    return BuildDepthsOutput(
        completed_routes=last_step.decoded_routes,
        next_frontier_routes=last_step.next_routes,
        last_logits=last_step.decode_output.next_token_logits,
        steps=steps,
    )


def select_routes_by_stage1_root(
    routes: Sequence[RouteState],
    selected_stage1_root_id: int,
) -> list[RouteState]:
    return [
        route for route in routes if route.stage1_root_id == selected_stage1_root_id
    ]


def collect_final_pending_candidates(
    completed_routes: Sequence[RouteState],
    last_logits: torch.Tensor,
    *,
    k: int,
) -> list[PendingCandidate]:
    return collect_pending_candidates(completed_routes, last_logits, k=k)


def filter_candidates_by_stage1_root(
    candidates: Sequence[PendingCandidate],
    selected_stage1_root_id: int,
) -> list[PendingCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.stage1_root_id == selected_stage1_root_id
    ]
