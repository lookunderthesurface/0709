from __future__ import annotations

import torch

from atlas_0709.flashinfer_paged.frontier import (
    GlobalTopKSelection,
    PerStage1RootTopKSelection,
    advance_frontier_one_token,
    materialize_pending_candidate_batch,
)
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.types import (
    DecodePhase,
    FrontierDecodeOutput,
    PendingCandidate,
    PendingCandidateBatch,
    PrefixKVView,
    RouteState,
)
from atlas_0709.flashinfer_paged.utils import global_topk_candidates


def _candidate_batch(
    candidates: list[PendingCandidate],
    *,
    candidates_per_parent: int,
) -> PendingCandidateBatch:
    return PendingCandidateBatch(
        parent_route_ids=torch.tensor(
            [candidate.parent_route_id for candidate in candidates],
            dtype=torch.long,
        ),
        stage1_root_ids=torch.tensor(
            [candidate.stage1_root_id for candidate in candidates],
            dtype=torch.long,
        ),
        parent_row_indices=torch.arange(len(candidates), dtype=torch.long),
        pending_token_ids=torch.tensor(
            [candidate.pending_token_id for candidate in candidates],
            dtype=torch.long,
        ),
        cumulative_logprobs=torch.tensor(
            [candidate.cumulative_logprob for candidate in candidates],
            dtype=torch.float64,
        ),
        parent_logprobs=torch.tensor(
            [candidate.parent_logprob for candidate in candidates],
            dtype=torch.float64,
        ),
        ranks_in_parent=torch.tensor(
            [candidate.rank_in_parent for candidate in candidates],
            dtype=torch.long,
        ),
        candidates_per_parent=candidates_per_parent,
    )


def _route(
    route_id: int,
    root_id: int,
    *,
    pending_token_id: int = 0,
    cumulative_logprob: float = 0.0,
) -> RouteState:
    return RouteState(
        route_id=route_id,
        stage1_root_id=root_id,
        parent_route_id=None,
        materialized_leaf_node_id=None,
        pending_token_id=pending_token_id,
        cumulative_logprob=cumulative_logprob,
        stage1_depth=0,
        stage2_depth=0,
        kv_view=PrefixKVView(committed_length=1).fork(),
    )


def test_global_gpu_order_matches_python_for_complete_score_ties() -> None:
    candidates = [
        PendingCandidate(20, 200, 7, -1.0, rank_in_parent=0),
        PendingCandidate(10, 100, 1, -1.0, rank_in_parent=1),
        PendingCandidate(10, 100, 9, -1.0, rank_in_parent=0),
        PendingCandidate(10, 100, 2, -1.0, rank_in_parent=0),
        PendingCandidate(5, 500, 0, -1.0, rank_in_parent=2),
        PendingCandidate(99, 900, 4, -0.5, rank_in_parent=3),
    ]
    batch = _candidate_batch(candidates, candidates_per_parent=1)
    policy = GlobalTopKSelection()

    selected_indices = policy.select_indices(batch, k=len(candidates), active_routes=[])
    actual = [candidates[index] for index in selected_indices.tolist()]
    expected = global_topk_candidates(candidates, num_selected=len(candidates))

    assert actual == expected
    assert [candidate.pending_token_id for candidate in actual] == [4, 0, 2, 9, 1, 7]


def test_per_root_gpu_order_matches_python_for_complete_score_ties() -> None:
    candidates = [
        PendingCandidate(20, 200, 8, -1.0, rank_in_parent=0),
        PendingCandidate(20, 200, 3, -1.0, rank_in_parent=1),
        PendingCandidate(10, 100, 9, -1.0, rank_in_parent=0),
        PendingCandidate(10, 100, 4, -1.0, rank_in_parent=1),
        PendingCandidate(21, 200, 7, -1.0, rank_in_parent=0),
        PendingCandidate(21, 200, 2, -1.0, rank_in_parent=1),
        PendingCandidate(11, 100, 6, -1.0, rank_in_parent=0),
        PendingCandidate(11, 100, 1, -1.0, rank_in_parent=1),
    ]
    active_routes = [
        _route(20, 200),
        _route(10, 100),
        _route(21, 200),
        _route(11, 100),
    ]
    batch = _candidate_batch(candidates, candidates_per_parent=2)
    policy = PerStage1RootTopKSelection()

    selected_indices = policy.select_indices(batch, k=2, active_routes=active_routes)
    actual = [candidates[index] for index in selected_indices.tolist()]
    expected = policy.select(candidates, k=2, active_routes=active_routes)

    assert actual == expected
    assert [candidate.stage1_root_id for candidate in actual] == [100, 100, 200, 200]
    assert [candidate.parent_route_id for candidate in actual] == [10, 10, 20, 20]


class _ReservingBackend:
    def __init__(self, store: KVTreeStore, logits: torch.Tensor) -> None:
        self.store = store
        self.logits = logits

    def decode_frontier_one_token(
        self,
        active_routes: list[RouteState],
        attention_backend=None,
    ) -> FrontierDecodeOutput:
        node_ids = self.store.reserve_node_ids(len(active_routes))
        return FrontierDecodeOutput(
            route_ids=torch.tensor([route.route_id for route in active_routes]),
            next_token_logits=self.logits.clone(),
            new_node_ids=torch.tensor(node_ids, dtype=torch.long),
        )


class _TensorOnlyNodeBackend:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def decode_frontier_one_token(
        self,
        active_routes: list[RouteState],
        attention_backend=None,
    ) -> FrontierDecodeOutput:
        return FrontierDecodeOutput(
            route_ids=torch.tensor([route.route_id for route in active_routes]),
            next_token_logits=self.logits.clone(),
            new_node_ids=torch.tensor([101, 102], dtype=torch.long),
        )


def _registered_routes(store: KVTreeStore) -> list[RouteState]:
    routes: list[RouteState] = []
    for token_id in (3, 4):
        route_id = store.allocate_route_id()
        route = _route(route_id, route_id, pending_token_id=token_id)
        routes.append(store.register_route(route))
    return routes


def test_frontier_depth_materializes_once_and_reuses_cpu_reserved_node_ids() -> None:
    store = KVTreeStore()
    active_routes = _registered_routes(store)
    logits = torch.tensor(
        [
            [0.0, 3.0, 3.0, 0.0, 0.0],
            [0.0, 3.0, 3.0, 0.0, 0.0],
        ]
    )

    step = advance_frontier_one_token(
        active_routes,
        k=2,
        route_store=store,
        model_backend=_ReservingBackend(store, logits),
        selection_policy=GlobalTopKSelection(phase=DecodePhase.STAGE1),
    )

    assert len(step.candidates) == 4
    assert len(step.selected_candidates) == 2
    assert step.selected_candidates == global_topk_candidates(
        step.candidates, num_selected=2
    )
    assert sorted(store.nodes) == [1, 2]
    assert step.selection_stats.candidate_count == 4
    assert step.selection_stats.selected_count == 2
    assert step.selection_stats.host_materialization_batches == 1
    assert step.selection_stats.host_materialization_elements == 4 * 8
    assert step.selection_stats.host_transfer_batches == 0
    assert step.selection_stats.new_node_ids_source == "route_store_counter"


def test_tensor_only_node_ids_share_the_single_packed_materialization() -> None:
    store = KVTreeStore()
    active_routes = _registered_routes(store)
    logits = torch.tensor(
        [
            [0.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )

    step = advance_frontier_one_token(
        active_routes,
        k=2,
        route_store=store,
        model_backend=_TensorOnlyNodeBackend(logits),
        selection_policy=GlobalTopKSelection(),
    )

    assert sorted(store.nodes) == [101, 102]
    assert step.selection_stats.host_materialization_batches == 1
    assert step.selection_stats.host_materialization_elements == (4 + 2) * 8
    assert step.selection_stats.new_node_ids_source == "packed_transfer"


def test_parent_logprob_remains_part_of_materialized_candidate() -> None:
    candidate = PendingCandidate(
        parent_route_id=7,
        stage1_root_id=3,
        pending_token_id=11,
        cumulative_logprob=-4.5,
        rank_in_parent=2,
        parent_logprob=-3.25,
    )
    batch = _candidate_batch([candidate], candidates_per_parent=1)
    materialized, selected, _, stats = materialize_pending_candidate_batch(
        batch,
        torch.tensor([0], dtype=torch.long),
    )

    assert materialized == [candidate]
    assert selected == [candidate]
    assert stats.host_materialization_batches == 1
