from __future__ import annotations

import math

import torch

from atlas_0709.flashinfer_paged.builders import (
    build_forest_one_depth,
    build_tree_one_depth,
    initialize_forest_routes,
    initialize_stage1_routes,
    select_routes_by_stage1_root,
)
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.sampling import (
    DrafterSamplingConfig,
    DrafterSamplingContext,
    batch_parent_token_candidates,
    single_parent_token_candidates,
)
from atlas_0709.flashinfer_paged.types import (
    DraftPrefixState,
    FrontierDecodeOutput,
    PrefixKVView,
    RouteState,
)


def _sampling(
    *,
    seed: int = 17,
    committed: tuple[int, ...] = (101, 102),
    temperature: float = 0.8,
) -> DrafterSamplingContext:
    return DrafterSamplingContext(
        config=DrafterSamplingConfig(
            do_sample=True,
            seed=seed,
            temperature=temperature,
        ),
        committed_token_ids=committed,
    )


def test_semantic_seed_is_invariant_to_prefix_route_split() -> None:
    before_commit = _sampling(committed=(11, 12))
    after_commit = _sampling(committed=(11, 12, 13))
    logits = torch.linspace(-2.0, 2.0, 97)

    before_logprobs, before_ids = single_parent_token_candidates(
        logits,
        k=7,
        sampling=before_commit,
        relative_parent_path=(13, 14),
    )
    after_logprobs, after_ids = single_parent_token_candidates(
        logits,
        k=7,
        sampling=after_commit,
        relative_parent_path=(14,),
    )

    assert before_commit.seed_for_parent((13, 14)) == after_commit.seed_for_parent((14,))
    assert torch.equal(before_ids, after_ids)
    assert torch.equal(before_logprobs, after_logprobs)


def test_sampling_is_reproducible_and_request_seed_changes_draws() -> None:
    logits = torch.zeros(128)
    first = single_parent_token_candidates(
        logits,
        k=12,
        sampling=_sampling(seed=123),
        relative_parent_path=(7, 8),
    )[1]
    repeated = single_parent_token_candidates(
        logits,
        k=12,
        sampling=_sampling(seed=123),
        relative_parent_path=(7, 8),
    )[1]
    other_seed = single_parent_token_candidates(
        logits,
        k=12,
        sampling=_sampling(seed=124),
        relative_parent_path=(7, 8),
    )[1]

    assert torch.equal(first, repeated)
    assert not torch.equal(first, other_seed)
    assert len(set(first.tolist())) == 12


def test_sampling_is_independent_of_frontier_batch_order_and_route_ids() -> None:
    context = _sampling(seed=9, committed=(1, 2, 3))
    logits_a = torch.linspace(-1.0, 1.0, 64)
    logits_b = torch.linspace(1.0, -1.0, 64)
    paths = ((4, 5), (6, 7))

    _, ids_ab = batch_parent_token_candidates(
        torch.stack((logits_a, logits_b)),
        k=5,
        sampling=context,
        relative_parent_paths=paths,
    )
    _, ids_ba = batch_parent_token_candidates(
        torch.stack((logits_b, logits_a)),
        k=5,
        sampling=context,
        relative_parent_paths=tuple(reversed(paths)),
    )

    assert torch.equal(ids_ab[0], ids_ba[1])
    assert torch.equal(ids_ab[1], ids_ba[0])
    _, duplicate_semantic_parent_ids = batch_parent_token_candidates(
        torch.stack((logits_a, logits_a)),
        k=5,
        sampling=context,
        relative_parent_paths=(paths[0], paths[0]),
    )
    # Different frontier rows (and therefore potentially different route/node
    # ids) receive the same draw when their semantic parent history is equal.
    assert torch.equal(
        duplicate_semantic_parent_ids[0], duplicate_semantic_parent_ids[1]
    )


def test_sampling_records_untempered_model_logprobs() -> None:
    logits = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])
    token_logprobs, token_ids = single_parent_token_candidates(
        logits,
        k=4,
        sampling=_sampling(temperature=2.5),
        relative_parent_path=(42,),
    )
    expected = torch.log_softmax(logits, dim=-1).gather(0, token_ids)

    assert torch.equal(token_logprobs, expected)


def test_greedy_candidate_mode_is_unchanged() -> None:
    logits = torch.tensor([0.5, 4.0, -2.0, 3.0, 1.0])
    actual_logprobs, actual_ids = single_parent_token_candidates(
        logits,
        k=3,
        sampling=DrafterSamplingContext(
            config=DrafterSamplingConfig(do_sample=False, seed=999, temperature=0.3),
            committed_token_ids=(1, 2),
        ),
        relative_parent_path=(3,),
    )
    expected_logprobs, expected_ids = torch.topk(
        torch.log_softmax(logits, dim=-1),
        k=3,
    )

    assert torch.equal(actual_ids, expected_ids)
    assert torch.equal(actual_logprobs, expected_logprobs)


class _RouteStoreBackend:
    def __init__(self, store: KVTreeStore, logits: torch.Tensor) -> None:
        self.store = store
        self.logits = logits

    def decode_frontier_one_token(
        self,
        active_routes: list[RouteState],
        attention_backend=None,
    ) -> FrontierDecodeOutput:
        if int(self.logits.shape[0]) == 1:
            next_logits = self.logits.expand(len(active_routes), -1).clone()
        else:
            next_logits = self.logits[: len(active_routes)].clone()
        node_ids = self.store.reserve_node_ids(len(active_routes))
        return FrontierDecodeOutput(
            route_ids=torch.tensor([route.route_id for route in active_routes]),
            next_token_logits=next_logits,
            new_node_ids=torch.tensor(node_ids, dtype=torch.long),
        )


def _prefix(logits: torch.Tensor) -> DraftPrefixState:
    return DraftPrefixState(
        token_ids=torch.tensor([90, 91], dtype=torch.long),
        prefix_kv_view=PrefixKVView(committed_length=2),
        next_token_logits=logits,
        committed_length=2,
    )


def test_route_token_logprobs_append_and_prune_with_forest_promotion() -> None:
    k = 2
    store = KVTreeStore()
    store.committed_token_ids = [90, 91]
    sampling = _sampling(seed=31, committed=(90, 91))
    logits = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]])
    backend = _RouteStoreBackend(store, logits)

    roots = initialize_stage1_routes(
        _prefix(logits),
        k=k,
        route_store=store,
        sampling=sampling,
    )
    assert all(len(route.token_logprobs) == 1 for route in roots)

    tree = build_tree_one_depth(
        roots,
        k=k,
        route_store=store,
        model_backend=backend,
        sampling=sampling,
    )
    assert all(len(route.token_logprobs) == route.stage1_depth for route in tree.decoded_routes)
    assert all(
        len(route.token_logprobs) == route.stage1_depth + 1
        for route in tree.next_routes
    )
    assert all(
        math.isclose(sum(route.token_logprobs), route.cumulative_logprob, abs_tol=1e-6)
        for route in tree.next_routes
    )

    forest_frontier = initialize_forest_routes(
        tree.decoded_routes,
        tree.decode_output.next_token_logits,
        k=k,
        route_store=store,
        sampling=sampling,
    )
    forest = build_forest_one_depth(
        forest_frontier,
        k=k,
        route_store=store,
        model_backend=backend,
        sampling=sampling,
    )
    selected_stage1 = tree.decoded_routes[0]
    retained = select_routes_by_stage1_root(
        forest.next_routes,
        selected_stage1_root_id=selected_stage1.route_id,
    )
    original_logprobs = [route.token_logprobs for route in retained]
    committed_depth = selected_stage1.stage1_depth
    promoted = store.promote_routes_after_commit(
        selected_stage1,
        retained,
        PrefixKVView(committed_length=2 + committed_depth),
    )

    for route, before in zip(promoted, original_logprobs):
        assert route.token_logprobs == before[committed_depth:]
        assert len(route.token_logprobs) == route.stage1_depth + 1
        assert math.isclose(
            sum(route.token_logprobs),
            route.cumulative_logprob - selected_stage1.cumulative_logprob,
            abs_tol=1e-6,
        )
        # cumulative_logprob remains the historical absolute route score; the
        # per-token tuple is intentionally relative to the new committed prefix.
