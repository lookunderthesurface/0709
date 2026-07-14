from __future__ import annotations

from types import SimpleNamespace

import torch

from atlas_0709.flashinfer_full_verify import prefix_layer_to_paged_kv
from atlas_0709.target_runtime import DirectFlashInferMaskedTreeVerifyBackend
from atlas_0709.target_runtime import VerifyRoutePayload
from atlas_0709.types import TargetRouteScore


def _make_backend() -> DirectFlashInferMaskedTreeVerifyBackend:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.config = SimpleNamespace(page_size=4)
    backend.prefix_token_ids = (1, 2, 3)
    backend._prefix_logits = torch.zeros((1, 8), dtype=torch.float32)
    backend._committed_verify_rounds = 0
    cache = torch.zeros((2, 2, 4, 1, 1), dtype=torch.float32)
    for position in range(8):
        page, offset = divmod(position, 4)
        cache[page, 0, offset, 0, 0] = 100 + position
        cache[page, 1, offset, 0, 0] = 200 + position
    backend._paged_prefix_kv = [cache]
    return backend


def test_selected_tree_nodes_are_compacted_into_committed_prefix() -> None:
    backend = _make_backend()
    logits = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    backend._commit_selected_path(
        token_ids=(9, 10),
        node_ids=(0, 2),
        verify_logits=logits,
    )

    cache = backend._paged_prefix_kv[0]
    assert cache[0, 0, 3, 0, 0].item() == 103
    assert cache[1, 0, 0, 0, 0].item() == 105
    assert cache[0, 1, 3, 0, 0].item() == 203
    assert cache[1, 1, 0, 0, 0].item() == 205
    assert backend.prefix_token_ids == (1, 2, 3, 9, 10)
    assert torch.equal(backend._prefix_logits, logits[2].unsqueeze(0))
    assert backend._committed_verify_rounds == 1


def test_target_page_capacity_grows_without_losing_prefix_kv() -> None:
    backend = _make_backend()
    before = backend._paged_prefix_kv[0].clone()

    backend._ensure_page_capacity(4)

    assert backend._paged_prefix_kv[0].shape[0] == 4
    assert torch.equal(backend._paged_prefix_kv[0][:2], before)


def test_hf_prefix_kv_is_written_in_flashinfer_page_order() -> None:
    key = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    value = (100 + torch.arange(5, dtype=torch.float32)).reshape(1, 1, 5, 1)

    cache = prefix_layer_to_paged_kv(
        key=key,
        value=value,
        prefix_len=5,
        node_count=2,
        page_size=4,
    )

    observed_key = [cache[position // 4, 0, position % 4, 0, 0].item() for position in range(5)]
    observed_value = [cache[position // 4, 1, position % 4, 0, 0].item() for position in range(5)]
    assert observed_key == [0, 1, 2, 3, 4]
    assert observed_value == [100, 101, 102, 103, 104]


def test_route_scoring_uses_shared_parent_node_logits() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.score_weights = None
    prefix_logits = torch.tensor([0.0, 3.0, 0.0, 0.0])
    verify_logits = torch.tensor(
        [
            [0.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    routes = [
        VerifyRoutePayload(route_id=1, token_ids=(1, 2), draft_logprob=-1.0),
        VerifyRoutePayload(route_id=2, token_ids=(1, 3), draft_logprob=-2.0),
    ]

    scores = backend._score_routes(
        routes=routes,
        paths=[route.token_ids for route in routes],
        prefix_logits=prefix_logits,
        verify_logits=verify_logits,
        route_to_node_paths=[[0, 1], [0, 2]],
    )

    prefix_lp = torch.log_softmax(prefix_logits, dim=-1)[1].item()
    parent_lp = torch.log_softmax(verify_logits[0], dim=-1)
    assert abs(scores[0].target_logprob - (prefix_lp + parent_lp[2].item())) < 1e-6
    assert abs(scores[1].target_logprob - (prefix_lp + parent_lp[3].item())) < 1e-6


def test_first_route_diagnostic_ignores_target_score() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.route_selection_policy = "first_route"
    scores = [
        TargetRouteScore(
            route_id=7,
            token_ids=(11,),
            target_logprob=-10.0,
            draft_logprob=-4.0,
        ),
        TargetRouteScore(
            route_id=3,
            token_ids=(12,),
            target_logprob=-0.1,
            draft_logprob=-0.2,
        ),
    ]

    assert backend._select_route(scores).route_id == 7
    assert backend._selection_policy_name() == "diagnostic_first_payload_route"


def test_target_best_policy_keeps_existing_tie_break() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.route_selection_policy = "target_best"
    backend.score_weights = None
    scores = [
        TargetRouteScore(
            route_id=9,
            token_ids=(11,),
            target_logprob=-1.0,
            draft_logprob=-2.0,
        ),
        TargetRouteScore(
            route_id=4,
            token_ids=(12,),
            target_logprob=-1.0,
            draft_logprob=-1.0,
        ),
    ]

    assert backend._select_route(scores).route_id == 4


def test_route_selection_policy_validation() -> None:
    assert (
        DirectFlashInferMaskedTreeVerifyBackend._validate_route_selection_policy(
            " FIRST_ROUTE "
        )
        == "first_route"
    )
    try:
        DirectFlashInferMaskedTreeVerifyBackend._validate_route_selection_policy(
            "sample"
        )
    except ValueError as exc:
        assert "target_best, target_sample, first_route" in str(exc)
    else:
        raise AssertionError("invalid route policy was accepted")


def test_target_route_sampling_is_seeded_and_payload_order_independent() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.route_selection_policy = "target_sample"
    backend.route_sampling_temperature = 1.0
    scores = [
        TargetRouteScore(
            route_id=8,
            token_ids=(30, 31),
            target_logprob=-1.0,
            draft_logprob=-0.8,
        ),
        TargetRouteScore(
            route_id=2,
            token_ids=(10, 11),
            target_logprob=-1.4,
            draft_logprob=-0.2,
        ),
        TargetRouteScore(
            route_id=5,
            token_ids=(20, 21),
            target_logprob=-0.7,
            draft_logprob=-0.5,
        ),
    ]

    selected_a, metadata_a = backend._select_route_with_metadata(
        scores,
        route_sampling_seed=1234,
        route_sampling_round=7,
    )
    selected_b, metadata_b = backend._select_route_with_metadata(
        list(reversed(scores)),
        route_sampling_seed=1234,
        route_sampling_round=7,
    )
    _, next_round_metadata = backend._select_route_with_metadata(
        scores,
        route_sampling_seed=1234,
        route_sampling_round=8,
    )

    assert selected_a.token_ids == selected_b.token_ids
    assert metadata_a["route_sampling_uniform"] == metadata_b["route_sampling_uniform"]
    assert (
        metadata_a["route_sampling_ordered_candidates"]
        == metadata_b["route_sampling_ordered_candidates"]
    )
    assert (
        metadata_a["route_sampling_uniform"]
        != next_round_metadata["route_sampling_uniform"]
    )
    assert metadata_a["route_sampling_scope"] == "verified_route_level_only"
    assert metadata_a["route_sampling_boundary_margin"] >= 0.0


def test_target_route_sampling_requires_explicit_seed_and_round() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.route_selection_policy = "target_sample"
    backend.route_sampling_temperature = 1.0
    scores = [
        TargetRouteScore(
            route_id=1,
            token_ids=(10,),
            target_logprob=-1.0,
            draft_logprob=-1.0,
        )
    ]

    try:
        backend._select_route(scores, route_sampling_seed=5)
    except ValueError as exc:
        assert "explicit route_sampling_seed" in str(exc)
    else:
        raise AssertionError("target_sample accepted a request without a round")


def test_route_intervention_metadata_compares_selected_to_draft_route0() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    route0 = TargetRouteScore(
        route_id=10,
        token_ids=(100, 101),
        target_logprob=-5.0,
        draft_logprob=-0.1,
        first_token_logprob=-2.0,
        selection_score=-5.0,
        draft_token_logprobs=(-0.05, -0.05),
    )
    selected = TargetRouteScore(
        route_id=20,
        token_ids=(200, 201),
        target_logprob=-1.0,
        draft_logprob=-2.0,
        first_token_logprob=-0.2,
        selection_score=-1.0,
        draft_token_logprobs=(-1.0, -1.0),
    )

    metadata = backend._route_intervention_metadata([route0, selected], selected)
    intervention = metadata["route_intervention"]

    assert metadata["selected_draft_rank"] == 2
    assert metadata["route0_route_id"] == 10
    assert metadata["draft_top_route_id"] == 10
    assert metadata["selected_route_changed_from_route0"] is True
    assert metadata["selected_first_token_changed_from_route0"] is True
    assert abs(metadata["selected_minus_route0_draft_first_token_logprob"] + 0.95) < 1e-8
    assert abs(metadata["selected_minus_route0_draft_logprob"] + 1.9) < 1e-8
    assert abs(metadata["selected_minus_route0_target_selection_score"] - 4.0) < 1e-8
    assert intervention["draft_token_logprobs_available"] is True
    assert intervention["selected_vs_draft_top"]["first_token_changed"] is True


def test_route_intervention_draft_rank_uses_current_path_not_promoted_constant() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    path_top_with_low_legacy_cumulative = TargetRouteScore(
        route_id=1,
        token_ids=(10, 11),
        target_logprob=-3.0,
        draft_logprob=-102.0,
        draft_token_logprobs=(-0.8, -1.2),
    )
    selected = TargetRouteScore(
        route_id=2,
        token_ids=(20, 21),
        target_logprob=-1.0,
        draft_logprob=-13.0,
        draft_token_logprobs=(-1.0, -2.0),
    )

    metadata = backend._route_intervention_metadata(
        [path_top_with_low_legacy_cumulative, selected],
        selected,
    )
    intervention = metadata["route_intervention"]

    assert metadata["draft_top_route_id"] == 1
    assert metadata["selected_draft_rank"] == 2
    assert metadata["selected_minus_route0_draft_logprob"] == -1.0
    assert (
        intervention["selected_vs_route0"]["legacy_cumulative_draft_logprob_delta"]
        == 89.0
    )


def test_target_best_tie_break_uses_current_path_not_promoted_constant() -> None:
    backend = object.__new__(DirectFlashInferMaskedTreeVerifyBackend)
    backend.route_selection_policy = "target_best"
    current_path_top = TargetRouteScore(
        route_id=1,
        token_ids=(10, 11),
        target_logprob=-1.0,
        draft_logprob=-100.0,
        draft_token_logprobs=(-0.1, -0.2),
    )
    legacy_cumulative_top = TargetRouteScore(
        route_id=2,
        token_ids=(20, 21),
        target_logprob=-1.0,
        draft_logprob=-2.0,
        draft_token_logprobs=(-1.0, -1.0),
    )

    selected = backend._select_route([legacy_cumulative_top, current_path_top])

    assert selected.route_id == 1


def test_route_sampling_temperature_validation() -> None:
    assert DirectFlashInferMaskedTreeVerifyBackend._validate_route_sampling_temperature(0.5) == 0.5
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        try:
            DirectFlashInferMaskedTreeVerifyBackend._validate_route_sampling_temperature(invalid)
        except ValueError as exc:
            assert "finite and positive" in str(exc)
        else:
            raise AssertionError(f"invalid route sampling temperature was accepted: {invalid}")


def test_selected_path_commit_is_truncated_by_budget_and_eos() -> None:
    tokens, nodes = DirectFlashInferMaskedTreeVerifyBackend._truncate_selected_path(
        token_ids=(10, 11, 12, 13),
        node_ids=(20, 21, 22, 23),
        max_tokens=3,
        eos_token_id=11,
    )

    assert tokens == (10, 11)
    assert nodes == (20, 21)


def test_selected_path_commit_rejects_empty_budget() -> None:
    try:
        DirectFlashInferMaskedTreeVerifyBackend._truncate_selected_path(
            token_ids=(10,),
            node_ids=(20,),
            max_tokens=0,
            eos_token_id=None,
        )
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("empty selected-path commit was accepted")
