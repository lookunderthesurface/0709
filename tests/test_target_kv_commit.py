from __future__ import annotations

from types import SimpleNamespace

import torch

from atlas_0709.flashinfer_full_verify import prefix_layer_to_paged_kv
from atlas_0709.target_runtime import DirectFlashInferMaskedTreeVerifyBackend
from atlas_0709.target_runtime import VerifyRoutePayload


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
