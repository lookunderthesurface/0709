from __future__ import annotations

from types import SimpleNamespace

import torch

from benchmarks.bench_atlas_0709_isolated_components import (
    PerRouteTop1Selection,
    measure_stepped_component,
)
from atlas_0709.flashinfer_paged.types import (
    DecodePhase,
    PendingCandidate,
    PendingCandidateBatch,
)


class FakeRoutePool:
    def __init__(self) -> None:
        # These values model prefill/setup work and must not leak into measured
        # hot-path deltas.
        self.counters = {"work": 100, "setup_only": 7}

    def hot_path_counters(self) -> dict[str, int]:
        return dict(self.counters)


def make_fake_state() -> SimpleNamespace:
    pool = FakeRoutePool()
    return SimpleNamespace(
        context=SimpleNamespace(
            backend=SimpleNamespace(route_pool=pool),
        ),
        step_index=0,
    )


def advance_fake_state(state: SimpleNamespace) -> SimpleNamespace:
    state.step_index += 1
    state.context.backend.route_pool.counters["work"] += state.step_index
    return SimpleNamespace(
        decode_output=SimpleNamespace(
            metadata={
                "hot_path_counters_total": dict(
                    state.context.backend.route_pool.counters
                )
            }
        ),
        selection_stats=SimpleNamespace(
            candidate_count=10 * state.step_index,
            selected_count=3,
            host_materialization_batches=1,
            host_materialization_elements=8 * state.step_index,
            host_transfer_batches=1,
            host_transfer_elements=8 * state.step_index,
        )
    )


def test_stepped_counter_samples_exclude_setup_and_track_each_depth(monkeypatch) -> None:
    sync_calls = 0

    def fake_sync_cuda() -> None:
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setattr(
        "benchmarks.bench_atlas_0709_isolated_components.sync_cuda",
        fake_sync_cuda,
    )

    result = measure_stepped_component(
        setup_fn=make_fake_state,
        step_fn=advance_fake_state,
        depth=2,
        warmup=1,
        iters=2,
    ).to_dict()

    depth1 = result["steps"][0]
    depth2 = result["steps"][1]
    assert depth1["hot_path_counter_delta"]["work"]["samples"] == [1, 1]
    assert depth1["hot_path_counter_cumulative"]["work"]["samples"] == [1, 1]
    assert depth2["hot_path_counter_delta"]["work"]["samples"] == [2, 2]
    assert depth2["hot_path_counter_cumulative"]["work"]["samples"] == [3, 3]
    assert result["hot_path_counter_total_delta"]["work"]["samples"] == [3, 3]
    assert depth1["hot_path_counter_delta"]["selection_host_transfer_batches"][
        "samples"
    ] == [1, 1]
    assert depth2["hot_path_counter_cumulative"][
        "selection_host_transfer_batches"
    ]["samples"] == [2, 2]
    assert result["hot_path_counter_total_delta"][
        "selection_host_transfer_elements"
    ]["samples"] == [24, 24]
    assert depth1["hot_path_counter_delta"]["total_host_transfer_batches"][
        "samples"
    ] == [1, 1]
    assert result["hot_path_counter_total_delta"][
        "total_host_transfer_batches"
    ]["samples"] == [2, 2]
    assert result["hot_path_counter_total_delta"][
        "total_host_transfer_elements"
    ]["samples"] == [24, 24]

    # The setup-only counter is present for auditability but its measured delta
    # is zero, proving the snapshot baseline was taken after setup/prefill.
    assert depth1["hot_path_counter_delta"]["setup_only"]["samples"] == [0, 0]
    assert result["hot_path_counter_total_delta"]["setup_only"]["samples"] == [0, 0]

    measurement = result["measurement_syncs"]
    assert measurement["warmup_after_workload_total"] == 1
    assert measurement["measured_per_iteration"] == 5
    assert measurement["measured_total"] == 10
    assert measurement["overall_total"] == 11
    assert measurement["excluded_from_hot_path_counters"] is True
    assert measurement["counter_summary_extraction_in_timed_region"] is False
    assert sync_calls == 11

    # Existing timing keys remain available for report consumers.
    assert set(result["total"]) == {"median_ms", "mean_ms", "samples_ms"}
    assert {"depth", "median_ms", "mean_ms", "samples_ms"} <= set(depth1)

    depth4_result = measure_stepped_component(
        setup_fn=make_fake_state,
        step_fn=advance_fake_state,
        depth=4,
        warmup=0,
        iters=1,
    ).to_dict()
    assert depth4_result["measurement_syncs"]["measured_per_iteration"] == 9
    assert depth4_result["measurement_syncs"]["measured_total"] == 9
    assert sync_calls == 20


def test_matched_ar_selection_keeps_one_greedy_child_per_route() -> None:
    active_routes = [SimpleNamespace(route_id=2), SimpleNamespace(route_id=1)]
    candidates = [
        PendingCandidate(
            parent_route_id=1,
            stage1_root_id=1,
            pending_token_id=9,
            cumulative_logprob=-1.0,
            rank_in_parent=1,
        ),
        PendingCandidate(
            parent_route_id=1,
            stage1_root_id=1,
            pending_token_id=8,
            cumulative_logprob=-1.0,
            rank_in_parent=0,
        ),
        PendingCandidate(
            parent_route_id=2,
            stage1_root_id=2,
            pending_token_id=7,
            cumulative_logprob=-2.0,
            rank_in_parent=0,
        ),
        PendingCandidate(
            parent_route_id=2,
            stage1_root_id=2,
            pending_token_id=6,
            cumulative_logprob=-0.5,
            rank_in_parent=1,
        ),
    ]
    policy = PerRouteTop1Selection(phase=DecodePhase.STAGE1)

    selected = policy.select(candidates, k=99, active_routes=active_routes)

    assert [(item.parent_route_id, item.pending_token_id) for item in selected] == [
        (2, 6),
        (1, 8),
    ]

    candidate_batch = PendingCandidateBatch(
        parent_route_ids=torch.tensor([2, 1]),
        stage1_root_ids=torch.tensor([2, 1]),
        parent_row_indices=torch.tensor([0, 1]),
        pending_token_ids=torch.tensor([6, 8]),
        cumulative_logprobs=torch.tensor([-0.5, -1.0]),
        parent_logprobs=torch.tensor([0.0, 0.0]),
        ranks_in_parent=torch.tensor([0, 0]),
        candidates_per_parent=1,
    )
    assert policy.select_indices(
        candidate_batch,
        k=1,
        active_routes=active_routes,
    ).tolist() == [0, 1]
