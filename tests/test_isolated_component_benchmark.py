from __future__ import annotations

from types import SimpleNamespace

import torch

from benchmarks.bench_atlas_0709_isolated_components import (
    NativeBatchARState,
    PerRouteTop1Selection,
    measure_paired_critical_path_components,
    measure_stepped_component,
    run_native_batch_ar_step,
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


def test_paired_critical_path_uses_abba_and_only_boundary_syncs(monkeypatch) -> None:
    setup_order: list[str] = []
    setup_counts = {"A": 0, "B": 0}
    sync_calls = 0

    def fake_sync_cuda() -> None:
        nonlocal sync_calls
        sync_calls += 1

    def setup(label: str) -> SimpleNamespace:
        setup_order.append(label)
        setup_counts[label] += 1
        state = make_fake_state()
        state.label = label
        return state

    monkeypatch.setattr(
        "benchmarks.bench_atlas_0709_isolated_components.sync_cuda",
        fake_sync_cuda,
    )
    paired = measure_paired_critical_path_components(
        setup_a=lambda: setup("A"),
        step_a=advance_fake_state,
        arm_a_name="native",
        setup_b=lambda: setup("B"),
        step_b=advance_fake_state,
        arm_b_name="atlas",
        depth=2,
        warmup=0,
        iters=4,
        order_seed=0,
    )

    assert setup_order == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert setup_counts == {"A": 4, "B": 4}
    assert paired.measurement_protocol["measured_round_orders"] == [
        "AB",
        "BA",
        "AB",
        "BA",
    ]
    assert paired.measurement_protocol["complete_abba_blocks"] == 2
    assert len(paired.arm_a.total.samples_ms) == 4
    assert len(paired.arm_b.total.samples_ms) == 4
    assert paired.arm_a.measurement_syncs["host_barriers_between_depths"] == 0
    assert paired.arm_a.measurement_syncs["measured_per_iteration"] == 2
    assert sync_calls == 16
    assert paired.arm_a.to_dict()["hot_path_counter_total_delta"]["work"][
        "samples"
    ] == [3, 3, 3, 3]
    assert len(paired.paired_comparison["blocks"]) == 2


def test_paired_critical_path_rejects_incomplete_abba_block() -> None:
    try:
        measure_paired_critical_path_components(
            setup_a=make_fake_state,
            step_a=advance_fake_state,
            arm_a_name="a",
            setup_b=make_fake_state,
            step_b=advance_fake_state,
            arm_b_name="b",
            depth=1,
            warmup=0,
            iters=3,
            order_seed=0,
        )
    except ValueError as exc:
        assert "positive even" in str(exc)
    else:
        raise AssertionError("incomplete ABBA block did not raise ValueError")


class FakeNativeReqPool:
    def __init__(self) -> None:
        self.table = torch.full((2, 16), -1, dtype=torch.long)
        self.write_calls = 0

    def write(self, indices, values: torch.Tensor) -> None:
        self.write_calls += 1
        self.table[indices] = values


class FakeNativeRoutePool:
    def __init__(self) -> None:
        self.page_size = 4
        self.use_page_attention_metadata = True
        self.validate_page_layout = True
        self.req_to_token_pool = FakeNativeReqPool()
        self.route_rows = {
            10: SimpleNamespace(written_length=4, pending_dirty_start=None),
            11: SimpleNamespace(written_length=4, pending_dirty_start=None),
        }
        self.route_slot_paths: dict[int, torch.Tensor] = {}
        self.route_tail_page_keys = {10: -10, 11: -11}
        self.pending_owned_slot_tensors: list[torch.Tensor] = []
        self.next_tail_page_key = 100
        self.last_allocate_args = None

    def allocate_decode_slots(self, *, seq_lens, last_locs) -> torch.Tensor:
        self.last_allocate_args = (list(seq_lens), last_locs.clone())
        return torch.tensor([8, 12], dtype=torch.long)

    def _record_pending_owned_slots(self, slots: torch.Tensor) -> None:
        self.pending_owned_slot_tensors.append(slots.clone())

    def _new_tail_page_key(self) -> int:
        self.next_tail_page_key += 1
        return -self.next_tail_page_key


class FakeNativeExecutor:
    def __init__(self) -> None:
        self.metadata = None
        self.input_ids = None

    def forward_frontier(self, *, input_ids, route_kv_metadata) -> torch.Tensor:
        self.input_ids = input_ids.clone()
        self.metadata = route_kv_metadata
        return torch.tensor(
            [[0.0, 1.0, 3.0], [4.0, 1.0, 0.0]],
            dtype=torch.float32,
        )


def test_native_batch_ar_step_is_batched_gpu_resident_shape_path() -> None:
    pool = FakeNativeRoutePool()
    executor = FakeNativeExecutor()
    state = NativeBatchARState(
        context=SimpleNamespace(
            backend=SimpleNamespace(route_pool=pool, executor=executor)
        ),
        route_ids=[10, 11],
        req_pool_indices=torch.tensor([0, 1], dtype=torch.long),
        slot_paths=[
            torch.tensor([0, 1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5, 6, 7], dtype=torch.long),
        ],
        input_ids=torch.tensor([5, 6], dtype=torch.long),
        sequence_length=4,
        counter_totals={
            "attention_metadata_builds": 0,
            "attention_metadata_token_indices": 0,
            "attention_metadata_page_indices": 0,
            "req_row_write_calls": 0,
            "req_row_elements_written": 0,
            "req_row_append_calls": 0,
            "req_row_append_elements": 0,
            "native_gpu_argmax_calls": 0,
            "native_req_row_batch_write_calls": 0,
        },
    )

    result = run_native_batch_ar_step(state)

    assert pool.last_allocate_args[0] == [5, 5]
    assert pool.last_allocate_args[1].tolist() == [3, 7]
    assert pool.req_to_token_pool.write_calls == 1
    assert pool.req_to_token_pool.table[:, 4].tolist() == [8, 12]
    assert state.sequence_length == 5
    assert state.input_ids.tolist() == [2, 0]
    assert [path.tolist() for path in state.slot_paths] == [
        [0, 1, 2, 3, 8],
        [4, 5, 6, 7, 12],
    ]
    assert pool.route_rows[10].written_length == 5
    assert pool.route_rows[11].written_length == 5
    assert len(pool.pending_owned_slot_tensors) == 1
    assert executor.input_ids.tolist() == [5, 6]
    metadata = executor.metadata
    assert metadata.seq_lens_cpu.tolist() == [5, 5]
    assert metadata.seq_lens_sum == 10
    assert metadata.positions.tolist() == [4, 4]
    assert metadata.attention_page_size == 4
    assert metadata.token_index_count == 10
    assert metadata.page_index_count == 4
    assert metadata.paged_decode_spec.kv_indices.tolist() == [0, 2, 1, 3]
    assert state.counter_totals["req_row_append_elements"] == 2
    assert state.counter_totals["native_gpu_argmax_calls"] == 1
    assert result.decode_output.metadata["candidate_host_transfer"] is False
