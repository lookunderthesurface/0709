from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from atlas_0709.flashinfer_paged import flashinfer_backends
from atlas_0709.flashinfer_paged.flashinfer_backends import (
    SGLangFlashInferPagedDecodeBackend,
    SGLangRouteKVMetadata,
)
from atlas_0709.flashinfer_paged.sglang_runtime import SGLangRoutePoolBridge
from atlas_0709.flashinfer_paged.types import PrefixKVView, RouteKVView, RouteState


class FakeKVCache:
    def __init__(self) -> None:
        self.values = torch.arange(512, dtype=torch.float32)

    def move_kv_cache(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        self.values[destination] = self.values[source]


class FakeAllocator:
    def __init__(self, page_size: int) -> None:
        self.page_size = int(page_size)
        self.next_page = 10
        self.kv_cache = FakeKVCache()

    def alloc(self, count: int) -> torch.Tensor:
        assert int(count) % self.page_size == 0
        page_count = int(count) // self.page_size
        pages = torch.arange(self.next_page, self.next_page + page_count)
        self.next_page += page_count
        offsets = torch.arange(self.page_size)
        return (pages[:, None] * self.page_size + offsets[None, :]).reshape(-1)

    def get_kvcache(self) -> FakeKVCache:
        return self.kv_cache


class FakeReqPool:
    def __init__(self) -> None:
        self.req_to_token = torch.full((8, 128), -1, dtype=torch.long)
        self.write_log: list[tuple[int, int, int, list[int]]] = []
        self.freed: list[object] = []

    def write(self, indices, values: torch.Tensor) -> None:
        self.req_to_token[indices] = values
        row, positions = indices
        if isinstance(row, int) and isinstance(positions, slice):
            self.write_log.append(
                (
                    int(row),
                    int(positions.start or 0),
                    int(positions.stop or 0),
                    values.tolist(),
                )
            )

    def free(self, handle: object) -> None:
        self.freed.append(handle)


class AlwaysFailReqPool(FakeReqPool):
    def write(self, indices, values: torch.Tensor) -> None:
        raise RuntimeError("synthetic write failure")


class FailSecondRowOnceReqPool(FakeReqPool):
    def __init__(self) -> None:
        super().__init__()
        # One logical write can attempt the slice API and then its tensor-index
        # fallback, so fail both calls before allowing a retry to proceed.
        self.failures_remaining = 2

    def write(self, indices, values: torch.Tensor) -> None:
        row = indices[0]
        targets_second_row = (
            int(row) == 1
            if isinstance(row, int)
            else bool(torch.all(row == 1).item())
        )
        if targets_second_row and self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("synthetic one-shot row failure")
        super().write(indices, values)


class RecyclingReqPool(FakeReqPool):
    def __init__(self) -> None:
        super().__init__()
        self.available = list(range(int(self.req_to_token.shape[0])))

    def alloc(self, handles: list[object]) -> torch.Tensor | None:
        if len(handles) > len(self.available):
            return None
        indices = [self.available.pop(0) for _ in handles]
        return torch.tensor(indices, dtype=torch.long)

    def free(self, handle: object) -> None:
        super().free(handle)
        req_pool_idx = getattr(handle, "req_pool_idx", None)
        if req_pool_idx is not None and int(req_pool_idx) not in self.available:
            self.available.append(int(req_pool_idx))
            self.available.sort()


class SliceRejectingReqPool(FakeReqPool):
    def write(self, indices, values: torch.Tensor) -> None:
        row, positions = indices
        if isinstance(row, int) and isinstance(positions, slice):
            raise TypeError("slice API intentionally unavailable")
        super().write(indices, values)


def make_bridge(*, req_pool: FakeReqPool | None = None) -> SGLangRoutePoolBridge:
    return SGLangRoutePoolBridge(
        req_to_token_pool=req_pool or FakeReqPool(),
        token_to_kv_pool_allocator=FakeAllocator(page_size=4),
        page_size=4,
        device="cpu",
        use_page_attention_metadata=False,
    )


def make_route(
    route_id: int,
    *,
    committed_length: int,
    node_ids: tuple[int, ...] = (),
    parent_route_id: int | None = None,
) -> RouteState:
    return RouteState(
        route_id=int(route_id),
        stage1_root_id=int(route_id),
        parent_route_id=parent_route_id,
        materialized_leaf_node_id=node_ids[-1] if node_ids else None,
        pending_token_id=7,
        cumulative_logprob=0.0,
        stage1_depth=len(node_ids),
        stage2_depth=0,
        kv_view=RouteKVView(
            prefix=PrefixKVView(committed_length=int(committed_length)),
            node_ids=node_ids,
        ),
    )


def test_new_row_full_init_then_inherited_row_appends_one_slot() -> None:
    bridge = make_bridge()
    parent = make_route(1, committed_length=4)
    bridge.route_slot_paths[1] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    bridge.bind_existing_route_row(1, 0)

    _, metadata = bridge.prepare_frontier(
        [parent],
        output_slot_ids=torch.tensor([4], dtype=torch.long),
    )

    assert bridge.req_to_token_pool.req_to_token[0, :5].tolist() == [0, 1, 2, 3, 4]
    assert metadata.seq_lens_sum == 5
    assert metadata.seq_lens_cpu is not None
    assert metadata.seq_lens_cpu.device.type == "cpu"
    assert metadata.seq_lens_cpu.tolist() == [5]
    assert bridge.route_rows[1].written_length == 5
    assert bridge.req_to_token_pool.write_log == [(0, 0, 5, [0, 1, 2, 3, 4])]

    child = make_route(
        2,
        committed_length=4,
        node_ids=(100,),
        parent_route_id=1,
    )
    bridge.reconcile_route_rows([parent], [child])
    bridge.prepare_frontier(
        [child],
        output_slot_ids=torch.tensor([5], dtype=torch.long),
    )

    assert bridge.req_to_token_pool.req_to_token[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert bridge.route_rows[2].written_length == 6
    assert bridge.req_to_token_pool.write_log[-1] == (0, 5, 6, [5])
    counters = bridge.hot_path_counters()
    assert counters["req_row_full_init_calls"] == 1
    assert counters["req_row_full_init_elements"] == 5
    assert counters["req_row_append_calls"] == 1
    assert counters["req_row_append_elements"] == 1
    assert counters["req_row_cow_patch_calls"] == 0
    assert counters["req_row_elements_written"] == 6


def test_inherited_cow_row_patches_only_dirty_tail_and_append() -> None:
    bridge = make_bridge()
    routes = [
        make_route(1, committed_length=5),
        make_route(2, committed_length=5),
    ]
    shared_path = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    for row_index, route in enumerate(routes):
        bridge.route_slot_paths[route.route_id] = shared_path
        bridge.bind_existing_route_row(
            route.route_id,
            row_index,
            written_length=5,
        )
        bridge.req_to_token_pool.req_to_token[row_index, :5] = shared_path

    bridge.prepare_frontier(
        routes,
        output_slot_ids=torch.tensor([5, 41], dtype=torch.long),
    )

    assert bridge.req_to_token_pool.req_to_token[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert bridge.req_to_token_pool.req_to_token[1, :6].tolist() == [0, 1, 2, 3, 40, 41]
    assert bridge.route_slot_paths[2].tolist() == [0, 1, 2, 3, 40, 41]
    assert bridge.req_to_token_pool.write_log == [
        (0, 5, 6, [5]),
        (1, 4, 6, [40, 41]),
    ]
    counters = bridge.hot_path_counters()
    assert counters["cow_pages_copied"] == 1
    assert counters["cow_tokens_copied"] == 1
    assert counters["req_row_full_init_calls"] == 0
    assert counters["req_row_append_calls"] == 1
    assert counters["req_row_append_elements"] == 1
    assert counters["req_row_cow_patch_calls"] == 1
    assert counters["req_row_cow_patch_elements"] == 2
    assert counters["req_row_elements_written"] == 3


def test_fork_reuses_first_child_row_and_fully_initializes_fresh_sibling() -> None:
    req_pool = RecyclingReqPool()
    bridge = make_bridge(req_pool=req_pool)
    parent = make_route(1, committed_length=5)
    shared_path = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    bridge.route_slot_paths[1] = shared_path
    bridge.bind_existing_route_row(1, 0, written_length=5)
    req_pool.available.remove(0)
    req_pool.req_to_token[0, :5] = shared_path
    children = [
        make_route(2, committed_length=5, parent_route_id=1),
        make_route(3, committed_length=5, parent_route_id=1),
    ]
    bridge.reconcile_route_rows([parent], children)

    bridge.prepare_frontier(
        children,
        output_slot_ids=torch.tensor([5, 41], dtype=torch.long),
    )

    assert bridge.route_rows[2].req_pool_index == 0
    assert bridge.route_rows[3].req_pool_index == 1
    assert req_pool.req_to_token[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert req_pool.req_to_token[1, :6].tolist() == [0, 1, 2, 3, 40, 41]
    counters = bridge.hot_path_counters()
    assert counters["req_row_append_calls"] == 1
    assert counters["req_row_append_elements"] == 1
    assert counters["req_row_full_init_calls"] == 1
    assert counters["req_row_full_init_elements"] == 6
    assert counters["req_row_cow_patch_calls"] == 0


def test_released_and_reused_row_is_fully_initialized() -> None:
    bridge = make_bridge()
    old_route = make_route(1, committed_length=5)
    old_handle = SimpleNamespace(req_pool_idx=0)
    bridge.bind_existing_route_row(
        1,
        0,
        handle=old_handle,
        written_length=5,
    )
    bridge.req_to_token_pool.req_to_token[0, :5] = torch.tensor([0, 1, 2, 3, 4])
    bridge.release_route_rows([old_route.route_id])

    new_route = make_route(2, committed_length=3)
    bridge.route_slot_paths[2] = torch.tensor([8, 9, 10], dtype=torch.long)
    bridge.bind_existing_route_row(2, 0)
    bridge.prepare_frontier(
        [new_route],
        output_slot_ids=torch.tensor([11], dtype=torch.long),
    )

    assert bridge.req_to_token_pool.freed == [old_handle]
    assert bridge.req_to_token_pool.req_to_token[0, :4].tolist() == [8, 9, 10, 11]
    assert bridge.req_to_token_pool.write_log == [(0, 0, 4, [8, 9, 10, 11])]
    assert bridge.route_rows[2].written_length == 4
    counters = bridge.hot_path_counters()
    assert counters["req_row_full_init_calls"] == 1
    assert counters["req_row_full_init_elements"] == 4
    assert counters["req_row_append_calls"] == 0


def test_allocator_recycled_row_resets_written_length_and_full_inits() -> None:
    req_pool = RecyclingReqPool()
    bridge = make_bridge(req_pool=req_pool)
    old_route = make_route(1, committed_length=3)
    bridge.route_slot_paths[1] = torch.tensor([8, 9, 10], dtype=torch.long)

    first_index = int(bridge.ensure_route_rows([old_route])[0])
    bridge.prepare_frontier(
        [old_route],
        output_slot_ids=torch.tensor([11], dtype=torch.long),
    )
    bridge.release_route_rows([old_route.route_id])

    new_route = make_route(2, committed_length=2)
    bridge.route_slot_paths[2] = torch.tensor([20, 21], dtype=torch.long)
    recycled_index = int(bridge.ensure_route_rows([new_route])[0])
    bridge.prepare_frontier(
        [new_route],
        output_slot_ids=torch.tensor([22], dtype=torch.long),
    )

    assert first_index == recycled_index == 0
    assert bridge.route_rows[2].written_length == 3
    assert req_pool.req_to_token[0, :3].tolist() == [20, 21, 22]
    counters = bridge.hot_path_counters()
    assert counters["req_row_full_init_calls"] == 2
    assert counters["req_row_full_init_elements"] == 7
    assert counters["req_row_append_calls"] == 0


def test_selected_route_promotion_does_not_rewrite_its_row() -> None:
    bridge = make_bridge()
    bridge.set_prefix_slots(torch.tensor([0, 1, 2, 3], dtype=torch.long))
    selected = make_route(1, committed_length=4, node_ids=(100,))
    bridge.route_slot_paths[1] = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    bridge.bind_existing_route_row(1, 0, written_length=5)
    bridge.req_to_token_pool.req_to_token[0, :5] = bridge.route_slot_paths[1]

    counters_before = bridge.hot_path_counters()
    bridge.commit_route_as_prefix(selected)
    bridge.retain_route_rows([selected.route_id])

    assert bridge.hot_path_counters() == counters_before
    assert bridge.req_to_token_pool.write_log == []
    assert bridge.route_rows[1].written_length == 5
    assert bridge.prefix_slot_ids.tolist() == [0, 1, 2, 3, 4]

    promoted = make_route(1, committed_length=5)
    bridge.prepare_frontier(
        [promoted],
        output_slot_ids=torch.tensor([5], dtype=torch.long),
    )
    assert bridge.req_to_token_pool.write_log == [(0, 5, 6, [5])]
    counters = bridge.hot_path_counters()
    assert counters["req_row_append_calls"] == 1
    assert counters["req_row_append_elements"] == 1
    assert counters["req_row_full_init_calls"] == 0


def test_failed_req_row_write_does_not_increment_counters() -> None:
    bridge = make_bridge(req_pool=AlwaysFailReqPool())

    try:
        bridge._write_req_token_slice(
            0,
            0,
            torch.tensor([1, 2], dtype=torch.long),
            write_kind="full_init",
        )
    except RuntimeError as exc:
        assert "failed to write" in str(exc)
    else:
        raise AssertionError("synthetic req-row failure was not propagated")

    counters = bridge.hot_path_counters()
    assert counters["req_row_write_calls"] == 0
    assert counters["req_row_elements_written"] == 0
    assert counters["req_row_full_init_calls"] == 0


def test_tensor_index_fallback_updates_row_and_counters_once() -> None:
    req_pool = SliceRejectingReqPool()
    bridge = make_bridge(req_pool=req_pool)

    bridge._write_req_token_slice(
        2,
        3,
        torch.tensor([17, 18], dtype=torch.long),
        write_kind="cow_patch",
    )

    assert req_pool.req_to_token[2, 3:5].tolist() == [17, 18]
    counters = bridge.hot_path_counters()
    assert counters["req_row_write_calls"] == 1
    assert counters["req_row_elements_written"] == 2
    assert counters["req_row_cow_patch_calls"] == 1
    assert counters["req_row_cow_patch_elements"] == 2


def test_failed_cow_patch_persists_dirty_range_for_retry() -> None:
    req_pool = FailSecondRowOnceReqPool()
    bridge = make_bridge(req_pool=req_pool)
    routes = [
        make_route(1, committed_length=5),
        make_route(2, committed_length=5),
    ]
    shared_path = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    for row_index, route in enumerate(routes):
        bridge.route_slot_paths[route.route_id] = shared_path
        bridge.bind_existing_route_row(
            route.route_id,
            row_index,
            written_length=5,
        )
        req_pool.req_to_token[row_index, :5] = shared_path

    try:
        bridge.prepare_frontier(
            routes,
            output_slot_ids=torch.tensor([5, 41], dtype=torch.long),
        )
    except RuntimeError as exc:
        assert "failed to write" in str(exc)
    else:
        raise AssertionError("synthetic COW patch failure was not propagated")

    # The physical route has already moved to its private tail.  The req row
    # has not, so the dirty offset must survive until a successful retry.
    assert bridge.route_slot_paths[2].tolist() == [0, 1, 2, 3, 40]
    assert bridge.route_rows[2].written_length == 5
    assert bridge.route_rows[2].pending_dirty_start == 4
    assert req_pool.req_to_token[1, :5].tolist() == [0, 1, 2, 3, 4]

    bridge.prepare_frontier(
        [routes[1]],
        output_slot_ids=torch.tensor([41], dtype=torch.long),
    )

    assert req_pool.req_to_token[1, :6].tolist() == [0, 1, 2, 3, 40, 41]
    assert bridge.route_rows[2].written_length == 6
    assert bridge.route_rows[2].pending_dirty_start is None
    assert req_pool.write_log[-1] == (1, 4, 6, [40, 41])
    counters = bridge.hot_path_counters()
    assert counters["req_row_write_calls"] == 2
    assert counters["req_row_append_calls"] == 1
    assert counters["req_row_cow_patch_calls"] == 1


def test_decode_backend_forwards_cached_cpu_sequence_metadata(monkeypatch) -> None:
    class CapturingForwardBatch:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class CapturingRunner:
        def __init__(self) -> None:
            self.forward_batch = None

        def forward(self, forward_batch):
            self.forward_batch = forward_batch
            return SimpleNamespace(next_token_logits=torch.ones((2, 4)))

    runner = CapturingRunner()
    backend = object.__new__(SGLangFlashInferPagedDecodeBackend)
    backend.model_runner = runner
    backend._ForwardBatch = CapturingForwardBatch
    backend._ForwardMode = SimpleNamespace(DECODE="decode")
    monkeypatch.setattr(flashinfer_backends, "_validate_1d_long", lambda *_: None)
    monkeypatch.setattr(
        flashinfer_backends,
        "_capture_hidden_mode_null",
        lambda: "null",
    )
    seq_lens_cpu = torch.tensor([5, 6], dtype=torch.long)
    metadata = SGLangRouteKVMetadata(
        req_pool_indices=torch.tensor([0, 1], dtype=torch.long),
        seq_lens=torch.tensor([5, 6], dtype=torch.long),
        out_cache_loc=torch.tensor([40, 44], dtype=torch.long),
        positions=torch.tensor([4, 5], dtype=torch.long),
        seq_lens_sum=11,
        seq_lens_cpu=seq_lens_cpu,
    )

    logits = backend.forward_frontier(
        input_ids=torch.tensor([7, 8], dtype=torch.long),
        route_kv_metadata=metadata,
    )

    assert logits.shape == (2, 4)
    assert runner.forward_batch.seq_lens_sum == 11
    assert runner.forward_batch.seq_lens_cpu is seq_lens_cpu

    try:
        backend.forward_frontier(
            input_ids=torch.tensor([7, 8], dtype=torch.long),
            route_kv_metadata=replace(
                metadata,
                seq_lens_cpu=torch.tensor([[5, 6]], dtype=torch.long),
            ),
        )
    except ValueError as exc:
        assert "shape [B]" in str(exc)
    else:
        raise AssertionError("non-vector seq_lens_cpu was accepted")

    try:
        backend.forward_frontier(
            input_ids=torch.tensor([7, 8], dtype=torch.long),
            route_kv_metadata=replace(
                metadata,
                seq_lens_cpu=torch.empty((2,), device="meta", dtype=torch.long),
            ),
        )
    except ValueError as exc:
        assert "CPU tensor" in str(exc)
    else:
        raise AssertionError("non-CPU seq_lens_cpu was accepted")
