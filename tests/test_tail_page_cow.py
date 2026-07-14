from types import SimpleNamespace

import torch

from atlas_0709.flashinfer_paged.sglang_runtime import SGLangRoutePoolBridge
from atlas_0709.flashinfer_paged.types import PrefixKVView, RouteKVView, RouteState


class FakeKVCache:
    def __init__(self) -> None:
        self.values = torch.arange(512, dtype=torch.float32)
        self.moves: list[tuple[list[int], list[int]]] = []

    def move_kv_cache(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        self.values[destination] = self.values[source]
        self.moves.append((destination.tolist(), source.tolist()))


class FakeAllocator:
    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.next_page = 10
        self.kv_cache = FakeKVCache()
        self.freed: list[int] = []

    def alloc(self, count: int) -> torch.Tensor:
        assert count % self.page_size == 0
        page_count = count // self.page_size
        pages = torch.arange(self.next_page, self.next_page + page_count)
        self.next_page += page_count
        offsets = torch.arange(self.page_size)
        return (pages[:, None] * self.page_size + offsets[None, :]).reshape(-1)

    def get_kvcache(self) -> FakeKVCache:
        return self.kv_cache

    def free(self, slots: torch.Tensor) -> None:
        self.freed.extend(slots.tolist())


class FakeReqPool:
    def __init__(self) -> None:
        self.req_to_token = torch.zeros((8, 128), dtype=torch.long)

    def write(self, indices, values: torch.Tensor) -> None:
        self.req_to_token[indices] = values

    def clear(self) -> None:
        pass


def make_bridge(page_size: int = 4) -> SGLangRoutePoolBridge:
    return SGLangRoutePoolBridge(
        req_to_token_pool=FakeReqPool(),
        token_to_kv_pool_allocator=FakeAllocator(page_size),
        page_size=page_size,
        device="cpu",
    )


def test_shared_partial_tail_copies_only_one_writer() -> None:
    bridge = make_bridge()
    shared_tail = torch.tensor([0, 1, 2], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(
        routes,
        [shared_tail, shared_tail],
    )

    assert copied[0] is shared_tail
    assert copied[1].tolist() == [40, 41, 42]
    assert bridge.cow_pages_copied == 1
    assert bridge.cow_tokens_copied == 3
    assert bridge.hot_path_counters()["hot_path_host_transfer_batches"] == 1
    assert bridge.hot_path_counters()["hot_path_host_transfer_elements"] == 2
    assert bridge.token_to_kv_pool_allocator.kv_cache.moves == [
        ([40, 41, 42], [0, 1, 2]),
    ]


def test_n_shared_partial_tail_writers_copy_n_minus_one_pages() -> None:
    bridge = make_bridge()
    shared_tail = torch.tensor([0, 1], dtype=torch.long)
    routes = [SimpleNamespace(route_id=index) for index in (1, 2, 3)]

    copied = bridge._copy_partial_tail_pages(
        routes,
        [shared_tail, shared_tail, shared_tail],
    )

    assert copied[0] is shared_tail
    assert copied[1].tolist() == [40, 41]
    assert copied[2].tolist() == [44, 45]
    assert bridge.cow_pages_copied == 2
    assert bridge.cow_tokens_copied == 4


def test_same_partial_page_with_different_valid_lengths_is_copied() -> None:
    bridge = make_bridge()
    short_tail = torch.tensor([0, 1], dtype=torch.long)
    long_tail = torch.tensor([0, 1, 2], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(routes, [short_tail, long_tail])

    assert copied[0] is short_tail
    assert copied[1].tolist() == [40, 41, 42]
    assert bridge.cow_pages_copied == 1
    assert bridge.cow_tokens_copied == 3
    assert bridge.token_to_kv_pool_allocator.kv_cache.moves == [
        ([40, 41, 42], [0, 1, 2]),
    ]


def test_prefix_slices_on_same_partial_page_share_tail_lease() -> None:
    bridge = make_bridge()
    bridge.set_prefix_slots(torch.tensor([0, 1, 2], dtype=torch.long))
    routes = [
        RouteState(
            route_id=1,
            stage1_root_id=1,
            parent_route_id=None,
            materialized_leaf_node_id=None,
            pending_token_id=7,
            cumulative_logprob=0.0,
            stage1_depth=0,
            stage2_depth=0,
            kv_view=RouteKVView(
                prefix=PrefixKVView(committed_length=2),
                node_ids=(),
            ),
        ),
        RouteState(
            route_id=2,
            stage1_root_id=2,
            parent_route_id=None,
            materialized_leaf_node_id=None,
            pending_token_id=8,
            cumulative_logprob=0.0,
            stage1_depth=0,
            stage2_depth=0,
            kv_view=RouteKVView(
                prefix=PrefixKVView(committed_length=3),
                node_ids=(),
            ),
        ),
    ]
    slot_paths = [bridge._slot_path_for_route(route) for route in routes]

    assert bridge.route_tail_page_keys[1] == bridge.route_tail_page_keys[2]
    copied = bridge._copy_partial_tail_pages(routes, slot_paths)

    assert copied[0].tolist() == [0, 1]
    assert copied[1].tolist() == [40, 41, 42]
    assert bridge.cow_pages_copied == 1
    assert bridge.hot_path_counters()["hot_path_host_transfer_batches"] == 0


def test_distinct_partial_tail_pages_are_not_copied_in_batch() -> None:
    bridge = make_bridge()
    first = torch.tensor([0, 1, 2], dtype=torch.long)
    second = torch.tensor([4, 5, 6], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(routes, [first, second])

    assert copied[0] is first
    assert copied[1] is second
    assert bridge.cow_pages_copied == 0
    assert bridge.cow_tokens_copied == 0
    assert bridge.token_to_kv_pool_allocator.kv_cache.moves == []


def test_private_cow_tails_append_without_being_copied_again() -> None:
    bridge = make_bridge(page_size=8)
    shared_tail = torch.tensor([0, 1, 2], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(routes, [shared_tail, shared_tail])
    assert bridge.cow_pages_copied == 1

    continued = [
        torch.cat((copied[0], torch.tensor([3], dtype=torch.long))),
        torch.cat((copied[1], torch.tensor([83], dtype=torch.long))),
    ]
    reused = bridge._copy_partial_tail_pages(routes, continued)

    assert reused[0] is continued[0]
    assert reused[1] is continued[1]
    assert bridge.cow_pages_copied == 1
    assert bridge.cow_tokens_copied == 3
    assert bridge.token_to_kv_pool_allocator.kv_cache.moves == [
        ([80, 81, 82], [0, 1, 2]),
    ]


def test_cow_branches_can_write_different_kv_without_contamination() -> None:
    bridge = make_bridge()
    shared_tail = torch.tensor([0, 1, 2], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(routes, [shared_tail, shared_tail])
    kv_cache = bridge.token_to_kv_pool_allocator.kv_cache
    first_output_slot = int(copied[0][-1]) + 1
    second_output_slot = int(copied[1][-1]) + 1
    kv_cache.values[first_output_slot] = 111.0
    kv_cache.values[second_output_slot] = 222.0

    assert copied[0].tolist() == [0, 1, 2]
    assert copied[1].tolist() == [40, 41, 42]
    assert kv_cache.values[copied[0]].tolist() == [0.0, 1.0, 2.0]
    assert kv_cache.values[copied[1]].tolist() == [0.0, 1.0, 2.0]
    assert float(kv_cache.values[first_output_slot]) == 111.0
    assert float(kv_cache.values[second_output_slot]) == 222.0


def test_shared_full_tail_page_does_not_need_cow() -> None:
    bridge = make_bridge()
    full_page = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(routes, [full_page, full_page])

    assert copied[0] is full_page
    assert copied[1] is full_page
    assert bridge.cow_pages_copied == 0


def test_batch1_ar_does_not_copy_a_partial_tail() -> None:
    bridge = make_bridge()
    partial_page = torch.tensor([0, 1, 2], dtype=torch.long)
    route = SimpleNamespace(route_id=1)

    copied = bridge._copy_partial_tail_pages([route], [partial_page])

    assert copied[0] is partial_page
    assert bridge.cow_pages_copied == 0


def test_page_reference_counts_and_selected_promotion_release() -> None:
    bridge = make_bridge()
    bridge.set_prefix_slots(torch.tensor([0, 1], dtype=torch.long))
    selected = RouteState(
        route_id=1,
        stage1_root_id=1,
        parent_route_id=None,
        materialized_leaf_node_id=1,
        pending_token_id=7,
        cumulative_logprob=0.0,
        stage1_depth=1,
        stage2_depth=0,
        kv_view=RouteKVView(
            prefix=PrefixKVView(committed_length=2),
            node_ids=(1,),
        ),
    )
    bridge.route_slot_paths = {
        1: torch.tensor([0, 1, 40], dtype=torch.long),
        2: torch.tensor([0, 1, 40, 41], dtype=torch.long),
        3: torch.tensor([0, 1, 44], dtype=torch.long),
    }
    bridge.owned_page_ids.update({10, 11})

    assert bridge.page_reference_counts() == {0: 4, 10: 2, 11: 1}

    bridge.commit_route_as_prefix(selected)
    bridge.retain_route_rows([2])
    assert bridge.page_reference_counts() == {0: 2, 10: 2}
    assert bridge.owned_page_ids == {0, 10, 11}

    released = bridge.release_unreferenced_node_slots([])

    assert bridge.page_reference_counts() == {0: 2, 10: 2}
    assert released == 1
    assert bridge.owned_page_ids == {0, 10}
    assert bridge.token_to_kv_pool_allocator.freed == [44]


def test_pending_owned_pages_are_materialized_once_before_release() -> None:
    bridge = make_bridge()
    bridge.set_prefix_slots(torch.tensor([0, 1], dtype=torch.long))
    bridge.route_slot_paths[1] = torch.tensor([0, 1, 40], dtype=torch.long)
    bridge._record_pending_owned_slots(torch.tensor([40, 41, 44], dtype=torch.long))

    released = bridge.release_unreferenced_node_slots([])

    assert released == 1
    assert bridge.pending_owned_slot_tensors == []
    assert bridge.owned_page_ids == {0, 10}
    assert bridge.token_to_kv_pool_allocator.freed == [44]
    assert bridge.release_unreferenced_node_slots([]) == 0
    assert bridge.token_to_kv_pool_allocator.freed == [44]


def test_tail_lease_propagates_through_reconcile_retain_commit_and_clear() -> None:
    bridge = make_bridge()
    bridge.set_prefix_slots(torch.tensor([40, 41, 42], dtype=torch.long))
    parent = RouteState(
        route_id=1,
        stage1_root_id=1,
        parent_route_id=None,
        materialized_leaf_node_id=None,
        pending_token_id=7,
        cumulative_logprob=0.0,
        stage1_depth=0,
        stage2_depth=0,
        kv_view=RouteKVView(
            prefix=PrefixKVView(committed_length=3),
            node_ids=(),
        ),
    )
    children = [
        RouteState(
            route_id=route_id,
            stage1_root_id=route_id,
            parent_route_id=1,
            materialized_leaf_node_id=None,
            pending_token_id=8,
            cumulative_logprob=0.0,
            stage1_depth=0,
            stage2_depth=0,
            kv_view=parent.kv_view.fork(),
        )
        for route_id in (2, 3)
    ]
    bridge._slot_path_for_route(parent)

    bridge.reconcile_route_rows([parent], children)
    parent_key = bridge.route_tail_page_keys[1]
    assert bridge.route_tail_page_keys[2] == parent_key
    assert bridge.route_tail_page_keys[3] == parent_key

    bridge.retain_route_rows([2])
    bridge.commit_route_as_prefix(children[0])

    assert bridge.prefix_slot_count == 3
    assert bridge.prefix_page_ids == {10}
    assert bridge.prefix_tail_page_key == parent_key
    assert bridge.prefix_partial_tail_keys == {0: parent_key}
    assert bridge.route_tail_page_keys == {2: parent_key}

    bridge.clear_physical_state()
    assert bridge.prefix_slot_count == 0
    assert bridge.prefix_page_ids == set()
    assert bridge.prefix_tail_page_key is None
    assert bridge.prefix_partial_tail_keys == {}
    assert bridge.route_tail_page_keys == {}


def test_req_row_full_rewrite_counters_record_written_elements() -> None:
    bridge = make_bridge()
    bridge._write_req_token_row(3, torch.tensor([4, 5, 6, 7], dtype=torch.long))
    bridge._write_req_token_row(3, torch.tensor([4, 5, 6, 7, 8], dtype=torch.long))

    counters = bridge.hot_path_counters()
    assert counters["req_row_write_calls"] == 2
    assert counters["req_row_elements_written"] == 9
    assert counters["req_row_full_rewrite_calls"] == 2
    assert counters["req_row_full_rewrite_elements"] == 9
