from types import SimpleNamespace

import torch

from atlas_0709.flashinfer_paged.sglang_runtime import SGLangRoutePoolBridge


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
    def clear(self) -> None:
        pass


def make_bridge(page_size: int = 4) -> SGLangRoutePoolBridge:
    return SGLangRoutePoolBridge(
        req_to_token_pool=FakeReqPool(),
        token_to_kv_pool_allocator=FakeAllocator(page_size),
        page_size=page_size,
        device="cpu",
    )


def test_partial_tail_is_copied_per_route() -> None:
    bridge = make_bridge()
    shared_tail = torch.tensor([0, 1, 2], dtype=torch.long)
    routes = [SimpleNamespace(route_id=1), SimpleNamespace(route_id=2)]

    copied = bridge._copy_partial_tail_pages(
        routes,
        [shared_tail, shared_tail],
    )

    assert copied[0].tolist() == [40, 41, 42]
    assert copied[1].tolist() == [44, 45, 46]
    assert bridge.cow_pages_copied == 2
    assert bridge.cow_tokens_copied == 6
    assert bridge.token_to_kv_pool_allocator.kv_cache.moves == [
        ([40, 41, 42], [0, 1, 2]),
        ([44, 45, 46], [0, 1, 2]),
    ]


def test_full_tail_page_remains_shared() -> None:
    bridge = make_bridge()
    full_page = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    route = SimpleNamespace(route_id=1)

    copied = bridge._copy_partial_tail_pages([route], [full_page])

    assert copied[0] is full_page
    assert bridge.cow_pages_copied == 0


def test_batch1_ar_does_not_copy_a_partial_tail() -> None:
    bridge = make_bridge()
    partial_page = torch.tensor([0, 1, 2], dtype=torch.long)
    route = SimpleNamespace(route_id=1)

    copied = bridge._copy_partial_tail_pages([route], [partial_page])

    assert copied[0] is partial_page
    assert bridge.cow_pages_copied == 0
