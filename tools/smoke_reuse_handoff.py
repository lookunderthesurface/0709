from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_0709.distributed_system import (
    DistributedAtlasConfig,
    PagedDistributedAtlasGenerator,
    PagedPrefixContext,
)
from atlas_0709.flashinfer_paged.builders import (
    build_forest_one_depth,
    build_tree_depths,
    initialize_forest_routes,
    initialize_stage1_routes,
)
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.types import (
    DraftPrefixState,
    FrontierDecodeOutput,
    PrefixKVView,
    RouteState,
)


class FakeReusableBackend:
    def __init__(self, store: KVTreeStore, prefix_len: int, vocab_size: int = 64) -> None:
        self.store = store
        self.prefix_len = int(prefix_len)
        self.vocab_size = int(vocab_size)

    def decode_frontier_one_token(
        self,
        active_routes: list[RouteState],
        attention_backend=None,
    ) -> FrontierDecodeOutput:
        rows = []
        for route in active_routes:
            center = (
                int(route.pending_token_id)
                + int(route.route_id)
                + int(route.total_materialized_depth())
            ) % self.vocab_size
            logits = -torch.arange(self.vocab_size, dtype=torch.float32)
            logits = torch.roll(logits, shifts=center)
            rows.append(logits)
        node_ids = self.store.reserve_node_ids(len(active_routes))
        return FrontierDecodeOutput(
            route_ids=torch.tensor([route.route_id for route in active_routes]),
            next_token_logits=torch.stack(rows),
            new_node_ids=torch.tensor(node_ids),
            metadata={"backend": "fake_reusable"},
        )

    def commit_stage1_and_promote(
        self,
        *,
        committed_route: RouteState,
        retained_routes: list[RouteState],
    ):
        committed_nodes = self.store.node_path_for_route(committed_route)
        self.store.commit_route(committed_route)
        self.prefix_len += len(committed_nodes)
        promoted = self.store.promote_routes_after_commit(
            committed_route,
            retained_routes,
            PrefixKVView(committed_length=self.prefix_len),
        )
        return promoted, {
            "committed_kv_tokens": len(committed_nodes),
            "retained_routes": len(promoted),
            "released_route_rows": 0,
            "released_kv_pages": 0,
        }


def prepare_case(*, k: int, d: int, prefix_len: int):
    store = KVTreeStore()
    store.committed_token_ids = [1] * prefix_len
    backend = FakeReusableBackend(store, prefix_len)
    prefix_logits = torch.linspace(0.0, 1.0, backend.vocab_size)
    prefix = DraftPrefixState(
        token_ids=torch.ones(prefix_len, dtype=torch.long),
        prefix_kv_view=PrefixKVView(committed_length=prefix_len),
        next_token_logits=prefix_logits,
        committed_length=prefix_len,
    )
    active = initialize_stage1_routes(prefix, k=k, route_store=store)
    stage1 = build_tree_depths(
        active,
        depth=d,
        k=k,
        route_store=store,
        model_backend=backend,
    )
    context = PagedPrefixContext(store=store, backend=backend, prefix=prefix)
    return context, stage1


def run_case(*, forest_depth: int, k: int = 3, d: int = 4, prefix_len: int = 8):
    context, stage1 = prepare_case(k=k, d=d, prefix_len=prefix_len)
    frontier = initialize_forest_routes(
        stage1.completed_routes,
        stage1.last_logits,
        k=k,
        route_store=context.store,
    )
    completed = None
    last_logits = None
    for _ in range(forest_depth):
        step = build_forest_one_depth(
            frontier,
            k=k,
            route_store=context.store,
            model_backend=context.backend,
        )
        completed = step.decoded_routes
        last_logits = step.decode_output.next_token_logits
        frontier = step.next_routes

    generator = object.__new__(PagedDistributedAtlasGenerator)
    generator.config = DistributedAtlasConfig(k=k, d=d, max_new_tokens=32)
    selected = stage1.completed_routes[0]
    handoff = generator._handoff_after_verify(
        ctx=context,
        selected_stage1_route=selected,
        forest_depth=forest_depth,
        forest_frontier=frontier,
        forest_completed=completed,
        forest_last_logits=last_logits,
    )

    assert len(handoff.stage1.completed_routes) == k
    assert {route.stage1_depth for route in handoff.stage1.completed_routes} == {d}
    assert {route.stage2_depth for route in handoff.stage1.completed_routes} == {0}
    assert handoff.reused_stage2_routes == k
    assert handoff.reused_stage2_depth == forest_depth
    assert context.backend.prefix_len == prefix_len + d
    return {
        "forest_depth_at_verify": forest_depth,
        "handoff_mode": handoff.mode,
        "reused_stage2_routes": handoff.reused_stage2_routes,
        "reused_stage2_depth": handoff.reused_stage2_depth,
        "remaining_tree_depth": handoff.remaining_tree_depth,
        "next_stage1_depth": handoff.stage1.completed_routes[0].stage1_depth,
        "new_prefix_len": context.backend.prefix_len,
    }


def main() -> int:
    result = {
        "partial_forest": run_case(forest_depth=1),
        "completed_forest": run_case(forest_depth=4),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
