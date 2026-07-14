from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Callable, Iterable, Sequence

from .types import DecodePhase, PendingCandidate, PrefixKVView, RouteKVView, RouteState


@dataclass
class KVTreeNode:
    node_id: int
    token_id: int
    parent_node_id: int | None
    stage1_root_id: int
    depth: int
    owner_route_id: int | None = None
    committed: bool = False


class KVTreeStore:
    """Logical route/KV topology store.

    This class deliberately stores node ids, token ids, and ownership metadata
    only. Fast backends can attach these logical node ids to physical KV pages.
    The reference backend uses the ids to reconstruct token paths.
    """

    def __init__(self) -> None:
        self._route_ids = count(1)
        self._node_ids = count(1)
        self.nodes: dict[int, KVTreeNode] = {}
        self.routes: dict[int, RouteState] = {}
        self.committed_token_ids: list[int] = []
        self._route_release_hooks: list[Callable[[Sequence[RouteState], Sequence[RouteState]], None]] = []

    def reset_speculative(self) -> None:
        self.nodes.clear()
        self.routes.clear()

    def add_route_release_hook(
        self,
        hook: Callable[[Sequence[RouteState], Sequence[RouteState]], None],
    ) -> None:
        self._route_release_hooks.append(hook)

    def allocate_route_id(self) -> int:
        return next(self._route_ids)

    def allocate_node_id(self) -> int:
        return next(self._node_ids)

    def reserve_node_ids(self, num_nodes: int) -> list[int]:
        return [self.allocate_node_id() for _ in range(num_nodes)]

    def prefix_view(self, committed_length: int | None = None) -> PrefixKVView:
        length = len(self.committed_token_ids) if committed_length is None else committed_length
        return PrefixKVView(committed_length=length)

    def register_route(self, route: RouteState) -> RouteState:
        self.routes[route.route_id] = route
        return route

    def get_route(self, route_id: int) -> RouteState:
        return self.routes[int(route_id)]

    def get_routes(self, route_ids: Iterable[int]) -> list[RouteState]:
        return [self.get_route(route_id) for route_id in route_ids]

    def mark_routes_materialized(
        self,
        routes: Sequence[RouteState],
        new_node_ids: Sequence[int],
        phase: DecodePhase,
    ) -> list[RouteState]:
        if len(routes) != len(new_node_ids):
            raise ValueError("routes and new_node_ids must have the same length")

        materialized: list[RouteState] = []
        for route, raw_node_id in zip(routes, new_node_ids):
            node_id = int(raw_node_id)
            parent_node_id = route.kv_view.node_ids[-1] if route.kv_view.node_ids else None
            next_depth = route.total_materialized_depth() + 1
            if route.token_logprobs and len(route.token_logprobs) != next_depth:
                raise ValueError(
                    "frontier route token_logprobs must include exactly its pending token: "
                    f"route_id={route.route_id}, logprobs={len(route.token_logprobs)}, "
                    f"next_materialized_depth={next_depth}"
                )
            self.nodes[node_id] = KVTreeNode(
                node_id=node_id,
                token_id=int(route.pending_token_id),
                parent_node_id=parent_node_id,
                stage1_root_id=route.stage1_root_id,
                depth=next_depth,
                owner_route_id=route.route_id,
            )

            route.materialized_leaf_node_id = node_id
            route.kv_view = route.kv_view.append_node(node_id)
            if phase == DecodePhase.STAGE1:
                route.stage1_depth += 1
            elif phase == DecodePhase.STAGE2:
                route.stage2_depth += 1
            else:
                raise ValueError(f"unsupported decode phase: {phase}")

            self.register_route(route)
            materialized.append(route)

        return materialized

    def materialize_route_descriptors(
        self,
        selected: Sequence[PendingCandidate],
        parent_routes: Sequence[RouteState],
    ) -> list[RouteState]:
        parents = {route.route_id: route for route in parent_routes}
        next_routes: list[RouteState] = []
        for candidate in selected:
            try:
                parent = parents[candidate.parent_route_id]
            except KeyError:
                parent = self.get_route(candidate.parent_route_id)

            route_id = self.allocate_route_id()
            if parent.token_logprobs and len(parent.token_logprobs) != parent.total_materialized_depth():
                raise ValueError(
                    "decoded parent token_logprobs must match its materialized path: "
                    f"route_id={parent.route_id}, logprobs={len(parent.token_logprobs)}, "
                    f"materialized_depth={parent.total_materialized_depth()}"
                )
            local_logprob = float(candidate.cumulative_logprob - candidate.parent_logprob)
            route = RouteState(
                route_id=route_id,
                stage1_root_id=candidate.stage1_root_id,
                parent_route_id=parent.route_id,
                materialized_leaf_node_id=parent.materialized_leaf_node_id,
                pending_token_id=int(candidate.pending_token_id),
                cumulative_logprob=float(candidate.cumulative_logprob),
                stage1_depth=parent.stage1_depth,
                stage2_depth=parent.stage2_depth,
                kv_view=parent.kv_view.fork(),
                token_logprobs=(
                    (*parent.token_logprobs, local_logprob)
                    if parent.token_logprobs
                    else ()
                ),
            )
            self.register_route(route)
            next_routes.append(route)
        return next_routes

    def release_routes_without_descendants(
        self,
        previous_routes: Sequence[RouteState],
        next_routes: Sequence[RouteState],
    ) -> None:
        """Hook for physical KV backends.

        The logical reference store keeps historical routes so verification and
        debugging can inspect them. A paged KV backend should override or wrap
        this hook to decrement page/node references for unneeded branches.
        """
        for hook in self._route_release_hooks:
            hook(previous_routes, next_routes)

    def tokens_for_node_ids(self, node_ids: Sequence[int]) -> tuple[int, ...]:
        return tuple(self.nodes[int(node_id)].token_id for node_id in node_ids)

    def tokens_for_view(self, kv_view: RouteKVView) -> tuple[int, ...]:
        return self.tokens_for_node_ids(kv_view.node_ids)

    def materialized_token_path(self, route: RouteState) -> tuple[int, ...]:
        return self.tokens_for_view(route.kv_view)

    def node_path_for_route(self, route: RouteState) -> tuple[int, ...]:
        return tuple(route.kv_view.node_ids)

    def pending_token_path(self, route: RouteState) -> tuple[int, ...]:
        return (*self.materialized_token_path(route), int(route.pending_token_id))

    def commit_route(self, route: RouteState, max_tokens: int | None = None) -> tuple[int, ...]:
        node_ids = self.node_path_for_route(route)
        if max_tokens is not None:
            node_ids = node_ids[:max_tokens]
        for node_id in node_ids:
            self.nodes[node_id].committed = True
        tokens = self.tokens_for_node_ids(node_ids)
        self.committed_token_ids.extend(tokens)
        return tokens

    def promote_routes_after_commit(
        self,
        committed_route: RouteState,
        routes: Sequence[RouteState],
        new_prefix: PrefixKVView,
    ) -> list[RouteState]:
        committed_node_ids = self.node_path_for_route(committed_route)
        promoted: list[RouteState] = []

        for route in routes:
            suffix_node_ids = route.kv_view.node_ids[len(committed_node_ids) :]
            if route.kv_view.node_ids[: len(committed_node_ids)] != committed_node_ids:
                raise ValueError("cannot promote a route outside the committed branch")

            if route.token_logprobs:
                materialized_depth = route.total_materialized_depth()
                valid_lengths = {materialized_depth, materialized_depth + 1}
                if len(route.token_logprobs) not in valid_lengths:
                    raise ValueError(
                        "promoted route token_logprobs do not match its token path: "
                        f"route_id={route.route_id}, logprobs={len(route.token_logprobs)}, "
                        f"valid_lengths={sorted(valid_lengths)}"
                    )
                route.token_logprobs = tuple(
                    route.token_logprobs[len(committed_node_ids) :]
                )

            route.kv_view = RouteKVView(prefix=new_prefix, node_ids=tuple(suffix_node_ids))
            route.materialized_leaf_node_id = suffix_node_ids[-1] if suffix_node_ids else None
            route.stage1_depth = route.stage2_depth
            route.stage2_depth = 0
            route.stage1_root_id = route.route_id
            route.parent_route_id = None
            self.register_route(route)
            promoted.append(route)

        return promoted

    def clone_route_with_new_pending(
        self,
        route: RouteState,
        pending_token_id: int,
        cumulative_logprob: float,
        stage1_root_id: int | None = None,
    ) -> RouteState:
        route_id = self.allocate_route_id()
        cloned = RouteState(
            route_id=route_id,
            stage1_root_id=route.stage1_root_id if stage1_root_id is None else stage1_root_id,
            parent_route_id=route.route_id,
            materialized_leaf_node_id=route.materialized_leaf_node_id,
            pending_token_id=int(pending_token_id),
            cumulative_logprob=float(cumulative_logprob),
            stage1_depth=route.stage1_depth,
            stage2_depth=route.stage2_depth,
            kv_view=route.kv_view.fork(),
            token_logprobs=(
                (
                    *route.token_logprobs,
                    float(cumulative_logprob - route.cumulative_logprob),
                )
                if route.token_logprobs
                else ()
            ),
        )
        return self.register_route(cloned)
