from __future__ import annotations

from itertools import count
from typing import Sequence

from .types import DecodePhase, PendingCandidate, RouteState


class RouteStore:
    """Small logical route store.

    This project intentionally tracks only route topology and token paths. A
    backend may attach physical KV cache state elsewhere, but the clean
    controller does not depend on optimized KV sharing.
    """

    def __init__(self) -> None:
        self._route_ids = count(1)
        self.routes: dict[int, RouteState] = {}

    def reset(self) -> None:
        self.routes.clear()

    def allocate_route_id(self) -> int:
        return next(self._route_ids)

    def register(self, route: RouteState) -> RouteState:
        self.routes[int(route.route_id)] = route
        return route

    def get(self, route_id: int) -> RouteState:
        return self.routes[int(route_id)]

    def materialize_pending(
        self,
        routes: Sequence[RouteState],
        *,
        phase: DecodePhase,
    ) -> list[RouteState]:
        decoded: list[RouteState] = []
        for route in routes:
            if phase == DecodePhase.STAGE1:
                stage1_depth = route.stage1_depth + 1
                stage2_depth = route.stage2_depth
            elif phase == DecodePhase.STAGE2:
                stage1_depth = route.stage1_depth
                stage2_depth = route.stage2_depth + 1
            else:
                raise ValueError(f"unsupported phase: {phase}")

            decoded_route = RouteState(
                route_id=route.route_id,
                stage1_root_id=route.stage1_root_id,
                parent_route_id=route.parent_route_id,
                materialized_tokens=route.pending_path,
                pending_token_id=route.pending_token_id,
                cumulative_logprob=route.cumulative_logprob,
                stage1_depth=stage1_depth,
                stage2_depth=stage2_depth,
            )
            self.register(decoded_route)
            decoded.append(decoded_route)
        return decoded

    def make_next_routes(
        self,
        selected: Sequence[PendingCandidate],
        *,
        parent_routes: Sequence[RouteState],
    ) -> list[RouteState]:
        parents = {route.route_id: route for route in parent_routes}
        next_routes: list[RouteState] = []
        for candidate in selected:
            parent = parents.get(candidate.parent_route_id) or self.get(candidate.parent_route_id)
            route = RouteState(
                route_id=self.allocate_route_id(),
                stage1_root_id=candidate.stage1_root_id,
                parent_route_id=parent.route_id,
                materialized_tokens=parent.materialized_tokens,
                pending_token_id=int(candidate.pending_token_id),
                cumulative_logprob=float(candidate.cumulative_logprob),
                stage1_depth=parent.stage1_depth,
                stage2_depth=parent.stage2_depth,
            )
            self.register(route)
            next_routes.append(route)
        return next_routes

    def promote_after_stage1_commit(
        self,
        selected_stage1_route: RouteState,
        routes: Sequence[RouteState],
    ) -> list[RouteState]:
        committed_prefix = selected_stage1_route.materialized_tokens[: selected_stage1_route.stage1_depth]
        promoted: list[RouteState] = []
        for route in routes:
            if route.materialized_tokens[: len(committed_prefix)] != committed_prefix:
                raise RuntimeError("cannot promote a route outside the selected stage-1 branch")
            suffix = tuple(route.materialized_tokens[len(committed_prefix) :])
            promoted_route = RouteState(
                route_id=route.route_id,
                stage1_root_id=route.route_id,
                parent_route_id=None,
                materialized_tokens=suffix,
                pending_token_id=route.pending_token_id,
                cumulative_logprob=route.cumulative_logprob,
                stage1_depth=route.stage2_depth,
                stage2_depth=0,
            )
            self.register(promoted_route)
            promoted.append(promoted_route)
        return promoted

