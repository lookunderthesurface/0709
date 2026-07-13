from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .backends import BatchDecodeBackend, BatchVerifyBackend
from .builders import build_forest_step, build_tree, initialize_forest_routes, initialize_stage1_routes
from .route_store import RouteStore
from .types import BuildResult, PrefixState, RoundTrace, RouteState
from .verify import BatchTargetVerifier


@dataclass(frozen=True)
class AtlasCleanConfig:
    k: int = 3
    d: int = 4
    max_new_tokens: int = 64
    eos_token_id: int | None = None

    def validate(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.d <= 0:
            raise ValueError("d must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass
class GenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: list[int]
    rounds: list[RoundTrace] = field(default_factory=list)

    @property
    def token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.generated_token_ids]


class AtlasCleanGenerator:
    def __init__(
        self,
        *,
        config: AtlasCleanConfig,
        drafter: BatchDecodeBackend,
        target: BatchVerifyBackend,
        route_store: RouteStore | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.drafter = drafter
        self.target = target
        self.verifier = BatchTargetVerifier(target)
        self.route_store = route_store or RouteStore()

    def generate(self, prompt_token_ids: Sequence[int]) -> GenerationResult:
        committed = [int(token_id) for token_id in prompt_token_ids]
        generated: list[int] = []
        traces: list[RoundTrace] = []
        self.route_store.reset()

        draft_prefix_state, _target_prefix_state = self._prefill_both(committed)
        active_routes = initialize_stage1_routes(
            draft_prefix_state,
            k=self.config.k,
            route_store=self.route_store,
        )
        stage1 = build_tree(
            active_routes,
            prefix_token_ids=committed,
            depth=self.config.d,
            k=self.config.k,
            route_store=self.route_store,
            drafter=self.drafter,
        )

        with ThreadPoolExecutor(max_workers=1) as verifier_pool:
            round_index = 0
            while len(generated) < self.config.max_new_tokens:
                verify_future = verifier_pool.submit(
                    self.verifier.verify,
                    prefix_token_ids=tuple(committed),
                    routes=list(stage1.completed_routes),
                )

                forest_frontier = initialize_forest_routes(
                    stage1.completed_routes,
                    stage1.last_logits,
                    k=self.config.k,
                    route_store=self.route_store,
                )
                forest_completed: list[RouteState] | None = None
                forest_last_logits: torch.Tensor | None = None
                forest_depth = 0

                while not verify_future.done() and forest_depth < self.config.d:
                    forest_step = build_forest_step(
                        forest_frontier,
                        prefix_token_ids=committed,
                        k=self.config.k,
                        route_store=self.route_store,
                        drafter=self.drafter,
                    )
                    forest_depth += 1
                    forest_completed = forest_step.decoded_routes
                    forest_frontier = forest_step.next_routes
                    forest_last_logits = forest_step.decode_output.logits

                verify_returned_before_forest_done = forest_depth < self.config.d and verify_future.done()
                verify_result = verify_future.result()
                selected_stage1_route = self.route_store.get(verify_result.selected_route_id)
                committed_now = self._commit_stage1_path(selected_stage1_route, generated)
                generated.extend(committed_now)
                committed.extend(committed_now)

                traces.append(
                    RoundTrace(
                        round_index=round_index,
                        selected_route_id=verify_result.selected_route_id,
                        committed_tokens=tuple(committed_now),
                        forest_depth=forest_depth,
                        verify_returned_before_forest_done=verify_returned_before_forest_done,
                        target_scores=tuple((score.route_id, score.target_logprob) for score in verify_result.scores),
                    )
                )
                round_index += 1

                if self._should_stop(committed_now, generated):
                    break

                self.target.prefill(tuple(committed))
                next_stage1 = self._next_stage1_after_commit(
                    committed=committed,
                    selected_stage1_route=selected_stage1_route,
                    forest_depth=forest_depth,
                    forest_frontier=forest_frontier,
                    forest_completed=forest_completed,
                    forest_last_logits=forest_last_logits,
                )
                if next_stage1 is None:
                    self.route_store.reset()
                    draft_prefix_state = self.drafter.prefill(committed)
                    active_routes = initialize_stage1_routes(
                        draft_prefix_state,
                        k=self.config.k,
                        route_store=self.route_store,
                    )
                    stage1 = build_tree(
                        active_routes,
                        prefix_token_ids=committed,
                        depth=self.config.d,
                        k=self.config.k,
                        route_store=self.route_store,
                        drafter=self.drafter,
                    )
                else:
                    stage1 = next_stage1

        return GenerationResult(
            prompt_token_ids=tuple(int(token_id) for token_id in prompt_token_ids),
            generated_token_ids=generated,
            rounds=traces,
        )

    def _prefill_both(self, committed: Sequence[int]) -> tuple[PrefixState, PrefixState]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            draft_future = pool.submit(self.drafter.prefill, tuple(committed))
            target_future = pool.submit(self.target.prefill, tuple(committed))
            return draft_future.result(), target_future.result()

    def _commit_stage1_path(self, selected_stage1_route: RouteState, generated: Sequence[int]) -> list[int]:
        path = list(selected_stage1_route.materialized_tokens[: selected_stage1_route.stage1_depth])
        remaining = self.config.max_new_tokens - len(generated)
        path = path[:remaining]
        if self.config.eos_token_id is not None:
            for idx, token_id in enumerate(path):
                if int(token_id) == self.config.eos_token_id:
                    return path[: idx + 1]
        return path

    def _should_stop(self, committed_now: Sequence[int], generated: Sequence[int]) -> bool:
        if len(generated) >= self.config.max_new_tokens:
            return True
        return self.config.eos_token_id is not None and any(
            int(token_id) == self.config.eos_token_id for token_id in committed_now
        )

    def _next_stage1_after_commit(
        self,
        *,
        committed: Sequence[int],
        selected_stage1_route: RouteState,
        forest_depth: int,
        forest_frontier: Sequence[RouteState],
        forest_completed: Sequence[RouteState] | None,
        forest_last_logits: torch.Tensor | None,
    ) -> BuildResult | None:
        selected_root_id = selected_stage1_route.route_id

        if forest_depth >= self.config.d:
            if forest_completed is None or forest_last_logits is None:
                return None
            indices = [
                idx
                for idx, route in enumerate(forest_completed)
                if route.stage1_root_id == selected_root_id
            ]
            if len(indices) != self.config.k:
                return None
            completed = [forest_completed[idx] for idx in indices]
            promoted = self.route_store.promote_after_stage1_commit(selected_stage1_route, completed)
            return BuildResult(
                completed_routes=promoted,
                next_frontier_routes=[],
                last_logits=forest_last_logits[indices],
                steps=[],
            )

        selected_frontier = [
            route for route in forest_frontier if route.stage1_root_id == selected_root_id
        ]
        if len(selected_frontier) != self.config.k:
            return None
        promoted_frontier = self.route_store.promote_after_stage1_commit(selected_stage1_route, selected_frontier)
        current_depth = promoted_frontier[0].stage1_depth
        remaining_depth = self.config.d - current_depth
        if remaining_depth <= 0:
            return None
        return build_tree(
            promoted_frontier,
            prefix_token_ids=committed,
            depth=remaining_depth,
            k=self.config.k,
            route_store=self.route_store,
            drafter=self.drafter,
        )
