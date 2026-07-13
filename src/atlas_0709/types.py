from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import torch


class DecodePhase(str, Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"


@dataclass(frozen=True)
class PrefixState:
    token_ids: tuple[int, ...]
    next_token_logits: torch.Tensor
    committed_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteState:
    route_id: int
    stage1_root_id: int
    parent_route_id: int | None
    materialized_tokens: tuple[int, ...]
    pending_token_id: int
    cumulative_logprob: float
    stage1_depth: int
    stage2_depth: int

    @property
    def pending_path(self) -> tuple[int, ...]:
        return (*self.materialized_tokens, int(self.pending_token_id))

    @property
    def total_depth(self) -> int:
        return self.stage1_depth + self.stage2_depth


@dataclass(frozen=True)
class PendingCandidate:
    parent_route_id: int
    stage1_root_id: int
    pending_token_id: int
    cumulative_logprob: float
    rank_in_parent: int


@dataclass(frozen=True)
class BatchDecodeOutput:
    route_ids: tuple[int, ...]
    logits: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontierStepResult:
    decoded_routes: list[RouteState]
    candidates: list[PendingCandidate]
    selected_candidates: list[PendingCandidate]
    next_routes: list[RouteState]
    decode_output: BatchDecodeOutput


@dataclass(frozen=True)
class BuildResult:
    completed_routes: list[RouteState]
    next_frontier_routes: list[RouteState]
    last_logits: torch.Tensor
    steps: list[FrontierStepResult]


@dataclass(frozen=True)
class TargetRouteScore:
    route_id: int
    token_ids: tuple[int, ...]
    target_logprob: float
    draft_logprob: float
    token_logprobs: tuple[float, ...] = ()
    first_token_logprob: float | None = None
    selection_score: float | None = None
    score_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class TargetVerifyResult:
    selected_route_id: int | None
    scores: list[TargetRouteScore]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decision: str = "select"
    fallback_token_ids: tuple[int, ...] = ()
    fallback_reason: str | None = None

    @property
    def selected_score(self) -> TargetRouteScore:
        if self.selected_route_id is None:
            raise KeyError("fallback result does not have a selected route")
        for score in self.scores:
            if score.route_id == self.selected_route_id:
                return score
        raise KeyError(self.selected_route_id)


@dataclass(frozen=True)
class RoundTrace:
    round_index: int
    selected_route_id: int
    committed_tokens: tuple[int, ...]
    forest_depth: int
    verify_returned_before_forest_done: bool
    target_scores: tuple[tuple[int, float], ...]


def route_paths(routes: Sequence[RouteState]) -> list[tuple[int, ...]]:
    return [tuple(route.materialized_tokens) for route in routes]
