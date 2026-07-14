from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import torch


class DecodePhase(str, Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"


@dataclass(frozen=True)
class PrefixKVView:
    committed_length: int
    page_ids: tuple[int, ...] = ()

    def fork(self) -> "RouteKVView":
        return RouteKVView(prefix=self, node_ids=())


@dataclass(frozen=True)
class RouteKVView:
    prefix: PrefixKVView
    node_ids: tuple[int, ...] = ()

    def fork(self) -> "RouteKVView":
        return RouteKVView(prefix=self.prefix, node_ids=tuple(self.node_ids))

    def append_node(self, node_id: int) -> "RouteKVView":
        return RouteKVView(
            prefix=self.prefix,
            node_ids=(*self.node_ids, int(node_id)),
        )

    def drop_committed_prefix_nodes(
        self,
        committed_node_ids: Sequence[int],
        new_prefix: PrefixKVView,
    ) -> "RouteKVView":
        committed = tuple(committed_node_ids)
        if committed and self.node_ids[: len(committed)] != committed:
            raise ValueError("route does not start with the committed node path")
        return RouteKVView(
            prefix=new_prefix,
            node_ids=tuple(self.node_ids[len(committed) :]),
        )


@dataclass
class RouteState:
    route_id: int
    stage1_root_id: int
    parent_route_id: int | None
    materialized_leaf_node_id: int | None
    pending_token_id: int
    cumulative_logprob: float
    stage1_depth: int
    stage2_depth: int
    kv_view: RouteKVView
    # Per-token Drafter model log probabilities for this route relative to
    # the current committed prefix.  A frontier route includes its pending
    # token; a completed/materialized route contains exactly its materialized
    # path.  The tuple is split/pruned together with the token path.
    token_logprobs: tuple[float, ...] = ()

    def total_materialized_depth(self) -> int:
        return self.stage1_depth + self.stage2_depth


@dataclass(frozen=True)
class PendingCandidate:
    parent_route_id: int
    stage1_root_id: int
    pending_token_id: int
    cumulative_logprob: float
    rank_in_parent: int = 0
    parent_logprob: float = 0.0


@dataclass(frozen=True)
class PendingCandidateBatch:
    """Device-resident candidate descriptors for one frontier depth."""

    parent_route_ids: torch.Tensor
    stage1_root_ids: torch.Tensor
    parent_row_indices: torch.Tensor
    pending_token_ids: torch.Tensor
    cumulative_logprobs: torch.Tensor
    parent_logprobs: torch.Tensor
    ranks_in_parent: torch.Tensor
    candidates_per_parent: int

    @property
    def candidate_count(self) -> int:
        return int(self.pending_token_ids.numel())


@dataclass(frozen=True)
class FrontierSelectionStats:
    candidate_count: int = 0
    selected_count: int = 0
    host_materialization_batches: int = 0
    host_materialization_elements: int = 0
    host_transfer_batches: int = 0
    host_transfer_elements: int = 0
    selection_device: str = "cpu"
    new_node_ids_source: str = "unknown"


@dataclass(frozen=True)
class DraftPrefixState:
    token_ids: torch.Tensor
    prefix_kv_view: PrefixKVView
    next_token_logits: torch.Tensor
    committed_length: int


@dataclass(frozen=True)
class FrontierDecodeOutput:
    route_ids: torch.Tensor
    next_token_logits: torch.Tensor
    new_node_ids: torch.Tensor | Sequence[int]
    new_node_ids_cpu: tuple[int, ...] | None = None
    attention_ms: float = 0.0
    model_ms: float = 0.0
    kv_append_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontierStepOutput:
    decoded_routes: list[RouteState]
    next_routes: list[RouteState]
    candidates: list[PendingCandidate]
    selected_candidates: list[PendingCandidate]
    decode_output: FrontierDecodeOutput
    selection_stats: FrontierSelectionStats = field(
        default_factory=FrontierSelectionStats
    )


@dataclass(frozen=True)
class BuildDepthsOutput:
    completed_routes: list[RouteState]
    next_frontier_routes: list[RouteState]
    last_logits: torch.Tensor
    steps: list[FrontierStepOutput]
