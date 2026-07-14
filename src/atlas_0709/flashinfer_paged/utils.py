from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch

from .types import PendingCandidate, PendingCandidateBatch


def topk_logprobs(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k <= 0:
        raise ValueError("k must be positive")
    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("topk_logprobs expects a single row or a 1D tensor")
        logits = logits[0]
    if logits.ndim != 1:
        raise ValueError(f"expected 1D logits, got shape {tuple(logits.shape)}")

    actual_k = min(k, logits.shape[-1])
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs, token_ids = torch.topk(logprobs, k=actual_k, dim=-1)
    return token_logprobs, token_ids


def global_topk_candidates(
    candidates: Sequence[PendingCandidate],
    num_selected: int,
) -> list[PendingCandidate]:
    if num_selected <= 0:
        raise ValueError("num_selected must be positive")
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.cumulative_logprob,
            -candidate.parent_route_id,
            -candidate.rank_in_parent,
            -candidate.pending_token_id,
        ),
        reverse=True,
    )
    return list(ordered[:num_selected])


def stable_candidate_order(
    candidates: PendingCandidateBatch,
    *,
    include_stage1_root: bool = False,
) -> torch.Tensor:
    """Return the existing candidate ordering without leaving the device.

    Python selection orders by score descending, then parent route, parent
    rank, and token id ascending.  Stable sorts are applied from the least to
    the most significant key so exact ties remain deterministic.  Forest
    selection adds the stage-1 root as the primary ascending key.
    """

    order = torch.arange(
        candidates.candidate_count,
        device=candidates.pending_token_ids.device,
        dtype=torch.long,
    )
    key_specs: list[tuple[torch.Tensor, bool]] = [
        (candidates.pending_token_ids, False),
        (candidates.ranks_in_parent, False),
        (candidates.parent_route_ids, False),
        (candidates.cumulative_logprobs, True),
    ]
    if include_stage1_root:
        key_specs.append((candidates.stage1_root_ids, False))

    for values, descending in key_specs:
        permutation = torch.argsort(
            values.index_select(0, order),
            dim=0,
            descending=descending,
            stable=True,
        )
        order = order.index_select(0, permutation)
    return order


def global_topk_candidate_indices(
    candidates: PendingCandidateBatch,
    *,
    num_selected: int,
) -> torch.Tensor:
    if num_selected <= 0:
        raise ValueError("num_selected must be positive")
    order = stable_candidate_order(candidates)
    return order[: min(int(num_selected), candidates.candidate_count)]


def per_root_topk_candidate_indices(
    candidates: PendingCandidateBatch,
    *,
    num_selected: int,
    root_candidate_counts: Sequence[tuple[int, int]],
) -> torch.Tensor:
    if num_selected <= 0:
        raise ValueError("num_selected must be positive")

    ordered_counts = sorted(
        (int(root_id), int(candidate_count))
        for root_id, candidate_count in root_candidate_counts
    )
    if (
        sum(candidate_count for _, candidate_count in ordered_counts)
        != candidates.candidate_count
    ):
        raise ValueError("root candidate counts do not cover the candidate batch")

    order = stable_candidate_order(candidates, include_stage1_root=True)
    selected_parts: list[torch.Tensor] = []
    offset = 0
    for _, candidate_count in ordered_counts:
        take = min(int(num_selected), candidate_count)
        selected_parts.append(order[offset : offset + take])
        offset += candidate_count
    if not selected_parts:
        return order[:0]
    return torch.cat(selected_parts, dim=0)


def group_candidates_by_stage1_root(
    candidates: Iterable[PendingCandidate],
) -> dict[int, list[PendingCandidate]]:
    grouped: dict[int, list[PendingCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.stage1_root_id].append(candidate)
    return dict(grouped)


def topk_pairs_to_python(
    token_ids: torch.Tensor,
    token_logprobs: torch.Tensor,
) -> list[tuple[int, float]]:
    """Materialize a top-k result with one host batch instead of scalar reads."""

    if token_ids.shape != token_logprobs.shape:
        raise ValueError("token ids and logprobs must have matching shapes")
    packed = torch.stack(
        (
            token_ids.to(dtype=torch.float64),
            token_logprobs.to(dtype=torch.float64),
        ),
        dim=-1,
    )
    rows = packed.detach().cpu().reshape(-1, 2).tolist()
    return [(int(token_id), float(token_logprob)) for token_id, token_logprob in rows]


def logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    logprobs = torch.log_softmax(logits, dim=-1)
    return float(logprobs[int(token_id)].detach().cpu())
