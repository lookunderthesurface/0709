from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch

from .types import PendingCandidate


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


def group_candidates_by_stage1_root(
    candidates: Iterable[PendingCandidate],
) -> dict[int, list[PendingCandidate]]:
    grouped: dict[int, list[PendingCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.stage1_root_id].append(candidate)
    return dict(grouped)


def tensor_to_int_list(values: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    logprobs = torch.log_softmax(logits, dim=-1)
    return float(logprobs[int(token_id)].detach().cpu())
