from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch

from .types import PendingCandidate


def topk_logprobs(logits: torch.Tensor, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 1:
        raise ValueError(f"topk_logprobs expects a 1D tensor, got {tuple(logits.shape)}")
    actual_k = min(int(k), int(logits.shape[-1]))
    return torch.topk(torch.log_softmax(logits, dim=-1), k=actual_k, dim=-1)


def batch_topk_logprobs(logits: torch.Tensor, *, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"batch_topk_logprobs expects a 2D tensor, got {tuple(logits.shape)}")
    actual_k = min(int(k), int(logits.shape[-1]))
    return torch.topk(torch.log_softmax(logits, dim=-1), k=actual_k, dim=-1)


def select_global_topk(candidates: Sequence[PendingCandidate], *, k: int) -> list[PendingCandidate]:
    return sorted(
        candidates,
        key=lambda item: (item.cumulative_logprob, -item.parent_route_id, -item.rank_in_parent),
        reverse=True,
    )[: int(k)]


def group_by_stage1_root(candidates: Iterable[PendingCandidate]) -> dict[int, list[PendingCandidate]]:
    grouped: dict[int, list[PendingCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[int(candidate.stage1_root_id)].append(candidate)
    return dict(grouped)


def select_topk_per_stage1_root(
    candidates: Sequence[PendingCandidate],
    *,
    active_root_ids: Sequence[int],
    k: int,
) -> list[PendingCandidate]:
    grouped = group_by_stage1_root(candidates)
    selected: list[PendingCandidate] = []
    for root_id in sorted({int(root_id) for root_id in active_root_ids}):
        root_candidates = grouped.get(root_id, [])
        if not root_candidates:
            raise RuntimeError(f"stage1 root {root_id} produced no candidates")
        selected.extend(select_global_topk(root_candidates, k=k))
    return selected

