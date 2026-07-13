from __future__ import annotations

from typing import Protocol, Sequence

import torch

from .types import BatchDecodeOutput, PrefixState, RouteState, TargetRouteScore, TargetVerifyResult
from .utils import topk_logprobs


class BatchDecodeBackend(Protocol):
    def prefill(self, prompt_token_ids: Sequence[int]) -> PrefixState:
        ...

    def decode_batch(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> BatchDecodeOutput:
        ...


class BatchVerifyBackend(Protocol):
    def prefill(self, prompt_token_ids: Sequence[int]) -> PrefixState:
        ...

    def verify_batch(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> TargetVerifyResult:
        ...


class DeterministicMockBackend:
    """Deterministic backend for controller smoke tests.

    The logits depend on the full logical token context, so tree/forest pruning
    and target verification are stable across runs without loading a real model.
    """

    def __init__(self, *, name: str, vocab_size: int = 128, salt: int = 0, device: str = "cpu") -> None:
        self.name = name
        self.vocab_size = int(vocab_size)
        self.salt = int(salt)
        self.device = torch.device(device)

    def prefill(self, prompt_token_ids: Sequence[int]) -> PrefixState:
        token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        logits = self._logits_for_context(token_ids)
        return PrefixState(
            token_ids=token_ids,
            next_token_logits=logits,
            committed_length=len(token_ids),
            metadata={"backend": "deterministic_mock", "name": self.name},
        )

    def decode_batch(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> BatchDecodeOutput:
        rows = [
            self._logits_for_context((*tuple(int(x) for x in prefix_token_ids), *route.pending_path))
            for route in routes
        ]
        logits = torch.stack(rows, dim=0)
        return BatchDecodeOutput(
            route_ids=tuple(route.route_id for route in routes),
            logits=logits,
            metadata={
                "backend": "deterministic_mock",
                "batch_size": len(routes),
                "ordinary_batch": True,
            },
        )

    def verify_batch(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> TargetVerifyResult:
        prefix = tuple(int(token_id) for token_id in prefix_token_ids)
        scores: list[TargetRouteScore] = []
        for route in routes:
            score = self._score_path(prefix, route.materialized_tokens[: route.stage1_depth])
            scores.append(
                TargetRouteScore(
                    route_id=route.route_id,
                    token_ids=route.materialized_tokens[: route.stage1_depth],
                    target_logprob=score,
                    draft_logprob=route.cumulative_logprob,
                )
            )
        selected = max(scores, key=lambda item: (item.target_logprob, item.draft_logprob, -item.route_id))
        return TargetVerifyResult(
            selected_route_id=selected.route_id,
            scores=scores,
            metadata={"backend": "deterministic_mock", "ordinary_batch_verify": True},
        )

    def _score_path(self, prefix: tuple[int, ...], path: tuple[int, ...]) -> float:
        context = prefix
        total = 0.0
        for token_id in path:
            logprobs, token_ids = topk_logprobs(self._logits_for_context(context), k=self.vocab_size)
            matches = (token_ids == int(token_id)).nonzero(as_tuple=False)
            if matches.numel() == 0:
                total += -100.0
            else:
                total += float(logprobs[int(matches[0, 0])].detach().cpu())
            context = (*context, int(token_id))
        return total

    def _logits_for_context(self, token_ids: Sequence[int]) -> torch.Tensor:
        total = sum((idx + 1) * int(token_id) for idx, token_id in enumerate(token_ids))
        center = (total + self.salt + 17 * len(token_ids)) % self.vocab_size
        ids = torch.arange(self.vocab_size, dtype=torch.float32, device=self.device)
        logits = -0.03 * torch.remainder(torch.abs(ids - center), self.vocab_size)
        for rank in range(min(8, self.vocab_size)):
            logits[(center + rank * 7 + self.salt) % self.vocab_size] += 2.0 - rank * 0.1
        return logits

