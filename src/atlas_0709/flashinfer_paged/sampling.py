from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Sequence

import torch


_SAMPLING_SEED_DOMAIN = b"atlas-0709-drafter-semantic-prefix-v1\0"
_TORCH_SEED_MASK = (1 << 63) - 1


@dataclass(frozen=True)
class DrafterSamplingConfig:
    """Request-level Drafter candidate sampling configuration.

    Sampling draws ``k`` distinct children from each parent with the
    Gumbel-top-k construction.  The recorded candidate scores remain the
    Drafter model's untempered log probabilities; ``temperature`` changes only
    the proposal distribution used to obtain a diverse candidate set.
    """

    do_sample: bool = False
    seed: int = 0
    temperature: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(float(self.temperature)) or float(self.temperature) <= 0.0:
            raise ValueError("drafter sampling temperature must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "do_sample": bool(self.do_sample),
            "seed": int(self.seed),
            "temperature": float(self.temperature),
            "candidate_sampling": (
                "semantic_prefix_gumbel_topk_without_replacement"
                if self.do_sample
                else "model_logprob_topk"
            ),
            "rng_key": (
                "request_seed_and_complete_semantic_parent_token_prefix"
                if self.do_sample
                else None
            ),
        }


@dataclass(frozen=True)
class DrafterSamplingContext:
    """Bind request sampling to the currently committed semantic prefix.

    A child's RNG key hashes ``committed_token_ids + relative_parent_path``.
    Consequently, speculative forest work and post-commit tree work use the
    same random draw whenever they represent the same absolute token history,
    even if scheduler timing changes how much forest work was completed.
    """

    config: DrafterSamplingConfig
    committed_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        self.config.validate()
        object.__setattr__(
            self,
            "committed_token_ids",
            tuple(int(token_id) for token_id in self.committed_token_ids),
        )

    def seed_for_parent(self, relative_parent_path: Sequence[int]) -> int:
        absolute_path = (
            *self.committed_token_ids,
            *(int(token_id) for token_id in relative_parent_path),
        )
        digest = hashlib.blake2b(digest_size=8)
        digest.update(_SAMPLING_SEED_DOMAIN)
        digest.update(struct.pack("<Q", int(self.config.seed) & ((1 << 64) - 1)))
        digest.update(struct.pack("<Q", len(absolute_path)))
        for token_id in absolute_path:
            if token_id < 0 or token_id > _TORCH_SEED_MASK:
                raise ValueError(f"token id is outside the supported int63 range: {token_id}")
            digest.update(struct.pack("<Q", token_id))
        return int.from_bytes(digest.digest(), "little") & _TORCH_SEED_MASK


def single_parent_token_candidates(
    logits: torch.Tensor,
    *,
    k: int,
    sampling: DrafterSamplingContext | None,
    relative_parent_path: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _normalize_logits(logits)
    if int(rows.shape[0]) != 1:
        raise ValueError("single-parent candidate selection expects exactly one logits row")
    token_logprobs, token_ids = batch_parent_token_candidates(
        rows,
        k=k,
        sampling=sampling,
        relative_parent_paths=(tuple(relative_parent_path),),
    )
    return token_logprobs[0], token_ids[0]


def batch_parent_token_candidates(
    logits: torch.Tensor,
    *,
    k: int,
    sampling: DrafterSamplingContext | None,
    relative_parent_paths: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one distinct candidate set per semantic parent.

    This function intentionally creates a fresh generator for every parent.
    A shared process-global generator would make results depend on frontier
    batch order and on how many speculative forest rows happened to execute.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    rows = _normalize_logits(logits)
    batch_size, vocab_size = (int(rows.shape[0]), int(rows.shape[1]))
    if len(relative_parent_paths) != batch_size:
        raise ValueError(
            "relative_parent_paths must contain one semantic path per logits row"
        )
    actual_k = min(int(k), vocab_size)
    model_logprobs = torch.log_softmax(rows, dim=-1)

    if sampling is None or not sampling.config.do_sample:
        return torch.topk(model_logprobs, k=actual_k, dim=-1)

    sampled_ids: list[torch.Tensor] = []
    temperature = float(sampling.config.temperature)
    for row_index, relative_path in enumerate(relative_parent_paths):
        generator = torch.Generator(device=rows.device)
        generator.manual_seed(sampling.seed_for_parent(relative_path))
        uniforms = torch.rand(
            (vocab_size,),
            device=rows.device,
            dtype=torch.float32,
            generator=generator,
        )
        uniforms = uniforms.clamp_min(torch.finfo(torch.float32).tiny)
        gumbels = -torch.log(-torch.log(uniforms))
        proposal_scores = rows[row_index].to(torch.float32) / temperature + gumbels
        sampled_ids.append(
            torch.topk(proposal_scores, k=actual_k, dim=-1).indices
        )

    token_ids = torch.stack(sampled_ids, dim=0)
    token_logprobs = model_logprobs.gather(dim=-1, index=token_ids)
    return token_logprobs, token_ids


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.unsqueeze(0)
    if logits.ndim != 2:
        raise ValueError(f"expected 1D or 2D logits, got shape {tuple(logits.shape)}")
    return logits
