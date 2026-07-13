from __future__ import annotations

from typing import Sequence

import torch

from .backends import BatchDecodeBackend, BatchVerifyBackend
from .types import BatchDecodeOutput, PrefixState, RouteState, TargetRouteScore, TargetVerifyResult


class HFRecomputeBatchBackend(BatchDecodeBackend, BatchVerifyBackend):
    """Clean ordinary-batch Hugging Face backend.

    This backend deliberately recomputes padded sequences for every batch call.
    It is useful for correctness smoke tests with real models, not for speed.
    """

    def __init__(
        self,
        *,
        model_path: str,
        dtype: str = "float16",
        device: str = "cuda",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = _dtype_from_name(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=None,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def prefill(self, prompt_token_ids: Sequence[int]) -> PrefixState:
        token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=input_ids, use_cache=False)
        logits = output.logits[0, -1, :].detach()
        return PrefixState(
            token_ids=token_ids,
            next_token_logits=logits,
            committed_length=len(token_ids),
            metadata={"backend": "hf_recompute_batch", "model_path": self.model_path},
        )

    @torch.inference_mode()
    def decode_batch(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> BatchDecodeOutput:
        sequences = [
            (*tuple(int(token_id) for token_id in prefix_token_ids), *route.pending_path)
            for route in routes
        ]
        input_ids, attention_mask, last_indices = self._pad_sequences(sequences)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits[torch.arange(len(sequences), device=self.device), last_indices, :].detach()
        return BatchDecodeOutput(
            route_ids=tuple(route.route_id for route in routes),
            logits=logits,
            metadata={
                "backend": "hf_recompute_batch",
                "ordinary_batch": True,
                "batch_size": len(routes),
            },
        )

    @torch.inference_mode()
    def verify_batch(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> TargetVerifyResult:
        prefix = tuple(int(token_id) for token_id in prefix_token_ids)
        paths = [route.materialized_tokens[: route.stage1_depth] for route in routes]
        sequences = [(*prefix, *path) for path in paths]
        input_ids, attention_mask, _ = self._pad_sequences(sequences)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits

        prefix_len = len(prefix)
        scores: list[TargetRouteScore] = []
        for row, (route, path) in enumerate(zip(routes, paths)):
            score = 0.0
            for offset, token_id in enumerate(path):
                logit_pos = prefix_len + offset - 1
                if logit_pos < 0:
                    raise RuntimeError("HF verify requires a non-empty prompt")
                logprobs = torch.log_softmax(logits[row, logit_pos, :], dim=-1)
                score += float(logprobs[int(token_id)].detach().cpu())
            scores.append(
                TargetRouteScore(
                    route_id=route.route_id,
                    token_ids=tuple(int(token_id) for token_id in path),
                    target_logprob=score,
                    draft_logprob=route.cumulative_logprob,
                )
            )

        selected = max(scores, key=lambda item: (item.target_logprob, item.draft_logprob, -item.route_id))
        return TargetVerifyResult(
            selected_route_id=selected.route_id,
            scores=scores,
            metadata={
                "backend": "hf_recompute_batch",
                "ordinary_batch_verify": True,
                "batch_size": len(routes),
            },
        )

    def encode_prompt(self, prompt: str) -> list[int]:
        return [int(token_id) for token_id in self.tokenizer.encode(prompt, add_special_tokens=False)]

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(int(token_id) for token_id in token_ids), skip_special_tokens=False)

    def _pad_sequences(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("sequences cannot be empty")
        max_len = max(len(seq) for seq in sequences)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=self.device)
        last_indices = []
        for row, seq in enumerate(sequences):
            values = torch.tensor([int(token_id) for token_id in seq], dtype=torch.long, device=self.device)
            input_ids[row, : values.numel()] = values
            attention_mask[row, : values.numel()] = 1
            last_indices.append(values.numel() - 1)
        return input_ids, attention_mask, torch.tensor(last_indices, dtype=torch.long, device=self.device)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")

