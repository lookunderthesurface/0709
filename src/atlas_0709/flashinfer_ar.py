from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .flashinfer_paged.builders import build_tree_one_depth, initialize_stage1_routes
from .flashinfer_paged.kv import KVTreeStore
from .flashinfer_paged.sglang_runtime import (
    SGLangFlashInferFrontierModelBackend,
    prefill_sglang_prefix,
    sglang_runner_component_report,
)


@dataclass(frozen=True)
class FlashInferARResult:
    generated_token_ids: list[int]
    finish_reason: str
    metadata: dict[str, object]


class FlashInferPagedGreedyARGenerator:
    """Strict batch-1 greedy AR over SGLang's FlashInfer paged KV runtime."""

    def __init__(
        self,
        *,
        runner,
        page_size: int,
        prefill_chunk_size: int,
    ) -> None:
        self.runner = runner
        self.page_size = int(page_size)
        self.prefill_chunk_size = int(prefill_chunk_size)

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> FlashInferARResult:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        store = KVTreeStore()
        backend = SGLangFlashInferFrontierModelBackend.from_runner(
            route_store=store,
            model_runner=self.runner,
            page_size=self.page_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        backend.route_pool.clear_physical_state()
        prefill = prefill_sglang_prefix(
            model_runner=self.runner,
            route_pool=backend.route_pool,
            prompt_token_ids=prompt_token_ids,
            chunk_size=self.prefill_chunk_size,
        )
        prefix = backend.attach_prefilled_prefix(
            prompt_token_ids=prefill.prompt_token_ids,
            prefix_slot_ids=prefill.prefix_slot_ids,
            next_token_logits=prefill.next_token_logits,
        )
        routes = initialize_stage1_routes(prefix, k=1, route_store=store)
        generated: list[int] = []
        finish_reason = "length"

        while len(generated) < int(max_new_tokens):
            token_id = int(routes[0].pending_token_id)
            generated.append(token_id)
            if eos_token_id is not None and token_id == int(eos_token_id):
                finish_reason = "eos"
                break
            if len(generated) >= int(max_new_tokens):
                break
            step = build_tree_one_depth(
                routes,
                k=1,
                route_store=store,
                model_backend=backend,
            )
            routes = step.next_routes

        return FlashInferARResult(
            generated_token_ids=generated,
            finish_reason=finish_reason,
            metadata={
                "backend": "sglang_flashinfer_paged_greedy_ar",
                "batch_size": 1,
                "paged_kv": True,
                "page_size": self.page_size,
                "do_sample": False,
                "runner": sglang_runner_component_report(self.runner),
            },
        )
