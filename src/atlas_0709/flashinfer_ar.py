from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .flashinfer_paged.builders import build_tree_one_depth, initialize_stage1_routes
from .flashinfer_paged.kv import KVTreeStore
from .flashinfer_paged.sampling import DrafterSamplingConfig, DrafterSamplingContext
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
    """Batch-1 greedy or reproducibly sampled AR over FlashInfer paged KV."""

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
        do_sample: bool = False,
        generation_seed: int = 0,
        temperature: float = 1.0,
    ) -> FlashInferARResult:
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        sampling_config = DrafterSamplingConfig(
            do_sample=bool(do_sample),
            seed=int(generation_seed),
            temperature=float(temperature),
        )
        sampling = DrafterSamplingContext(
            config=sampling_config,
            committed_token_ids=tuple(int(token_id) for token_id in prompt_token_ids),
        )

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
        routes = initialize_stage1_routes(
            prefix,
            k=1,
            route_store=store,
            sampling=sampling,
        )
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
                sampling=sampling,
            )
            routes = step.next_routes

        return FlashInferARResult(
            generated_token_ids=generated,
            finish_reason=finish_reason,
            metadata={
                "backend": (
                    "sglang_flashinfer_paged_sampled_ar"
                    if do_sample
                    else "sglang_flashinfer_paged_greedy_ar"
                ),
                "batch_size": 1,
                "paged_kv": True,
                "page_size": self.page_size,
                **sampling_config.to_dict(),
                "generation_seed": int(generation_seed),
                "runner": sglang_runner_component_report(self.runner),
            },
        )
