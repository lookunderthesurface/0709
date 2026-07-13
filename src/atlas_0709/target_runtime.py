from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .flashinfer_full_verify import (
    FullDecodeState,
    FlashInferFullVerifyConfig,
    FullVerifyState,
    build_deduplicated_tree_spec_from_paths,
    build_paged_prefix_kv,
    ceil_div,
    dtype_from_name,
    load_model,
    make_decode_wrappers,
    make_prefill_wrapper,
    next_argmax_inputs,
    plan_prefill_wrapper,
    prefill_cache,
    run_flashinfer_full_ar_decode,
    run_flashinfer_full_verify,
)
from .types import PrefixState, TargetRouteScore, TargetVerifyResult


@dataclass(frozen=True)
class VerifyRoutePayload:
    route_id: int
    token_ids: tuple[int, ...]
    draft_logprob: float


class DirectFlashInferMaskedTreeVerifyBackend:
    """Target-side persistent full-model masked tree verifier.

    The backend is designed to be hosted in its own process. It keeps the
    target model resident, accepts a committed prefix through ``prefill``, and
    verifies the k stage-1 drafter paths with one FlashInfer masked forward.
    Shared path nodes are deduplicated, and the selected path's physical KV is
    compacted into the committed prefix before the verify response is returned.
    """

    def __init__(
        self,
        *,
        model_path: str,
        config: FlashInferFullVerifyConfig,
        score_weights: Sequence[float] | None = None,
        fallback_threshold: float | None = None,
        first_token_threshold: float | None = None,
        fallback_ar_tokens: int = 4,
    ) -> None:
        config.validate()
        self.model_path = model_path
        self.config = config
        self.score_weights = self._validate_score_weights(score_weights)
        self.fallback_threshold = (
            None if fallback_threshold is None else float(fallback_threshold)
        )
        self.first_token_threshold = (
            None if first_token_threshold is None else float(first_token_threshold)
        )
        self.fallback_ar_tokens = int(fallback_ar_tokens)
        if self.fallback_ar_tokens <= 0:
            raise ValueError("fallback_ar_tokens must be positive")
        self.dtype = dtype_from_name(config.dtype)
        self.tokenizer, self.model = load_model(
            model_path,
            dtype=self.dtype,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )
        import flashinfer

        self.flashinfer = flashinfer
        self._wrapper = make_prefill_wrapper(args=self.config, flashinfer=self.flashinfer)
        self.prefix_token_ids: tuple[int, ...] = ()
        self._paged_prefix_kv: list[torch.Tensor] | None = None
        self._prefix_logits: torch.Tensor | None = None
        self._committed_verify_rounds = 0

    @torch.inference_mode()
    def prefill(self, prompt_token_ids: Sequence[int]) -> PrefixState:
        token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        if not token_ids:
            raise ValueError("target prefill requires a non-empty prompt")
        past, prefix_logits = prefill_cache(
            self.model,
            token_ids,
            batch_size=1,
            device=self.config.device,
        )
        paged_prefix_kv = build_paged_prefix_kv(
            past_key_values=past,
            prefix_len=len(token_ids),
            node_count=0,
            page_size=self.config.page_size,
            num_layers=len(self.model.model.layers),
        )
        self.prefix_token_ids = token_ids
        self._paged_prefix_kv = paged_prefix_kv
        self._prefix_logits = prefix_logits.detach()
        self._committed_verify_rounds = 0
        return PrefixState(
            token_ids=token_ids,
            next_token_logits=self._prefix_logits[0],
            committed_length=len(token_ids),
            metadata=self.runtime_metadata(),
        )

    @torch.inference_mode()
    def verify_payloads(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[VerifyRoutePayload],
        fallback_max_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> TargetVerifyResult:
        prefix = tuple(int(token_id) for token_id in prefix_token_ids)
        if self._paged_prefix_kv is None or self._prefix_logits is None:
            self.prefill(prefix)
        elif prefix != self.prefix_token_ids:
            raise RuntimeError(
                "target committed prefix mismatch; incremental verify cannot silently rebuild KV: "
                f"request_len={len(prefix)}, target_len={len(self.prefix_token_ids)}"
            )
        if not routes:
            raise ValueError("verify requires at least one route")

        paths = [tuple(int(token_id) for token_id in route.token_ids) for route in routes]
        path_lengths = {len(path) for path in paths}
        if len(path_lengths) != 1:
            raise ValueError(f"all verify routes must have the same depth, got {sorted(path_lengths)}")
        depth = next(iter(path_lengths))
        if depth <= 0:
            raise ValueError("verify routes must contain at least one token")

        tree, route_to_node_paths = build_deduplicated_tree_spec_from_paths(
            paths=paths,
            prefix_len=len(prefix),
            page_size=self.config.page_size,
            device=self.config.device,
        )
        self._ensure_page_capacity(tree.total_pages)
        assert self._paged_prefix_kv is not None
        state = FullVerifyState(tree=tree, paged_prefix_kv=self._paged_prefix_kv)
        plan_prefill_wrapper(self._wrapper, tree, args=self.config, config=self.model.config, dtype=self.dtype)
        logits = run_flashinfer_full_verify(
            model=self.model,
            wrapper=self._wrapper,
            state=state,
            prefix_len=len(prefix),
            page_size=self.config.page_size,
        )

        scores = self._score_routes(
            routes=routes,
            paths=paths,
            prefix_logits=self._prefix_logits[0],
            verify_logits=logits,
            route_to_node_paths=route_to_node_paths,
        )
        selected = max(
            scores,
            key=lambda item: (
                self._selection_score(item),
                item.draft_logprob,
                -item.route_id,
            ),
        )
        fallback_reason = self._fallback_reason(selected)
        prefix_len_before = len(self.prefix_token_ids)
        common_metadata = {
            **self.runtime_metadata(),
            "route_count": len(routes),
            "path_depth": int(depth),
            "node_count": int(tree.node_count),
            "unmerged_path_node_count": int(len(routes) * depth),
            "kv_len": int(tree.kv_len),
            "prefix_len_before": int(prefix_len_before),
            "selection_policy": self._selection_policy_name(),
            "selection_score_name": (
                "weighted_target_logprob" if self.score_weights is not None else "target_logprob"
            ),
            "fallback_threshold": self.fallback_threshold,
            "first_token_threshold": self.first_token_threshold,
            "fallback_ar_tokens": int(self.fallback_ar_tokens),
        }
        if fallback_reason is not None:
            generated = self._generate_fallback_ar(
                max_tokens=(
                    self.fallback_ar_tokens
                    if fallback_max_tokens is None
                    else min(int(fallback_max_tokens), self.fallback_ar_tokens)
                ),
                eos_token_id=eos_token_id,
            )
            return TargetVerifyResult(
                selected_route_id=None,
                scores=scores,
                decision="fallback_ar",
                fallback_token_ids=tuple(generated),
                fallback_reason=fallback_reason,
                metadata={
                    **common_metadata,
                    "prefix_len_after": int(len(self.prefix_token_ids)),
                    "target_kv_commit": "fallback_ar_in_place",
                    "target_reprefill_after_verify": False,
                    "fallback_triggered": True,
                    "fallback_reason": fallback_reason,
                    "fallback_token_count": int(len(generated)),
                    "best_route_id": int(selected.route_id),
                    "best_selection_score": float(self._selection_score(selected)),
                    "best_first_token_logprob": (
                        None
                        if selected.first_token_logprob is None
                        else float(selected.first_token_logprob)
                    ),
                },
            )
        selected_index = next(
            index for index, route in enumerate(routes) if int(route.route_id) == int(selected.route_id)
        )
        self._commit_selected_path(
            token_ids=paths[selected_index],
            node_ids=route_to_node_paths[selected_index],
            verify_logits=logits,
        )
        return TargetVerifyResult(
            selected_route_id=selected.route_id,
            scores=scores,
            metadata={
                **common_metadata,
                "prefix_len_after": int(len(self.prefix_token_ids)),
                "target_kv_commit": "selected_tree_path_in_place",
                "target_reprefill_after_verify": False,
                "fallback_triggered": False,
                "best_route_id": int(selected.route_id),
                "best_selection_score": float(self._selection_score(selected)),
                "best_first_token_logprob": (
                    None
                    if selected.first_token_logprob is None
                    else float(selected.first_token_logprob)
                ),
            },
        )

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "backend": "direct_flashinfer_full_llama_masked_verify",
            "model": self.model_path,
            "paged_kv": True,
            "custom_mask": True,
            "packed_custom_mask": self.config.use_packed_custom_mask,
            "rope_applied": True,
            "logits_aligned": None,
            "runtime_alignment_checked": False,
            "semantic_correctness_required": True,
            "page_size": self.config.page_size,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "flashinfer_version": getattr(self.flashinfer, "__version__", "unknown"),
            "persistent_target_paged_kv": True,
            "deduplicated_tree_nodes": True,
            "selection_policy": self._selection_policy_name(),
            "score_weights": (
                None if self.score_weights is None else [float(value) for value in self.score_weights]
            ),
            "fallback_threshold": self.fallback_threshold,
            "first_token_threshold": self.first_token_threshold,
            "fallback_ar_tokens": int(self.fallback_ar_tokens),
            "committed_verify_rounds": self._committed_verify_rounds,
            "committed_prefix_length": len(self.prefix_token_ids),
        }

    @staticmethod
    def _validate_score_weights(weights: Sequence[float] | None) -> tuple[float, ...] | None:
        if weights is None:
            return None
        values = tuple(float(value) for value in weights)
        if not values:
            return None
        if any(value < 0.0 for value in values):
            raise ValueError("score weights must be non-negative")
        total = sum(values)
        if total <= 0.0:
            raise ValueError("at least one score weight must be positive")
        return tuple(value / total for value in values)

    def _weights_for_depth(self, depth: int) -> tuple[float, ...] | None:
        if self.score_weights is None:
            return None
        if len(self.score_weights) != int(depth):
            raise ValueError(
                "score weights must match verify route depth: "
                f"weights={len(self.score_weights)}, depth={int(depth)}"
            )
        return self.score_weights

    def _selection_policy_name(self) -> str:
        if self.score_weights is None:
            return "best_of_n_target_path_logprob"
        return "best_of_n_weighted_target_path_logprob"

    @staticmethod
    def _selection_score(score: TargetRouteScore) -> float:
        if score.selection_score is None:
            return float(score.target_logprob)
        return float(score.selection_score)

    def _fallback_reason(self, selected: TargetRouteScore) -> str | None:
        if (
            self.first_token_threshold is not None
            and selected.first_token_logprob is not None
            and float(selected.first_token_logprob) < float(self.first_token_threshold)
        ):
            return "first_token_below_threshold"
        if (
            self.fallback_threshold is not None
            and self._selection_score(selected) < float(self.fallback_threshold)
        ):
            return "path_score_below_threshold"
        return None

    def _score_routes(
        self,
        *,
        routes: Sequence[VerifyRoutePayload],
        paths: Sequence[Sequence[int]],
        prefix_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        route_to_node_paths: Sequence[Sequence[int]],
    ) -> list[TargetRouteScore]:
        scores: list[TargetRouteScore] = []
        prefix_logprobs = torch.log_softmax(prefix_logits, dim=-1)
        verify_logprobs = torch.log_softmax(verify_logits, dim=-1)
        for route, path, route_node_ids in zip(routes, paths, route_to_node_paths):
            token_logprobs: list[float] = []
            for offset, token_id in enumerate(path):
                if offset == 0:
                    token_logprob = prefix_logprobs[int(token_id)]
                else:
                    prev_node_row = int(route_node_ids[offset - 1])
                    token_logprob = verify_logprobs[prev_node_row, int(token_id)]
                token_logprobs.append(float(token_logprob.detach().cpu()))
            target_logprob = sum(token_logprobs)
            weights = self._weights_for_depth(len(token_logprobs))
            selection_score = (
                target_logprob
                if weights is None
                else sum(weight * value for weight, value in zip(weights, token_logprobs))
            )
            scores.append(
                TargetRouteScore(
                    route_id=int(route.route_id),
                    token_ids=tuple(int(token_id) for token_id in path),
                    target_logprob=target_logprob,
                    draft_logprob=float(route.draft_logprob),
                    token_logprobs=tuple(token_logprobs),
                    first_token_logprob=token_logprobs[0],
                    selection_score=selection_score,
                    score_weights=() if weights is None else tuple(weights),
                )
            )
        return scores

    def _generate_fallback_ar(
        self,
        *,
        max_tokens: int,
        eos_token_id: int | None,
    ) -> list[int]:
        if self._paged_prefix_kv is None or self._prefix_logits is None:
            raise RuntimeError("target fallback AR requires initialized prefix KV")
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            return []
        prefix_len = len(self.prefix_token_ids)
        required_pages = ceil_div(prefix_len + max_tokens, int(self.config.page_size))
        self._ensure_page_capacity(required_pages)
        generated: list[int] = []
        input_ids = next_argmax_inputs(self._prefix_logits)
        for step_idx in range(max_tokens):
            token_id = int(input_ids.reshape(-1)[0].detach().cpu())
            generated.append(token_id)
            wrappers = make_decode_wrappers(
                args=self.config,
                config=self.model.config,
                dtype=self.dtype,
                flashinfer=self.flashinfer,
                prefix_len=prefix_len + step_idx,
                steps=1,
            )
            state = FullDecodeState(input_ids=input_ids, paged_prefix_kv=self._paged_prefix_kv)
            logits = run_flashinfer_full_ar_decode(
                model=self.model,
                wrappers=wrappers,
                state=state,
                prefix_len=prefix_len + step_idx,
                page_size=self.config.page_size,
            )
            self._prefix_logits = logits.detach()
            if eos_token_id is not None and int(token_id) == int(eos_token_id):
                break
            input_ids = next_argmax_inputs(logits)
        self.prefix_token_ids = (
            *self.prefix_token_ids,
            *(int(token_id) for token_id in generated),
        )
        self._committed_verify_rounds += 1
        return generated

    def _ensure_page_capacity(self, required_pages: int) -> None:
        if self._paged_prefix_kv is None:
            raise RuntimeError("target prefill must initialize paged prefix KV")
        required_pages = int(required_pages)
        grown: list[torch.Tensor] = []
        for layer_kv in self._paged_prefix_kv:
            if int(layer_kv.shape[0]) >= required_pages:
                grown.append(layer_kv)
                continue
            new_shape = (required_pages, *layer_kv.shape[1:])
            expanded = torch.zeros(new_shape, dtype=layer_kv.dtype, device=layer_kv.device)
            expanded[: int(layer_kv.shape[0])].copy_(layer_kv)
            grown.append(expanded)
        self._paged_prefix_kv = grown

    def _commit_selected_path(
        self,
        *,
        token_ids: Sequence[int],
        node_ids: Sequence[int],
        verify_logits: torch.Tensor,
    ) -> None:
        if self._paged_prefix_kv is None:
            raise RuntimeError("target paged prefix KV is not initialized")
        if len(token_ids) != len(node_ids) or not node_ids:
            raise ValueError("selected path tokens and node ids must have equal non-zero length")

        prefix_len = len(self.prefix_token_ids)
        device = self._paged_prefix_kv[0].device
        source_positions = torch.tensor(
            [prefix_len + int(node_id) for node_id in node_ids],
            dtype=torch.long,
            device=device,
        )
        destination_positions = torch.arange(
            prefix_len,
            prefix_len + len(node_ids),
            dtype=torch.long,
            device=device,
        )
        source_pages = torch.div(
            source_positions,
            int(self.config.page_size),
            rounding_mode="floor",
        )
        source_offsets = torch.remainder(source_positions, int(self.config.page_size))
        destination_pages = torch.div(
            destination_positions,
            int(self.config.page_size),
            rounding_mode="floor",
        )
        destination_offsets = torch.remainder(destination_positions, int(self.config.page_size))
        for layer_kv in self._paged_prefix_kv:
            selected_kv = layer_kv[source_pages, :, source_offsets].clone()
            layer_kv[destination_pages, :, destination_offsets] = selected_kv

        self.prefix_token_ids = (
            *self.prefix_token_ids,
            *(int(token_id) for token_id in token_ids),
        )
        last_node_id = int(node_ids[-1])
        self._prefix_logits = verify_logits[last_node_id].detach().unsqueeze(0)
        self._committed_verify_rounds += 1

def verify_payload_from_mapping(item: dict[str, object]) -> VerifyRoutePayload:
    return VerifyRoutePayload(
        route_id=int(item["route_id"]),
        token_ids=tuple(int(token_id) for token_id in item["token_ids"]),
        draft_logprob=float(item.get("draft_logprob", 0.0)),
    )
