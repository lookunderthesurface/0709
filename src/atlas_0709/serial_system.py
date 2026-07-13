from __future__ import annotations

import time
from typing import Sequence

import torch

from .distributed_system import (
    DistributedGenerationResult,
    DistributedRoundTrace,
    PagedDistributedAtlasGenerator,
)
from .flashinfer_paged.builders import build_tree_depths, initialize_stage1_routes
from .flashinfer_paged.types import DecodePhase, DraftPrefixState, PrefixKVView, RouteState
from .rpc import selected_route_id_from_response


class SerialAtlasGenerator(PagedDistributedAtlasGenerator):
    """Quality-matched serial baseline: build tree, then Target verify.

    This intentionally omits forest construction, asynchronous overlap, and
    stage-2 KV handoff. It keeps the same Drafter, Target, route scoring, and
    fallback behavior as the distributed generator so benchmark quality comparisons
    isolate scheduling overhead on a shared GPU.
    """

    def generate(self, prompt_token_ids: Sequence[int]) -> DistributedGenerationResult:
        committed = [int(token_id) for token_id in prompt_token_ids]
        generated: list[int] = []
        traces: list[DistributedRoundTrace] = []
        target_health = self.target_client.health()

        prefill_start = time.perf_counter()
        self.target_client.prefill(committed)
        target_prefill_elapsed_s = time.perf_counter() - prefill_start

        drafter_prefill_start = time.perf_counter()
        ctx = self._prepare_prefix(committed)
        drafter_prefill_elapsed_s = time.perf_counter() - drafter_prefill_start
        prefix = ctx.prefix

        round_index = 0
        drafter_elapsed_s = 0.0
        target_elapsed_s = 0.0
        total_start = time.perf_counter()
        while len(generated) < self.config.max_new_tokens:
            drafter_start = time.perf_counter()
            active_routes = initialize_stage1_routes(
                prefix,
                k=self.config.k,
                route_store=ctx.store,
            )
            stage1 = build_tree_depths(
                active_routes,
                depth=self.config.d,
                k=self.config.k,
                route_store=ctx.store,
                model_backend=ctx.backend,
            )
            # SGLang model calls enqueue CUDA work. Synchronize at the phase
            # boundary so pending Drafter kernels are not charged to the
            # following Target RPC on a shared GPU.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            drafter_elapsed_s += time.perf_counter() - drafter_start

            remaining_tokens = self.config.max_new_tokens - len(generated)
            verify_start = time.perf_counter()
            verify_response = self.target_client.verify(
                prefix_token_ids=committed,
                routes=self._route_payloads(ctx.store, stage1.completed_routes),
                fallback_max_tokens=min(
                    int(self.config.fallback_ar_tokens), int(remaining_tokens)
                ),
                eos_token_id=self.config.eos_token_id,
            )
            verify_elapsed_s = time.perf_counter() - verify_start
            target_elapsed_s += verify_elapsed_s
            target_verify_metadata = dict(verify_response.get("metadata", {}))
            decision = str(verify_response.get("decision", "select"))

            selected_route_id: int | None = None
            fallback_reason: str | None = None
            fallback_token_ids: list[int] = []
            if decision == "fallback_ar":
                fallback_token_ids = [
                    int(token_id) for token_id in verify_response.get("fallback_token_ids", [])
                ]
                fallback_reason = (
                    None
                    if verify_response.get("fallback_reason") is None
                    else str(verify_response.get("fallback_reason"))
                )
                if not fallback_token_ids:
                    raise RuntimeError("serial Target fallback returned no tokens")
                if len(fallback_token_ids) > remaining_tokens:
                    raise RuntimeError(
                        "serial Target fallback returned more tokens than the remaining budget: "
                        f"returned={len(fallback_token_ids)}, remaining={remaining_tokens}"
                    )
                committed_now = list(fallback_token_ids)
                handoff_mode = "serial_fallback_ar_append_persistent_kv"
            elif decision == "select":
                selected_route_id = selected_route_id_from_response(verify_response)
                selected_route = self._find_route(stage1, selected_route_id)
                full_commit = list(
                    ctx.store.materialized_token_path(selected_route)[: self.config.d]
                )
                committed_now = self._truncate_commit(full_commit, generated)
                if not committed_now:
                    raise RuntimeError("serial Target selected an empty route commit")
                handoff_mode = "serial_tree_only_persistent_kv"
            else:
                raise RuntimeError(f"unknown target verify decision: {decision!r}")

            generated.extend(committed_now)
            committed.extend(committed_now)
            should_stop = self._should_stop(committed_now, generated)
            handoff_start = time.perf_counter()
            if decision == "select":
                selected_index = next(
                    index
                    for index, route in enumerate(stage1.completed_routes)
                    if int(route.route_id) == int(selected_route_id)
                )
                if committed_now != full_commit:
                    # A partial final commit ends generation, so no next-prefix
                    # state is needed and committing extra KV would be wrong.
                    physical_stats = {
                        "committed_kv_tokens": 0,
                        "released_route_rows": 0,
                        "released_kv_pages": 0,
                    }
                else:
                    physical_stats = self._commit_route_as_prefix(
                        ctx=ctx,
                        route=selected_route,
                    )
                    prefix = self._persistent_prefix_state(
                        ctx=ctx,
                        next_token_logits=stage1.last_logits[selected_index],
                    )
            else:
                prefix, physical_stats = self._append_fallback_tokens(
                    ctx=ctx,
                    token_ids=committed_now,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            drafter_elapsed_s += time.perf_counter() - handoff_start
            traces.append(
                DistributedRoundTrace(
                    round_index=round_index,
                    selected_route_id=selected_route_id,
                    committed_tokens=list(committed_now),
                    decision=decision,
                    fallback_reason=fallback_reason,
                    fallback_token_ids=list(fallback_token_ids),
                    forest_depth=0,
                    verify_returned_before_forest_done=False,
                    target_scores=list(verify_response.get("scores", [])),
                    handoff_mode=handoff_mode,
                    reused_stage2_routes=0,
                    reused_stage2_depth=0,
                    remaining_tree_depth=0,
                    committed_kv_tokens=physical_stats["committed_kv_tokens"],
                    released_route_rows=physical_stats["released_route_rows"],
                    released_kv_pages=physical_stats["released_kv_pages"],
                    target_unique_tree_nodes=int(target_verify_metadata.get("node_count", 0)),
                    target_unmerged_path_nodes=int(
                        target_verify_metadata.get("unmerged_path_node_count", 0)
                    ),
                    target_prefix_len_before=int(
                        target_verify_metadata.get("prefix_len_before", len(committed) - len(committed_now))
                    ),
                    target_prefix_len_after=int(
                        target_verify_metadata.get("prefix_len_after", len(committed))
                    ),
                )
            )
            round_index += 1
            if should_stop:
                break

        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(committed, skip_special_tokens=False)
        total_elapsed_s = time.perf_counter() - total_start
        generated_tokens = len(generated)
        return DistributedGenerationResult(
            prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
            generated_token_ids=generated,
            text=text,
            rounds=traces,
            metadata={
                "system": "atlas_serial_tree_verify",
                "execution_mode": "serial_tree_verify",
                "async": False,
                "build_forest": False,
                "prefill_overlap": False,
                "verify_overlap": False,
                "serial_phase_order": "drafter_build_tree_then_target_verify",
                "drafter_backend": "sglang_flashinfer_paged_decode",
                "target_backend": "direct_flashinfer_full_llama_masked_verify",
                "target_transport": str(target_health.get("transport", "http")),
                "target_health": target_health,
                "k": self.config.k,
                "d": self.config.d,
                "max_new_tokens": self.config.max_new_tokens,
                "fallback_ar_tokens": int(self.config.fallback_ar_tokens),
                "drafter_prefill_elapsed_s": drafter_prefill_elapsed_s,
                "rebuild_drafter_prefix_after_commit": False,
                "cross_round_drafter_prefix_kv_reuse": True,
                "cross_round_stage2_kv_reuse": False,
                "cross_round_target_kv_reuse": True,
                "target_selection_policy": "best_of_n_full_path_with_optional_target_ar_fallback",
                "target_prefill_elapsed_s": target_prefill_elapsed_s,
                "last_verify_elapsed_s": verify_elapsed_s if traces else None,
                # All three throughputs intentionally use the same numerator:
                # tokens actually committed to the generated response. Target
                # prompt prefill is reported separately and excluded here.
                "generated_tokens": generated_tokens,
                "drafter_elapsed_s": drafter_elapsed_s,
                "target_elapsed_s": target_elapsed_s,
                "total_elapsed_s": total_elapsed_s,
                "drafter_tokens_per_second": (
                    generated_tokens / drafter_elapsed_s if drafter_elapsed_s else None
                ),
                "target_tokens_per_second": (
                    generated_tokens / target_elapsed_s if target_elapsed_s else None
                ),
                "overall_tokens_per_second": (
                    generated_tokens / total_elapsed_s if total_elapsed_s else None
                ),
                "throughput_token_definition": "committed_generated_tokens",
                "throughput_excludes_target_prompt_prefill": True,
                "throughput_excludes_drafter_prompt_prefill": True,
                "route_tail_page_cow": True,
            },
        )

    @staticmethod
    def _persistent_prefix_state(*, ctx, next_token_logits: torch.Tensor) -> DraftPrefixState:
        logits = next_token_logits.detach()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        token_ids = tuple(int(token_id) for token_id in ctx.backend.prefix_token_ids)
        return DraftPrefixState(
            token_ids=torch.tensor(token_ids, device=logits.device, dtype=torch.long),
            prefix_kv_view=PrefixKVView(committed_length=len(token_ids)),
            next_token_logits=logits,
            committed_length=len(token_ids),
        )

    @staticmethod
    def _commit_route_as_prefix(*, ctx, route: RouteState) -> dict[str, int]:
        _, stats = ctx.backend.commit_stage1_and_promote(
            committed_route=route,
            retained_routes=[],
        )
        ctx.store.reset_speculative()
        return stats

    def _append_fallback_tokens(
        self,
        *,
        ctx,
        token_ids: Sequence[int],
    ) -> tuple[DraftPrefixState, dict[str, int]]:
        """Materialize known Target AR tokens into the persistent Drafter KV."""
        if not token_ids:
            raise ValueError("fallback append requires at least one token")
        route_id = ctx.store.allocate_route_id()
        route = ctx.store.register_route(
            RouteState(
                route_id=route_id,
                stage1_root_id=route_id,
                parent_route_id=None,
                materialized_leaf_node_id=None,
                pending_token_id=int(token_ids[0]),
                cumulative_logprob=0.0,
                stage1_depth=0,
                stage2_depth=0,
                kv_view=ctx.store.prefix_view().fork(),
            )
        )
        logits: torch.Tensor | None = None
        for index, token_id in enumerate(token_ids):
            route.pending_token_id = int(token_id)
            output = ctx.backend.decode_frontier_one_token([route])
            ctx.store.mark_routes_materialized(
                [route],
                [int(output.new_node_ids.reshape(-1)[0])],
                phase=DecodePhase.STAGE1,
            )
            logits = output.next_token_logits
        assert logits is not None
        stats = self._commit_route_as_prefix(ctx=ctx, route=route)
        return self._persistent_prefix_state(ctx=ctx, next_token_logits=logits[0]), stats
