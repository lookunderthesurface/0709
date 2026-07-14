from __future__ import annotations

import argparse
import gc
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from .edge_timeline import (
    EdgeBlackboxTimeline,
    EdgeRoundTimeline,
    EdgeSpan,
    write_timeline_json,
    write_timeline_svg,
)
from .flashinfer_paged.builders import (
    build_forest_one_depth,
    build_tree_depths,
    build_tree_one_depth,
    initialize_forest_routes,
    initialize_stage1_routes,
    select_routes_by_stage1_root,
)
from .flashinfer_paged.kv import KVTreeStore
from .flashinfer_paged.sampling import (
    DrafterSamplingConfig,
    DrafterSamplingContext,
)
from .flashinfer_paged.sglang_runtime import (
    SGLangFlashInferFrontierModelBackend,
    SGLangRunnerConfig,
    create_sglang_model_runner,
    prefill_sglang_prefix,
    sglang_runner_component_report,
)
from .flashinfer_paged.types import BuildDepthsOutput, DraftPrefixState, RouteState
from .rpc import RemoteTargetClient, selected_route_id_from_response


@dataclass
class PagedPrefixContext:
    store: KVTreeStore
    backend: SGLangFlashInferFrontierModelBackend
    prefix: DraftPrefixState


@dataclass(frozen=True)
class DistributedAtlasConfig:
    k: int = 3
    d: int = 4
    max_new_tokens: int = 64
    eos_token_id: int | None = None
    fallback_ar_tokens: int = 4
    generation_seed: int = 0
    drafter_do_sample: bool = False
    drafter_temperature: float = 1.0
    fixed_forest_depth: int | None = None
    validate_state_alignment: bool = False

    def validate(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.d <= 0:
            raise ValueError("d must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.fallback_ar_tokens <= 0:
            raise ValueError("fallback_ar_tokens must be positive")
        if (
            not math.isfinite(float(self.drafter_temperature))
            or float(self.drafter_temperature) <= 0.0
        ):
            raise ValueError("drafter_temperature must be finite and positive")
        if self.fixed_forest_depth is not None and not (
            0 <= int(self.fixed_forest_depth) <= int(self.d)
        ):
            raise ValueError("fixed_forest_depth must be between 0 and d")


@dataclass
class DistributedRoundTrace:
    round_index: int
    selected_route_id: int | None
    committed_tokens: list[int]
    decision: str
    fallback_reason: str | None
    fallback_token_ids: list[int]
    forest_depth: int
    verify_returned_before_forest_done: bool
    target_scores: list[dict[str, object]]
    handoff_mode: str
    reused_stage2_routes: int
    reused_stage2_depth: int
    remaining_tree_depth: int
    committed_kv_tokens: int
    released_route_rows: int
    released_kv_pages: int
    target_unique_tree_nodes: int
    target_unmerged_path_nodes: int
    target_prefix_len_before: int
    target_prefix_len_after: int
    target_decision_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StageHandoff:
    stage1: BuildDepthsOutput
    mode: str
    reused_stage2_routes: int
    reused_stage2_depth: int
    remaining_tree_depth: int
    physical_stats: dict[str, int]
    commit_prune_span: EdgeSpan | None = None
    tree_step_spans: tuple[EdgeSpan, ...] = ()


@dataclass
class DistributedGenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    text: str | None
    rounds: list[DistributedRoundTrace] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "token_ids": [*self.prompt_token_ids, *self.generated_token_ids],
            "text": self.text,
            "rounds": [
                {
                    "round_index": trace.round_index,
                    "selected_route_id": trace.selected_route_id,
                    "committed_tokens": trace.committed_tokens,
                    "decision": trace.decision,
                    "fallback_reason": trace.fallback_reason,
                    "fallback_token_ids": trace.fallback_token_ids,
                    "forest_depth": trace.forest_depth,
                    "verify_returned_before_forest_done": trace.verify_returned_before_forest_done,
                    "target_scores": trace.target_scores,
                    "handoff_mode": trace.handoff_mode,
                    "reused_stage2_routes": trace.reused_stage2_routes,
                    "reused_stage2_depth": trace.reused_stage2_depth,
                    "remaining_tree_depth": trace.remaining_tree_depth,
                    "committed_kv_tokens": trace.committed_kv_tokens,
                    "released_route_rows": trace.released_route_rows,
                    "released_kv_pages": trace.released_kv_pages,
                    "target_unique_tree_nodes": trace.target_unique_tree_nodes,
                    "target_unmerged_path_nodes": trace.target_unmerged_path_nodes,
                    "target_prefix_len_before": trace.target_prefix_len_before,
                    "target_prefix_len_after": trace.target_prefix_len_after,
                    "target_decision_metadata": trace.target_decision_metadata,
                }
                for trace in self.rounds
            ],
            "metadata": self.metadata,
        }


class PagedDistributedAtlasGenerator:
    def __init__(
        self,
        *,
        config: DistributedAtlasConfig,
        runner,
        page_size: int,
        prefill_chunk_size: int,
        target_client: RemoteTargetClient,
        tokenizer=None,
    ) -> None:
        self.config = config
        self.runner = runner
        self.page_size = int(page_size)
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.target_client = target_client
        self.tokenizer = tokenizer
        self.last_timeline: EdgeBlackboxTimeline | None = None
        self._timeline_origin_s: float | None = None
        self.config.validate()

    def generate(self, prompt_token_ids: Sequence[int]) -> DistributedGenerationResult:
        committed = [int(token_id) for token_id in prompt_token_ids]
        generated: list[int] = []
        traces: list[DistributedRoundTrace] = []
        target_fallback_ar_profiles: list[dict[str, object]] = []
        target_health = self.target_client.health()
        target_runtime_metadata = dict(target_health.get("metadata", {}))
        self._timeline_origin_s = time.perf_counter()
        timeline_rounds: list[EdgeRoundTimeline] = []
        initial_sampling = self._drafter_sampling_context(committed)

        prebuilt_forest_frontier: list[RouteState] | None = None
        prebuilt_forest_completed: list[RouteState] | None = None
        prebuilt_forest_last_logits: torch.Tensor | None = None
        prebuilt_forest_depth = 0
        prebuilt_forest_spans: list[EdgeSpan] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            target_prefill_future = pool.submit(
                self._timed_call,
                "cloud_prompt_blackbox",
                self.target_client.prefill,
                committed,
            )
            ctx_future = pool.submit(
                self._timed_call,
                "edge_prompt_prefill",
                self._prepare_prefix,
                committed,
            )
            ctx, edge_prefill_span = ctx_future.result()

            initial_tree_start = self._timeline_now()
            active_routes = initialize_stage1_routes(
                ctx.prefix,
                k=self.config.k,
                route_store=ctx.store,
                sampling=initial_sampling,
            )
            stage1 = build_tree_depths(
                active_routes,
                depth=self.config.d,
                k=self.config.k,
                route_store=ctx.store,
                model_backend=ctx.backend,
                sampling=initial_sampling,
            )
            initial_tree_span = EdgeSpan(
                name="edge_initial_stage1_tree",
                start_s=initial_tree_start,
                end_s=self._timeline_now(),
            )

            if (
                self.config.fixed_forest_depth is None
                and not target_prefill_future.done()
            ):
                prebuilt_forest_frontier = initialize_forest_routes(
                    stage1.completed_routes,
                    stage1.last_logits,
                    k=self.config.k,
                    route_store=ctx.store,
                    sampling=initial_sampling,
                )
                while (
                    not target_prefill_future.done()
                    and prebuilt_forest_depth < self.config.d
                ):
                    step_start_s = self._timeline_now()
                    forest_step = build_forest_one_depth(
                        prebuilt_forest_frontier,
                        k=self.config.k,
                        route_store=ctx.store,
                        model_backend=ctx.backend,
                        sampling=initial_sampling,
                    )
                    prebuilt_forest_depth += 1
                    prebuilt_forest_spans.append(
                        EdgeSpan(
                            name=f"edge_forest_step_{prebuilt_forest_depth}",
                            start_s=step_start_s,
                            end_s=self._timeline_now(),
                        )
                    )
                    prebuilt_forest_completed = forest_step.decoded_routes
                    prebuilt_forest_last_logits = (
                        forest_step.decode_output.next_token_logits
                    )
                    prebuilt_forest_frontier = forest_step.next_routes

            _, cloud_prefill_span = target_prefill_future.result()

        round_index = 0
        with ThreadPoolExecutor(max_workers=1) as verifier_pool:
            while len(generated) < self.config.max_new_tokens:
                round_sampling = self._drafter_sampling_context(committed)
                round_prefix_len = len(committed)
                verify_routes = self._route_payloads(ctx.store, stage1.completed_routes)
                verify_submit_s = self._timeline_now()
                remaining_tokens = self.config.max_new_tokens - len(generated)
                verify_future = verifier_pool.submit(
                    self._verify_blackbox_call,
                    committed,
                    verify_routes,
                    remaining_tokens,
                    round_index,
                )
                if round_index == 0 and prebuilt_forest_frontier is not None:
                    forest_depth = prebuilt_forest_depth
                    forest_completed = prebuilt_forest_completed
                    forest_last_logits = prebuilt_forest_last_logits
                    forest_step_spans = list(prebuilt_forest_spans)
                    forest_frontier = prebuilt_forest_frontier
                else:
                    forest_depth = 0
                    forest_completed = None
                    forest_last_logits = None
                    forest_step_spans = []
                    forest_frontier = initialize_forest_routes(
                        stage1.completed_routes,
                        stage1.last_logits,
                        k=self.config.k,
                        route_store=ctx.store,
                        sampling=round_sampling,
                    )
                while self._should_build_forest_step(
                    verify_done=verify_future.done(),
                    forest_depth=forest_depth,
                ):
                    forest_step_start = self._timeline_now()
                    forest_step = build_forest_one_depth(
                        forest_frontier,
                        k=self.config.k,
                        route_store=ctx.store,
                        model_backend=ctx.backend,
                        sampling=round_sampling,
                    )
                    forest_depth += 1
                    forest_step_spans.append(
                        EdgeSpan(
                            name=f"edge_forest_step_{forest_depth}",
                            start_s=forest_step_start,
                            end_s=self._timeline_now(),
                        )
                    )
                    forest_completed = forest_step.decoded_routes
                    forest_last_logits = forest_step.decode_output.next_token_logits
                    forest_frontier = forest_step.next_routes

                verify_returned_before_forest_done = forest_depth < self.config.d and verify_future.done()
                verify_response, verify_done_s = verify_future.result()
                verify_blackbox_span = EdgeSpan(
                    name="cloud_verify_blackbox",
                    start_s=verify_submit_s,
                    end_s=verify_done_s,
                )
                target_verify_metadata = dict(verify_response.get("metadata", {}))
                fallback_ar_profile = target_verify_metadata.get("fallback_ar_profile")
                if isinstance(fallback_ar_profile, dict):
                    target_fallback_ar_profiles.append(
                        {"round_index": round_index, **fallback_ar_profile}
                    )

                decision = str(verify_response.get("decision", "select"))
                selected_route_id: int | None = None
                committed_now: list[int]
                fallback_reason: str | None = None
                fallback_token_ids: list[int] = []
                handoff_mode = "stopped"
                reused_stage2_routes = 0
                reused_stage2_depth = 0
                remaining_tree_depth = 0
                physical_stats = {
                    "committed_kv_tokens": 0,
                    "released_route_rows": 0,
                    "released_kv_pages": 0,
                }
                handoff_span: EdgeSpan | None = None
                post_prune_tree_step_spans: list[EdgeSpan] = []

                if decision == "fallback_ar":
                    fallback_token_ids = [
                        int(token_id) for token_id in verify_response.get("fallback_token_ids", [])
                    ]
                    fallback_reason = (
                        None
                        if verify_response.get("fallback_reason") is None
                        else str(verify_response.get("fallback_reason"))
                    )
                    if len(fallback_token_ids) > remaining_tokens:
                        raise RuntimeError(
                            "target fallback returned more tokens than the remaining budget: "
                            f"returned={len(fallback_token_ids)}, remaining={remaining_tokens}"
                        )
                    committed_now = list(fallback_token_ids)
                    generated.extend(committed_now)
                    committed.extend(committed_now)
                    should_stop = not committed_now or self._should_stop(committed_now, generated)
                    handoff_mode = "fallback_ar_stopped" if should_stop else "fallback_ar_extend_prefix_kv"
                    if not should_stop:
                        extend_start_s = self._timeline_now()
                        stage1, physical_stats = self._fallback_extend_handoff(
                            ctx=ctx,
                            fallback_token_ids=committed_now,
                        )
                        handoff_span = EdgeSpan(
                            name="edge_fallback_extend_then_build_tree",
                            start_s=extend_start_s,
                            end_s=self._timeline_now(),
                        )
                elif decision == "select":
                    selected_route_id = selected_route_id_from_response(verify_response)
                    selected_route = self._find_route(stage1, selected_route_id)
                    full_commit = list(
                        ctx.store.materialized_token_path(selected_route)[: self.config.d]
                    )
                    committed_now = self._truncate_commit(full_commit, generated)
                    generated.extend(committed_now)
                    committed.extend(committed_now)

                    should_stop = not committed_now or self._should_stop(committed_now, generated)

                    if not should_stop:
                        if committed_now != full_commit:
                            raise RuntimeError(
                                "cannot reuse stage-2 KV after a partial stage-1 commit"
                            )
                        handoff = self._handoff_after_verify(
                            ctx=ctx,
                            selected_stage1_route=selected_route,
                            forest_depth=forest_depth,
                            forest_frontier=forest_frontier,
                            forest_completed=forest_completed,
                            forest_last_logits=forest_last_logits,
                            sampling=self._drafter_sampling_context(committed),
                        )
                        handoff_span = handoff.commit_prune_span
                        post_prune_tree_step_spans = list(handoff.tree_step_spans)
                        stage1 = handoff.stage1
                        handoff_mode = handoff.mode
                        reused_stage2_routes = handoff.reused_stage2_routes
                        reused_stage2_depth = handoff.reused_stage2_depth
                        remaining_tree_depth = handoff.remaining_tree_depth
                        physical_stats = handoff.physical_stats
                else:
                    raise RuntimeError(f"unknown target verify decision: {decision!r}")

                if self.config.validate_state_alignment:
                    self._validate_target_prefix_lengths(
                        target_verify_metadata,
                        expected_before=round_prefix_len,
                        expected_after=len(committed),
                    )
                    if not should_stop:
                        self._validate_drafter_prefix_state(
                            ctx,
                            committed,
                            active_routes=[
                                *stage1.completed_routes,
                                *stage1.next_frontier_routes,
                            ],
                        )

                traces.append(
                    DistributedRoundTrace(
                        round_index=round_index,
                        selected_route_id=selected_route_id,
                        committed_tokens=[int(token_id) for token_id in committed_now],
                        decision=decision,
                        fallback_reason=fallback_reason,
                        fallback_token_ids=list(fallback_token_ids),
                        forest_depth=forest_depth,
                        verify_returned_before_forest_done=verify_returned_before_forest_done,
                        target_scores=list(verify_response.get("scores", [])),
                        handoff_mode=handoff_mode,
                        reused_stage2_routes=reused_stage2_routes,
                        reused_stage2_depth=reused_stage2_depth,
                        remaining_tree_depth=remaining_tree_depth,
                        committed_kv_tokens=int(physical_stats["committed_kv_tokens"]),
                        released_route_rows=int(physical_stats["released_route_rows"]),
                        released_kv_pages=int(physical_stats["released_kv_pages"]),
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
                        target_decision_metadata=self._compact_target_decision_metadata(
                            target_verify_metadata
                        ),
                    )
                )
                timeline_rounds.append(
                    EdgeRoundTimeline(
                        round_index=round_index,
                        verify_blackbox=verify_blackbox_span,
                        forest_steps=forest_step_spans,
                        post_prune_tree_steps=post_prune_tree_step_spans,
                        handoff=handoff_span,
                        forest_depth=forest_depth,
                        accepted_tokens=len(committed_now),
                        generated_tokens=len(generated),
                        handoff_mode=handoff_mode,
                        verify_returned_before_forest_done=verify_returned_before_forest_done,
                    )
                )
                round_index += 1

                if should_stop:
                    break

        elapsed_s = self._timeline_now()
        timeline = EdgeBlackboxTimeline(
            edge_prefill=edge_prefill_span,
            cloud_prefill_blackbox=cloud_prefill_span,
            initial_tree=initial_tree_span,
            rounds=timeline_rounds,
            elapsed_s=elapsed_s,
            generated_tokens=len(generated),
        )
        self.last_timeline = timeline

        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(committed, skip_special_tokens=False)
        return DistributedGenerationResult(
            prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
            generated_token_ids=generated,
            text=text,
            rounds=traces,
            metadata={
                "system": "atlas_distributed_two_process",
                "drafter_backend": "sglang_flashinfer_paged_decode",
                "target_backend": "direct_flashinfer_full_llama_masked_verify",
                "target_health": target_health,
                "k": self.config.k,
                "d": self.config.d,
                "max_new_tokens": self.config.max_new_tokens,
                "fallback_ar_tokens": int(self.config.fallback_ar_tokens),
                "generation_seed": int(self.config.generation_seed),
                "drafter_sampling": self._drafter_sampling_config().to_dict(),
                "fixed_forest_depth": self.config.fixed_forest_depth,
                "validate_state_alignment": bool(self.config.validate_state_alignment),
                "rebuild_drafter_prefix_after_commit": False,
                "rebuild_drafter_prefix_after_fallback": False,
                "fallback_drafter_handoff": "single_multi_token_extend",
                "target_fallback_ar_profiles": target_fallback_ar_profiles,
                "rebuild_target_prefix_after_commit": False,
                "cross_round_stage2_kv_reuse": True,
                "target_selection_policy": target_runtime_metadata.get(
                    "selection_policy",
                    "best_of_n_full_path_with_optional_target_ar_fallback",
                ),
                "target_route_selection_policy": target_runtime_metadata.get(
                    "route_selection_policy"
                ),
                "cross_round_target_kv_reuse": True,
                "route_tail_page_cow": True,
                "branch_safe_page_size": True,
                "forest_cancel_granularity": "between_decode_steps",
                "edge_blackbox_timeline": timeline.summary(),
                "drafter_runner": sglang_runner_component_report(self.runner),
            },
        )

    def _timeline_now(self) -> float:
        if self._timeline_origin_s is None:
            raise RuntimeError("edge timeline clock has not been initialized")
        return time.perf_counter() - self._timeline_origin_s

    def _timed_call(self, name: str, fn, *args):
        start_s = self._timeline_now()
        result = fn(*args)
        return result, EdgeSpan(name=name, start_s=start_s, end_s=self._timeline_now())

    def _verify_blackbox_call(
        self,
        committed: Sequence[int],
        routes: Sequence[dict[str, object]],
        remaining_tokens: int,
        round_index: int,
    ) -> tuple[dict[str, object], float]:
        response = self.target_client.verify(
            prefix_token_ids=committed,
            routes=routes,
            fallback_max_tokens=min(int(self.config.fallback_ar_tokens), int(remaining_tokens)),
            selected_path_max_tokens=int(remaining_tokens),
            eos_token_id=self.config.eos_token_id,
            route_sampling_seed=int(self.config.generation_seed),
            route_sampling_round=int(round_index),
        )
        return response, self._timeline_now()

    def _should_build_forest_step(
        self,
        *,
        verify_done: bool,
        forest_depth: int,
    ) -> bool:
        if int(forest_depth) >= int(self.config.d):
            return False
        fixed_depth = self.config.fixed_forest_depth
        if fixed_depth is not None:
            return int(forest_depth) < int(fixed_depth)
        return not bool(verify_done)

    @staticmethod
    def _validate_target_prefix_lengths(
        metadata: dict[str, object],
        *,
        expected_before: int,
        expected_after: int,
    ) -> None:
        observed_before = int(metadata.get("prefix_len_before", -1))
        observed_after = int(metadata.get("prefix_len_after", -1))
        if observed_before != int(expected_before) or observed_after != int(expected_after):
            raise RuntimeError(
                "Target committed-prefix length diverged from the coordinator: "
                f"target={observed_before}->{observed_after}, "
                f"coordinator={int(expected_before)}->{int(expected_after)}"
            )

    @staticmethod
    def _validate_drafter_prefix_state(
        ctx: PagedPrefixContext,
        committed: Sequence[int],
        *,
        active_routes: Sequence[RouteState],
    ) -> None:
        expected = [int(token_id) for token_id in committed]
        backend_tokens = [int(token_id) for token_id in ctx.backend.prefix_token_ids]
        store_tokens = [int(token_id) for token_id in ctx.store.committed_token_ids]
        prefix_slots = ctx.backend.route_pool.prefix_slot_ids
        if backend_tokens != expected or store_tokens != expected:
            raise RuntimeError(
                "Drafter logical committed prefix diverged from the coordinator"
            )
        if prefix_slots is None or int(prefix_slots.numel()) != len(expected):
            observed_slots = None if prefix_slots is None else int(prefix_slots.numel())
            raise RuntimeError(
                "Drafter physical prefix length diverged from the coordinator: "
                f"slots={observed_slots}, tokens={len(expected)}"
            )
        for route in active_routes:
            route_id = int(route.route_id)
            slot_path = ctx.backend.route_pool.route_slot_paths.get(route_id)
            if slot_path is None:
                raise RuntimeError(f"Drafter route {route_id} has no physical slot path")
            actual_slots = [int(slot_id) for slot_id in slot_path.detach().cpu().tolist()]
            expected_suffix: list[int] = []
            for node_id in route.kv_view.node_ids:
                slot_id = ctx.backend.route_pool.node_slot_ids.get(int(node_id))
                if slot_id is None:
                    raise RuntimeError(
                        f"Drafter route {route_id} node {int(node_id)} has no physical KV slot"
                    )
                if isinstance(slot_id, torch.Tensor):
                    expected_suffix.append(int(slot_id.detach().cpu()))
                else:
                    expected_suffix.append(int(slot_id))
            committed_length = int(route.kv_view.prefix.committed_length)
            if actual_slots[committed_length:] != expected_suffix:
                raise RuntimeError(
                    f"Drafter route {route_id} physical slots do not match its logical node path"
                )
            row = ctx.backend.route_pool.route_rows.get(route_id)
            if row is not None and int(row.written_length) != len(actual_slots):
                raise RuntimeError(
                    f"Drafter route {route_id} req-row length is inconsistent: "
                    f"row={int(row.written_length)}, slots={len(actual_slots)}"
                )
            req_table = getattr(
                ctx.backend.route_pool.req_to_token_pool,
                "req_to_token",
                None,
            )
            if row is not None and isinstance(req_table, torch.Tensor):
                req_slots = [
                    int(slot_id)
                    for slot_id in req_table[
                        int(row.req_pool_index), : len(actual_slots)
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if req_slots != actual_slots:
                    raise RuntimeError(
                        f"Drafter route {route_id} req row does not match its physical slot path"
                    )

    def _prepare_prefix(self, committed: Sequence[int]) -> PagedPrefixContext:
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
            prompt_token_ids=committed,
            chunk_size=self.prefill_chunk_size,
        )
        prefix = backend.attach_prefilled_prefix(
            prompt_token_ids=prefill.prompt_token_ids,
            prefix_slot_ids=prefill.prefix_slot_ids,
            next_token_logits=prefill.next_token_logits,
        )
        store.committed_token_ids = [int(token_id) for token_id in committed]
        return PagedPrefixContext(store=store, backend=backend, prefix=prefix)

    def _drafter_sampling_config(self) -> DrafterSamplingConfig:
        return DrafterSamplingConfig(
            do_sample=bool(self.config.drafter_do_sample),
            seed=int(self.config.generation_seed),
            temperature=float(self.config.drafter_temperature),
        )

    def _drafter_sampling_context(
        self,
        committed: Sequence[int],
    ) -> DrafterSamplingContext:
        return DrafterSamplingContext(
            config=self._drafter_sampling_config(),
            committed_token_ids=tuple(int(token_id) for token_id in committed),
        )

    @staticmethod
    def _compact_target_decision_metadata(
        metadata: dict[str, object],
    ) -> dict[str, object]:
        keys = (
            "selection_policy",
            "route_sampling_enabled",
            "route_sampling_scope",
            "route_sampling_seed",
            "route_sampling_round",
            "route_sampling_temperature",
            "route_sampling_uniform",
            "route_sampling_selected_probability",
            "route_sampling_selected_cdf_low",
            "route_sampling_selected_cdf_high",
            "route_sampling_boundary_margin",
            "route_sampling_ordered_candidates",
            "selected_draft_rank",
            "route0_route_id",
            "draft_top_route_id",
            "selected_route_changed_from_route0",
            "selected_first_token_changed_from_route0",
            "selected_minus_route0_draft_first_token_logprob",
            "selected_minus_route0_draft_logprob",
            "selected_minus_route0_target_selection_score",
            "route_intervention",
        )
        return {key: metadata[key] for key in keys if key in metadata}

    def _fallback_extend_handoff(
        self,
        *,
        ctx: PagedPrefixContext,
        fallback_token_ids: Sequence[int],
    ) -> tuple[BuildDepthsOutput, dict[str, int]]:
        """Discard speculative work, extend persistent prefix KV, and rebuild stage-1."""
        suffix = [int(token_id) for token_id in fallback_token_ids]
        if not suffix:
            raise ValueError("fallback handoff requires at least one Target token")

        # No forest kernel is in flight here: the verify/forest loop only
        # checks the decision after the current full decode step has returned.
        # Dropping every route row first makes route_slot_paths stop retaining
        # speculative pages; committed prefix pages remain protected by
        # route_pool.prefix_page_ids.
        released_rows = ctx.backend.route_pool.retain_route_rows([])
        released_pages = ctx.backend.route_pool.release_unreferenced_node_slots([])
        ctx.store.committed_token_ids.extend(suffix)
        ctx.store.reset_speculative()

        next_token_logits = ctx.backend.append_known_tokens_as_prefix(suffix)
        prefix_tokens = tuple(int(token_id) for token_id in ctx.backend.prefix_token_ids)
        prefix = DraftPrefixState(
            token_ids=torch.tensor(
                prefix_tokens,
                device=next_token_logits.device,
                dtype=torch.long,
            ),
            prefix_kv_view=ctx.store.prefix_view(),
            next_token_logits=(
                next_token_logits.unsqueeze(0)
                if next_token_logits.ndim == 1
                else next_token_logits
            ),
            committed_length=len(prefix_tokens),
        )
        active_routes = initialize_stage1_routes(
            prefix,
            k=self.config.k,
            route_store=ctx.store,
            sampling=self._drafter_sampling_context(ctx.store.committed_token_ids),
        )
        stage1 = build_tree_depths(
            active_routes,
            depth=self.config.d,
            k=self.config.k,
            route_store=ctx.store,
            model_backend=ctx.backend,
            sampling=self._drafter_sampling_context(ctx.store.committed_token_ids),
        )
        return stage1, {
            "committed_kv_tokens": len(suffix),
            "retained_routes": 0,
            "released_route_rows": released_rows,
            "released_kv_pages": released_pages,
        }

    def _handoff_after_verify(
        self,
        *,
        ctx: PagedPrefixContext,
        selected_stage1_route: RouteState,
        forest_depth: int,
        forest_frontier: Sequence[RouteState],
        forest_completed: Sequence[RouteState] | None,
        forest_last_logits: torch.Tensor | None,
        sampling: DrafterSamplingContext,
    ) -> StageHandoff:
        selected_root_id = int(selected_stage1_route.route_id)
        timeline_active = getattr(self, "_timeline_origin_s", None) is not None

        def clock_now() -> float:
            if timeline_active:
                return self._timeline_now()
            return time.perf_counter()

        if forest_depth >= self.config.d:
            if forest_completed is None or forest_last_logits is None:
                raise RuntimeError("completed forest is missing its decoded routes or logits")
            selected_indices = [
                index
                for index, route in enumerate(forest_completed)
                if int(route.stage1_root_id) == selected_root_id
            ]
            if len(selected_indices) != self.config.k:
                raise RuntimeError(
                    "completed forest does not contain k routes for the selected stage-1 root: "
                    f"expected={self.config.k}, got={len(selected_indices)}"
                )
            retained = [forest_completed[index] for index in selected_indices]
            commit_start_s = clock_now()
            promoted, physical_stats = ctx.backend.commit_stage1_and_promote(
                committed_route=selected_stage1_route,
                retained_routes=retained,
            )
            commit_span = EdgeSpan(
                name="edge_commit_prune",
                start_s=commit_start_s,
                end_s=clock_now(),
            )
            return StageHandoff(
                stage1=BuildDepthsOutput(
                    completed_routes=promoted,
                    next_frontier_routes=[],
                    last_logits=forest_last_logits[selected_indices],
                    steps=[],
                ),
                mode="completed_forest_reused",
                reused_stage2_routes=len(promoted),
                reused_stage2_depth=self.config.d,
                remaining_tree_depth=0,
                physical_stats=physical_stats,
                commit_prune_span=commit_span,
            )

        retained = select_routes_by_stage1_root(
            forest_frontier,
            selected_stage1_root_id=selected_root_id,
        )
        if len(retained) != self.config.k:
            raise RuntimeError(
                "partial forest does not contain k active routes for the selected stage-1 root: "
                f"expected={self.config.k}, got={len(retained)}"
            )
        commit_start_s = clock_now()
        promoted, physical_stats = ctx.backend.commit_stage1_and_promote(
            committed_route=selected_stage1_route,
            retained_routes=retained,
        )
        commit_span = EdgeSpan(
            name="edge_commit_prune",
            start_s=commit_start_s,
            end_s=clock_now(),
        )
        promoted_depths = {int(route.stage1_depth) for route in promoted}
        if promoted_depths != {int(forest_depth)}:
            raise RuntimeError(
                "promoted route depth does not match completed forest depth: "
                f"route_depths={sorted(promoted_depths)}, forest_depth={forest_depth}"
            )
        remaining_depth = self.config.d - int(forest_depth)
        tree_steps = []
        tree_step_spans: list[EdgeSpan] = []
        frontier = list(promoted)
        for step_index in range(remaining_depth):
            step_start_s = clock_now()
            tree_step = build_tree_one_depth(
                frontier,
                k=self.config.k,
                route_store=ctx.store,
                model_backend=ctx.backend,
                sampling=sampling,
            )
            tree_step_spans.append(
                EdgeSpan(
                    name=f"edge_post_prune_tree_step_{step_index + 1}",
                    start_s=step_start_s,
                    end_s=clock_now(),
                )
            )
            tree_steps.append(tree_step)
            frontier = tree_step.next_routes
        last_step = tree_steps[-1]
        stage1 = BuildDepthsOutput(
            completed_routes=last_step.decoded_routes,
            next_frontier_routes=last_step.next_routes,
            last_logits=last_step.decode_output.next_token_logits,
            steps=tree_steps,
        )
        return StageHandoff(
            stage1=stage1,
            mode="partial_forest_pruned_then_build_tree",
            reused_stage2_routes=len(promoted),
            reused_stage2_depth=int(forest_depth),
            remaining_tree_depth=remaining_depth,
            physical_stats=physical_stats,
            commit_prune_span=commit_span,
            tree_step_spans=tuple(tree_step_spans),
        )

    def _route_payloads(
        self,
        store: KVTreeStore,
        routes: Sequence[RouteState],
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for route in routes:
            draft_token_logprobs = tuple(
                float(value)
                for value in route.token_logprobs[: int(route.stage1_depth)]
            )
            payloads.append(
                {
                    "route_id": int(route.route_id),
                    "token_ids": [int(token_id) for token_id in store.materialized_token_path(route)[: route.stage1_depth]],
                    # Preserve the historical cumulative beam score for
                    # debugging. Target/evaluators use draft_token_logprobs
                    # for the current-prefix path score after promotion.
                    "draft_logprob": float(route.cumulative_logprob),
                    "draft_token_logprobs": list(draft_token_logprobs),
                }
            )
        return payloads

    @staticmethod
    def _find_route(stage1: BuildDepthsOutput, route_id: int) -> RouteState:
        for route in stage1.completed_routes:
            if int(route.route_id) == int(route_id):
                return route
        raise KeyError(f"selected route_id={route_id} is not in the completed stage1 routes")

    def _should_stop(self, committed_now: Sequence[int], generated: Sequence[int]) -> bool:
        if len(generated) >= self.config.max_new_tokens:
            return True
        return self.config.eos_token_id is not None and any(
            int(token_id) == int(self.config.eos_token_id) for token_id in committed_now
        )

    def _truncate_commit(
        self,
        token_ids: Sequence[int],
        generated: Sequence[int],
    ) -> list[int]:
        remaining = self.config.max_new_tokens - len(generated)
        result = [int(token_id) for token_id in token_ids[:remaining]]
        if self.config.eos_token_id is not None:
            for index, token_id in enumerate(result):
                if int(token_id) == int(self.config.eos_token_id):
                    return result[: index + 1]
        return result


def parse_token_ids(
    raw: str | None,
    *,
    tokenizer,
    prompt: str,
    prefix_len: int | None = None,
    repeat_token_id: int | None = None,
) -> list[int]:
    if repeat_token_id is not None:
        if prefix_len is None or prefix_len <= 0:
            raise RuntimeError("--repeat-token-id requires a positive --prefix-len")
        return [int(repeat_token_id)] * int(prefix_len)
    if raw:
        tokens = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        tokens = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
    if not tokens:
        raise RuntimeError("prompt token list is empty")
    if prefix_len is None:
        return tokens
    repeated = (tokens * ((int(prefix_len) + len(tokens) - 1) // len(tokens)))[: int(prefix_len)]
    return repeated


def required_context_length(prompt_len: int, *, max_new_tokens: int, depth: int) -> int:
    return int(prompt_len) + int(max_new_tokens) + 2 * int(depth)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ATLAS drafter/coordinator process.")
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--target-url", required=True, help="Example: http://127.0.0.1:18080")
    parser.add_argument("--prompt", default="ATLAS distributed generation.")
    parser.add_argument("--prompt-token-ids", default=None)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--repeat-token-id", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument(
        "--fixed-forest-depth",
        type=int,
        default=None,
        help=(
            "Correctness/replay diagnostic: build exactly this many forest depths "
            "per round instead of stopping on Target wall-clock completion."
        ),
    )
    parser.add_argument(
        "--validate-state-alignment",
        action="store_true",
        help=(
            "Correctness diagnostic: assert Target/Drafter/coordinator committed "
            "prefix lengths and tokens after every non-terminal handoff."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument(
        "--fallback-ar-tokens",
        type=int,
        default=4,
        help="Maximum Target AR tokens the Edge will accept on a low-confidence fallback.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=0,
        help=(
            "Request-level seed used by semantic-prefix Drafter sampling and "
            "Target route sampling. It is sent explicitly on every verify round."
        ),
    )
    parser.add_argument(
        "--drafter-do-sample",
        action="store_true",
        help=(
            "Sample each Drafter parent candidate set without replacement. RNG "
            "is keyed by the complete semantic token prefix, so forest timing "
            "does not advance or reorder the generation random stream."
        ),
    )
    parser.add_argument(
        "--drafter-temperature",
        type=float,
        default=1.0,
        help="Proposal temperature for --drafter-do-sample.",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Drafter KV page size. Partial tail pages are copied on route fork.",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help=(
            "Run this many unmeasured in-process warmups before the real timeline. "
            "Use 0 only when intentionally measuring first-forward cold start."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--timeline-svg",
        default=None,
        help="Write an Edge-observed SVG timeline; Cloud is treated as a black box.",
    )
    parser.add_argument(
        "--timeline-json",
        default=None,
        help="Write the raw Edge-observed timeline measurements as JSON.",
    )
    parser.add_argument("--timeline-width", type=int, default=1600)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the drafter process.")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    tokenizer = AutoTokenizer.from_pretrained(args.drafter_model, trust_remote_code=args.trust_remote_code)
    prompt_token_ids = parse_token_ids(
        args.prompt_token_ids,
        tokenizer=tokenizer,
        prompt=args.prompt,
        prefix_len=args.prefix_len,
        repeat_token_id=args.repeat_token_id,
    )
    warmup_generated_tokens = (
        max(
            2 * int(args.d),
            int(args.page_size) + 2 * int(args.d),
        )
        if args.warmup_runs
        else 0
    )
    context_generation_budget = max(
        int(args.max_new_tokens),
        warmup_generated_tokens,
    )
    min_context = required_context_length(
        len(prompt_token_ids),
        max_new_tokens=context_generation_budget,
        depth=args.d,
    )
    context_length = min_context if args.context_length is None else max(int(args.context_length), min_context)
    if args.context_length is not None and int(args.context_length) < min_context:
        print(
            f"[warn] --context-length {args.context_length} is too small; using {context_length} "
            f"(prompt_len={len(prompt_token_ids)}, generation_budget={context_generation_budget}, "
            f"d={args.d}, warmup_runs={args.warmup_runs}).",
            flush=True,
        )

    runner_config = SGLangRunnerConfig(
        model_path=args.drafter_model,
        dtype=args.dtype,
        context_length=context_length,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        gpu_id=args.gpu_id,
        nccl_port=args.nccl_port,
        trust_remote_code=args.trust_remote_code,
    )
    runner = None
    try:
        runner = create_sglang_model_runner(runner_config, initialize=True)
        target_client = RemoteTargetClient(args.target_url, timeout=args.timeout)
        warmup_elapsed_s = 0.0
        for warmup_index in range(int(args.warmup_runs)):
            print(
                f"[warmup] run {warmup_index + 1}/{args.warmup_runs}: "
                f"k={args.k}, d={args.d}, generated_tokens={warmup_generated_tokens}",
                flush=True,
            )
            warmup_start_s = time.perf_counter()
            warmup_generator = PagedDistributedAtlasGenerator(
                config=DistributedAtlasConfig(
                    k=args.k,
                    d=args.d,
                    max_new_tokens=warmup_generated_tokens,
                    eos_token_id=None,
                    fallback_ar_tokens=args.fallback_ar_tokens,
                    generation_seed=args.generation_seed,
                    drafter_do_sample=args.drafter_do_sample,
                    drafter_temperature=args.drafter_temperature,
                    fixed_forest_depth=args.fixed_forest_depth,
                    validate_state_alignment=args.validate_state_alignment,
                ),
                runner=runner,
                page_size=args.page_size,
                prefill_chunk_size=args.prefill_chunk_size,
                target_client=target_client,
                tokenizer=None,
            )
            warmup_generator.generate(prompt_token_ids)
            torch.cuda.synchronize()
            run_elapsed_s = time.perf_counter() - warmup_start_s
            warmup_elapsed_s += run_elapsed_s
            print(
                f"[warmup] completed run {warmup_index + 1} in "
                f"{run_elapsed_s * 1000.0:.1f} ms; resetting KV/session for measurement",
                flush=True,
            )

        generator = PagedDistributedAtlasGenerator(
            config=DistributedAtlasConfig(
                k=args.k,
                d=args.d,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=args.eos_token_id,
                fallback_ar_tokens=args.fallback_ar_tokens,
                generation_seed=args.generation_seed,
                drafter_do_sample=args.drafter_do_sample,
                drafter_temperature=args.drafter_temperature,
                fixed_forest_depth=args.fixed_forest_depth,
                validate_state_alignment=args.validate_state_alignment,
            ),
            runner=runner,
            page_size=args.page_size,
            prefill_chunk_size=args.prefill_chunk_size,
            target_client=target_client,
            tokenizer=tokenizer,
        )
        result = generator.generate(prompt_token_ids)
        result.metadata["warmup_runs"] = int(args.warmup_runs)
        result.metadata["warmup_generated_tokens_per_run"] = warmup_generated_tokens
        result.metadata["warmup_elapsed_s"] = warmup_elapsed_s
        result.metadata["warmup_excluded_from_timeline"] = True
        text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        print(text)
        if args.json_out:
            Path(args.json_out).write_text(text + "\n", encoding="utf-8")
        if generator.last_timeline is None:
            raise RuntimeError("generation completed without an Edge timeline")
        if args.timeline_svg:
            write_timeline_svg(
                args.timeline_svg,
                generator.last_timeline,
                width=args.timeline_width,
            )
            print(f"[timeline] wrote SVG: {args.timeline_svg}", flush=True)
        if args.timeline_json:
            write_timeline_json(args.timeline_json, generator.last_timeline)
            print(f"[timeline] wrote JSON: {args.timeline_json}", flush=True)
    finally:
        runner = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        cleanup_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
