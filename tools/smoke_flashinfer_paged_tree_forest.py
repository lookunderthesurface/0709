from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atlas_0709.flashinfer_paged.builders import (
    build_forest_depths,
    build_tree_depths,
    initialize_forest_routes,
    initialize_stage1_routes,
)
from atlas_0709.flashinfer_paged.kv import KVTreeStore
from atlas_0709.flashinfer_paged.sglang_runtime import (
    SGLangFlashInferFrontierModelBackend,
    SGLangRunnerConfig,
    create_sglang_model_runner,
    prefill_sglang_prefix,
    sglang_runner_component_report,
)
from atlas_0709.flashinfer_paged.types import DecodePhase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test real SGLang/FlashInfer paged KV tree and forest construction."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="ATLAS staged decode smoke test.")
    parser.add_argument("--prompt-token-ids", default=None, help="Comma-separated token ids. Overrides --prompt.")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", "--depth", dest="depth", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--skip-forest", action="store_true")
    parser.add_argument(
        "--legacy-token-attention-metadata",
        action="store_true",
        help="Use SGLang's legacy page_size=1 token-index updater for A/B validation.",
    )
    parser.add_argument(
        "--skip-page-layout-validation",
        action="store_true",
        help="Skip the physical page-layout assertion (benchmark diagnostics only).",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--check-hf-logits",
        action="store_true",
        help="Load a second HF model and compare every tree/forest frontier against full-history logits.",
    )
    parser.add_argument("--alignment-atol", type=float, default=1.0)
    parser.add_argument("--alignment-rtol", type=float, default=0.05)
    return parser


def parse_token_ids(raw: str | None, *, model: str, prompt: str) -> list[int]:
    if raw is not None:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer produced an empty prompt")
    return [int(token_id) for token_id in token_ids]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def required_context_length(prompt_len: int, depth: int) -> int:
    return int(prompt_len) + 2 * int(depth)


def resolve_context_length(requested: int, *, prompt_len: int, depth: int) -> int:
    required = required_context_length(prompt_len, depth)
    if int(requested) < required:
        print(
            "[warn] --context-length is too small for this workload; "
            f"using {required} instead of {int(requested)} "
            f"(prompt_len={int(prompt_len)}, d={int(depth)})."
        )
        return required
    return int(requested)


def summarize_build(build: Any) -> dict[str, Any]:
    steps = []
    for index, step in enumerate(build.steps, start=1):
        steps.append(
            {
                "step": index,
                "decoded_route_count": len(step.decoded_routes),
                "next_route_count": len(step.next_routes),
                "candidate_count": len(step.candidates),
                "selected_candidate_count": len(step.selected_candidates),
                "selected_candidate_signature": [
                    {
                        "parent_route_id": int(candidate.parent_route_id),
                        "stage1_root_id": int(candidate.stage1_root_id),
                        "pending_token_id": int(candidate.pending_token_id),
                        "rank_in_parent": int(candidate.rank_in_parent),
                        "cumulative_logprob": float(candidate.cumulative_logprob),
                    }
                    for candidate in step.selected_candidates
                ],
                "next_token_top1_ids": [
                    int(token_id)
                    for token_id in torch.argmax(
                        step.decode_output.next_token_logits,
                        dim=-1,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "logits_shape": list(step.decode_output.next_token_logits.shape),
                "model_ms": step.decode_output.model_ms,
                "metadata": dict(step.decode_output.metadata),
            }
        )
    return {
        "completed_route_count": len(build.completed_routes),
        "next_frontier_count": len(build.next_frontier_routes),
        "last_logits_shape": list(build.last_logits.shape),
        "sum_model_ms": sum(float(step.decode_output.model_ms) for step in build.steps),
        "steps": steps,
    }


@torch.inference_mode()
def check_build_alignment(
    *,
    name: str,
    build: Any,
    route_store: KVTreeStore,
    prompt_token_ids: list[int],
    reference_model: Any,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    step_reports: list[dict[str, Any]] = []
    all_passed = True
    for step_index, step in enumerate(build.steps, start=1):
        histories = [
            [
                *prompt_token_ids,
                *route_store.materialized_token_path(route),
            ]
            for route in step.decoded_routes
        ]
        lengths = {len(history) for history in histories}
        if len(lengths) != 1:
            raise RuntimeError(f"{name} step {step_index} has unequal route history lengths")
        input_ids = torch.tensor(histories, dtype=torch.long, device=reference_model.device)
        reference = reference_model(input_ids=input_ids, use_cache=False).logits[:, -1, :].float()
        candidate = step.decode_output.next_token_logits.float()
        abs_diff = (candidate - reference).abs()
        tolerance = float(atol) + float(rtol) * reference.abs()
        logits_within_tolerance = bool(torch.all(abs_diff <= tolerance).item())
        top1_match_rate = float(
            torch.argmax(candidate, dim=-1)
            .eq(torch.argmax(reference, dim=-1))
            .float()
            .mean()
            .item()
        )
        passed = bool(logits_within_tolerance and top1_match_rate == 1.0)
        all_passed = all_passed and passed
        step_reports.append(
            {
                "step": step_index,
                "passed": passed,
                "logits_within_tolerance": logits_within_tolerance,
                "batch_size": len(histories),
                "history_len": next(iter(lengths)),
                "max_abs_diff": float(abs_diff.max().item()),
                "mean_abs_diff": float(abs_diff.mean().item()),
                "top1_match_rate": top1_match_rate,
            }
        )
    return {
        "name": name,
        "passed": all_passed,
        "atol": float(atol),
        "rtol": float(rtol),
        "steps": step_reports,
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        prompt_token_ids = parse_token_ids(args.prompt_token_ids, model=args.model, prompt=args.prompt)
        context_length = resolve_context_length(
            args.context_length,
            prompt_len=len(prompt_token_ids),
            depth=args.depth,
        )
        config = SGLangRunnerConfig(
            model_path=args.model,
            dtype=args.dtype,
            context_length=context_length,
            page_size=args.page_size,
            mem_fraction_static=args.mem_fraction_static,
            max_running_requests=args.max_running_requests,
            max_total_tokens=args.max_total_tokens,
            gpu_id=args.gpu_id,
            nccl_port=args.nccl_port,
            use_page_attention_metadata=not args.legacy_token_attention_metadata,
            validate_page_layout=not args.skip_page_layout_validation,
        )
        runner = create_sglang_model_runner(config, initialize=True)
        route_store = KVTreeStore()
        frontier_backend = SGLangFlashInferFrontierModelBackend.from_runner(
            route_store=route_store,
            model_runner=runner,
            page_size=args.page_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        sync_cuda()
        prefill_start = time.perf_counter()
        prefill = prefill_sglang_prefix(
            model_runner=runner,
            route_pool=frontier_backend.route_pool,
            prompt_token_ids=prompt_token_ids,
        )
        sync_cuda()
        prefill_wall_ms = (time.perf_counter() - prefill_start) * 1000.0

        draft_prefix = frontier_backend.attach_prefilled_prefix(
            prompt_token_ids=prefill.prompt_token_ids,
            prefix_slot_ids=prefill.prefix_slot_ids,
            next_token_logits=prefill.next_token_logits,
        )
        route_store.committed_token_ids = list(prompt_token_ids)
        stage1_routes = initialize_stage1_routes(draft_prefix, k=args.k, route_store=route_store)

        sync_cuda()
        tree_start = time.perf_counter()
        tree = build_tree_depths(
            stage1_routes,
            depth=args.depth,
            k=args.k,
            route_store=route_store,
            model_backend=frontier_backend,
            phase=DecodePhase.STAGE1,
        )
        sync_cuda()
        tree_wall_ms = (time.perf_counter() - tree_start) * 1000.0

        forest = None
        forest_report = None
        if not args.skip_forest:
            forest_routes = initialize_forest_routes(
                tree.completed_routes,
                tree.last_logits,
                k=args.k,
                route_store=route_store,
            )
            sync_cuda()
            forest_start = time.perf_counter()
            forest = build_forest_depths(
                forest_routes,
                depth=args.depth,
                k=args.k,
                route_store=route_store,
                model_backend=frontier_backend,
            )
            sync_cuda()
            forest_wall_ms = (time.perf_counter() - forest_start) * 1000.0
            forest_report = {
                "initial_route_count": len(forest_routes),
                "wall_ms": forest_wall_ms,
                **summarize_build(forest),
            }

        alignment = None
        if args.check_hf_logits:
            reference_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=getattr(torch, args.dtype),
                trust_remote_code=False,
            ).to("cuda")
            reference_model.eval()
            alignment = {
                "tree": check_build_alignment(
                    name="tree",
                    build=tree,
                    route_store=route_store,
                    prompt_token_ids=prompt_token_ids,
                    reference_model=reference_model,
                    atol=args.alignment_atol,
                    rtol=args.alignment_rtol,
                ),
                "forest": (
                    check_build_alignment(
                        name="forest",
                        build=forest,
                        route_store=route_store,
                        prompt_token_ids=prompt_token_ids,
                        reference_model=reference_model,
                        atol=args.alignment_atol,
                        rtol=args.alignment_rtol,
                    )
                    if forest is not None
                    else None
                ),
            }
            if forest is not None:
                selected_stage1 = tree.completed_routes[0]
                selected_indices = [
                    index
                    for index, route in enumerate(forest.completed_routes)
                    if int(route.stage1_root_id) == int(selected_stage1.route_id)
                ]
                if len(selected_indices) != args.k:
                    raise RuntimeError(
                        "forest does not contain k completed routes under the selected stage-1 root"
                    )
                retained = [forest.completed_routes[index] for index in selected_indices]
                promoted, promotion_stats = frontier_backend.commit_stage1_and_promote(
                    committed_route=selected_stage1,
                    retained_routes=retained,
                )
                next_forest_routes = initialize_forest_routes(
                    promoted,
                    forest.last_logits[selected_indices],
                    k=args.k,
                    route_store=route_store,
                )
                post_commit = build_forest_depths(
                    next_forest_routes,
                    depth=1,
                    k=args.k,
                    route_store=route_store,
                    model_backend=frontier_backend,
                )
                alignment["post_commit_forest"] = check_build_alignment(
                    name="post_commit_forest",
                    build=post_commit,
                    route_store=route_store,
                    prompt_token_ids=list(route_store.committed_token_ids),
                    reference_model=reference_model,
                    atol=args.alignment_atol,
                    rtol=args.alignment_rtol,
                )
                alignment["promotion_stats"] = promotion_stats
            else:
                alignment["post_commit_forest"] = None
            alignment["passed"] = bool(
                alignment["tree"]["passed"]
                and (
                    alignment["forest"] is None
                    or alignment["forest"]["passed"]
                )
                and (
                    alignment["post_commit_forest"] is None
                    or alignment["post_commit_forest"]["passed"]
                )
            )

        report = {
            **sglang_runner_component_report(runner),
            "backend": "sglang_flashinfer_paged_decode",
            "reference_only": False,
            "paged_kv": True,
            "cascade": False,
            "k": args.k,
            "d": args.depth,
            "prompt_len": len(prompt_token_ids),
            "attention_metadata_mode": (
                "legacy_token_page_size_1"
                if args.legacy_token_attention_metadata
                else f"physical_page_size_{args.page_size}"
            ),
            "page_layout_validation": not args.skip_page_layout_validation,
            "requested_context_length": args.context_length,
            "effective_context_length": context_length,
            "required_context_length": required_context_length(len(prompt_token_ids), args.depth),
            "prefill": {
                **prefill.forward_metadata,
                "wall_ms": prefill_wall_ms,
                "req_pool_index": prefill.req_pool_index,
                "prefix_slot_shape": list(prefill.prefix_slot_ids.shape),
                "next_token_logits_shape": list(prefill.next_token_logits.shape),
            },
            "tree": {
                "initial_route_count": len(stage1_routes),
                "wall_ms": tree_wall_ms,
                **summarize_build(tree),
            },
            "forest": forest_report,
            "logit_alignment": alignment,
            "route_pool": {
                "req_pool_row_count": len(frontier_backend.route_pool.route_rows),
                "node_slot_count": len(frontier_backend.route_pool.node_slot_ids),
                "route_slot_path_count": len(frontier_backend.route_pool.route_slot_paths),
                "owned_page_count": len(frontier_backend.route_pool.owned_page_ids),
                "cow_pages_copied": frontier_backend.route_pool.cow_pages_copied,
                "cow_tokens_copied": frontier_backend.route_pool.cow_tokens_copied,
                "prefix_slot_count": int(prefill.prefix_slot_ids.numel()),
                "max_running_requests": args.max_running_requests,
                "max_total_tokens": args.max_total_tokens,
                "hot_path_counters": frontier_backend.route_pool.hot_path_counters(),
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if args.json_out:
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[json] wrote {output}", flush=True)
        if alignment is not None and not alignment["passed"]:
            raise RuntimeError(f"route logits are not aligned: {alignment}")
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
