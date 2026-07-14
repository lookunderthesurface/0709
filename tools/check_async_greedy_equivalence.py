from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Sequence


# This must be set before the first CUDA context is created. It is recorded in
# the report even though the current ATLAS generation path uses no sampling.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


DEFAULT_PROMPTS = (
    "Explain why the sky appears blue in two concise sentences.",
    "Compute 37 * 19 step by step and state the final answer.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that k=1 async ATLAS with a first-route Target is exactly "
            "equivalent to the Drafter's independent greedy paged AR."
        )
    )
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--target-url", default="http://127.0.0.1:18109")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text; repeat this option for multiple cases.",
    )
    parser.add_argument(
        "--prompt-token-ids",
        action="append",
        default=None,
        help="Comma-separated exact token IDs; repeat for multiple cases.",
    )
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument(
        "--forest-depths",
        default="0,1,2,3,4",
        help="Comma-separated fixed forest depths to test; every value must be in [0,d].",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--target-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help=(
            "Ask PyTorch to reject known nondeterministic operations. FlashInfer "
            "custom kernels may not be covered, so fresh-process replay is still required."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out", required=True)
    return parser


def configure_reproducibility(seed: int, *, deterministic_algorithms: bool) -> dict[str, object]:
    import torch

    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        numpy_seeded = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": numpy_seeded,
        "torch_seeded": True,
        "cuda_seeded": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(deterministic_algorithms),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32": False,
        "algorithmic_sampling": False,
    }


def encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": str(prompt)}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError):
        encoded = tokenizer.encode(
            f"User: {prompt}\nAssistant:",
            add_special_tokens=True,
        )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def parse_token_ids(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("prompt-token-ids cannot be empty")
    return values


def token_hash(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def first_mismatch(left: Sequence[int], right: Sequence[int]) -> dict[str, int | None] | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return {
                "index": index,
                "ar_token_id": int(left_token),
                "atlas_token_id": int(right_token),
            }
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "ar_token_id": None if len(left) <= len(right) else int(left[len(right)]),
            "atlas_token_id": None if len(right) <= len(left) else int(right[len(left)]),
        }
    return None


def main() -> int:
    args = build_parser().parse_args()
    if args.d <= 0 or args.max_new_tokens <= 0 or args.repeats <= 0:
        raise SystemExit("d, max-new-tokens, and repeats must be positive")
    forest_depths = [
        int(part.strip()) for part in args.forest_depths.split(",") if part.strip()
    ]
    if not forest_depths or any(depth < 0 or depth > args.d for depth in forest_depths):
        raise SystemExit("every forest depth must be between 0 and d")

    reproducibility = configure_reproducibility(
        args.seed,
        deterministic_algorithms=args.deterministic_algorithms,
    )

    import torch
    from transformers import AutoTokenizer

    from atlas_0709.distributed_system import (
        DistributedAtlasConfig,
        PagedDistributedAtlasGenerator,
        cleanup_cuda,
    )
    from atlas_0709.flashinfer_ar import FlashInferPagedGreedyARGenerator
    from atlas_0709.flashinfer_paged.sglang_runtime import (
        SGLangRunnerConfig,
        create_sglang_model_runner,
        sglang_runner_component_report,
    )
    from atlas_0709.rpc import RemoteTargetClient

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    target = RemoteTargetClient(args.target_url, timeout=args.target_timeout)
    health = target.health()
    target_metadata = dict(health.get("metadata", {}))
    if target_metadata.get("route_selection_policy") != "first_route":
        raise SystemExit(
            "Target must be started with --route-selection-policy first_route"
        )
    if target_metadata.get("fallback_threshold") is not None or target_metadata.get(
        "first_token_threshold"
    ) is not None:
        raise SystemExit("Target first-route diagnostic must have fallback disabled")

    tokenizer = AutoTokenizer.from_pretrained(
        args.drafter_model,
        trust_remote_code=args.trust_remote_code,
    )
    target_tokenizer_vocab_size = target_metadata.get("tokenizer_vocab_size")
    if (
        target_tokenizer_vocab_size is not None
        and int(target_tokenizer_vocab_size) != int(len(tokenizer))
    ):
        raise SystemExit(
            "Drafter and Target tokenizer vocab sizes differ: "
            f"drafter={len(tokenizer)}, target={target_tokenizer_vocab_size}"
        )
    target_eos_token_id = target_metadata.get("tokenizer_eos_token_id")
    if (
        target_eos_token_id is not None
        and tokenizer.eos_token_id is not None
        and int(target_eos_token_id) != int(tokenizer.eos_token_id)
    ):
        raise SystemExit(
            "Drafter and Target tokenizer EOS ids differ: "
            f"drafter={tokenizer.eos_token_id}, target={target_eos_token_id}"
        )
    prompt_cases: list[tuple[str, list[int]]] = []
    for index, raw in enumerate(args.prompt_token_ids or []):
        prompt_cases.append((f"token_ids_{index}", parse_token_ids(raw)))
    prompts = (
        args.prompt
        if args.prompt is not None
        else (() if args.prompt_token_ids else DEFAULT_PROMPTS)
    )
    for index, prompt in enumerate(prompts):
        prompt_cases.append((f"prompt_{index}", encode_prompt(tokenizer, prompt)))

    runner = None
    try:
        runner = create_sglang_model_runner(
            SGLangRunnerConfig(
                model_path=args.drafter_model,
                dtype=args.dtype,
                context_length=args.context_length,
                page_size=args.page_size,
                mem_fraction_static=args.mem_fraction_static,
                max_running_requests=args.max_running_requests,
                max_total_tokens=args.max_total_tokens,
                gpu_id=args.gpu_id,
                nccl_port=args.nccl_port,
                trust_remote_code=args.trust_remote_code,
            ),
            initialize=True,
        )
        ar_generator = FlashInferPagedGreedyARGenerator(
            runner=runner,
            page_size=args.page_size,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        case_reports: list[dict[str, object]] = []
        all_exact = True
        for case_name, prompt_token_ids in prompt_cases:
            if len(prompt_token_ids) + args.max_new_tokens > args.context_length:
                raise RuntimeError(f"{case_name} exceeds context length")

            ar_runs: list[list[int]] = []
            for _ in range(args.repeats):
                configure_reproducibility(
                    args.seed,
                    deterministic_algorithms=args.deterministic_algorithms,
                )
                result = ar_generator.generate(
                    prompt_token_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=args.eos_token_id,
                )
                torch.cuda.synchronize()
                ar_runs.append([int(token_id) for token_id in result.generated_token_ids])

            depth_reports: list[dict[str, object]] = []
            for forest_depth in forest_depths:
                atlas_runs: list[list[int]] = []
                trace_runs: list[list[dict[str, object]]] = []
                for _ in range(args.repeats):
                    configure_reproducibility(
                        args.seed,
                        deterministic_algorithms=args.deterministic_algorithms,
                    )
                    generator = PagedDistributedAtlasGenerator(
                        config=DistributedAtlasConfig(
                            k=1,
                            d=args.d,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=args.eos_token_id,
                            fallback_ar_tokens=1,
                            fixed_forest_depth=forest_depth,
                            validate_state_alignment=True,
                        ),
                        runner=runner,
                        page_size=args.page_size,
                        prefill_chunk_size=args.prefill_chunk_size,
                        target_client=target,
                        tokenizer=None,
                    )
                    result = generator.generate(prompt_token_ids)
                    torch.cuda.synchronize()
                    atlas_runs.append(
                        [int(token_id) for token_id in result.generated_token_ids]
                    )
                    trace_runs.append(
                        [
                            {
                                "round": int(trace.round_index),
                                "forest_depth": int(trace.forest_depth),
                                "handoff_mode": str(trace.handoff_mode),
                                "committed_tokens": [
                                    int(token_id) for token_id in trace.committed_tokens
                                ],
                                "target_prefix_len_before": int(
                                    trace.target_prefix_len_before
                                ),
                                "target_prefix_len_after": int(
                                    trace.target_prefix_len_after
                                ),
                            }
                            for trace in result.rounds
                        ]
                    )

                paired_exact = [
                    ar_runs[index] == atlas_runs[index]
                    for index in range(args.repeats)
                ]
                atlas_repeat_exact = all(
                    run == atlas_runs[0] for run in atlas_runs[1:]
                )
                depth_exact = all(paired_exact) and atlas_repeat_exact
                all_exact = all_exact and depth_exact
                depth_reports.append(
                    {
                        "fixed_forest_depth": forest_depth,
                        "exact_match_by_repeat": paired_exact,
                        "atlas_repeat_exact": atlas_repeat_exact,
                        "atlas_token_hashes": [token_hash(run) for run in atlas_runs],
                        "first_mismatch_by_repeat": [
                            first_mismatch(ar_runs[index], atlas_runs[index])
                            for index in range(args.repeats)
                        ],
                        "trace_runs": trace_runs,
                    }
                )

            ar_repeat_exact = all(run == ar_runs[0] for run in ar_runs[1:])
            all_exact = all_exact and ar_repeat_exact
            case_reports.append(
                {
                    "name": case_name,
                    "prompt_length": len(prompt_token_ids),
                    "prompt_token_hash": token_hash(prompt_token_ids),
                    "ar_repeat_exact": ar_repeat_exact,
                    "ar_token_hashes": [token_hash(run) for run in ar_runs],
                    "generated_tokens": [len(run) for run in ar_runs],
                    "depths": depth_reports,
                }
            )

        report = {
            "passed": bool(all_exact),
            "claim": "k1_first_route_async_equals_drafter_greedy_ar",
            "drafter_model": args.drafter_model,
            "target_health": health,
            "k": 1,
            "d": args.d,
            "forest_depths": forest_depths,
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "eos_token_id": args.eos_token_id,
            "page_size": args.page_size,
            "dtype": args.dtype,
            "reproducibility": reproducibility,
            "runner": sglang_runner_component_report(runner),
            "cases": case_reports,
        }
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if not all_exact:
            raise RuntimeError("k=1 first-route ATLAS diverged from Drafter greedy AR")
        return 0
    finally:
        runner = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        cleanup_cuda()


if __name__ == "__main__":
    raise SystemExit(main())
