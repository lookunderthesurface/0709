from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mtbench_deepseek_eval import (
    DEFAULT_QUESTION_FILE,
    DEFAULT_QUESTION_URL,
    append_jsonl,
    chat_token_ids,
    load_questions,
    question_references,
    question_turns,
    read_jsonl,
    run_judging,
    selected_stop_token_id,
    stable_id,
    summarize,
    utc_now,
    write_json,
)


def generation_signature(
    args: argparse.Namespace,
    questions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_candidates = {
        candidate_id: model_path
        for candidate_id, model_path in model_candidates(args)
    }
    return {
        "system": "sglang_flashinfer_paged_greedy_ar_mtbench_baselines",
        "candidates": selected_candidates,
        "question_ids": [row["question_id"] for row in questions],
        "turn_mode": args.turn_mode,
        "system_prompt": args.system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": args.eos_token_id,
        "dtype": args.dtype,
        "backend": "sglang_flashinfer_paged_greedy_ar",
        "page_size": args.page_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "do_sample": False,
        "temperature": None,
    }


def model_candidates(args: argparse.Namespace) -> list[tuple[str, str]]:
    candidates = [
        ("ar_1b", str(args.model_1b)),
        ("ar_8b", str(args.model_8b)),
    ]
    if args.candidate == "both":
        return candidates
    selected_id = f"ar_{args.candidate}"
    return [item for item in candidates if item[0] == selected_id]


def run_one_model(
    *,
    args: argparse.Namespace,
    questions: Sequence[Mapping[str, Any]],
    run_id: str,
    candidate_id: str,
    model_path: str,
    generations_path: Path,
    generation_failures_path: Path,
    completed: set[tuple[str, str]],
) -> None:
    import torch
    from transformers import AutoTokenizer

    from atlas_0709.distributed_system import cleanup_cuda
    from atlas_0709.flashinfer_ar import FlashInferPagedGreedyARGenerator
    from atlas_0709.flashinfer_paged.sglang_runtime import (
        SGLangRunnerConfig,
        create_sglang_model_runner,
    )

    print(f"[generation] loading candidate={candidate_id} model={model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    stop_token_id = selected_stop_token_id(tokenizer, args.eos_token_id)
    runner_config = SGLangRunnerConfig(
        model_path=model_path,
        dtype=args.dtype,
        context_length=args.context_length,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
        max_total_tokens=args.max_total_tokens,
        gpu_id=args.gpu_id,
        nccl_port=args.nccl_port,
        trust_remote_code=args.trust_remote_code,
    )
    runner = create_sglang_model_runner(runner_config, initialize=True)
    generator = FlashInferPagedGreedyARGenerator(
        runner=runner,
        page_size=args.page_size,
        prefill_chunk_size=args.prefill_chunk_size,
    )

    def generate_ids(prompt_ids: Sequence[int], max_new_tokens: int):
        return generator.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop_token_id,
        )

    warmup_messages: list[dict[str, str]] = []
    if args.system_prompt:
        warmup_messages.append({"role": "system", "content": args.system_prompt})
    warmup_messages.append({"role": "user", "content": "Briefly say hello."})
    warmup_prompt_ids = chat_token_ids(tokenizer, warmup_messages)
    for warmup_index in range(args.warmup_runs):
        started = time.perf_counter()
        generate_ids(warmup_prompt_ids, min(16, args.max_new_tokens))
        torch.cuda.synchronize()
        print(
            f"[generation] candidate={candidate_id} warmup "
            f"{warmup_index + 1}/{args.warmup_runs}: "
            f"{(time.perf_counter() - started) * 1000.0:.1f} ms",
            flush=True,
        )

    total = len(questions)
    for question_index, question in enumerate(questions, start=1):
        question_id = str(question["question_id"])
        completion_key = (candidate_id, question_id)
        if completion_key in completed:
            print(
                f"[generation {question_index}/{total}] candidate={candidate_id} "
                f"q={question_id} resume-skip",
                flush=True,
            )
            continue
        messages: list[dict[str, str]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        answers: list[str] = []
        stats: list[dict[str, Any]] = []
        try:
            turns = question_turns(question, args.turn_mode)
            for turn_index, user_text in enumerate(turns, start=1):
                messages.append({"role": "user", "content": user_text})
                prompt_ids = chat_token_ids(tokenizer, messages)
                required_context = len(prompt_ids) + args.max_new_tokens
                if required_context > args.context_length:
                    raise RuntimeError(
                        f"prompt requires context_length>={required_context}, "
                        f"configured={args.context_length}"
                    )
                started = time.perf_counter()
                generation_result = generate_ids(prompt_ids, args.max_new_tokens)
                torch.cuda.synchronize()
                elapsed_s = time.perf_counter() - started
                generated_ids = generation_result.generated_token_ids
                answer = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                answers.append(answer)
                messages.append({"role": "assistant", "content": answer})
                stats.append(
                    {
                        "turn": turn_index,
                        "prompt_tokens": len(prompt_ids),
                        "generated_tokens": len(generated_ids),
                        "elapsed_s": elapsed_s,
                        "tokens_per_second": len(generated_ids) / elapsed_s if elapsed_s else None,
                        "finish_reason": generation_result.finish_reason,
                        "stop_token_id": stop_token_id,
                        "backend": generation_result.metadata,
                    }
                )
                print(
                    f"[generation {question_index}/{total}] candidate={candidate_id} "
                    f"q={question_id} turn={turn_index} tokens={len(generated_ids)} "
                    f"finish={generation_result.finish_reason} "
                    f"elapsed={elapsed_s:.3f}s",
                    flush=True,
                )
            append_jsonl(
                generations_path,
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "model_id": candidate_id,
                    "model_path": model_path,
                    "question_id": question["question_id"],
                    "category": question.get("category", "unknown"),
                    "questions": turns,
                    "references": question_references(question, len(answers)),
                    "answers": answers,
                    "choices": [{"index": 0, "turns": answers}],
                    "generation_stats": stats,
                    "decoding": {
                        "type": "autoregressive_greedy",
                        "backend": "sglang_flashinfer_paged_greedy_ar",
                        "paged_kv": True,
                        "do_sample": False,
                        "temperature": None,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "created_at": utc_now(),
                },
            )
        except Exception as exc:
            append_jsonl(
                generation_failures_path,
                {
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "question_id": question["question_id"],
                    "error": repr(exc),
                    "created_at": utc_now(),
                },
            )
            raise

    runner = None
    generator = None
    del tokenizer
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    cleanup_cuda()


def generate_ar_answers(
    *,
    args: argparse.Namespace,
    questions: Sequence[Mapping[str, Any]],
    run_id: str,
    generations_path: Path,
    generation_failures_path: Path,
) -> None:
    if not args.resume_generate:
        generations_path.unlink(missing_ok=True)
        generation_failures_path.unlink(missing_ok=True)
    existing = [
        row for row in read_jsonl(generations_path) if str(row.get("run_id")) == run_id
    ]
    completed = {
        (str(row.get("candidate_id")), str(row["question_id"]))
        for row in existing
    } if args.resume_generate else set()
    process_context = multiprocessing.get_context("spawn")
    for candidate_id, model_path in model_candidates(args):
        expected = {
            (candidate_id, str(question["question_id"]))
            for question in questions
        }
        if expected.issubset(completed):
            print(
                f"[generation] candidate={candidate_id} already complete; "
                "skipping model load",
                flush=True,
            )
            continue
        worker = process_context.Process(
            target=run_one_model,
            kwargs={
                "args": args,
                "questions": list(questions),
                "run_id": run_id,
                "candidate_id": candidate_id,
                "model_path": model_path,
                "generations_path": generations_path,
                "generation_failures_path": generation_failures_path,
                "completed": completed,
            },
            name=f"atlas-mtbench-{candidate_id}",
        )
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(
                f"FlashInfer AR worker for {candidate_id} failed "
                f"with exit code {worker.exitcode}"
            )
        completed.update(expected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score fair SGLang/FlashInfer paged greedy 1B and 8B AR baselines "
            "on MT-Bench with DeepSeek."
        )
    )
    parser.add_argument("--model-1b", default=None)
    parser.add_argument("--model-8b", default=None)
    parser.add_argument(
        "--candidate",
        choices=["1b", "8b", "both"],
        default="both",
        help="Generate one AR candidate or both sequentially.",
    )
    parser.add_argument("--questions-file", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--questions-url", default=DEFAULT_QUESTION_URL)
    parser.add_argument("--output-dir", default="../0709_outputs/mtbench_ar_baselines")
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--turn-mode", choices=["first", "conversation"], default="conversation")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--resume-generate", action="store_true")

    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--judge-reasoning-effort", choices=["high", "max"], default="high")
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=4096,
        help="DeepSeek reasoning plus final JSON budget.",
    )
    parser.add_argument("--judge-timeout", type=float, default=180.0)
    parser.add_argument("--judge-retries", type=int, default=4)
    parser.add_argument("--judge-retry-delay", type=float, default=2.0)
    parser.add_argument("--judge-repeats", type=int, default=1)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--resume-judge", action="store_true")
    parser.add_argument("--download-timeout", type=float, default=60.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate in {"1b", "both"} and not args.model_1b:
        raise SystemExit("--model-1b is required for --candidate 1b/both")
    if args.candidate in {"8b", "both"} and not args.model_8b:
        raise SystemExit("--model-8b is required for --candidate 8b/both")
    for name in (
        "max_new_tokens",
        "context_length",
        "page_size",
        "prefill_chunk_size",
        "max_running_requests",
        "max_total_tokens",
        "judge_max_tokens",
        "judge_timeout",
        "judge_repeats",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_runs < 0 or args.judge_retries < 0:
        raise SystemExit("--warmup-runs and --judge-retries must be non-negative")
    if args.skip_generate and args.resume_generate:
        raise SystemExit("--skip-generate and --resume-generate cannot be combined")
    if args.skip_judge and args.resume_judge:
        raise SystemExit("--skip-judge and --resume-judge cannot be combined")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args, output_dir)
    signature = generation_signature(args, questions)
    run_id = stable_id(signature)
    generations_path = output_dir / "generations.jsonl"
    judgments_path = output_dir / "judgments.jsonl"
    generation_failures_path = output_dir / "generation_failures.jsonl"
    judge_failures_path = output_dir / "judge_failures.jsonl"
    summary_path = output_dir / "summary.json"

    write_json(
        output_dir / f"run_{run_id}.json",
        {
            "run_id": run_id,
            "generation": signature,
            "fairness": {
                "same_questions": True,
                "same_turn_context": True,
                "same_chat_template_policy": True,
                "same_greedy_decoding": True,
                "same_max_new_tokens": True,
                "candidate_identity_hidden_from_judge": True,
                "same_judge_prompt_and_model": True,
            },
            "judge": {
                "base_url": args.judge_base_url,
                "model": args.judge_model,
                "thinking": args.judge_thinking,
                "reasoning_effort": args.judge_reasoning_effort,
                "max_tokens": args.judge_max_tokens,
                "repeats": args.judge_repeats,
                "response_format": "json_object",
            },
            "created_at": utc_now(),
        },
    )
    print(f"[run] generation_run_id={run_id}", flush=True)
    if not args.skip_judge and not os.environ.get(args.deepseek_api_key_env):
        raise SystemExit(
            f"{args.deepseek_api_key_env} is not set. Export a fresh key, "
            "or add --skip-judge."
        )
    if not args.skip_generate:
        generate_ar_answers(
            args=args,
            questions=questions,
            run_id=run_id,
            generations_path=generations_path,
            generation_failures_path=generation_failures_path,
        )
    elif not any(
        str(row.get("run_id")) == run_id for row in read_jsonl(generations_path)
    ):
        raise SystemExit(
            f"--skip-generate was set, but {generations_path} has no rows for run_id={run_id}"
        )

    judge_run_id: str | None = None
    if not args.skip_judge:
        judge_run_id = run_judging(
            args=args,
            run_id=run_id,
            generations_path=generations_path,
            judgments_path=judgments_path,
            judge_failures_path=judge_failures_path,
        )
    summary = summarize(
        run_id=run_id,
        judge_run_id=judge_run_id,
        questions=questions,
        generations_path=generations_path,
        judgments_path=judgments_path,
        generation_failures_path=generation_failures_path,
        judge_failures_path=judge_failures_path,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[summary] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
