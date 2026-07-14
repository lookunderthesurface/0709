from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DATA_FILE = "/home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl"
DEFAULT_SYSTEM_PROMPT = ""
LLAMA_FINAL_RE = re.compile(r"The final answer is\s+((?:-?[$0-9.,]{2,})|(?:-?[0-9]+))")
LLAMA_FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"),
]
FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def optional_float(value: str) -> float | None:
    """Parse a float while allowing an explicit disabled value on the CLI."""
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite or one of: none, off, disabled")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def exponential_path_weights(depth: int, alpha: float) -> tuple[float, ...]:
    """Return normalized w[i] = exp(-alpha * i), i in [0, depth)."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    raw = [math.exp(-alpha * index) for index in range(depth)]
    total = math.fsum(raw)
    return tuple(value / total for value in raw)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("question"), str):
                raise RuntimeError(f"invalid GSM8K row at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    match = NUMBER_RE.search(value.replace(",", ""))
    if not match:
        return None
    text = match.group(0)
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number == 0:
        return "0"
    if number.is_integer():
        return str(int(number))
    return format(number, ".15g")


def extract_answer(text: str, *, allow_last_number: bool) -> str | None:
    llama = LLAMA_FINAL_RE.findall(text)
    if llama:
        return normalize_number(llama[-1])
    marked = FINAL_ANSWER_RE.findall(text)
    if marked:
        return normalize_number(marked[-1])
    if allow_last_number:
        numbers = NUMBER_RE.findall(text.replace(",", ""))
        return normalize_number(numbers[-1]) if numbers else None
    return None


def build_messages(question: str, args: argparse.Namespace) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    if args.protocol == "llama-8shot":
        instruction = (
            "Given the following problem, reason and give a final answer to the problem.\n"
            f"Problem: {question}\nYour response should end with \"The final answer is [answer]\" "
            "where [answer] is the response to the problem.\n"
        )
        for sample_question, sample_answer in LLAMA_FEWSHOT:
            sample_instruction = (
                "Given the following problem, reason and give a final answer to the problem.\n"
                f"Problem: {sample_question}\nYour response should end with \"The final answer is [answer]\" "
                "where [answer] is the response to the problem.\n"
            )
            messages.extend(({"role": "user", "content": sample_instruction},
                             {"role": "assistant", "content": sample_answer}))
        messages.append({"role": "user", "content": instruction})
    else:
        messages.append({"role": "user", "content": question})
    return messages


def chat_token_ids(tokenizer: Any, question: str, args: argparse.Namespace) -> list[int]:
    messages = build_messages(question, args)
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except (AttributeError, ValueError):
        rendered = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        encoded = tokenizer.encode(rendered + "\nAssistant:", add_special_tokens=True)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def stop_token_id(tokenizer: Any, override: int | None) -> int | None:
    if override is not None:
        return override
    token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(token_id, int) and token_id >= 0 and token_id != getattr(tokenizer, "unk_token_id", None):
        return token_id
    return getattr(tokenizer, "eos_token_id", None)


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.data_file)
    if not path.is_file():
        raise SystemExit(f"GSM8K data file does not exist: {path}")
    rows = read_jsonl(path)
    end = len(rows) if args.end is None else min(args.end, len(rows))
    selected = rows[max(0, args.start):end]
    if args.max_examples is not None:
        selected = selected[:args.max_examples]
    if not selected:
        raise SystemExit("selected GSM8K range is empty")
    return selected


def create_ar(args: argparse.Namespace):
    from transformers import AutoTokenizer
    from atlas_0709.flashinfer_ar import FlashInferPagedGreedyARGenerator
    from atlas_0709.flashinfer_paged.sglang_runtime import SGLangRunnerConfig, create_sglang_model_runner

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    runner = create_sglang_model_runner(SGLangRunnerConfig(
        model_path=args.model, dtype=args.dtype, context_length=args.context_length,
        page_size=args.page_size, mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests, max_total_tokens=args.max_total_tokens,
        gpu_id=args.gpu_id, nccl_port=args.nccl_port,
        trust_remote_code=args.trust_remote_code,
    ), initialize=True)
    generator = FlashInferPagedGreedyARGenerator(
        runner=runner, page_size=args.page_size, prefill_chunk_size=args.prefill_chunk_size
    )
    eos = stop_token_id(tokenizer, args.eos_token_id)

    def generate(prompt_ids: Sequence[int], max_new_tokens: int | None = None):
        result = generator.generate(prompt_ids, max_new_tokens=max_new_tokens or args.max_new_tokens, eos_token_id=eos)
        return result.generated_token_ids, {"finish_reason": result.finish_reason, "backend": result.metadata}

    return tokenizer, generate, runner


def create_atlas(args: argparse.Namespace, *, serial: bool = False):
    from transformers import AutoTokenizer
    from atlas_0709.distributed_system import DistributedAtlasConfig, PagedDistributedAtlasGenerator
    from atlas_0709.flashinfer_paged.sglang_runtime import SGLangRunnerConfig, create_sglang_model_runner
    from atlas_0709.rpc import InProcessTargetClient, RemoteTargetClient

    atlas_generator_class = PagedDistributedAtlasGenerator
    if serial:
        from atlas_0709.serial_system import SerialAtlasGenerator

        atlas_generator_class = SerialAtlasGenerator

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if serial:
        from atlas_0709.flashinfer_full_verify import FlashInferFullVerifyConfig
        from atlas_0709.target_runtime import DirectFlashInferMaskedTreeVerifyBackend

        weights = tuple(
            float(value.strip())
            for value in args.path_score_weights.split(",")
            if value.strip()
        )
        target = InProcessTargetClient(
            DirectFlashInferMaskedTreeVerifyBackend(
                model_path=args.target_model,
                config=FlashInferFullVerifyConfig(
                    k=args.k, d=args.d, prefix_len=args.context_length,
                    page_size=args.page_size, dtype=args.dtype, device="cuda",
                    workspace_mb=args.target_workspace_mb,
                    trust_remote_code=args.trust_remote_code,
                    use_packed_custom_mask=args.use_packed_custom_mask,
                    check_logit_alignment=False,
                ),
                score_weights=weights,
                fallback_threshold=args.fallback_threshold,
                first_token_threshold=args.first_token_threshold,
                fallback_ar_tokens=args.fallback_ar_tokens,
                route_selection_policy=args.route_selection_policy,
            )
        )
    else:
        target = RemoteTargetClient(args.target_url, timeout=args.target_timeout)
    health = target.health()
    runner = create_sglang_model_runner(SGLangRunnerConfig(
        model_path=args.model, dtype=args.dtype, context_length=args.context_length,
        page_size=args.page_size, mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests, max_total_tokens=args.max_total_tokens,
        gpu_id=args.gpu_id, nccl_port=args.nccl_port,
        trust_remote_code=args.trust_remote_code,
    ), initialize=True)
    eos = stop_token_id(tokenizer, args.eos_token_id)

    def generate(prompt_ids: Sequence[int], max_new_tokens: int | None = None):
        generator = atlas_generator_class(
            config=DistributedAtlasConfig(
                k=args.k,
                d=args.d,
                max_new_tokens=max_new_tokens or args.max_new_tokens,
                eos_token_id=eos,
                fallback_ar_tokens=args.fallback_ar_tokens,
                fixed_forest_depth=args.fixed_forest_depth,
                validate_state_alignment=args.validate_state_alignment,
            ),
            runner=runner, page_size=args.page_size, prefill_chunk_size=args.prefill_chunk_size,
            target_client=target, tokenizer=None,
        )
        result = generator.generate(prompt_ids)
        return result.generated_token_ids, {
            "rounds": len(result.rounds),
            "target_health": health,
            "execution_mode": result.metadata.get("execution_mode", "async_distributed"),
            "generator_metadata": result.metadata,
        }

    return tokenizer, generate, runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact-match GSM8K evaluation for ATLAS 0709 and AR baselines.")
    parser.add_argument("--backend", choices=["ar", "atlas", "atlas_serial"], required=True)
    parser.add_argument("--model", required=True, help="AR model or ATLAS drafter model")
    parser.add_argument("--target-url", default="http://127.0.0.1:18090")
    parser.add_argument("--target-model", default="/home/hwc/models/Meta-Llama-3.1-8B-Instruct",
                        help="In-process Target model used by atlas_serial; no Target server is needed")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--protocol", choices=["llama-8shot", "zero-shot"], default="llama-8shot",
                        help="Default reproduces Meta/lm-eval gsm8k_cot_llama")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-marker", action="store_true", help="Count answers without #### as extraction failures")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument(
        "--fixed-forest-depth",
        type=int,
        default=None,
        help=(
            "For async correctness/replay runs, build exactly this many forest "
            "depths per round. Ignored by atlas_serial."
        ),
    )
    parser.add_argument(
        "--validate-state-alignment",
        action="store_true",
        help="Assert cross-model prefix invariants after every active ATLAS handoff.",
    )
    parser.add_argument("--fallback-ar-tokens", type=int, default=4)
    parser.add_argument("--path-score-weights", default="0.45,0.30,0.17,0.08")
    parser.add_argument(
        "--path-weight-alpha", type=non_negative_float,
        help=(
            "Use normalized exponential weights w[i]=exp(-alpha*i)/sum(w); "
            "when set, this overrides --path-score-weights"
        ),
    )
    parser.add_argument(
        "--fallback-threshold", type=optional_float, default=-0.50,
        help="Weighted path-score fallback threshold; use 'none' to disable this trigger",
    )
    parser.add_argument(
        "--first-token-threshold", type=optional_float, default=-0.70,
        help="First-token fallback threshold; use 'none' to disable this trigger",
    )
    parser.add_argument("--target-workspace-mb", type=int, default=128)
    parser.add_argument("--use-packed-custom-mask", action="store_true")
    parser.add_argument(
        "--route-selection-policy",
        choices=["target_best", "first_route"],
        default="target_best",
        help=(
            "Target route policy for atlas_serial. For remote atlas runs this is "
            "configured on the Target server and recorded from /health."
        ),
    )
    parser.add_argument("--target-timeout", type=float, default=600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for name in ("max_new_tokens", "context_length", "page_size", "prefill_chunk_size", "max_total_tokens", "k", "d"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.path_weight_alpha is not None:
        resolved_weights = exponential_path_weights(args.d, args.path_weight_alpha)
        args.path_score_weights = ",".join(format(value, ".17g") for value in resolved_weights)
    examples = load_examples(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    failures_path = output_dir / "failures.jsonl"
    if not args.resume:
        predictions_path.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed = {int(row["index"]) for row in existing}

    import torch
    from atlas_0709.distributed_system import cleanup_cuda
    if args.backend == "ar":
        tokenizer, generate, runner = create_ar(args)
    elif args.backend == "atlas_serial":
        tokenizer, generate, runner = create_atlas(args, serial=True)
    else:
        tokenizer, generate, runner = create_atlas(args)
    warmup_ids = chat_token_ids(tokenizer, "Jan has 2 apples and buys 3 more. How many apples?", args)
    for _ in range(args.warmup_runs):
        generate(warmup_ids, min(32, args.max_new_tokens))
    torch.cuda.synchronize()

    try:
        for offset, example in enumerate(examples):
            index = args.start + offset
            if index in completed:
                continue
            gold = extract_answer(str(example.get("answer", "")), allow_last_number=False)
            if gold is None:
                raise RuntimeError(f"gold answer at index {index} has no valid #### marker")
            try:
                prompt_ids = chat_token_ids(tokenizer, example["question"], args)
                if len(prompt_ids) + args.max_new_tokens > args.context_length:
                    raise RuntimeError("prompt plus generation exceeds context length")
                started = time.perf_counter()
                generated_ids, metadata = generate(prompt_ids)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                text = tokenizer.decode(generated_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False).strip()
                prediction = extract_answer(text, allow_last_number=not args.strict_marker)
                row = {"index": index, "backend": args.backend, "model": args.model,
                       "question": example["question"], "gold": gold, "prediction": prediction,
                       "correct": prediction == gold, "response": text,
                       "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated_ids),
                       "elapsed_s": elapsed,
                       "tokens_per_second": len(generated_ids) / elapsed if elapsed else None,
                       "metadata": metadata, "created_at": utc_now()}
                append_jsonl(predictions_path, row)
                print(f"[{offset + 1}/{len(examples)}] index={index} pred={prediction} gold={gold} "
                      f"correct={row['correct']} tokens={len(generated_ids)}", flush=True)
            except Exception as exc:
                append_jsonl(failures_path, {"index": index, "error": repr(exc), "created_at": utc_now()})
                raise
    finally:
        runner = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        cleanup_cuda()

    rows = [row for row in read_jsonl(predictions_path) if args.start <= int(row["index"]) and
            (args.end is None or int(row["index"]) < args.end)]
    if args.max_examples is not None:
        wanted = {args.start + i for i in range(len(examples))}
        rows = [row for row in rows if int(row["index"]) in wanted]
    speeds = [float(row["tokens_per_second"]) for row in rows if row.get("tokens_per_second") is not None]
    serial_metrics = []
    if args.backend == "atlas_serial":
        serial_metrics = [
            row.get("metadata", {}).get("generator_metadata", {}) for row in rows
        ]

    def aggregate_serial_speed(time_key: str) -> float | None:
        elapsed = sum(float(meta.get(time_key, 0.0)) for meta in serial_metrics)
        tokens = sum(int(meta.get("generated_tokens", 0)) for meta in serial_metrics)
        return tokens / elapsed if elapsed else None

    serial_tokens = sum(int(meta.get("generated_tokens", 0)) for meta in serial_metrics)
    serial_rounds = sum(int(meta.get("rounds", 0)) for meta in serial_metrics)
    serial_fallback_rounds = sum(
        int(meta.get("fallback_rounds", 0)) for meta in serial_metrics
    )
    serial_drafter_elapsed_s = sum(
        float(meta.get("drafter_elapsed_s", 0.0)) for meta in serial_metrics
    )
    serial_target_elapsed_s = sum(
        float(meta.get("target_elapsed_s", 0.0)) for meta in serial_metrics
    )
    serial_total_elapsed_s = sum(
        float(meta.get("total_elapsed_s", 0.0)) for meta in serial_metrics
    )
    serial_fallback_appended_tokens = sum(
        int(meta.get("fallback_appended_tokens", 0)) for meta in serial_metrics
    )
    serial_fallback_append_elapsed_s = sum(
        float(meta.get("fallback_append_elapsed_s", 0.0)) for meta in serial_metrics
    )

    correct = sum(bool(row["correct"]) for row in rows)
    total_generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in rows)
    total_elapsed_s = sum(float(row.get("elapsed_s", 0.0)) for row in rows)
    summary = {"backend": args.backend, "model": args.model, "data_file": args.data_file,
               "count": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None,
               "extraction_failures": sum(row.get("prediction") is None for row in rows),
               "mean_tokens_per_second": statistics.fmean(speeds) if speeds else None,
               "generated_tokens": total_generated_tokens,
               "elapsed_s": total_elapsed_s,
               "end_to_end_tokens_per_second": (
                   total_generated_tokens / total_elapsed_s if total_elapsed_s else None
               ),
               "serial_speed": ({
                   "generated_tokens": serial_tokens,
                   "rounds": serial_rounds,
                   "fallback_rounds": serial_fallback_rounds,
                   "fallback_rate": (
                       serial_fallback_rounds / serial_rounds if serial_rounds else None
                   ),
                   "drafter_elapsed_s": serial_drafter_elapsed_s,
                   "target_elapsed_s": serial_target_elapsed_s,
                   "total_elapsed_s": serial_total_elapsed_s,
                   "fallback_appended_tokens": serial_fallback_appended_tokens,
                   "fallback_append_elapsed_s": serial_fallback_append_elapsed_s,
                   "fallback_append_tokens_per_second": (
                       serial_fallback_appended_tokens / serial_fallback_append_elapsed_s
                       if serial_fallback_append_elapsed_s else None
                   ),
                   "fallback_drafter_append_mode": "single_multi_token_extend",
                   "drafter_tokens_per_second": aggregate_serial_speed("drafter_elapsed_s"),
                   "target_tokens_per_second": aggregate_serial_speed("target_elapsed_s"),
                   "overall_tokens_per_second": aggregate_serial_speed("total_elapsed_s"),
                   "token_definition": "committed_generated_tokens",
                   "excludes_target_prompt_prefill": True,
               } if serial_metrics else None),
               "settings": {"protocol": args.protocol, "num_fewshot": 8 if args.protocol == "llama-8shot" else 0,
                            "max_new_tokens": args.max_new_tokens, "strict_marker": args.strict_marker,
                             "k": args.k if args.backend.startswith("atlas") else None,
                             "d": args.d if args.backend.startswith("atlas") else None,
                             "fixed_forest_depth": (
                                 args.fixed_forest_depth if args.backend == "atlas" else None
                             ),
                            "path_score_weights": (
                                args.path_score_weights if args.backend.startswith("atlas") else None
                            ),
                            "path_weight_alpha": (
                                args.path_weight_alpha if args.backend.startswith("atlas") else None
                            ),
                            "fallback_threshold": (
                                args.fallback_threshold if args.backend.startswith("atlas") else None
                            ),
                             "first_token_threshold": (
                                 args.first_token_threshold if args.backend.startswith("atlas") else None
                             ),
                             "fallback_ar_tokens": (
                                 args.fallback_ar_tokens if args.backend.startswith("atlas") else None
                             ),
                             "route_selection_policy": (
                                 args.route_selection_policy if args.backend.startswith("atlas") else None
                             )}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
