#!/usr/bin/env python3
"""Controlled serial GSM8K sweep for Target fallback.

Only the weighted path-score fallback threshold changes between runs. Path
weights and every other generation setting stay fixed, while the independent
first-token fallback trigger is disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = "/home/hwc/workspace/thirdparty/grade-school-math/grade_school_math/data/test.jsonl"


def exponential_weights(depth: int, alpha: float) -> tuple[float, ...]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("path alpha must be a finite non-negative number")
    raw = [math.exp(-alpha * index) for index in range(depth)]
    total = math.fsum(raw)
    return tuple(value / total for value in raw)


def parse_threshold(value: str) -> float | None:
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("threshold must be finite or 'none'")
    return parsed


def value_tag(value: float | None) -> str:
    if value is None:
        return "disabled"
    return format(value, ".12g").replace("-", "m").replace("+", "p").replace(".", "p")


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--drafter-model", default="/home/hwc/models/Llama-3.2-1B-Instruct")
    parser.add_argument("--target-model", default="/home/hwc/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT.parent / "0709_outputs" / "gsm8k_serial_fallback_sweep"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--fallback-ar-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29500)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--target-workspace-mb", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--protocol", choices=["llama-8shot", "zero-shot"], default="llama-8shot")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict-marker", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-packed-custom-mask", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-variable serial GSM8K sweep: fallback threshold -> speed and accuracy."
    )
    add_common_eval_args(parser)
    parser.add_argument(
        "--thresholds", nargs="+", type=parse_threshold,
        default=[None, -1.0, -0.75, -0.50, -0.25],
        help="Path-score thresholds. Use 'none' for the fallback-disabled baseline.",
    )
    parser.add_argument(
        "--path-alpha", type=float, default=0.50,
        help="Fixed exponential path-weight decay used for every threshold.",
    )
    return parser


def base_eval_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "gsm8k_eval.py"),
        "--backend", "atlas_serial",
        "--model", args.drafter_model,
        "--target-model", args.target_model,
        "--data-file", args.data_file,
        "--gpu-id", str(args.gpu_id),
        "--nccl-port", str(args.nccl_port),
        "--k", str(args.k),
        "--d", str(args.d),
        "--fallback-ar-tokens", str(args.fallback_ar_tokens),
        "--max-new-tokens", str(args.max_new_tokens),
        "--context-length", str(args.context_length),
        "--page-size", str(args.page_size),
        "--prefill-chunk-size", str(args.prefill_chunk_size),
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--max-running-requests", str(args.max_running_requests),
        "--max-total-tokens", str(args.max_total_tokens),
        "--warmup-runs", str(args.warmup_runs),
        "--target-workspace-mb", str(args.target_workspace_mb),
        "--dtype", args.dtype,
        "--protocol", args.protocol,
        "--start", str(args.start),
        "--first-token-threshold", "none",
    ]
    if args.end is not None:
        command.extend(("--end", str(args.end)))
    if args.max_examples is not None:
        command.extend(("--max-examples", str(args.max_examples)))
    for enabled, flag in (
        (args.resume, "--resume"),
        (args.strict_marker, "--strict-marker"),
        (args.trust_remote_code, "--trust-remote-code"),
        (args.use_packed_custom_mask, "--use-packed-custom-mask"),
    ):
        if enabled:
            command.append(flag)
    return command


def result_row(threshold: float | None, summary: dict[str, Any]) -> dict[str, Any]:
    speed = summary.get("serial_speed") or {}
    count = summary.get("count")
    elapsed_s = summary.get("elapsed_s")
    return {
        "fallback_threshold": threshold,
        "fallback_enabled": threshold is not None,
        "count": count,
        "correct": summary.get("correct"),
        "accuracy": summary.get("accuracy"),
        "generated_tokens": summary.get("generated_tokens"),
        "fallback_rounds": speed.get("fallback_rounds"),
        "rounds": speed.get("rounds"),
        "fallback_rate": speed.get("fallback_rate"),
        "fallback_appended_tokens": speed.get("fallback_appended_tokens"),
        "end_to_end_tokens_per_second": summary.get("end_to_end_tokens_per_second"),
        "serial_overall_tokens_per_second": speed.get("overall_tokens_per_second"),
        "serial_drafter_tokens_per_second": speed.get("drafter_tokens_per_second"),
        "serial_target_tokens_per_second": speed.get("target_tokens_per_second"),
        "elapsed_s": elapsed_s,
        "mean_elapsed_s_per_example": (
            elapsed_s / count if isinstance(elapsed_s, (int, float)) and count else None
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.k <= 0 or args.d <= 0 or args.fallback_ar_tokens <= 0:
        raise SystemExit("k, d, and fallback-ar-tokens must be positive")
    if len(set(args.thresholds)) != len(args.thresholds):
        raise SystemExit("--thresholds contains duplicate values")

    weights = exponential_weights(args.d, args.path_alpha)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_command = base_eval_command(args)
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        run_dir = output_root / f"threshold_{value_tag(threshold)}"
        threshold_text = "none" if threshold is None else format(threshold, ".17g")
        command = base_command + [
            "--output-dir", str(run_dir),
            "--path-weight-alpha", format(args.path_alpha, ".17g"),
            "--fallback-threshold", threshold_text,
        ]
        runs.append({"fallback_threshold": threshold, "output_dir": str(run_dir), "command": command})
        print(
            f"\n=== fallback_threshold={threshold_text}; fixed_alpha={args.path_alpha:g}; "
            f"weights={','.join(format(value, '.17g') for value in weights)} ===",
            flush=True,
        )
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(result_row(threshold, summary))

    experiment = {
        "experiment": "serial_gsm8k_fallback_threshold_sweep",
        "changed_variable": "fallback_threshold",
        "first_token_threshold": None,
        "fixed_path_alpha": args.path_alpha,
        "fixed_path_score_weights": weights,
        "fixed_fallback_ar_tokens": args.fallback_ar_tokens,
        "runs": runs,
    }
    (output_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        return 0

    (output_root / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nConsolidated results: {output_root / 'results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
