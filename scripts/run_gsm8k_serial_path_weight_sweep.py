#!/usr/bin/env python3
"""Controlled serial GSM8K sweep for exponential path-score weights.

Only alpha changes between runs. Both Target fallback triggers are disabled,
so alpha affects route selection without also changing fallback frequency.
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
    """Return normalized w[i] = exp(-alpha * i), i in [0, depth)."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number")
    raw = [math.exp(-alpha * index) for index in range(depth)]
    total = math.fsum(raw)
    return tuple(value / total for value in raw)


def value_tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace("+", "p").replace(".", "p")


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--drafter-model", default="/home/hwc/models/Llama-3.2-1B-Instruct")
    parser.add_argument("--target-model", default="/home/hwc/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT.parent / "0709_outputs" / "gsm8k_serial_path_weight_sweep"))
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
        description="One-variable serial GSM8K sweep: exponential path alpha -> accuracy."
    )
    add_common_eval_args(parser)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.50, 0.75, 1.0],
        help="Non-negative exponential decay coefficients to test.",
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
        "--fallback-threshold", "none",
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


def result_row(alpha: float, weights: tuple[float, ...], summary: dict[str, Any]) -> dict[str, Any]:
    speed = summary.get("serial_speed") or {}
    return {
        "alpha": alpha,
        "path_score_weights": ",".join(format(value, ".12g") for value in weights),
        "weights_sum": math.fsum(weights),
        "count": summary.get("count"),
        "correct": summary.get("correct"),
        "accuracy": summary.get("accuracy"),
        "extraction_failures": summary.get("extraction_failures"),
        "fallback_rounds": speed.get("fallback_rounds"),
        "fallback_rate": speed.get("fallback_rate"),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.k <= 0 or args.d <= 0:
        raise SystemExit("k and d must be positive")
    if any(not math.isfinite(alpha) or alpha < 0.0 for alpha in args.alphas):
        raise SystemExit("all --alphas must be finite and non-negative")
    if len(set(args.alphas)) != len(args.alphas):
        raise SystemExit("--alphas contains duplicate values")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_command = base_eval_command(args)
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for alpha in args.alphas:
        weights = exponential_weights(args.d, alpha)
        weights_text = ",".join(format(value, ".17g") for value in weights)
        run_dir = output_root / f"alpha_{value_tag(alpha)}"
        command = base_command + [
            "--output-dir", str(run_dir),
            "--path-weight-alpha", format(alpha, ".17g"),
        ]
        runs.append({"alpha": alpha, "path_score_weights": weights, "output_dir": str(run_dir), "command": command})
        print(f"\n=== alpha={alpha:g}; weights={weights_text}; sum={math.fsum(weights):.17g} ===", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(result_row(alpha, weights, summary))

    experiment = {
        "experiment": "serial_gsm8k_exponential_path_weight_sweep",
        "changed_variable": "path_weight_alpha",
        "weight_formula": "w_i = exp(-alpha * i) / sum_j exp(-alpha * j), i=0..d-1",
        "fallback_threshold": None,
        "first_token_threshold": None,
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
