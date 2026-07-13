from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_0709.flashinfer_full_verify import (
    FlashInferFullVerifyConfig,
    FlashInferFullVerifyRunner,
    parse_prompt_token_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke the src-integrated Direct FlashInfer full masked verify runner."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--repeat-token-id", type=int, default=42)
    parser.add_argument("--prompt", default="ATLAS full masked verify src smoke.")
    parser.add_argument("--prompt-token-ids", default=None)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--use-packed-custom-mask", action="store_true")
    parser.add_argument("--skip-logit-alignment-check", action="store_true")
    parser.add_argument("--fail-on-logit-mismatch", action="store_true")
    parser.add_argument("--alignment-atol", type=float, default=1.0)
    parser.add_argument("--alignment-rtol", type=float, default=0.05)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = FlashInferFullVerifyConfig(
        k=args.k,
        d=args.d,
        prefix_len=args.prefix_len,
        repeat_token_id=args.repeat_token_id,
        page_size=args.page_size,
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        workspace_mb=args.workspace_mb,
        flashinfer_backend=args.flashinfer_backend,
        trust_remote_code=args.trust_remote_code,
        use_packed_custom_mask=args.use_packed_custom_mask,
        check_logit_alignment=not args.skip_logit_alignment_check,
        fail_on_logit_mismatch=args.fail_on_logit_mismatch,
        alignment_atol=args.alignment_atol,
        alignment_rtol=args.alignment_rtol,
    )
    tokenizer, runner = FlashInferFullVerifyRunner.from_model_path(args.model, config=config)
    prompt_token_ids = parse_prompt_token_ids(
        tokenizer=tokenizer,
        prompt=args.prompt,
        prompt_token_ids=args.prompt_token_ids,
        prefix_len=args.prefix_len,
        repeat_token_id=args.repeat_token_id,
    )
    report = runner.benchmark(prompt_token_ids).to_dict()
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
