from __future__ import annotations

import argparse

from .flashinfer_full_verify import FlashInferFullVerifyConfig
from .rpc import TargetServerApp, serve_target
from .target_runtime import DirectFlashInferMaskedTreeVerifyBackend


def parse_score_weights(raw: str | None) -> tuple[float, ...] | None:
    if raw is None or not raw.strip():
        return None
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        return None
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ATLAS target verify server.")
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Use 0.0.0.0 for remote clients.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--repeat-token-id", type=int, default=42)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--use-packed-custom-mask", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--path-score-weights",
        default=None,
        help=(
            "Comma-separated per-token path score weights, e.g. 0.45,0.30,0.17,0.08. "
            "Weights are normalized internally. Omit to keep the original unweighted path logprob."
        ),
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=None,
        help=(
            "Reject all draft paths and run Target AR if the best selection score is below this value. "
            "With --path-score-weights this is a weighted per-token logprob; otherwise it is the path logprob sum."
        ),
    )
    parser.add_argument(
        "--first-token-threshold",
        type=float,
        default=None,
        help="Reject all draft paths and run Target AR if the selected path's first token logprob is below this value.",
    )
    parser.add_argument(
        "--fallback-ar-tokens",
        type=int,
        default=4,
        help="Maximum number of greedy Target AR tokens to return on fallback.",
    )
    parser.add_argument(
        "--profile-fallback-ar",
        action="store_true",
        help=(
            "Synchronize CUDA to report fallback wrapper setup, decode planning, and model-forward "
            "times. Disabled by default because the synchronization changes production timing."
        ),
    )
    parser.add_argument(
        "--route-selection-policy",
        choices=["target_best", "first_route"],
        default="target_best",
        help=(
            "Route decision policy. first_route is a correctness diagnostic that "
            "always commits the first payload route and requires fallback thresholds "
            "to be omitted."
        ),
    )
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
        warmup=1,
        iters=1,
        workspace_mb=args.workspace_mb,
        flashinfer_backend=args.flashinfer_backend,
        trust_remote_code=args.trust_remote_code,
        use_packed_custom_mask=args.use_packed_custom_mask,
        check_logit_alignment=False,
    )
    backend = DirectFlashInferMaskedTreeVerifyBackend(
        model_path=args.model,
        config=config,
        score_weights=parse_score_weights(args.path_score_weights),
        fallback_threshold=args.fallback_threshold,
        first_token_threshold=args.first_token_threshold,
        fallback_ar_tokens=args.fallback_ar_tokens,
        profile_fallback_ar=args.profile_fallback_ar,
        route_selection_policy=args.route_selection_policy,
    )
    serve_target(TargetServerApp(backend), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
