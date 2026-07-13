from __future__ import annotations

import argparse
import json
from typing import Sequence

from .backends import DeterministicMockBackend
from .controller import AtlasCleanConfig, AtlasCleanGenerator
from .hf_backend import HFRecomputeBatchBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the clean ATLAS staged decode controller.")
    parser.add_argument("--backend", choices=["mock", "hf"], default="mock")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--prompt-token-ids", default="1,2,3,4")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--drafter-model", default=None)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--mock-vocab-size", type=int, default=128)
    return parser


def parse_token_ids(raw: str) -> list[int]:
    token_ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not token_ids:
        raise ValueError("prompt-token-ids cannot be empty")
    return token_ids


def make_backends(args) -> tuple[object, object, list[int]]:
    if args.backend == "mock":
        prompt_token_ids = parse_token_ids(args.prompt_token_ids)
        return (
            DeterministicMockBackend(name="drafter", vocab_size=args.mock_vocab_size, salt=3),
            DeterministicMockBackend(name="target", vocab_size=args.mock_vocab_size, salt=19),
            prompt_token_ids,
        )

    if args.drafter_model is None or args.target_model is None:
        raise RuntimeError("--backend hf requires --drafter-model and --target-model")
    drafter = HFRecomputeBatchBackend(
        model_path=args.drafter_model,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    target = HFRecomputeBatchBackend(
        model_path=args.target_model,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.prompt is not None:
        prompt_token_ids = drafter.encode_prompt(args.prompt)
    else:
        prompt_token_ids = parse_token_ids(args.prompt_token_ids)
    return drafter, target, prompt_token_ids


def result_to_jsonable(result, *, drafter=None) -> dict[str, object]:
    data = {
        "prompt_token_ids": list(result.prompt_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "token_ids": result.token_ids,
        "rounds": [
            {
                "round_index": trace.round_index,
                "selected_route_id": trace.selected_route_id,
                "committed_tokens": list(trace.committed_tokens),
                "forest_depth": trace.forest_depth,
                "verify_returned_before_forest_done": trace.verify_returned_before_forest_done,
                "target_scores": [
                    {"route_id": route_id, "target_logprob": score}
                    for route_id, score in trace.target_scores
                ],
            }
            for trace in result.rounds
        ],
    }
    if drafter is not None and hasattr(drafter, "decode_tokens"):
        data["text"] = drafter.decode_tokens(result.token_ids)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    drafter, target, prompt_token_ids = make_backends(args)
    config = AtlasCleanConfig(
        k=args.k,
        d=args.d,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=args.eos_token_id,
    )
    generator = AtlasCleanGenerator(config=config, drafter=drafter, target=target)
    result = generator.generate(prompt_token_ids)
    report = {
        "config": {
            "k": config.k,
            "d": config.d,
            "max_new_tokens": config.max_new_tokens,
            "eos_token_id": config.eos_token_id,
            "tree_decode": f"ordinary_batch_{config.k}",
            "forest_decode": f"ordinary_batch_{config.k * config.k}",
            "target_verify": "ordinary_batch_stage1_paths",
        },
        "result": result_to_jsonable(result, drafter=drafter),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

