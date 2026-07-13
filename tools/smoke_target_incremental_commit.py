from __future__ import annotations

import argparse
import json

import torch

from atlas_0709.flashinfer_full_verify import FlashInferFullVerifyConfig, prefill_cache
from atlas_0709.target_runtime import (
    DirectFlashInferMaskedTreeVerifyBackend,
    VerifyRoutePayload,
)


def candidate_routes(logits: torch.Tensor, *, depth: int, vocab_size: int) -> list[VerifyRoutePayload]:
    seeds = torch.topk(logits.float(), k=3, dim=-1).indices.detach().cpu().tolist()
    shared = int(seeds[0])
    paths = [
        tuple([shared, *[(shared + offset + 1) % vocab_size for offset in range(depth - 1)]]),
        tuple([shared, *[(shared + offset + 1) % vocab_size for offset in range(max(0, depth - 2))], (shared + 17) % vocab_size]),
        tuple([int(seeds[1]), *[(int(seeds[1]) + 3 * offset + 1) % vocab_size for offset in range(depth - 1)]]),
    ]
    return [
        VerifyRoutePayload(route_id=index + 1, token_ids=path, draft_logprob=-float(index))
        for index, path in enumerate(paths)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke persistent ATLAS 0709 Target KV across two verify rounds.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-token-ids", default="1,2,3,4")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-logit-diff", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prompt = tuple(int(part.strip()) for part in args.prompt_token_ids.split(",") if part.strip())
    config = FlashInferFullVerifyConfig(
        k=3,
        d=args.d,
        prefix_len=len(prompt),
        page_size=args.page_size,
        dtype=args.dtype,
        device=args.device,
        workspace_mb=args.workspace_mb,
        trust_remote_code=args.trust_remote_code,
        check_logit_alignment=False,
    )
    backend = DirectFlashInferMaskedTreeVerifyBackend(model_path=args.model, config=config)
    prefix = backend.prefill(prompt)
    vocab_size = int(backend.model.config.vocab_size)

    round1 = backend.verify_payloads(
        prefix_token_ids=prompt,
        routes=candidate_routes(prefix.next_token_logits, depth=args.d, vocab_size=vocab_size),
    )
    committed_after_round1 = backend.prefix_token_ids
    _, reference_logits = prefill_cache(
        backend.model,
        committed_after_round1,
        batch_size=1,
        device=args.device,
    )
    candidate_logits = backend._prefix_logits
    assert candidate_logits is not None
    max_abs_diff = float(
        (candidate_logits.float() - reference_logits.float()).abs().max().detach().cpu()
    )
    top1_match = bool(
        torch.argmax(candidate_logits, dim=-1)
        .eq(torch.argmax(reference_logits, dim=-1))
        .all()
        .detach()
        .cpu()
    )
    if max_abs_diff > float(args.max_logit_diff) or not top1_match:
        raise RuntimeError(
            f"committed Target KV logits mismatch: max_abs_diff={max_abs_diff}, top1_match={top1_match}"
        )

    round2 = backend.verify_payloads(
        prefix_token_ids=committed_after_round1,
        routes=candidate_routes(candidate_logits[0], depth=args.d, vocab_size=vocab_size),
    )
    result = {
        "round1_selected_route_id": round1.selected_route_id,
        "round2_selected_route_id": round2.selected_route_id,
        "round1_unique_nodes": round1.metadata["node_count"],
        "round1_unmerged_nodes": round1.metadata["unmerged_path_node_count"],
        "prefix_len_initial": len(prompt),
        "prefix_len_after_round1": len(committed_after_round1),
        "prefix_len_after_round2": len(backend.prefix_token_ids),
        "committed_verify_rounds": backend._committed_verify_rounds,
        "round1_commit_logit_max_abs_diff": max_abs_diff,
        "round1_commit_logit_top1_match": top1_match,
        "persistent_target_paged_kv": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
