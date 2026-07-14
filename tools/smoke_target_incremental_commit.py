from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from atlas_0709.flashinfer_full_verify import (
    FlashInferFullVerifyConfig,
    build_paged_prefix_kv,
    prefill_cache,
)
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
    parser.add_argument("--max-kv-diff", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--route-selection-policy",
        choices=["target_best", "first_route"],
        default="first_route",
    )
    parser.add_argument("--json-out", default=None)
    return parser


def canonical_prefix_kv_diff(
    *,
    candidate: list[torch.Tensor],
    reference: list[torch.Tensor],
    prefix_len: int,
    page_size: int,
) -> tuple[float, float]:
    positions = torch.arange(prefix_len, device=candidate[0].device, dtype=torch.long)
    pages = torch.div(positions, int(page_size), rounding_mode="floor")
    offsets = torch.remainder(positions, int(page_size))
    max_abs_diff = 0.0
    abs_sum = 0.0
    element_count = 0
    for candidate_layer, reference_layer in zip(candidate, reference):
        candidate_used = candidate_layer[pages, :, offsets].float()
        reference_used = reference_layer[pages, :, offsets].float()
        diff = (candidate_used - reference_used).abs()
        max_abs_diff = max(max_abs_diff, float(diff.max().detach().cpu()))
        abs_sum += float(diff.sum().detach().cpu())
        element_count += int(diff.numel())
    return max_abs_diff, abs_sum / element_count if element_count else 0.0


def main() -> int:
    args = build_parser().parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
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
    backend = DirectFlashInferMaskedTreeVerifyBackend(
        model_path=args.model,
        config=config,
        route_selection_policy=args.route_selection_policy,
    )
    prefix = backend.prefill(prompt)
    vocab_size = int(backend.model.config.vocab_size)
    round_reports: list[dict[str, object]] = []
    for round_index in range(args.rounds):
        prefix_before = backend.prefix_token_ids
        candidate_logits = backend._prefix_logits
        assert candidate_logits is not None
        verify_result = backend.verify_payloads(
            prefix_token_ids=prefix_before,
            routes=candidate_routes(
                candidate_logits[0],
                depth=args.d,
                vocab_size=vocab_size,
            ),
        )
        committed_prefix = backend.prefix_token_ids
        reference_past, reference_logits = prefill_cache(
            backend.model,
            committed_prefix,
            batch_size=1,
            device=args.device,
        )
        reference_paged = build_paged_prefix_kv(
            past_key_values=reference_past,
            prefix_len=len(committed_prefix),
            node_count=0,
            page_size=args.page_size,
            num_layers=len(backend.model.model.layers),
        )
        candidate_logits = backend._prefix_logits
        candidate_kv = backend._paged_prefix_kv
        assert candidate_logits is not None and candidate_kv is not None
        max_logit_diff = float(
            (candidate_logits.float() - reference_logits.float()).abs().max().detach().cpu()
        )
        mean_logit_diff = float(
            (candidate_logits.float() - reference_logits.float()).abs().mean().detach().cpu()
        )
        top1_match = bool(
            torch.argmax(candidate_logits, dim=-1)
            .eq(torch.argmax(reference_logits, dim=-1))
            .all()
            .detach()
            .cpu()
        )
        max_kv_diff, mean_kv_diff = canonical_prefix_kv_diff(
            candidate=candidate_kv,
            reference=reference_paged,
            prefix_len=len(committed_prefix),
            page_size=args.page_size,
        )
        round_report = {
            "round_index": round_index,
            "selected_route_id": verify_result.selected_route_id,
            "prefix_len_before": len(prefix_before),
            "prefix_len_after": len(committed_prefix),
            "unique_nodes": verify_result.metadata["node_count"],
            "unmerged_nodes": verify_result.metadata["unmerged_path_node_count"],
            "commit_logit_max_abs_diff": max_logit_diff,
            "commit_logit_mean_abs_diff": mean_logit_diff,
            "commit_logit_top1_match": top1_match,
            "canonical_kv_max_abs_diff": max_kv_diff,
            "canonical_kv_mean_abs_diff": mean_kv_diff,
        }
        round_reports.append(round_report)
        if (
            max_logit_diff > float(args.max_logit_diff)
            or max_kv_diff > float(args.max_kv_diff)
            or not top1_match
        ):
            raise RuntimeError(f"committed Target KV mismatch: {round_report}")

    result = {
        "passed": True,
        "route_selection_policy": args.route_selection_policy,
        "rounds_requested": args.rounds,
        "prefix_len_initial": len(prompt),
        "prefix_len_final": len(backend.prefix_token_ids),
        "committed_verify_rounds": backend._committed_verify_rounds,
        "persistent_target_paged_kv": True,
        "rounds": round_reports,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out is not None:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
