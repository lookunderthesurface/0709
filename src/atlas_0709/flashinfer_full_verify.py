from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

"""Direct FlashInfer full-model verify components.

This module is intentionally not the default target verifier. It is a reusable
source module for speed experiments that compare three operations on the same
manual Llama + FlashInfer execution chain:

* Direct FlashInfer AR decode.
* Direct FlashInfer linear causal verify.
* Direct FlashInfer packed ancestor-mask verify.

Suffix Q/K states are RoPE-rotated before entering FlashInfer. The benchmark
also performs an untimed HF logits alignment check for the masked tree verify
path and reports the measured diff in metadata.
"""


@dataclass(frozen=True)
class TimedResult:
    median_ms: float
    mean_ms: float
    samples_ms: list[float]


@dataclass(frozen=True)
class FlashInferFullVerifyConfig:
    k: int = 3
    d: int = 4
    prefix_len: int = 8192
    repeat_token_id: int = 42
    page_size: int = 16
    dtype: str = "float16"
    device: str = "cuda"
    warmup: int = 1
    iters: int = 5
    workspace_mb: int = 128
    flashinfer_backend: str = "auto"
    trust_remote_code: bool = False
    use_packed_custom_mask: bool = False
    check_logit_alignment: bool = True
    fail_on_logit_mismatch: bool = False
    alignment_atol: float = 1.0
    alignment_rtol: float = 0.05

    @property
    def semantic_correctness_required(self) -> bool:
        return self.check_logit_alignment

    @property
    def logits_aligned(self) -> bool:
        return False

    @property
    def rope_applied(self) -> bool:
        return True

    def validate(self) -> None:
        if self.k <= 0 or self.d <= 0:
            raise ValueError("k and d must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.warmup < 0 or self.iters <= 0:
            raise ValueError("warmup must be >= 0 and iters must be positive")
        if self.workspace_mb <= 0:
            raise ValueError("workspace_mb must be positive")
        if self.alignment_atol < 0 or self.alignment_rtol < 0:
            raise ValueError("alignment tolerances must be non-negative")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Direct FlashInfer full verify")


@dataclass(frozen=True)
class FlashInferFullVerifyBenchmarkResult:
    metadata: dict[str, object]
    target_ar_decode_1_token: TimedResult
    target_ar_decode_2_tokens: TimedResult
    target_hf_verify: TimedResult
    flashinfer_full_ar_decode_1_token: TimedResult
    flashinfer_full_ar_decode_2_tokens: TimedResult
    flashinfer_full_linear_verify: TimedResult
    flashinfer_masked_verify_full: TimedResult
    tree_metadata: dict[str, object]
    linear_metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        masked = self.flashinfer_masked_verify_full
        fi_ar1 = self.flashinfer_full_ar_decode_1_token
        fi_ar2 = self.flashinfer_full_ar_decode_2_tokens
        linear = self.flashinfer_full_linear_verify
        hf_ar1 = self.target_ar_decode_1_token
        hf_ar2 = self.target_ar_decode_2_tokens
        hf_verify = self.target_hf_verify
        k = int(self.metadata["k"])
        d = int(self.metadata["d"])
        return {
            **self.metadata,
            **self.tree_metadata,
            **self.linear_metadata,
            "results": {
                "target_ar_decode_1_token": result_dict(hf_ar1),
                "target_ar_decode_2_tokens": result_dict(hf_ar2),
                "flashinfer_full_ar_decode_1_token": result_dict(fi_ar1),
                "flashinfer_full_ar_decode_2_tokens": result_dict(fi_ar2),
                "target_hf_verify": result_dict(hf_verify),
                f"target_hf_verify_k{k}_paths_d{d}": result_dict(hf_verify),
                "flashinfer_full_linear_verify": result_dict(linear),
                f"flashinfer_full_linear_verify_k{k}_d{d}": result_dict(linear),
                "flashinfer_masked_verify_full": result_dict(masked),
                f"flashinfer_masked_verify_full_k{k}_d{d}": result_dict(masked),
                "ratios_median": {
                    "flashinfer_verify_over_hf_ar_decode_1": masked.median_ms / hf_ar1.median_ms,
                    "flashinfer_verify_over_hf_ar_decode_2": masked.median_ms / hf_ar2.median_ms,
                    "flashinfer_verify_over_flashinfer_ar_decode_1": masked.median_ms / fi_ar1.median_ms,
                    "flashinfer_verify_over_flashinfer_ar_decode_2": masked.median_ms / fi_ar2.median_ms,
                    "flashinfer_verify_over_flashinfer_linear_verify": masked.median_ms / linear.median_ms,
                    "flashinfer_verify_over_hf_verify": masked.median_ms / hf_verify.median_ms,
                },
                "goal_flashinfer_verify_lt_flashinfer_ar_decode_2": masked.median_ms < fi_ar2.median_ms,
                "goal_flashinfer_verify_lt_flashinfer_linear_verify": masked.median_ms < linear.median_ms,
            },
        }


@dataclass(frozen=True)
class TreeSpec:
    flat_token_ids: torch.Tensor
    parent_indices: list[int]
    depths: torch.Tensor
    packed_mask: torch.Tensor
    bool_mask: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    total_pages: int
    node_count: int
    kv_len: int


@dataclass(frozen=True)
class LinearVerifySpec:
    flat_token_ids: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    total_pages: int
    node_count: int
    kv_len_per_path: int
    prefix_pages: int
    suffix_pages: int
    path_count: int
    path_depth: int


@dataclass
class FullVerifyState:
    tree: TreeSpec
    paged_prefix_kv: list[torch.Tensor]


@dataclass
class FullLinearVerifyState:
    spec: LinearVerifySpec
    paged_prefix_kv: list[torch.Tensor]


@dataclass
class FullDecodeState:
    input_ids: torch.Tensor
    paged_prefix_kv: list[torch.Tensor]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-model target verify with FlashInfer packed ancestor-mask attention. "
            "Suffix Q/K uses RoPE and the benchmark reports HF logits alignment."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--repeat-token-id", type=int, default=42)
    parser.add_argument("--prompt", default="ATLAS full masked verify.")
    parser.add_argument("--prompt-token-ids", default=None)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
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


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotary_module(model, layer):
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "rotary_emb"):
        return base.rotary_emb
    attn = getattr(layer, "self_attn", None)
    if attn is not None and hasattr(attn, "rotary_emb"):
        return attn.rotary_emb
    return None


def _manual_rope_cos_sin(
    *,
    model,
    positions: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = float(getattr(model.config, "rope_theta", 10000.0))
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, int(head_dim), 2, device=device, dtype=torch.float32) / int(head_dim))
    )
    freqs = torch.outer(positions.to(device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rope_cos_sin(
    *,
    model,
    layer,
    positions: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.to(device=device, dtype=torch.long).reshape(-1)
    rotary = _rotary_module(model, layer)
    if rotary is None:
        return _manual_rope_cos_sin(
            model=model,
            positions=positions,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    position_ids = positions.reshape(1, -1)
    hidden_size = int(getattr(model.config, "hidden_size", int(head_dim)))
    dummy = torch.empty((1, int(positions.numel()), hidden_size), dtype=dtype, device=device)
    try:
        cos, sin = rotary(dummy, position_ids)
    except TypeError:
        seq_len = int(positions.max().detach().cpu().item()) + 1
        dummy_heads = torch.empty((1, 1, seq_len, int(head_dim)), dtype=dtype, device=device)
        cos, sin = rotary(dummy_heads, seq_len=seq_len)
        cos = cos.index_select(0, positions)
        sin = sin.index_select(0, positions)

    cos = cos.to(device=device, dtype=dtype).reshape(-1, int(head_dim))
    sin = sin.to(device=device, dtype=dtype).reshape(-1, int(head_dim))
    if int(cos.shape[0]) != int(positions.numel()):
        if int(cos.shape[0]) > int(positions.max().detach().cpu().item()):
            cos = cos.index_select(0, positions)
            sin = sin.index_select(0, positions)
        else:
            raise RuntimeError(
                f"rotary embedding returned {int(cos.shape[0])} positions for {int(positions.numel())} queries"
            )
    return cos, sin


def apply_rope_to_qk(
    *,
    model,
    layer,
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope_cos_sin(
        model=model,
        layer=layer,
        positions=positions,
        head_dim=head_dim,
        dtype=q.dtype,
        device=q.device,
    )
    return apply_rope_with_cos_sin(q=q, k=k, cos=cos, sin=sin)


def apply_rope_with_cos_sin(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_with_setup(
    setup_fn: Callable[[], object],
    run_fn: Callable[[object], None],
    *,
    warmup: int,
    iters: int,
) -> TimedResult:
    for _ in range(warmup):
        state = setup_fn()
        sync_cuda()
        run_fn(state)
        sync_cuda()
        del state
    samples: list[float] = []
    for _ in range(iters):
        state = setup_fn()
        sync_cuda()
        start = time.perf_counter()
        run_fn(state)
        sync_cuda()
        samples.append((time.perf_counter() - start) * 1000.0)
        del state
    return TimedResult(
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.fmean(samples)),
        samples_ms=samples,
    )


def result_dict(result: TimedResult) -> dict[str, object]:
    return {
        "median_ms": result.median_ms,
        "mean_ms": result.mean_ms,
        "samples_ms": result.samples_ms,
    }


def print_result(name: str, result: TimedResult) -> None:
    print(f"{name:<44} median={result.median_ms:9.3f} ms mean={result.mean_ms:9.3f} ms")


def parse_prompt_token_ids(
    *,
    tokenizer,
    prompt: str,
    prompt_token_ids: str | None,
    prefix_len: int,
    repeat_token_id: int,
) -> list[int]:
    if prompt_token_ids:
        base = [int(part.strip()) for part in prompt_token_ids.split(",") if part.strip()]
    elif repeat_token_id is not None:
        base = [int(repeat_token_id)]
    else:
        base = [int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False)]
    if not base:
        raise RuntimeError("empty prompt token list")
    repeats = (int(prefix_len) + len(base) - 1) // len(base)
    return (base * repeats)[: int(prefix_len)]


def load_model(model_path: str, *, dtype: torch.dtype, device: str, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def model_forward(model, input_ids: torch.Tensor, past_key_values=None):
    kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    try:
        return model(**kwargs, return_legacy_cache=True)
    except TypeError:
        return model(**kwargs)


@torch.inference_mode()
def prefill_cache(model, prompt_token_ids: Sequence[int], *, batch_size: int, device: str):
    row = torch.tensor([int(token_id) for token_id in prompt_token_ids], dtype=torch.long, device=device)
    input_ids = row.unsqueeze(0).repeat(int(batch_size), 1)
    output = model_forward(model, input_ids)
    return output.past_key_values, output.logits[:, -1, :].detach()


@torch.inference_mode()
def decode_one(model, past_key_values, input_ids: torch.Tensor):
    output = model_forward(model, input_ids[:, None], past_key_values=past_key_values)
    return output.past_key_values, output.logits[:, -1, :].detach()


def next_argmax_inputs(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1).to(dtype=torch.long)


def bench_ar_decode(
    model,
    prompt_token_ids: Sequence[int],
    *,
    steps: int,
    device: str,
    warmup: int,
    iters: int,
) -> TimedResult:
    def setup():
        past, logits = prefill_cache(model, prompt_token_ids, batch_size=1, device=device)
        return past, next_argmax_inputs(logits)

    def run_state(state) -> None:
        past, input_ids = state
        for _ in range(int(steps)):
            past, logits_i = decode_one(model, past, input_ids)
            input_ids = next_argmax_inputs(logits_i)

    return measure_with_setup(setup, run_state, warmup=warmup, iters=iters)


def bench_hf_verify_append(
    model,
    prompt_token_ids: Sequence[int],
    *,
    k: int,
    depth: int,
    device: str,
    warmup: int,
    iters: int,
) -> TimedResult:
    def setup():
        past, logits = prefill_cache(model, prompt_token_ids, batch_size=k, device=device)
        _, token_ids = torch.topk(torch.log_softmax(logits, dim=-1), k=int(depth), dim=-1)
        verify_tokens = token_ids[:, : int(depth)].to(dtype=torch.long)
        return past, verify_tokens

    def run_state(state) -> None:
        past, verify_tokens = state
        model_forward(model, verify_tokens, past_key_values=past)

    return measure_with_setup(setup, run_state, warmup=warmup, iters=iters)


def ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def last_page_len(length: int, page_size: int) -> int:
    rem = int(length) % int(page_size)
    return int(page_size) if rem == 0 else rem


def packbits(mask: torch.Tensor) -> torch.Tensor:
    try:
        from flashinfer.quantization import packbits as flashinfer_packbits
    except Exception:
        from flashinfer import packbits as flashinfer_packbits
    return flashinfer_packbits(mask)


def build_path_tree_tokens(
    *,
    prefix_logits: torch.Tensor,
    k: int,
    depth: int,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    vocab_size = int(prefix_logits.shape[-1])
    _, seeds = torch.topk(torch.log_softmax(prefix_logits[0], dim=-1), k=int(k), dim=-1)
    token_ids: list[int] = []
    parent_indices: list[int] = []
    depths: list[int] = []
    for route_idx in range(int(k)):
        parent = -1
        seed = int(seeds[route_idx])
        for depth_idx in range(int(depth)):
            node_idx = len(token_ids)
            token_ids.append((seed + 13 * depth_idx + 17 * route_idx) % vocab_size)
            parent_indices.append(parent)
            depths.append(depth_idx)
            parent = node_idx
    device = prefix_logits.device
    return (
        torch.tensor(token_ids, dtype=torch.long, device=device),
        parent_indices,
        torch.tensor(depths, dtype=torch.long, device=device),
    )


def build_path_tree_tokens_from_paths(
    paths: Sequence[Sequence[int]],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    if not paths:
        raise ValueError("paths must not be empty")
    path_lengths = {len(path) for path in paths}
    if len(path_lengths) != 1:
        raise ValueError(f"all paths must have the same length, got {sorted(path_lengths)}")
    depth = next(iter(path_lengths))
    if depth <= 0:
        raise ValueError("paths must contain at least one token")

    token_ids: list[int] = []
    parent_indices: list[int] = []
    depths: list[int] = []
    for path in paths:
        parent = -1
        for depth_idx, token_id in enumerate(path):
            node_idx = len(token_ids)
            token_ids.append(int(token_id))
            parent_indices.append(parent)
            depths.append(int(depth_idx))
            parent = node_idx
    return (
        torch.tensor(token_ids, dtype=torch.long, device=device),
        parent_indices,
        torch.tensor(depths, dtype=torch.long, device=device),
    )


def build_deduplicated_path_tree_tokens(
    paths: Sequence[Sequence[int]],
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[list[int]]]:
    """Build a prefix trie and return one physical node per unique token prefix."""
    if not paths:
        raise ValueError("paths must not be empty")
    path_lengths = {len(path) for path in paths}
    if len(path_lengths) != 1:
        raise ValueError(f"all paths must have the same length, got {sorted(path_lengths)}")
    depth = next(iter(path_lengths))
    if depth <= 0:
        raise ValueError("paths must contain at least one token")

    token_ids: list[int] = []
    parent_indices: list[int] = []
    depths: list[int] = []
    children_by_node: list[dict[int, int]] = []
    root_children: dict[int, int] = {}
    route_to_node_paths: list[list[int]] = []

    for path in paths:
        parent = -1
        children = root_children
        route_nodes: list[int] = []
        for depth_idx, raw_token_id in enumerate(path):
            token_id = int(raw_token_id)
            node_idx = children.get(token_id)
            if node_idx is None:
                node_idx = len(token_ids)
                children[token_id] = node_idx
                token_ids.append(token_id)
                parent_indices.append(parent)
                depths.append(int(depth_idx))
                children_by_node.append({})
            route_nodes.append(node_idx)
            parent = node_idx
            children = children_by_node[node_idx]
        route_to_node_paths.append(route_nodes)

    return (
        torch.tensor(token_ids, dtype=torch.long, device=device),
        parent_indices,
        torch.tensor(depths, dtype=torch.long, device=device),
        route_to_node_paths,
    )


def ancestors_inclusive(node_idx: int, parent_indices: Sequence[int]) -> list[int]:
    out: list[int] = []
    cur = int(node_idx)
    while cur >= 0:
        out.append(cur)
        cur = int(parent_indices[cur])
    return out


def build_tree_spec(
    *,
    prefix_logits: torch.Tensor,
    prefix_len: int,
    k: int,
    depth: int,
    page_size: int,
) -> TreeSpec:
    if int(prefix_len) % int(page_size) != 0:
        raise ValueError("prefix_len must be page-aligned for Direct FlashInfer full verify")
    flat_token_ids, parent_indices, depths = build_path_tree_tokens(
        prefix_logits=prefix_logits,
        k=k,
        depth=depth,
    )
    device = prefix_logits.device
    node_count = int(flat_token_ids.numel())
    kv_len = int(prefix_len) + node_count
    total_pages = ceil_div(kv_len, page_size)

    mask = torch.zeros((node_count, kv_len), dtype=torch.bool, device=device)
    mask[:, : int(prefix_len)] = True
    for node_idx in range(node_count):
        for ancestor_idx in ancestors_inclusive(node_idx, parent_indices):
            mask[node_idx, int(prefix_len) + ancestor_idx] = True
    flat_mask = mask.flatten()
    return TreeSpec(
        flat_token_ids=flat_token_ids,
        parent_indices=parent_indices,
        depths=depths,
        packed_mask=packbits(flat_mask),
        bool_mask=flat_mask,
        qo_indptr=torch.tensor([0, node_count], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, total_pages], dtype=torch.int32, device=device),
        kv_indices=torch.arange(total_pages, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor([last_page_len(kv_len, page_size)], dtype=torch.int32, device=device),
        total_pages=total_pages,
        node_count=node_count,
        kv_len=kv_len,
    )


def build_tree_spec_from_paths(
    *,
    paths: Sequence[Sequence[int]],
    prefix_len: int,
    page_size: int,
    device: torch.device | str,
) -> TreeSpec:
    flat_token_ids, parent_indices, depths = build_path_tree_tokens_from_paths(paths, device=device)
    node_count = int(flat_token_ids.numel())
    kv_len = int(prefix_len) + node_count
    total_pages = ceil_div(kv_len, page_size)

    mask = torch.zeros((node_count, kv_len), dtype=torch.bool, device=device)
    mask[:, : int(prefix_len)] = True
    for node_idx in range(node_count):
        for ancestor_idx in ancestors_inclusive(node_idx, parent_indices):
            mask[node_idx, int(prefix_len) + ancestor_idx] = True
    flat_mask = mask.flatten()
    return TreeSpec(
        flat_token_ids=flat_token_ids,
        parent_indices=parent_indices,
        depths=depths,
        packed_mask=packbits(flat_mask),
        bool_mask=flat_mask,
        qo_indptr=torch.tensor([0, node_count], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, total_pages], dtype=torch.int32, device=device),
        kv_indices=torch.arange(total_pages, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor([last_page_len(kv_len, page_size)], dtype=torch.int32, device=device),
        total_pages=total_pages,
        node_count=node_count,
        kv_len=kv_len,
    )


def build_deduplicated_tree_spec_from_paths(
    *,
    paths: Sequence[Sequence[int]],
    prefix_len: int,
    page_size: int,
    device: torch.device | str,
) -> tuple[TreeSpec, list[list[int]]]:
    flat_token_ids, parent_indices, depths, route_to_node_paths = build_deduplicated_path_tree_tokens(
        paths,
        device=device,
    )
    node_count = int(flat_token_ids.numel())
    kv_len = int(prefix_len) + node_count
    total_pages = ceil_div(kv_len, page_size)

    mask = torch.zeros((node_count, kv_len), dtype=torch.bool, device=device)
    mask[:, : int(prefix_len)] = True
    for node_idx in range(node_count):
        for ancestor_idx in ancestors_inclusive(node_idx, parent_indices):
            mask[node_idx, int(prefix_len) + ancestor_idx] = True
    flat_mask = mask.flatten()
    tree = TreeSpec(
        flat_token_ids=flat_token_ids,
        parent_indices=parent_indices,
        depths=depths,
        packed_mask=packbits(flat_mask),
        bool_mask=flat_mask,
        qo_indptr=torch.tensor([0, node_count], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, total_pages], dtype=torch.int32, device=device),
        kv_indices=torch.arange(total_pages, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor([last_page_len(kv_len, page_size)], dtype=torch.int32, device=device),
        total_pages=total_pages,
        node_count=node_count,
        kv_len=kv_len,
    )
    return tree, route_to_node_paths


def build_linear_verify_spec(
    *,
    prefix_logits: torch.Tensor,
    prefix_len: int,
    k: int,
    depth: int,
    page_size: int,
) -> LinearVerifySpec:
    if int(prefix_len) % int(page_size) != 0:
        raise ValueError("prefix_len must be page-aligned for Direct FlashInfer full verify")
    flat_token_ids, _parent_indices, _depths = build_path_tree_tokens(
        prefix_logits=prefix_logits,
        k=k,
        depth=depth,
    )
    device = prefix_logits.device
    prefix_pages = int(prefix_len) // int(page_size)
    suffix_pages = ceil_div(int(depth), int(page_size))
    total_pages = prefix_pages + int(k) * suffix_pages
    page_lists: list[int] = []
    indptr = [0]
    shared_pages = list(range(prefix_pages))
    for path_idx in range(int(k)):
        private_start = prefix_pages + path_idx * suffix_pages
        page_lists.extend(shared_pages)
        page_lists.extend(range(private_start, private_start + suffix_pages))
        indptr.append(len(page_lists))
    qo_indptr = [path_idx * int(depth) for path_idx in range(int(k) + 1)]
    kv_len = int(prefix_len) + int(depth)
    return LinearVerifySpec(
        flat_token_ids=flat_token_ids,
        qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32, device=device),
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=torch.tensor(page_lists, dtype=torch.int32, device=device),
        kv_last_page_len=torch.tensor([last_page_len(kv_len, page_size)] * int(k), dtype=torch.int32, device=device),
        total_pages=total_pages,
        node_count=int(flat_token_ids.numel()),
        kv_len_per_path=kv_len,
        prefix_pages=prefix_pages,
        suffix_pages=suffix_pages,
        path_count=int(k),
        path_depth=int(depth),
    )


def legacy_layer_kv(past_key_values, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        layer = legacy[layer_idx]
        return layer[0], layer[1]
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        for key_name, value_name in (
            ("keys", "values"),
            ("key_cache", "value_cache"),
            ("k_cache", "v_cache"),
        ):
            if hasattr(layer, key_name) and hasattr(layer, value_name):
                return getattr(layer, key_name), getattr(layer, value_name)
        if hasattr(layer, "get_keys") and hasattr(layer, "get_values"):
            return layer.get_keys(), layer.get_values()
    layer = past_key_values[layer_idx]
    if not isinstance(layer, (tuple, list)) or len(layer) < 2:
        raise TypeError(
            "expected legacy tuple past_key_values or a DynamicCache exposing "
            "to_legacy_cache/key_cache/value_cache/layers"
        )
    return layer[0], layer[1]


def prefix_layer_to_paged_kv(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_len: int,
    node_count: int,
    page_size: int,
    total_pages: int | None = None,
) -> torch.Tensor:
    if key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError("full masked verify expects batch-1 prefix KV")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"expected key/value shape [1, H, T, D], got {tuple(key.shape)} and {tuple(value.shape)}")
    _, kv_heads, key_len, head_dim = key.shape
    if int(key_len) != int(prefix_len):
        raise ValueError(f"prefix cache length mismatch: expected {prefix_len}, got {key_len}")
    if total_pages is None:
        total_pages = ceil_div(int(prefix_len) + int(node_count), int(page_size))
    prefix_pages = ceil_div(int(prefix_len), int(page_size))
    kv_cache = torch.zeros(
        total_pages,
        2,
        int(page_size),
        int(kv_heads),
        int(head_dim),
        dtype=key.dtype,
        device=key.device,
    )
    key_tokens = key[0].permute(1, 0, 2).contiguous()
    value_tokens = value[0].permute(1, 0, 2).contiguous()
    positions = torch.arange(int(prefix_len), device=key.device, dtype=torch.long)
    page_ids = torch.div(positions, int(page_size), rounding_mode="floor")
    offsets = torch.remainder(positions, int(page_size))
    kv_cache[page_ids, 0, offsets] = key_tokens
    kv_cache[page_ids, 1, offsets] = value_tokens
    return kv_cache


def build_paged_prefix_kv(
    *,
    past_key_values,
    prefix_len: int,
    node_count: int,
    page_size: int,
    num_layers: int,
    total_pages: int | None = None,
) -> list[torch.Tensor]:
    paged: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        key, value = legacy_layer_kv(past_key_values, layer_idx)
        paged.append(
            prefix_layer_to_paged_kv(
                key=key,
                value=value,
                prefix_len=prefix_len,
                node_count=node_count,
                page_size=page_size,
                total_pages=total_pages,
            )
        )
    return paged


def make_prefill_wrapper(*, args, flashinfer):
    workspace = torch.empty(int(args.workspace_mb) * 1024 * 1024, dtype=torch.uint8, device=args.device)
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend=args.flashinfer_backend)


def make_decode_wrapper(*, args, flashinfer):
    workspace = torch.empty(int(args.workspace_mb) * 1024 * 1024, dtype=torch.uint8, device=args.device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend=args.flashinfer_backend,
    )


def plan_decode_wrapper(
    wrapper,
    *,
    seq_len: int,
    args,
    config,
    dtype: torch.dtype,
) -> None:
    head_dim = getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))
    pages = ceil_div(seq_len, int(args.page_size))
    device = torch.device(args.device)
    indptr = torch.tensor([0, pages], dtype=torch.int32, device=device)
    indices = torch.arange(pages, dtype=torch.int32, device=device)
    last_len = torch.tensor([last_page_len(seq_len, int(args.page_size))], dtype=torch.int32, device=device)
    attempts = [
        {"pos_encoding_mode": "NONE", "data_type": dtype},
        {"pos_encoding_mode": "NONE", "q_data_type": dtype},
        {"data_type": dtype},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            wrapper.plan(
                indptr,
                indices,
                last_len,
                int(config.num_attention_heads),
                int(config.num_key_value_heads),
                int(head_dim),
                int(args.page_size),
                **kwargs,
            )
            return
        except (AttributeError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(f"BatchDecodeWithPagedKVCacheWrapper.plan failed: {last_error!r}")


def make_decode_wrappers(
    *,
    args,
    config,
    dtype: torch.dtype,
    flashinfer,
    prefix_len: int,
    steps: int,
) -> list[object]:
    wrappers = []
    for step_idx in range(int(steps)):
        wrapper = make_decode_wrapper(args=args, flashinfer=flashinfer)
        plan_decode_wrapper(
            wrapper,
            seq_len=int(prefix_len) + step_idx + 1,
            args=args,
            config=config,
            dtype=dtype,
        )
        wrappers.append(wrapper)
    return wrappers


def plan_prefill_wrapper(wrapper, tree: TreeSpec, *, args, config, dtype: torch.dtype) -> None:
    head_dim = getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))
    attempts = [
        {"q_data_type": dtype, "kv_data_type": dtype},
        {"q_data_type": str(dtype).replace("torch.", ""), "kv_data_type": str(dtype).replace("torch.", "")},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            mask_kwargs = (
                {"packed_custom_mask": tree.packed_mask}
                if args.use_packed_custom_mask
                else {"custom_mask": tree.bool_mask}
            )
            wrapper.plan(
                tree.qo_indptr,
                tree.kv_indptr,
                tree.kv_indices,
                tree.kv_last_page_len,
                int(config.num_attention_heads),
                int(config.num_key_value_heads),
                int(head_dim),
                int(args.page_size),
                causal=False,
                pos_encoding_mode="NONE",
                **mask_kwargs,
                **kwargs,
            )
            return
        except (AttributeError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(f"BatchPrefillWithPagedKVCacheWrapper.plan failed: {last_error!r}")


def plan_linear_prefill_wrapper(wrapper, spec: LinearVerifySpec, *, args, config, dtype: torch.dtype) -> None:
    head_dim = getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))
    attempts = [
        {"q_data_type": dtype, "kv_data_type": dtype},
        {"q_data_type": str(dtype).replace("torch.", ""), "kv_data_type": str(dtype).replace("torch.", "")},
        {},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            wrapper.plan(
                spec.qo_indptr,
                spec.kv_indptr,
                spec.kv_indices,
                spec.kv_last_page_len,
                int(config.num_attention_heads),
                int(config.num_key_value_heads),
                int(head_dim),
                int(args.page_size),
                causal=True,
                pos_encoding_mode="NONE",
                **kwargs,
            )
            return
        except (AttributeError, TypeError) as exc:
            last_error = exc
    raise RuntimeError(f"BatchPrefillWithPagedKVCacheWrapper.plan failed for linear verify: {last_error!r}")


def first_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"expected tensor output, got {type(value).__name__}")


def write_suffix_kv(
    kv_cache: torch.Tensor,
    *,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefix_len: int,
    page_size: int,
    start_pos: int | None = None,
) -> None:
    node_count = int(key_states.shape[0])
    start = int(prefix_len) if start_pos is None else int(start_pos)
    positions = torch.arange(start, start + node_count, device=key_states.device)
    page_ids = torch.div(positions, int(page_size), rounding_mode="floor").to(dtype=torch.long)
    offsets = torch.remainder(positions, int(page_size)).to(dtype=torch.long)
    kv_cache[page_ids, 0, offsets] = key_states
    kv_cache[page_ids, 1, offsets] = value_states


def write_linear_suffix_kv(
    kv_cache: torch.Tensor,
    *,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    prefix_len: int,
    page_size: int,
    spec: LinearVerifySpec,
) -> None:
    row = 0
    for path_idx in range(spec.path_count):
        private_start = spec.prefix_pages + path_idx * spec.suffix_pages
        for depth_idx in range(spec.path_depth):
            page_id = private_start + depth_idx // int(page_size)
            offset = (int(prefix_len) + depth_idx) % int(page_size)
            kv_cache[page_id, 0, offset] = key_states[row]
            kv_cache[page_id, 1, offset] = value_states[row]
            row += 1


@torch.inference_mode()
def run_flashinfer_full_ar_decode(
    *,
    model,
    wrappers: Sequence[object],
    state: FullDecodeState,
    prefix_len: int,
    page_size: int,
) -> torch.Tensor:
    config = model.config
    base = model.model
    head_dim = getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    input_ids = state.input_ids
    logits = None

    for step_idx, wrapper in enumerate(wrappers):
        hidden_states = base.embed_tokens(input_ids)
        positions = torch.tensor([int(prefix_len) + int(step_idx)], device=hidden_states.device, dtype=torch.long)
        cos, sin = rope_cos_sin(
            model=model,
            layer=base.layers[0],
            positions=positions,
            head_dim=int(head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for layer_idx, layer in enumerate(base.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            q = layer.self_attn.q_proj(normed).reshape(1, num_q_heads, int(head_dim)).contiguous()
            k = layer.self_attn.k_proj(normed).reshape(1, num_kv_heads, int(head_dim)).contiguous()
            v = layer.self_attn.v_proj(normed).reshape(1, num_kv_heads, int(head_dim)).contiguous()
            q, k = apply_rope_with_cos_sin(q=q, k=k, cos=cos, sin=sin)

            kv_cache = state.paged_prefix_kv[layer_idx]
            write_suffix_kv(
                kv_cache,
                key_states=k,
                value_states=v,
                prefix_len=prefix_len,
                page_size=page_size,
                start_pos=int(prefix_len) + step_idx,
            )
            attn = first_tensor(wrapper.run(q, kv_cache))
            attn = attn.reshape(1, int(config.hidden_size))
            hidden_states = residual + layer.self_attn.o_proj(attn)

            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = base.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        input_ids = torch.argmax(logits, dim=-1).to(dtype=torch.long)
    if logits is None:
        raise RuntimeError("decode steps must be positive")
    return logits


@torch.inference_mode()
def run_flashinfer_full_linear_verify(
    *,
    model,
    wrapper,
    state: FullLinearVerifyState,
    prefix_len: int,
    page_size: int,
) -> torch.Tensor:
    config = model.config
    base = model.model
    head_dim = getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    hidden_states = base.embed_tokens(state.spec.flat_token_ids)
    positions = int(prefix_len) + torch.arange(
        int(state.spec.path_depth),
        device=hidden_states.device,
        dtype=torch.long,
    ).repeat(int(state.spec.path_count))
    cos, sin = rope_cos_sin(
        model=model,
        layer=base.layers[0],
        positions=positions,
        head_dim=int(head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for layer_idx, layer in enumerate(base.layers):
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        q = layer.self_attn.q_proj(normed).reshape(state.spec.node_count, num_q_heads, int(head_dim)).contiguous()
        k = layer.self_attn.k_proj(normed).reshape(state.spec.node_count, num_kv_heads, int(head_dim)).contiguous()
        v = layer.self_attn.v_proj(normed).reshape(state.spec.node_count, num_kv_heads, int(head_dim)).contiguous()
        q, k = apply_rope_with_cos_sin(q=q, k=k, cos=cos, sin=sin)

        kv_cache = state.paged_prefix_kv[layer_idx]
        write_linear_suffix_kv(
            kv_cache,
            key_states=k,
            value_states=v,
            prefix_len=prefix_len,
            page_size=page_size,
            spec=state.spec,
        )
        attn = first_tensor(wrapper.run(q, kv_cache))
        attn = attn.reshape(state.spec.node_count, int(config.hidden_size))
        hidden_states = residual + layer.self_attn.o_proj(attn)

        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    hidden_states = base.norm(hidden_states)
    return model.lm_head(hidden_states)


@torch.inference_mode()
def run_flashinfer_full_verify(
    *,
    model,
    wrapper,
    state: FullVerifyState,
    prefix_len: int,
    page_size: int,
) -> torch.Tensor:
    config = model.config
    base = model.model
    head_dim = getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads))
    num_q_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    hidden_states = base.embed_tokens(state.tree.flat_token_ids)
    positions = int(prefix_len) + state.tree.depths.to(device=hidden_states.device, dtype=torch.long)
    cos, sin = rope_cos_sin(
        model=model,
        layer=base.layers[0],
        positions=positions,
        head_dim=int(head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for layer_idx, layer in enumerate(base.layers):
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)
        q = layer.self_attn.q_proj(normed).reshape(state.tree.node_count, num_q_heads, int(head_dim)).contiguous()
        k = layer.self_attn.k_proj(normed).reshape(state.tree.node_count, num_kv_heads, int(head_dim)).contiguous()
        v = layer.self_attn.v_proj(normed).reshape(state.tree.node_count, num_kv_heads, int(head_dim)).contiguous()
        q, k = apply_rope_with_cos_sin(q=q, k=k, cos=cos, sin=sin)

        kv_cache = state.paged_prefix_kv[layer_idx]
        write_suffix_kv(
            kv_cache,
            key_states=k,
            value_states=v,
            prefix_len=prefix_len,
            page_size=page_size,
        )
        attn = first_tensor(wrapper.run(q, kv_cache))
        attn = attn.reshape(state.tree.node_count, int(config.hidden_size))
        hidden_states = residual + layer.self_attn.o_proj(attn)

        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

    hidden_states = base.norm(hidden_states)
    return model.lm_head(hidden_states)


def logits_alignment_summary(
    *,
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    candidate = candidate_logits.detach().float()
    reference = reference_logits.detach().float()
    if candidate.shape != reference.shape:
        return {
            "passed": False,
            "reason": f"shape mismatch: candidate={tuple(candidate.shape)}, reference={tuple(reference.shape)}",
        }
    abs_diff = (candidate - reference).abs()
    tolerance = float(atol) + float(rtol) * reference.abs()
    passed = bool(torch.all(abs_diff <= tolerance).detach().cpu().item())
    top1_match = torch.argmax(candidate, dim=-1).eq(torch.argmax(reference, dim=-1)).float()
    return {
        "passed": passed,
        "shape": list(candidate.shape),
        "max_abs_diff": float(abs_diff.max().detach().cpu().item()),
        "mean_abs_diff": float(abs_diff.mean().detach().cpu().item()),
        "max_allowed_diff": float(tolerance.max().detach().cpu().item()),
        "top1_match_rate": float(top1_match.mean().detach().cpu().item()),
    }


@torch.inference_mode()
def reference_hf_verify_logits(
    *,
    model,
    prompt_token_ids: Sequence[int],
    verify_tokens: torch.Tensor,
    device: str,
) -> torch.Tensor:
    past, _prefix_logits = prefill_cache(
        model,
        prompt_token_ids,
        batch_size=int(verify_tokens.shape[0]),
        device=device,
    )
    output = model_forward(model, verify_tokens.to(device=device, dtype=torch.long), past_key_values=past)
    return output.logits.detach()


@torch.inference_mode()
def check_flashinfer_full_verify_alignment(
    *,
    model,
    prompt_token_ids: Sequence[int],
    args,
    dtype: torch.dtype,
    flashinfer,
) -> dict[str, object]:
    tree_state = setup_flashinfer_state(model=model, prompt_token_ids=prompt_token_ids, args=args)
    verify_tokens = tree_state.tree.flat_token_ids.reshape(int(args.k), int(args.d))
    hf_logits = reference_hf_verify_logits(
        model=model,
        prompt_token_ids=prompt_token_ids,
        verify_tokens=verify_tokens,
        device=args.device,
    ).reshape(tree_state.tree.node_count, -1)

    tree_wrapper = make_prefill_wrapper(args=args, flashinfer=flashinfer)
    plan_prefill_wrapper(tree_wrapper, tree_state.tree, args=args, config=model.config, dtype=dtype)
    tree_logits = run_flashinfer_full_verify(
        model=model,
        wrapper=tree_wrapper,
        state=tree_state,
        prefix_len=len(prompt_token_ids),
        page_size=args.page_size,
    )
    tree_summary = logits_alignment_summary(
        candidate_logits=tree_logits,
        reference_logits=hf_logits,
        atol=args.alignment_atol,
        rtol=args.alignment_rtol,
    )
    del tree_state

    linear_state = setup_flashinfer_linear_state(model=model, prompt_token_ids=prompt_token_ids, args=args)
    linear_wrapper = make_prefill_wrapper(args=args, flashinfer=flashinfer)
    plan_linear_prefill_wrapper(linear_wrapper, linear_state.spec, args=args, config=model.config, dtype=dtype)
    linear_logits = run_flashinfer_full_linear_verify(
        model=model,
        wrapper=linear_wrapper,
        state=linear_state,
        prefix_len=len(prompt_token_ids),
        page_size=args.page_size,
    )
    linear_summary = logits_alignment_summary(
        candidate_logits=linear_logits,
        reference_logits=hf_logits,
        atol=args.alignment_atol,
        rtol=args.alignment_rtol,
    )
    del linear_state

    passed = bool(tree_summary.get("passed")) and bool(linear_summary.get("passed"))
    return {
        "checked": True,
        "passed": passed,
        "atol": float(args.alignment_atol),
        "rtol": float(args.alignment_rtol),
        "reference_backend": "hf_append_verify",
        "candidate_backends": [
            "direct_flashinfer_full_linear_verify",
            "direct_flashinfer_full_masked_tree_verify",
        ],
        "masked_tree": tree_summary,
        "linear": linear_summary,
    }


def setup_flashinfer_state(
    *,
    model,
    prompt_token_ids: Sequence[int],
    args,
) -> FullVerifyState:
    past, prefix_logits = prefill_cache(model, prompt_token_ids, batch_size=1, device=args.device)
    tree = build_tree_spec(
        prefix_logits=prefix_logits,
        prefix_len=len(prompt_token_ids),
        k=args.k,
        depth=args.d,
        page_size=args.page_size,
    )
    paged_prefix_kv = build_paged_prefix_kv(
        past_key_values=past,
        prefix_len=len(prompt_token_ids),
        node_count=tree.node_count,
        page_size=args.page_size,
        num_layers=len(model.model.layers),
    )
    return FullVerifyState(tree=tree, paged_prefix_kv=paged_prefix_kv)


def setup_flashinfer_linear_state(
    *,
    model,
    prompt_token_ids: Sequence[int],
    args,
) -> FullLinearVerifyState:
    past, prefix_logits = prefill_cache(model, prompt_token_ids, batch_size=1, device=args.device)
    spec = build_linear_verify_spec(
        prefix_logits=prefix_logits,
        prefix_len=len(prompt_token_ids),
        k=args.k,
        depth=args.d,
        page_size=args.page_size,
    )
    paged_prefix_kv = build_paged_prefix_kv(
        past_key_values=past,
        prefix_len=len(prompt_token_ids),
        node_count=spec.node_count,
        page_size=args.page_size,
        num_layers=len(model.model.layers),
        total_pages=spec.total_pages,
    )
    return FullLinearVerifyState(spec=spec, paged_prefix_kv=paged_prefix_kv)


def setup_flashinfer_decode_state(
    *,
    model,
    prompt_token_ids: Sequence[int],
    steps: int,
    args,
) -> FullDecodeState:
    past, prefix_logits = prefill_cache(model, prompt_token_ids, batch_size=1, device=args.device)
    input_ids = next_argmax_inputs(prefix_logits)
    paged_prefix_kv = build_paged_prefix_kv(
        past_key_values=past,
        prefix_len=len(prompt_token_ids),
        node_count=int(steps),
        page_size=args.page_size,
        num_layers=len(model.model.layers),
    )
    return FullDecodeState(input_ids=input_ids, paged_prefix_kv=paged_prefix_kv)


def bench_flashinfer_full_ar_decode(
    *,
    model,
    prompt_token_ids: Sequence[int],
    steps: int,
    args,
    dtype: torch.dtype,
    flashinfer,
) -> TimedResult:
    wrappers = make_decode_wrappers(
        args=args,
        config=model.config,
        dtype=dtype,
        flashinfer=flashinfer,
        prefix_len=len(prompt_token_ids),
        steps=steps,
    )

    def setup():
        return setup_flashinfer_decode_state(
            model=model,
            prompt_token_ids=prompt_token_ids,
            steps=steps,
            args=args,
        )

    def run_state(state) -> None:
        run_flashinfer_full_ar_decode(
            model=model,
            wrappers=wrappers,
            state=state,
            prefix_len=len(prompt_token_ids),
            page_size=args.page_size,
        )

    return measure_with_setup(setup, run_state, warmup=args.warmup, iters=args.iters)


def bench_flashinfer_full_linear_verify(
    *,
    model,
    prompt_token_ids: Sequence[int],
    args,
    dtype: torch.dtype,
    flashinfer,
) -> tuple[TimedResult, dict[str, object]]:
    state0 = setup_flashinfer_linear_state(model=model, prompt_token_ids=prompt_token_ids, args=args)
    wrapper = make_prefill_wrapper(args=args, flashinfer=flashinfer)
    plan_linear_prefill_wrapper(wrapper, state0.spec, args=args, config=model.config, dtype=dtype)
    metadata = {
        "linear_node_count": state0.spec.node_count,
        "linear_kv_len_per_path": state0.spec.kv_len_per_path,
        "linear_total_pages": state0.spec.total_pages,
        "linear_path_count": state0.spec.path_count,
        "linear_path_depth": state0.spec.path_depth,
    }
    del state0

    def setup():
        return setup_flashinfer_linear_state(model=model, prompt_token_ids=prompt_token_ids, args=args)

    def run_state(state) -> None:
        run_flashinfer_full_linear_verify(
            model=model,
            wrapper=wrapper,
            state=state,
            prefix_len=len(prompt_token_ids),
            page_size=args.page_size,
        )

    return measure_with_setup(setup, run_state, warmup=args.warmup, iters=args.iters), metadata


def bench_flashinfer_full_verify(
    *,
    model,
    prompt_token_ids: Sequence[int],
    args,
    dtype: torch.dtype,
    flashinfer,
) -> tuple[TimedResult, dict[str, object]]:
    state0 = setup_flashinfer_state(model=model, prompt_token_ids=prompt_token_ids, args=args)
    wrapper = make_prefill_wrapper(args=args, flashinfer=flashinfer)
    plan_prefill_wrapper(wrapper, state0.tree, args=args, config=model.config, dtype=dtype)
    metadata = {
        "node_count": state0.tree.node_count,
        "kv_len": state0.tree.kv_len,
        "total_pages": state0.tree.total_pages,
        "mask_elements": int(state0.tree.bool_mask.numel()),
        "mask_true_count": int(state0.tree.bool_mask.sum().detach().cpu().item()),
        "path_count": args.k,
        "path_depth": args.d,
    }
    del state0

    def setup():
        return setup_flashinfer_state(model=model, prompt_token_ids=prompt_token_ids, args=args)

    def run_state(state) -> None:
        run_flashinfer_full_verify(
            model=model,
            wrapper=wrapper,
            state=state,
            prefix_len=len(prompt_token_ids),
            page_size=args.page_size,
        )

    return measure_with_setup(setup, run_state, warmup=args.warmup, iters=args.iters), metadata


class FlashInferFullVerifyRunner:
    """Reusable runner for Direct FlashInfer full-model masked tree verify."""

    def __init__(
        self,
        *,
        model,
        config: FlashInferFullVerifyConfig,
        model_name: str | None = None,
        flashinfer_module=None,
    ) -> None:
        config.validate()
        self.model = model
        self.config = config
        self.model_name = model_name or getattr(model, "name_or_path", None) or type(model).__name__
        self.flashinfer = flashinfer_module
        if self.flashinfer is None:
            import flashinfer

            self.flashinfer = flashinfer
        self.dtype = dtype_from_name(config.dtype)

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        *,
        config: FlashInferFullVerifyConfig,
    ) -> tuple[object, "FlashInferFullVerifyRunner"]:
        config.validate()
        tokenizer, model = load_model(
            model_path,
            dtype=dtype_from_name(config.dtype),
            device=config.device,
            trust_remote_code=config.trust_remote_code,
        )
        return tokenizer, cls(model=model, config=config, model_name=model_path)

    def benchmark(self, prompt_token_ids: Sequence[int]) -> FlashInferFullVerifyBenchmarkResult:
        prompt = [int(token_id) for token_id in prompt_token_ids]
        if len(prompt) <= 0:
            raise ValueError("prompt_token_ids must not be empty")
        if len(prompt) % int(self.config.page_size) != 0:
            raise ValueError("prompt token count must be divisible by page_size")
        metadata = self._metadata(prompt)

        hf_ar1 = bench_ar_decode(
            self.model,
            prompt,
            steps=1,
            device=self.config.device,
            warmup=self.config.warmup,
            iters=self.config.iters,
        )
        hf_ar2 = bench_ar_decode(
            self.model,
            prompt,
            steps=2,
            device=self.config.device,
            warmup=self.config.warmup,
            iters=self.config.iters,
        )
        hf_verify = bench_hf_verify_append(
            self.model,
            prompt,
            k=self.config.k,
            depth=self.config.d,
            device=self.config.device,
            warmup=self.config.warmup,
            iters=self.config.iters,
        )
        fi_ar1 = bench_flashinfer_full_ar_decode(
            model=self.model,
            prompt_token_ids=prompt,
            steps=1,
            args=self.config,
            dtype=self.dtype,
            flashinfer=self.flashinfer,
        )
        fi_ar2 = bench_flashinfer_full_ar_decode(
            model=self.model,
            prompt_token_ids=prompt,
            steps=2,
            args=self.config,
            dtype=self.dtype,
            flashinfer=self.flashinfer,
        )
        linear_verify, linear_metadata = bench_flashinfer_full_linear_verify(
            model=self.model,
            prompt_token_ids=prompt,
            args=self.config,
            dtype=self.dtype,
            flashinfer=self.flashinfer,
        )
        masked_verify, tree_metadata = bench_flashinfer_full_verify(
            model=self.model,
            prompt_token_ids=prompt,
            args=self.config,
            dtype=self.dtype,
            flashinfer=self.flashinfer,
        )
        if self.config.check_logit_alignment:
            alignment = check_flashinfer_full_verify_alignment(
                model=self.model,
                prompt_token_ids=prompt,
                args=self.config,
                dtype=self.dtype,
                flashinfer=self.flashinfer,
            )
            metadata["logit_alignment"] = alignment
            metadata["logits_aligned"] = bool(alignment.get("passed"))
            if self.config.fail_on_logit_mismatch and not bool(alignment.get("passed")):
                raise RuntimeError(f"Direct FlashInfer full verify logits are not aligned: {alignment}")
        else:
            metadata["logit_alignment"] = {
                "checked": False,
                "reason": "disabled by --skip-logit-alignment-check",
            }
            metadata["logits_aligned"] = False
        return FlashInferFullVerifyBenchmarkResult(
            metadata=metadata,
            target_ar_decode_1_token=hf_ar1,
            target_ar_decode_2_tokens=hf_ar2,
            target_hf_verify=hf_verify,
            flashinfer_full_ar_decode_1_token=fi_ar1,
            flashinfer_full_ar_decode_2_tokens=fi_ar2,
            flashinfer_full_linear_verify=linear_verify,
            flashinfer_masked_verify_full=masked_verify,
            tree_metadata=tree_metadata,
            linear_metadata=linear_metadata,
        )

    def _metadata(self, prompt_token_ids: Sequence[int]) -> dict[str, object]:
        head_dim = getattr(
            self.model.config,
            "head_dim",
            int(self.model.config.hidden_size) // int(self.model.config.num_attention_heads),
        )
        return {
            "backend": "direct_flashinfer_full_llama_masked_verify",
            "model": self.model_name,
            "semantic_correctness_required": self.config.semantic_correctness_required,
            "logits_aligned": False,
            "rope_applied": self.config.rope_applied,
            "paged_kv": True,
            "custom_mask": True,
            "packed_custom_mask": self.config.use_packed_custom_mask,
            "cascade": False,
            "prefix_len": len(prompt_token_ids),
            "k": self.config.k,
            "d": self.config.d,
            "page_size": self.config.page_size,
            "dtype": self.config.dtype,
            "device": self.config.device,
            "flashinfer_version": getattr(self.flashinfer, "__version__", "unknown"),
            "torch_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "layers": int(self.model.config.num_hidden_layers),
            "hidden_size": int(self.model.config.hidden_size),
            "num_attention_heads": int(self.model.config.num_attention_heads),
            "num_key_value_heads": int(self.model.config.num_key_value_heads),
            "head_dim": int(head_dim),
            "warmup": self.config.warmup,
            "iters": self.config.iters,
            "prefill_excluded_from_timing": True,
            "prefix_kv_page_conversion_excluded_from_timing": True,
            "fair_flashinfer_ar_baseline_included": True,
            "fair_flashinfer_linear_verify_baseline_included": True,
            "logit_alignment_check_enabled": self.config.check_logit_alignment,
            "fail_on_logit_mismatch": self.config.fail_on_logit_mismatch,
            "use_packed_custom_mask": self.config.use_packed_custom_mask,
            "alignment_atol": self.config.alignment_atol,
            "alignment_rtol": self.config.alignment_rtol,
        }


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_benchmark_report(result: FlashInferFullVerifyBenchmarkResult) -> None:
    metadata = result.metadata
    k = int(metadata["k"])
    d = int(metadata["d"])
    hf_ar1 = result.target_ar_decode_1_token
    hf_ar2 = result.target_ar_decode_2_tokens
    hf_verify = result.target_hf_verify
    fi_ar1 = result.flashinfer_full_ar_decode_1_token
    fi_ar2 = result.flashinfer_full_ar_decode_2_tokens
    linear = result.flashinfer_full_linear_verify
    masked = result.flashinfer_masked_verify_full

    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))

    print("\n=== Target AR / HF verify baselines ===")
    print_result("target_ar_decode_1_token", hf_ar1)
    print_result("target_ar_decode_2_tokens", hf_ar2)
    print_result(f"target_hf_verify_k{k}_paths_d{d}", hf_verify)

    print("\n=== Direct FlashInfer full-model AR baselines ===")
    print_result("flashinfer_full_ar_decode_1_token", fi_ar1)
    print_result("flashinfer_full_ar_decode_2_tokens", fi_ar2)

    print("\n=== Direct FlashInfer full-model verify baseline ===")
    print_result(f"flashinfer_full_linear_verify_k{k}_d{d}", linear)

    print("\n=== FlashInfer full-model masked verify ===")
    print_result(f"flashinfer_masked_verify_full_k{k}_d{d}", masked)
    print("-- ratios, median --")
    print(f"flashinfer_verify / hf_ar_decode_1    {masked.median_ms / hf_ar1.median_ms:9.3f}x")
    print(f"flashinfer_verify / hf_ar_decode_2    {masked.median_ms / hf_ar2.median_ms:9.3f}x")
    print(f"flashinfer_verify / fi_ar_decode_1    {masked.median_ms / fi_ar1.median_ms:9.3f}x")
    print(f"flashinfer_verify / fi_ar_decode_2    {masked.median_ms / fi_ar2.median_ms:9.3f}x")
    print(f"flashinfer_verify / fi_linear_verify  {masked.median_ms / linear.median_ms:9.3f}x")
    print(f"flashinfer_verify / hf_verify         {masked.median_ms / hf_verify.median_ms:9.3f}x")
    print(f"goal flashinfer_verify < fi_ar_decode_2  {'PASS' if masked.median_ms < fi_ar2.median_ms else 'FAIL'}")
    alignment = metadata.get("logit_alignment")
    if isinstance(alignment, dict) and alignment.get("checked"):
        masked_alignment = alignment.get("masked_tree", {})
        print("-- logits alignment --")
        print(f"masked_tree aligned                  {'PASS' if alignment.get('passed') else 'FAIL'}")
        if isinstance(masked_alignment, dict) and "max_abs_diff" in masked_alignment:
            print(f"masked_tree max_abs_diff             {float(masked_alignment['max_abs_diff']):9.6f}")
            print(f"masked_tree top1_match_rate          {float(masked_alignment['top1_match_rate']):9.6f}")


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
    result = runner.benchmark(prompt_token_ids)
    print_benchmark_report(result)
    report = result.to_dict()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    free_model(runner.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
