from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .runtime_metadata import package_version
from .sglang_page_attention import AtlasPagedDecodeSpec


@dataclass(frozen=True)
class SGLangRouteKVMetadata:
    """Per-frontier metadata required by SGLang decode.

    ``req_pool_indices`` index SGLang's ReqToTokenPool rows. Those rows must
    already contain the logical token-to-KV-slot path for each route. 
    ``out_cache_loc`` is the newly allocated KV slot for the pending token of
    each route.
    """

    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    orig_seq_lens: torch.Tensor | None = None
    seq_lens_sum: int | None = None
    seq_lens_cpu: torch.Tensor | None = None
    attention_page_size: int = 1
    token_index_count: int = 0
    page_index_count: int = 0
    paged_decode_spec: AtlasPagedDecodeSpec | None = None


class FlashInferBackendBase:
    paged_kv_enabled = True

    def __init__(self, *, model_runner: Any, page_size: int) -> None:
        self.model_runner = model_runner
        self.page_size = int(page_size)
        self.flashinfer_version = package_version("flashinfer-python") or package_version("flashinfer")
        if self.flashinfer_version is None:
            raise RuntimeError("FlashInfer backend requested, but flashinfer-python is not installed.")

    def runtime_metadata(self, *, cascade_level: int, verify_backend: str) -> dict[str, Any]:
        return {
            "model_runner_class": type(self.model_runner).__name__,
            "attention_backend_class": _runner_attr_class_name(
                self.model_runner,
                ("attn_backend", "attention_backend", "decode_attn_backend", "attn_backends"),
            ),
            "kv_cache_class": _runner_attr_class_name(
                self.model_runner,
                ("token_to_kv_pool", "token_to_kv_pool_allocator", "kv_pool_allocator"),
            ),
            "flashinfer_version": self.flashinfer_version,
            "page_size": self.page_size,
            "paged_kv_enabled": True,
            "cascade_level": cascade_level,
            "verify_backend": verify_backend,
        }


class SGLangFlashInferPagedDecodeBackend(FlashInferBackendBase):
    """Paged decode through an already initialized SGLang ModelRunner.

    This backend is the first real execution bridge. It does not instantiate a
    server or allocate routes by itself; ATLAS owns route topology, while the
    caller must provide SGLang req-pool rows and newly allocated KV slots.
    """

    def __init__(self, *, model_runner: Any, page_size: int) -> None:
        super().__init__(model_runner=model_runner, page_size=page_size)
        self._ForwardBatch = _import_attr_any(
            [
                "sglang.srt.model_executor.forward_batch_info",
                "sglang.srt.managers.schedule_batch",
            ],
            "ForwardBatch",
        )
        self._ForwardMode = _import_attr_any(
            [
                "sglang.srt.model_executor.forward_batch_info",
                "sglang.srt.managers.schedule_batch",
            ],
            "ForwardMode",
        )

    def forward_frontier(
        self,
        *,
        input_ids: torch.Tensor,
        route_kv_metadata: SGLangRouteKVMetadata,
    ) -> torch.Tensor:
        metadata = route_kv_metadata
        _validate_1d_long("input_ids", input_ids)
        _validate_1d_long("req_pool_indices", metadata.req_pool_indices)
        _validate_1d_long("seq_lens", metadata.seq_lens)
        _validate_1d_long("out_cache_loc", metadata.out_cache_loc)
        _validate_1d_long("positions", metadata.positions)

        batch_size = int(input_ids.numel())
        if not (
            metadata.req_pool_indices.numel()
            == metadata.seq_lens.numel()
            == metadata.out_cache_loc.numel()
            == metadata.positions.numel()
            == batch_size
        ):
            raise ValueError("SGLang frontier tensors must all have shape [B]")

        forward_batch = self._ForwardBatch(
            forward_mode=_decode_forward_mode(self._ForwardMode),
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=metadata.seq_lens,
            out_cache_loc=metadata.out_cache_loc,
            seq_lens_sum=(
                int(metadata.seq_lens_sum)
                if metadata.seq_lens_sum is not None
                else int(metadata.seq_lens.sum().item())
            ),
            orig_seq_lens=metadata.orig_seq_lens,
            positions=metadata.positions,
            seq_lens_cpu=(
                metadata.seq_lens_cpu
                if metadata.seq_lens_cpu is not None
                else metadata.seq_lens.detach().cpu()
            ),
            spec_info=metadata.paged_decode_spec,
            capture_hidden_mode=_capture_hidden_mode_null(),
        )
        output = self.model_runner.forward(forward_batch)
        return _extract_next_token_logits(output)


def _import_attr(module_name: str, attr_name: str) -> Any:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as exc:
        raise RuntimeError(f"failed to import {module_name}.{attr_name}: {exc!r}") from exc


def _import_attr_any(module_names: list[str], attr_name: str) -> Any:
    errors: list[str] = []
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            return getattr(module, attr_name)
        except Exception as exc:
            errors.append(f"{module_name}.{attr_name}: {exc!r}")
    raise RuntimeError("failed to import any SGLang symbol: " + "; ".join(errors))


def _runner_attr_class_name(obj: Any, names: tuple[str, ...]) -> str:
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and value:
            return ",".join(type(item).__name__ for item in value)
        return type(value).__name__
    return "NoneType"


def _validate_1d_long(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must have shape [B], got {tuple(tensor.shape)}")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be int32/int64, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _decode_forward_mode(ForwardMode: Any) -> Any:
    for name in ("DECODE", "decode", "Decode"):
        if hasattr(ForwardMode, name):
            return getattr(ForwardMode, name)
    for item in ForwardMode:
        if getattr(item, "name", "").lower() == "decode":
            return item
    raise RuntimeError("SGLang ForwardMode has no DECODE value")


def _capture_hidden_mode_null() -> Any:
    CaptureHiddenMode = _import_attr_any(
        [
            "sglang.srt.model_executor.forward_batch_info",
            "sglang.srt.managers.schedule_batch",
        ],
        "CaptureHiddenMode",
    )
    return getattr(CaptureHiddenMode, "NULL")


def _extract_next_token_logits(output: Any) -> torch.Tensor:
    candidates = [
        output,
        getattr(output, "logits_output", None),
        getattr(output, "output", None),
    ]
    attr_names = [
        "next_token_logits",
        "next_token_logits_buffer",
        "logits",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for attr_name in attr_names:
            value = getattr(candidate, attr_name, None)
            if isinstance(value, torch.Tensor):
                return value
    raise RuntimeError(
        "could not extract next-token logits from SGLang ModelRunner output"
    )
