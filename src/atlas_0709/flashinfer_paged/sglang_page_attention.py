from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch


@dataclass(frozen=True)
class AtlasPagedDecodeSpec:
    """0709-only page metadata carried through SGLang's ``ForwardBatch``."""

    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    page_size: int


def reshape_token_kv_cache_as_pages(
    cache: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Expose SGLang's flat NHD token pool as FlashInfer NHD physical pages."""

    if cache.ndim == 4:
        if int(cache.shape[1]) != int(page_size):
            raise RuntimeError(
                "existing paged KV cache has the wrong page size: "
                f"shape={tuple(cache.shape)}, expected_page_size={int(page_size)}"
            )
        return cache
    if cache.ndim != 3:
        raise RuntimeError(
            "ATLAS paged FlashInfer decode expects a flat [slots, heads, dim] "
            f"or paged [pages, page_size, heads, dim] KV tensor, got {tuple(cache.shape)}"
        )
    if int(cache.shape[0]) % int(page_size) != 0:
        raise RuntimeError(
            "flat SGLang KV pool length is not divisible by page_size: "
            f"slots={int(cache.shape[0])}, page_size={int(page_size)}"
        )
    return cache.view(
        int(cache.shape[0]) // int(page_size),
        int(page_size),
        int(cache.shape[1]),
        int(cache.shape[2]),
    )


def install_atlas_paged_decode_attention(
    attention_backend: Any,
    *,
    page_size: int,
) -> Any:
    """Install the minimal 0709 page-size decode path on SGLang 0.5.14.

    SGLang's stock FlashInfer updater treats every token slot as a page of size
    one. ATLAS carries an explicit, already validated physical page table in
    ``ForwardBatch.spec_info``. Only batches carrying ``AtlasPagedDecodeSpec``
    use this path; prefill/extend and unrelated SGLang batches retain the stock
    implementation.
    """

    if int(page_size) <= 1:
        return attention_backend
    installed_page_size = getattr(
        attention_backend,
        "atlas_paged_decode_page_size",
        None,
    )
    if installed_page_size is not None:
        if int(installed_page_size) != int(page_size):
            raise RuntimeError(
                "ATLAS paged decode was already installed with a different page size: "
                f"installed={int(installed_page_size)}, requested={int(page_size)}"
            )
        return attention_backend

    if getattr(attention_backend, "dispatch_reason", None) is not None:
        raise RuntimeError(
            "the 0709 dedicated page metadata path currently supports the single-wrapper "
            "full-attention Drafter only"
        )
    updater = getattr(attention_backend, "indices_updater_decode", None)
    if updater is None or not callable(getattr(updater, "call_begin_forward", None)):
        raise RuntimeError("SGLang FlashInfer decode updater is unavailable")

    original_call_begin_forward = updater.call_begin_forward

    def call_begin_forward(
        self: Any,
        wrapper: Any,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor | None,
        spec_info: Any,
        seq_lens_cpu: torch.Tensor | None,
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: int | None = None,
        disable_split_kv: bool | None = None,
    ) -> None:
        if not isinstance(spec_info, AtlasPagedDecodeSpec):
            return original_call_begin_forward(
                wrapper,
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                kv_indptr,
                kv_start_idx,
                spec_info,
                seq_lens_cpu,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
            )
        if int(spec_info.page_size) != int(page_size):
            raise RuntimeError(
                "ATLAS ForwardBatch page size does not match the installed decode backend: "
                f"metadata={int(spec_info.page_size)}, backend={int(page_size)}"
            )
        if kv_start_idx is not None or use_sliding_window_kv_pool:
            raise RuntimeError("ATLAS page metadata does not support sliding-window remapping")
        if bool(getattr(wrapper, "is_cuda_graph_enabled", False)):
            raise RuntimeError("ATLAS page metadata currently supports eager decode only")

        wrapper.begin_forward(
            spec_info.kv_indptr,
            spec_info.kv_indices,
            spec_info.kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            int(page_size),
            data_type=self.data_type,
            q_data_type=self.q_data_type,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            disable_split_kv=(
                disable_split_kv if disable_split_kv is not None else False
            ),
        )

    updater.call_begin_forward = MethodType(call_begin_forward, updater)

    original_forward_decode = attention_backend.forward_decode

    def forward_decode(
        self: Any,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: Any,
        forward_batch: Any,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        spec_info = getattr(forward_batch, "spec_info", None)
        if not isinstance(spec_info, AtlasPagedDecodeSpec):
            return original_forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
            )

        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if k is not None:
            if v is None:
                raise RuntimeError("value tensor is required when key tensor is present")
            if save_kv_cache:
                KVWriteLoc = _import_attr(
                    "sglang.srt.mem_cache.memory_pool",
                    "KVWriteLoc",
                )
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        paged_cache = (
            reshape_token_kv_cache_as_pages(key_cache, page_size=int(page_size)),
            reshape_token_kv_cache_as_pages(value_cache, page_size=int(page_size)),
        )
        output = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            paged_cache,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        return output.view(-1, layer.tp_q_head_num * layer.head_dim)

    attention_backend.forward_decode = MethodType(forward_decode, attention_backend)
    setattr(attention_backend, "atlas_paged_decode_page_size", int(page_size))
    setattr(attention_backend, "atlas_paged_decode_enabled", True)
    return attention_backend


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
