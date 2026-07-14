from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class FlashInferPagedKVMetadata:
    """FlashInfer page-table metadata derived from physical KV slot paths.

    ``kv_indptr`` counts pages, ``kv_page_indices`` contains physical page ids,
    and ``kv_last_page_len`` contains the number of valid tokens in each
    route's final page. ``token_index_count`` records the size of the legacy
    token-level representation for counter/benchmark comparisons.
    """

    kv_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    seq_lens: torch.Tensor
    page_size: int
    token_index_count: int
    layout_validated: bool

    @property
    def page_index_count(self) -> int:
        return int(self.kv_page_indices.numel())


def build_flashinfer_paged_kv_metadata(
    slot_paths: Sequence[torch.Tensor],
    *,
    page_size: int,
    validate_layout: bool = True,
) -> FlashInferPagedKVMetadata:
    """Compress token-slot paths into physical FlashInfer page tables.

    A division by ``page_size`` is valid only after proving that every logical
    page starts on a physical page boundary and that all valid slots inside the
    page are contiguous. Validation is vectorized; CUDA uses an asynchronous
    device assertion and therefore performs no scalar host read.
    """

    if int(page_size) <= 0:
        raise ValueError("page_size must be positive")
    if not slot_paths:
        raise ValueError("slot_paths cannot be empty")

    first = slot_paths[0]
    if first.ndim != 1:
        raise ValueError(f"slot paths must be 1-D, got {tuple(first.shape)}")
    if first.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"slot paths must be int32/int64, got {first.dtype}")

    device = first.device
    page_counts: list[int] = []
    seq_lens: list[int] = []
    page_indices: list[torch.Tensor] = []
    layout_checks: list[torch.Tensor] = []
    token_index_count = 0

    for slot_path in slot_paths:
        if slot_path.ndim != 1:
            raise ValueError(f"slot paths must be 1-D, got {tuple(slot_path.shape)}")
        if slot_path.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"slot paths must be int32/int64, got {slot_path.dtype}")
        if slot_path.device != device:
            raise ValueError("all slot paths must be on the same device")

        seq_len = int(slot_path.numel())
        if seq_len <= 0:
            raise ValueError("FlashInfer paged decode requires non-empty slot paths")
        num_pages = (seq_len + int(page_size) - 1) // int(page_size)
        page_starts = slot_path[:: int(page_size)]
        if int(page_starts.numel()) != num_pages:
            raise RuntimeError("internal page-count mismatch")

        page_counts.append(num_pages)
        seq_lens.append(seq_len)
        token_index_count += seq_len
        page_indices.append(
            torch.div(page_starts, int(page_size), rounding_mode="floor").to(torch.int32)
        )

        if validate_layout:
            offsets = torch.arange(seq_len, device=device, dtype=slot_path.dtype)
            offsets = torch.remainder(offsets, int(page_size))
            expanded_starts = torch.repeat_interleave(
                page_starts,
                int(page_size),
            )[:seq_len]
            layout_checks.append(
                torch.logical_and(
                    torch.remainder(page_starts, int(page_size)).eq(0).all(),
                    slot_path.eq(expanded_starts + offsets).all(),
                )
            )

    if validate_layout:
        _assert_layout_valid(torch.stack(layout_checks).all())

    indptr_values = [0]
    for count in page_counts:
        indptr_values.append(indptr_values[-1] + int(count))
    last_page_lens = [
        seq_len - (page_count - 1) * int(page_size)
        for seq_len, page_count in zip(seq_lens, page_counts)
    ]

    return FlashInferPagedKVMetadata(
        kv_indptr=torch.tensor(indptr_values, device=device, dtype=torch.int32),
        kv_page_indices=torch.cat(page_indices, dim=0),
        kv_last_page_len=torch.tensor(last_page_lens, device=device, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        page_size=int(page_size),
        token_index_count=int(token_index_count),
        layout_validated=bool(validate_layout),
    )


def _assert_layout_valid(condition: torch.Tensor) -> None:
    message = (
        "physical KV slot paths are not page-aligned contiguous pages; "
        "cannot safely compress token slots into FlashInfer page indices"
    )
    if condition.device.type == "cpu":
        if not bool(condition.item()):
            raise RuntimeError(message)
        return
    assert_async = getattr(torch, "_assert_async", None)
    if callable(assert_async):
        assert_async(condition, message)
        return
    raise RuntimeError(
        "CUDA page-layout validation requires torch._assert_async to avoid "
        "a device-to-host synchronization"
    )
