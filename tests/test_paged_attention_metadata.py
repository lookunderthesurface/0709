from types import SimpleNamespace

import torch

from atlas_0709.flashinfer_paged.paged_metadata import (
    build_flashinfer_paged_kv_metadata,
)
from atlas_0709.flashinfer_paged.sglang_page_attention import (
    AtlasPagedDecodeSpec,
    install_atlas_paged_decode_attention,
    reshape_token_kv_cache_as_pages,
)


def _slot_path(page_ids: list[int], length: int, page_size: int = 16) -> torch.Tensor:
    slots = [
        page_id * page_size + offset
        for page_id in page_ids
        for offset in range(page_size)
    ]
    return torch.tensor(slots[:length], dtype=torch.long)


def test_page_metadata_boundaries_and_noncontiguous_physical_pages() -> None:
    lengths = [1, 15, 16, 17, 31, 32]
    paths = [
        _slot_path([11 + 3 * index, 31 + 5 * index], length)
        for index, length in enumerate(lengths)
    ]

    metadata = build_flashinfer_paged_kv_metadata(paths, page_size=16)

    expected_counts = [(length + 15) // 16 for length in lengths]
    expected_indptr = [0]
    for count in expected_counts:
        expected_indptr.append(expected_indptr[-1] + count)
    assert metadata.kv_indptr.tolist() == expected_indptr
    assert metadata.kv_last_page_len.tolist() == [1, 15, 16, 1, 15, 16]
    assert metadata.seq_lens.tolist() == lengths
    assert metadata.token_index_count == sum(lengths)
    assert metadata.page_index_count == sum(expected_counts)
    assert metadata.layout_validated


def test_long_prefix_reduces_metadata_indices_by_page_size() -> None:
    path = _slot_path(list(range(100, 612)), 8192)
    metadata = build_flashinfer_paged_kv_metadata([path], page_size=16)

    assert metadata.token_index_count == 8192
    assert metadata.page_index_count == 512
    assert metadata.kv_indptr.tolist() == [0, 512]
    assert metadata.kv_last_page_len.tolist() == [16]


def test_layout_validation_rejects_unaligned_or_noncontiguous_slots() -> None:
    invalid_paths = [
        torch.tensor([161, 162, 163], dtype=torch.long),
        torch.tensor([160, 161, 165], dtype=torch.long),
        torch.tensor([160 + index for index in range(16)] + [193], dtype=torch.long),
    ]
    for path in invalid_paths:
        try:
            build_flashinfer_paged_kv_metadata([path], page_size=16)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid physical page layout was accepted: {path.tolist()}")


def test_flat_sglang_kv_cache_is_viewed_as_physical_pages() -> None:
    cache = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
    paged = reshape_token_kv_cache_as_pages(cache, page_size=4)

    assert paged.shape == (2, 4, 2, 3)
    assert paged.data_ptr() == cache.data_ptr()
    assert paged[1, 2].tolist() == cache[6].tolist()


def test_dedicated_updater_passes_real_page_size_and_preserves_legacy_fallback() -> None:
    class FakeUpdater:
        num_qo_heads = 8
        num_kv_heads = 2
        head_dim = 64
        data_type = torch.float16
        q_data_type = torch.float16

        def __init__(self) -> None:
            self.legacy_calls = 0

        def call_begin_forward(self, *args, **kwargs) -> None:
            self.legacy_calls += 1

    class FakeWrapper:
        is_cuda_graph_enabled = False

        def __init__(self) -> None:
            self.args = None
            self.kwargs = None

        def begin_forward(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    updater = FakeUpdater()
    backend = SimpleNamespace(
        dispatch_reason=None,
        indices_updater_decode=updater,
        forward_decode=lambda *args, **kwargs: None,
    )
    install_atlas_paged_decode_attention(backend, page_size=16)

    spec = AtlasPagedDecodeSpec(
        kv_indptr=torch.tensor([0, 2], dtype=torch.int32),
        kv_indices=torch.tensor([5, 9], dtype=torch.int32),
        kv_last_page_len=torch.tensor([3], dtype=torch.int32),
        page_size=16,
    )
    wrapper = FakeWrapper()
    updater.call_begin_forward(
        wrapper,
        torch.tensor([1]),
        torch.tensor([19]),
        19,
        torch.tensor([0, 19]),
        None,
        spec,
        torch.tensor([19]),
    )

    assert wrapper.args[0] is spec.kv_indptr
    assert wrapper.args[1] is spec.kv_indices
    assert wrapper.args[2] is spec.kv_last_page_len
    assert wrapper.args[6] == 16
    assert updater.legacy_calls == 0

    updater.call_begin_forward(
        wrapper,
        torch.tensor([1]),
        torch.tensor([19]),
        19,
        torch.tensor([0, 19]),
        None,
        None,
        torch.tensor([19]),
    )
    assert updater.legacy_calls == 1
