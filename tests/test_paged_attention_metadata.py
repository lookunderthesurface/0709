import torch

from atlas_0709.flashinfer_paged.paged_metadata import (
    build_flashinfer_paged_kv_metadata,
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
        torch.tensor([160 + index for index in range(16)] + [192], dtype=torch.long),
    ]
    for path in invalid_paths:
        try:
            build_flashinfer_paged_kv_metadata([path], page_size=16)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid physical page layout was accepted: {path.tolist()}")
