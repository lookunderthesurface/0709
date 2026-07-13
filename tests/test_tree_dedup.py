from __future__ import annotations

from atlas_0709.flashinfer_full_verify import build_deduplicated_path_tree_tokens


def test_deduplicated_tree_reuses_shared_token_prefixes() -> None:
    tokens, parents, depths, route_nodes = build_deduplicated_path_tree_tokens(
        [
            (10, 20, 30, 40),
            (10, 20, 31, 41),
            (11, 21, 32, 42),
        ],
        device="cpu",
    )

    assert tokens.tolist() == [10, 20, 30, 40, 31, 41, 11, 21, 32, 42]
    assert parents == [-1, 0, 1, 2, 1, 4, -1, 6, 7, 8]
    assert depths.tolist() == [0, 1, 2, 3, 2, 3, 0, 1, 2, 3]
    assert route_nodes == [
        [0, 1, 2, 3],
        [0, 1, 4, 5],
        [6, 7, 8, 9],
    ]


def test_deduplicated_tree_keeps_identical_routes_logically_distinct() -> None:
    tokens, parents, depths, route_nodes = build_deduplicated_path_tree_tokens(
        [(7, 8), (7, 8)],
        device="cpu",
    )

    assert tokens.tolist() == [7, 8]
    assert parents == [-1, 0]
    assert depths.tolist() == [0, 1]
    assert route_nodes == [[0, 1], [0, 1]]
