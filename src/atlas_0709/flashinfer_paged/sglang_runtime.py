from __future__ import annotations

import importlib
import inspect
import pkgutil
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .flashinfer_backends import (
    SGLangFlashInferPagedDecodeBackend,
    SGLangRouteKVMetadata,
    _extract_next_token_logits,
    _import_attr_any,
)
from .kv import KVTreeStore
from .paged_metadata import build_flashinfer_paged_kv_metadata
from .sglang_page_attention import AtlasPagedDecodeSpec, install_atlas_paged_decode_attention
from .types import DraftPrefixState, FrontierDecodeOutput, PrefixKVView, RouteState


@dataclass(frozen=True)
class SGLangRunnerConfig:
    model_path: str
    tokenizer_path: str | None = None
    dtype: str = "float16"
    kv_cache_dtype: str = "auto"
    context_length: int | None = None
    page_size: int = 16
    mem_fraction_static: float = 0.75
    max_running_requests: int | None = None
    max_total_tokens: int | None = None
    gpu_id: int = 0
    tp_size: int = 1
    pp_size: int = 1
    nccl_port: int = 29500
    trust_remote_code: bool = False
    load_format: str = "auto"
    model_impl: str = "auto"
    device: str = "cuda"
    skip_tokenizer_init: bool = True
    disable_radix_cache: bool = True
    disable_cuda_graph: bool = True
    skip_server_warmup: bool = True
    use_page_attention_metadata: bool = True
    validate_page_layout: bool = True


@dataclass
class SGLangRouteRow:
    route_id: int
    req_pool_index: int
    handle: Any
    written_length: int = 0
    pending_dirty_start: int | None = None


@dataclass(frozen=True)
class SGLangMemoryPoolBundle:
    req_to_token_pool: Any
    token_to_kv_pool: Any
    token_to_kv_pool_allocator: Any
    max_running_requests: int
    max_context_len: int
    max_total_tokens: int
    head_num: int
    head_dim: int
    layer_num: int
    dtype: torch.dtype


@dataclass(frozen=True)
class SGLangPrefillResult:
    prompt_token_ids: tuple[int, ...]
    prefix_slot_ids: torch.Tensor
    next_token_logits: torch.Tensor
    req_pool_index: int
    forward_metadata: dict[str, Any]


@dataclass
class _PoolReqHandle:
    rid: str
    req_pool_idx: int | None = None


@dataclass
class SGLangRoutePoolBridge:
    """Bind ATLAS logical routes to SGLang req rows and physical KV slots."""

    req_to_token_pool: Any
    token_to_kv_pool_allocator: Any
    page_size: int
    max_context_len: int | None = None
    device: str | torch.device = "cuda"
    prefix_slot_ids: torch.Tensor | None = None
    prefix_slot_count: int = 0
    prefix_page_ids: set[int] = field(default_factory=set)
    route_rows: dict[int, SGLangRouteRow] = field(default_factory=dict)
    node_slot_ids: dict[int, int | torch.Tensor] = field(default_factory=dict)
    route_slot_paths: dict[int, torch.Tensor] = field(default_factory=dict)
    route_tail_page_keys: dict[int, int] = field(default_factory=dict)
    prefix_tail_page_key: int | None = None
    prefix_partial_tail_keys: dict[int, int] = field(default_factory=dict)
    next_tail_page_key: int = 1
    owned_page_ids: set[int] = field(default_factory=set)
    pending_owned_slot_tensors: list[torch.Tensor] = field(default_factory=list)
    cow_pages_copied: int = 0
    cow_tokens_copied: int = 0
    attention_metadata_builds: int = 0
    attention_metadata_token_indices: int = 0
    attention_metadata_page_indices: int = 0
    req_row_write_calls: int = 0
    req_row_elements_written: int = 0
    req_row_full_rewrite_calls: int = 0
    req_row_full_rewrite_elements: int = 0
    req_row_full_init_calls: int = 0
    req_row_full_init_elements: int = 0
    req_row_append_calls: int = 0
    req_row_append_elements: int = 0
    req_row_cow_patch_calls: int = 0
    req_row_cow_patch_elements: int = 0
    hot_path_host_transfer_batches: int = 0
    hot_path_host_transfer_elements: int = 0
    use_page_attention_metadata: bool = True
    validate_page_layout: bool = True

    @classmethod
    def from_runner(
        cls,
        model_runner: Any,
        *,
        page_size: int,
        device: str | torch.device = "cuda",
    ) -> "SGLangRoutePoolBridge":
        req_pool = _first_existing_attr(model_runner, ("req_to_token_pool",))
        allocator = _first_existing_attr(
            model_runner,
            ("token_to_kv_pool_allocator", "token_to_kv_pool_alloc", "kv_pool_allocator"),
        )
        if req_pool is None:
            raise RuntimeError("SGLang ModelRunner does not expose req_to_token_pool")
        if allocator is None:
            raise RuntimeError("SGLang ModelRunner does not expose token_to_kv_pool_allocator")
        actual_page_size = getattr(model_runner, "atlas_pool_page_size", None)
        if actual_page_size is None:
            token_pool = _first_existing_attr(model_runner, ("token_to_kv_pool", "kv_pool", "kvcache"))
            actual_page_size = getattr(token_pool, "page_size", None)
        if actual_page_size is not None and int(actual_page_size) != int(page_size):
            raise RuntimeError(
                "ATLAS page-size argument does not match the physical SGLang KV pool: "
                f"requested={int(page_size)}, physical={int(actual_page_size)}"
            )
        max_context_len = getattr(model_runner, "atlas_max_context_len", None)
        if max_context_len is None:
            max_context_len = _req_to_token_pool_context_capacity(req_pool)
        return cls(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            max_context_len=int(max_context_len) if max_context_len is not None else None,
            device=device,
            use_page_attention_metadata=bool(
                getattr(model_runner, "atlas_use_page_attention_metadata", True)
            ),
            validate_page_layout=bool(
                getattr(model_runner, "atlas_validate_page_layout", True)
            ),
        )

    def set_prefix_slots(self, slot_ids: Sequence[int] | torch.Tensor) -> None:
        self.prefix_slot_ids = _as_1d_long_tensor(slot_ids, device=self.device)
        self._refresh_prefix_cpu_metadata()
        self.owned_page_ids = set(self.prefix_page_ids)
        self.pending_owned_slot_tensors.clear()
        self.prefix_partial_tail_keys.clear()
        self.prefix_tail_page_key = (
            self._new_tail_page_key()
            if self.prefix_slot_count
            and self.prefix_slot_count % self.page_size
            else None
        )
        if self.prefix_tail_page_key is not None:
            tail_page_index = (self.prefix_slot_count - 1) // self.page_size
            self.prefix_partial_tail_keys[tail_page_index] = self.prefix_tail_page_key

    def _refresh_prefix_cpu_metadata(self) -> None:
        if self.prefix_slot_ids is None or int(self.prefix_slot_ids.numel()) == 0:
            self.prefix_slot_count = 0
            self.prefix_page_ids.clear()
            return
        self.prefix_slot_count = int(self.prefix_slot_ids.numel())
        page_ids = torch.div(
            self.prefix_slot_ids[:: self.page_size],
            self.page_size,
            rounding_mode="floor",
        ).to(dtype=torch.long)
        self.prefix_page_ids = {
            int(page_id)
            for page_id in _int_list(page_ids)
        }

    def _new_tail_page_key(self) -> int:
        # Negative lease ids cannot collide with fallback physical page ids,
        # which are non-negative.
        key = -int(self.next_tail_page_key)
        self.next_tail_page_key += 1
        return key

    def _record_pending_owned_slots(self, slot_ids: torch.Tensor) -> None:
        if int(slot_ids.numel()) > 0:
            self.pending_owned_slot_tensors.append(slot_ids.detach().reshape(-1))

    def _materialize_pending_owned_pages(self) -> None:
        """Update CPU ownership only at pruning/report boundaries.

        Decode keeps newly allocated slot ids on device.  Their page ids are
        not needed for attention or request-row writes, so transferring them
        in every frontier would add a second D2H fence beside candidate
        materialization.  Pruning and diagnostics flush all pending slots in
        one batch instead.
        """

        if not self.pending_owned_slot_tensors:
            return
        slots = torch.cat(self.pending_owned_slot_tensors, dim=0)
        self.owned_page_ids.update(
            int(slot_id) // self.page_size
            for slot_id in _int_list(slots)
        )
        self.pending_owned_slot_tensors.clear()

    def owned_page_count(self) -> int:
        self._materialize_pending_owned_pages()
        return len(self.owned_page_ids)

    def bind_existing_route_row(
        self,
        route_id: int,
        req_pool_index: int,
        handle: Any | None = None,
        *,
        written_length: int = 0,
    ) -> None:
        if int(written_length) < 0:
            raise ValueError("written_length must be non-negative")
        if handle is None:
            handle = _PoolReqHandle(rid=f"atlas-{int(route_id)}", req_pool_idx=int(req_pool_index))
        self.route_rows[int(route_id)] = SGLangRouteRow(
            route_id=int(route_id),
            req_pool_index=int(req_pool_index),
            handle=handle,
            written_length=int(written_length),
        )

    def ensure_route_rows(self, routes: Sequence[RouteState]) -> torch.Tensor:
        missing = [route for route in routes if int(route.route_id) not in self.route_rows]
        if missing:
            handles = [_PoolReqHandle(rid=f"atlas-{int(route.route_id)}") for route in missing]
            indices = self.req_to_token_pool.alloc(handles)
            if indices is None:
                raise RuntimeError("SGLang ReqToTokenPool.alloc returned None")
            indices_list = _int_list(indices)
            if len(indices_list) != len(missing):
                raise RuntimeError("SGLang ReqToTokenPool.alloc returned the wrong number of rows")
            for route, handle, req_pool_index in zip(missing, handles, indices_list):
                handle.req_pool_idx = int(req_pool_index)
                self.bind_existing_route_row(route.route_id, int(req_pool_index), handle=handle)

        return torch.tensor(
            [self.route_rows[int(route.route_id)].req_pool_index for route in routes],
            device=self.device,
            dtype=torch.long,
        )

    def release_route_rows(self, route_ids: Sequence[int]) -> None:
        for route_id in route_ids:
            row = self.route_rows.pop(int(route_id), None)
            if row is not None and hasattr(self.req_to_token_pool, "free"):
                self.req_to_token_pool.free(row.handle)

    def retain_route_rows(self, route_ids: Sequence[int]) -> int:
        retained = {int(route_id) for route_id in route_ids}
        released = [
            int(route_id)
            for route_id in self.route_rows
            if int(route_id) not in retained
        ]
        self.release_route_rows(released)
        self.route_slot_paths = {
            route_id: slot_path
            for route_id, slot_path in self.route_slot_paths.items()
            if route_id in retained
        }
        self.route_tail_page_keys = {
            route_id: tail_key
            for route_id, tail_key in self.route_tail_page_keys.items()
            if route_id in retained
        }
        return len(released)

    def register_node_slots(self, node_ids: Sequence[int] | torch.Tensor, slot_ids: Sequence[int] | torch.Tensor) -> None:
        node_list = _int_list(node_ids)
        if isinstance(slot_ids, torch.Tensor):
            slot_values: list[int | torch.Tensor] = [
                slot_ids.reshape(-1)[index]
                for index in range(int(slot_ids.numel()))
            ]
        else:
            slot_values = [int(slot_id) for slot_id in slot_ids]
        if len(node_list) != len(slot_values):
            raise ValueError("node_ids and slot_ids must have the same length")
        for node_id, slot_id in zip(node_list, slot_values):
            self.node_slot_ids[int(node_id)] = slot_id

    def reconcile_route_rows(
        self,
        previous_routes: Sequence[RouteState],
        next_routes: Sequence[RouteState],
    ) -> None:
        children_by_parent: dict[int, list[RouteState]] = {}
        for child in next_routes:
            if child.parent_route_id is None:
                continue
            children_by_parent.setdefault(int(child.parent_route_id), []).append(child)

        for parent in previous_routes:
            parent_id = int(parent.route_id)
            row = self.route_rows.pop(parent_id, None)
            children = children_by_parent.get(parent_id, [])
            parent_slot_path = self.route_slot_paths.get(parent_id)
            if parent_slot_path is not None:
                for child in children:
                    self.route_slot_paths[int(child.route_id)] = parent_slot_path
                    parent_tail_key = self.route_tail_page_keys.get(parent_id)
                    if parent_tail_key is not None:
                        self.route_tail_page_keys[int(child.route_id)] = parent_tail_key
            if row is None:
                continue

            if children:
                child = children[0]
                self.route_rows[int(child.route_id)] = SGLangRouteRow(
                    route_id=int(child.route_id),
                    req_pool_index=row.req_pool_index,
                    handle=row.handle,
                    written_length=row.written_length,
                    pending_dirty_start=row.pending_dirty_start,
                )
                continue

            if hasattr(self.req_to_token_pool, "free"):
                self.req_to_token_pool.free(row.handle)

    def commit_route_as_prefix(self, route: RouteState) -> torch.Tensor:
        """Promote a materialized route path without copying its physical KV."""
        old_prefix_len = 0 if self.prefix_slot_ids is None else int(self.prefix_slot_ids.numel())
        full_slot_path = self._slot_path_for_route(route)
        expected_len = old_prefix_len + len(route.kv_view.node_ids)
        if int(full_slot_path.numel()) != expected_len:
            raise RuntimeError(
                "cannot promote route KV to prefix: "
                f"slot_path_len={int(full_slot_path.numel())}, expected={expected_len}"
            )
        self.prefix_slot_ids = full_slot_path
        self._refresh_prefix_cpu_metadata()
        self.prefix_partial_tail_keys.clear()
        self.prefix_tail_page_key = (
            self.route_tail_page_keys.get(int(route.route_id))
            if int(full_slot_path.numel()) % self.page_size
            else None
        )
        if self.prefix_tail_page_key is not None:
            tail_page_index = (int(full_slot_path.numel()) - 1) // self.page_size
            self.prefix_partial_tail_keys[tail_page_index] = self.prefix_tail_page_key
        return full_slot_path

    def page_reference_counts(self) -> dict[int, int]:
        """Return read-only page reference counts for ownership validation.

        A reference is counted once per physical path view, not once per token
        slot.  The committed prefix contributes one reference to every page it
        reaches, and every retained route path contributes another.  Physical
        release remains the mark-and-sweep operation below; this helper is for
        invariants, tests, and diagnostics only.  Read-only references do not
        by themselves determine which active route holds the tail writer lease.
        """

        counts: dict[int, int] = {}
        paths = [self.prefix_slot_ids, *self.route_slot_paths.values()]
        for page_ids in self._page_id_sets_for_paths(paths):
            for page_id in page_ids:
                counts[page_id] = counts.get(page_id, 0) + 1
        return dict(sorted(counts.items()))

    def _page_id_sets_for_paths(
        self,
        paths: Sequence[torch.Tensor | None],
    ) -> list[set[int]]:
        page_counts: list[int] = []
        page_parts: list[torch.Tensor] = []
        for path in paths:
            if path is None or int(path.numel()) == 0:
                page_counts.append(0)
                continue
            sampled_slots = torch.cat((path[:: self.page_size], path[-1:]))
            page_ids = torch.div(
                sampled_slots,
                self.page_size,
                rounding_mode="floor",
            ).to(dtype=torch.long)
            page_counts.append(int(page_ids.numel()))
            page_parts.append(page_ids)
        flat_page_ids = _int_list(torch.cat(page_parts)) if page_parts else []
        result: list[set[int]] = []
        offset = 0
        for count in page_counts:
            result.append(
                {int(page_id) for page_id in flat_page_ids[offset : offset + count]}
            )
            offset += count
        return result

    def release_unreferenced_node_slots(self, live_node_ids: Sequence[int]) -> int:
        """Release pages that contain no committed-prefix or live-route slots.

        SGLang's paged allocator frees whole pages even when passed one token
        slot. A page is therefore returned only when none of its slots remains
        reachable. Dead node mappings on a partially live page are forgotten
        but the physical page stays allocated.
        """
        self._materialize_pending_owned_pages()
        live_nodes = {int(node_id) for node_id in live_node_ids}
        live_pages = set(self.prefix_page_ids)
        for page_ids in self._page_id_sets_for_paths(
            list(self.route_slot_paths.values())
        ):
            live_pages.update(page_ids)
        releasable_pages = sorted(self.owned_page_ids - live_pages)

        for node_id in list(self.node_slot_ids):
            if int(node_id) not in live_nodes:
                self.node_slot_ids.pop(int(node_id), None)

        if releasable_pages:
            free_slots = torch.tensor(
                [page_id * self.page_size for page_id in releasable_pages],
                device=self.device,
                dtype=torch.long,
            )
            free = getattr(self.token_to_kv_pool_allocator, "free", None)
            if not callable(free):
                raise RuntimeError("SGLang token-to-KV allocator does not expose free()")
            free(free_slots)
            self.owned_page_ids.difference_update(releasable_pages)
        return len(releasable_pages)

    def clear_physical_state(self) -> None:
        self.prefix_slot_ids = None
        self.prefix_slot_count = 0
        self.prefix_page_ids.clear()
        self.prefix_tail_page_key = None
        self.prefix_partial_tail_keys.clear()
        self.route_rows.clear()
        self.node_slot_ids.clear()
        self.route_slot_paths.clear()
        self.route_tail_page_keys.clear()
        self.next_tail_page_key = 1
        self.owned_page_ids.clear()
        self.pending_owned_slot_tensors.clear()
        self.cow_pages_copied = 0
        self.cow_tokens_copied = 0
        self.attention_metadata_builds = 0
        self.attention_metadata_token_indices = 0
        self.attention_metadata_page_indices = 0
        self.req_row_write_calls = 0
        self.req_row_elements_written = 0
        self.req_row_full_rewrite_calls = 0
        self.req_row_full_rewrite_elements = 0
        self.req_row_full_init_calls = 0
        self.req_row_full_init_elements = 0
        self.req_row_append_calls = 0
        self.req_row_append_elements = 0
        self.req_row_cow_patch_calls = 0
        self.req_row_cow_patch_elements = 0
        self.hot_path_host_transfer_batches = 0
        self.hot_path_host_transfer_elements = 0
        clear_req_pool = getattr(self.req_to_token_pool, "clear", None)
        if callable(clear_req_pool):
            clear_req_pool()
        clear_allocator = getattr(self.token_to_kv_pool_allocator, "clear", None)
        if callable(clear_allocator):
            clear_allocator()

    def allocate_token_slots(self, count: int, *, mode: str = "decode") -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        slots = _allocate_kv_slots(
            self.token_to_kv_pool_allocator,
            int(count),
            mode=mode,
        )
        slot_tensor = _slot_tensor_from_allocator_result(slots, device=self.device)
        if int(slot_tensor.numel()) < int(count):
            raise RuntimeError(
                "SGLang token slot allocator returned too few slots: "
                f"needed={int(count)}, got_shape={tuple(slot_tensor.shape)}, mode={mode}, "
                f"result_type={type(slots).__name__}"
            )
        return slot_tensor[: int(count)]

    def allocate_extend_slots(
        self,
        *,
        prefix_lens: Sequence[int],
        seq_lens: Sequence[int],
        last_locs: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        prefix_lens_cpu = torch.tensor(list(prefix_lens), dtype=torch.long)
        seq_lens_cpu = torch.tensor(list(seq_lens), dtype=torch.long)
        prefix_lens_tensor = prefix_lens_cpu.to(device=self.device)
        seq_lens_tensor = seq_lens_cpu.to(device=self.device)
        last_locs_tensor = _as_1d_long_tensor(last_locs, device=self.device)
        if not (
            prefix_lens_tensor.numel()
            == seq_lens_tensor.numel()
            == last_locs_tensor.numel()
        ):
            raise ValueError("prefix_lens, seq_lens, and last_locs must have the same length")
        extend_num_tokens = sum(
            int(seq_len) - int(prefix_len)
            for prefix_len, seq_len in zip(prefix_lens, seq_lens)
        )
        if extend_num_tokens <= 0:
            raise ValueError("extend_num_tokens must be positive")

        alloc_extend = getattr(self.token_to_kv_pool_allocator, "alloc_extend", None)
        if callable(alloc_extend):
            result = alloc_extend(
                prefix_lens_tensor,
                prefix_lens_cpu,
                seq_lens_tensor,
                seq_lens_cpu,
                last_locs_tensor,
                extend_num_tokens,
            )
            if result is not None:
                slots = _slot_tensor_from_allocator_result(result, device=self.device)
                if int(slots.numel()) >= extend_num_tokens:
                    return slots[:extend_num_tokens]
        return self.allocate_token_slots(extend_num_tokens, mode="extend")

    def allocate_decode_slots(
        self,
        *,
        seq_lens: Sequence[int],
        last_locs: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        seq_lens_cpu = torch.tensor(list(seq_lens), dtype=torch.long)
        seq_lens_tensor = seq_lens_cpu.to(device=self.device)
        last_locs_tensor = _as_1d_long_tensor(last_locs, device=self.device)
        if seq_lens_tensor.numel() != last_locs_tensor.numel():
            raise ValueError("seq_lens and last_locs must have the same length")

        alloc_decode = getattr(self.token_to_kv_pool_allocator, "alloc_decode", None)
        if callable(alloc_decode):
            result = alloc_decode(
                seq_lens_tensor,
                seq_lens_cpu,
                last_locs_tensor,
            )
            if result is not None:
                slots = _slot_tensor_from_allocator_result(result, device=self.device)
                if int(slots.numel()) >= int(seq_lens_tensor.numel()):
                    return slots[: int(seq_lens_tensor.numel())]
        return self.allocate_token_slots(int(seq_lens_tensor.numel()), mode="decode")

    def _allocate_full_pages(self, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0, self.page_size), device=self.device, dtype=torch.long)
        alloc = getattr(self.token_to_kv_pool_allocator, "alloc", None)
        if not callable(alloc):
            raise RuntimeError("SGLang token-to-KV allocator does not expose alloc()")
        result = alloc(int(count) * self.page_size)
        if result is None:
            raise RuntimeError(f"SGLang KV pool cannot allocate {int(count)} COW pages")
        slots = _slot_tensor_from_allocator_result(result, device=self.device)
        expected = int(count) * self.page_size
        if int(slots.numel()) != expected:
            raise RuntimeError(
                "SGLang full-page allocation returned the wrong number of slots: "
                f"expected={expected}, got={int(slots.numel())}"
            )
        pages = slots.reshape(int(count), self.page_size)
        page_ids = torch.div(pages, self.page_size, rounding_mode="floor")
        _assert_device_true(
            page_ids.eq(page_ids[:, :1]).all(),
            "SGLang allocator returned a non-contiguous COW page",
        )
        self._record_pending_owned_slots(pages[:, 0])
        return pages

    def _copy_kv_slots(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        get_kvcache = getattr(self.token_to_kv_pool_allocator, "get_kvcache", None)
        kv_cache = get_kvcache() if callable(get_kvcache) else None
        move_kv_cache = getattr(kv_cache, "move_kv_cache", None)
        if not callable(move_kv_cache):
            raise RuntimeError(
                "SGLang KV pool does not expose move_kv_cache(); "
                "tail-page copy-on-write cannot be performed"
            )
        move_kv_cache(destination.to(dtype=torch.long), source.to(dtype=torch.long))

    def _copy_partial_tail_pages(
        self,
        routes: Sequence[RouteState],
        slot_paths: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        copied_paths, _ = self._copy_partial_tail_pages_with_dirty_starts(
            routes,
            slot_paths,
        )
        return copied_paths

    def _copy_partial_tail_pages_with_dirty_starts(
        self,
        routes: Sequence[RouteState],
        slot_paths: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[int, int]]:
        """Copy shared partial tails and return changed req-row offsets.

        The returned mapping is keyed by the route's position in ``routes``.
        Its value is the first logical token position whose physical slot id
        changed, so an already initialized request row can patch only the COW
        tail instead of replaying its full prefix.
        """
        if len(routes) != len(slot_paths):
            raise ValueError("routes and slot_paths must have the same length")
        if len(routes) <= 1:
            return list(slot_paths), {}

        # Only active writers that point at the same partial physical page need
        # copy-on-write.  Distinct partial pages are already private.  For N
        # writers sharing one page, one writer can safely append at its first
        # unused offset while the remaining N-1 writers get private copies.
        # Grouping solely by page also handles aliased views with different
        # valid tail lengths; each copied view preserves its own valid length.
        partial_groups: dict[int, list[tuple[int, torch.Tensor, int]]] = {}
        missing_tail_keys: list[
            tuple[int, torch.Tensor, int, int, torch.Tensor]
        ] = []
        result = list(slot_paths)
        for index, slot_path in enumerate(slot_paths):
            tail_len = int(slot_path.numel()) % self.page_size
            if not tail_len:
                continue
            route_id = int(routes[index].route_id)
            tail_page_key = self.route_tail_page_keys.get(route_id)
            if tail_page_key is None:
                # Compatibility path for externally injected/test slot paths.
                # Production paths carry a CPU lease key through every fork,
                # so this batched fallback is not part of the decode hot path.
                source_tail = slot_path[-tail_len:]
                source_page_ids = torch.div(
                    source_tail,
                    self.page_size,
                    rounding_mode="floor",
                )
                _assert_device_true(
                    source_page_ids.eq(source_page_ids[:1]).all(),
                    "route partial tail spans multiple physical pages",
                )
                missing_tail_keys.append(
                    (index, slot_path, tail_len, route_id, source_page_ids[:1])
                )
                continue
            partial_groups.setdefault(tail_page_key, []).append(
                (index, slot_path, tail_len)
            )

        if missing_tail_keys:
            fallback_page_keys = _int_list(
                torch.cat([entry[4] for entry in missing_tail_keys], dim=0)
            )
            self.hot_path_host_transfer_batches += 1
            self.hot_path_host_transfer_elements += len(fallback_page_keys)
            for entry, tail_page_key in zip(missing_tail_keys, fallback_page_keys):
                index, slot_path, tail_len, route_id, _ = entry
                self.route_tail_page_keys[route_id] = int(tail_page_key)
                partial_groups.setdefault(int(tail_page_key), []).append(
                    (index, slot_path, tail_len)
                )

        copy_plan = [entry for group in partial_groups.values() for entry in group[1:]]
        if not copy_plan:
            return result, {}

        destination_pages = self._allocate_full_pages(len(copy_plan))
        dirty_starts: dict[int, int] = {}
        for destination_page, (index, slot_path, tail_len) in zip(
            destination_pages,
            copy_plan,
        ):
            source_tail = slot_path[-tail_len:]
            destination_tail = destination_page[:tail_len]
            self._copy_kv_slots(destination_tail, source_tail)
            copied_path = torch.cat((slot_path[:-tail_len], destination_tail), dim=0)
            route_id = int(routes[index].route_id)
            self.route_slot_paths[route_id] = copied_path
            self.route_tail_page_keys[route_id] = self._new_tail_page_key()
            result[index] = copied_path
            dirty_starts[index] = int(slot_path.numel()) - int(tail_len)
            self.cow_pages_copied += 1
            self.cow_tokens_copied += int(tail_len)
        return result, dirty_starts

    def prepare_frontier(
        self,
        active_routes: Sequence[RouteState],
        *,
        output_slot_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SGLangRouteKVMetadata]:
        if not active_routes:
            raise ValueError("active_routes cannot be empty")
        req_pool_indices = self.ensure_route_rows(active_routes)
        input_ids = torch.tensor(
            [int(route.pending_token_id) for route in active_routes],
            device=self.device,
            dtype=torch.long,
        )
        slot_paths = [self._slot_path_for_route(route) for route in active_routes]
        slot_paths, cow_dirty_starts = self._copy_partial_tail_pages_with_dirty_starts(
            active_routes,
            slot_paths,
        )
        for route_index, dirty_start in cow_dirty_starts.items():
            row = self.route_rows[int(active_routes[route_index].route_id)]
            if int(row.written_length) > 0:
                row.pending_dirty_start = (
                    int(dirty_start)
                    if row.pending_dirty_start is None
                    else min(int(row.pending_dirty_start), int(dirty_start))
                )
        seq_lens = [int(slot_path.numel()) + 1 for slot_path in slot_paths]
        last_locs_tensor = torch.cat(
            [
                slot_path[-1:].to(dtype=torch.long)
                if int(slot_path.numel())
                else torch.full(
                    (1,),
                    -1,
                    device=self.device,
                    dtype=torch.long,
                )
                for slot_path in slot_paths
            ],
            dim=0,
        )

        if output_slot_ids is None:
            output_slot_ids = self.allocate_decode_slots(
                seq_lens=seq_lens,
                last_locs=last_locs_tensor,
            )
        else:
            output_slot_ids = _as_1d_long_tensor(output_slot_ids, device=self.device)
        if int(output_slot_ids.numel()) != len(active_routes):
            raise ValueError("output_slot_ids must have one slot per active route")
        if len(active_routes) > 1:
            duplicates = output_slot_ids[:, None].eq(output_slot_ids[None, :])
            duplicates.fill_diagonal_(False)
            _assert_device_true(
                duplicates.logical_not().all(),
                "SGLang allocated duplicate output KV slots across forked routes; "
                "tail-page copy-on-write is not active",
            )
        self._record_pending_owned_slots(output_slot_ids)

        positions: list[int] = []
        full_slot_paths: list[torch.Tensor] = []

        req_pool_index_list = [
            self.route_rows[int(route.route_id)].req_pool_index
            for route in active_routes
        ]
        for route_index, (route, slot_path, req_pool_index) in enumerate(
            zip(active_routes, slot_paths, req_pool_index_list)
        ):
            output_slot = output_slot_ids[route_index : route_index + 1]
            full_slot_path = torch.cat(
                [
                    slot_path,
                    output_slot,
                ],
                dim=0,
            )
            row = self.route_rows[int(route.route_id)]
            previous_len = int(slot_path.numel())
            if int(row.written_length) == 0:
                write_start = 0
                write_values = full_slot_path
                write_kind = "full_init"
            else:
                if int(row.written_length) != previous_len:
                    raise RuntimeError(
                        "SGLang route-row state does not match its physical KV path: "
                        f"route_id={int(route.route_id)}, "
                        f"written_length={int(row.written_length)}, "
                        f"physical_path_length={previous_len}"
                    )
                dirty_start = row.pending_dirty_start
                if dirty_start is None:
                    write_start = previous_len
                    write_values = full_slot_path[previous_len:]
                    write_kind = "append"
                else:
                    write_start = int(dirty_start)
                    write_values = full_slot_path[write_start:]
                    write_kind = "cow_patch"
            self._write_req_token_slice(
                int(req_pool_index),
                write_start,
                write_values,
                write_kind=write_kind,
            )
            row.written_length = int(full_slot_path.numel())
            row.pending_dirty_start = None
            self.route_slot_paths[int(route.route_id)] = full_slot_path
            if (
                previous_len % self.page_size == 0
                or int(route.route_id) not in self.route_tail_page_keys
            ):
                self.route_tail_page_keys[int(route.route_id)] = (
                    self._new_tail_page_key()
                )
            seq_len = int(full_slot_path.numel())
            positions.append(seq_len - 1)
            full_slot_paths.append(full_slot_path)

        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.long)
        seq_lens_tensor = seq_lens_cpu.to(device=self.device)
        token_index_count = int(sum(seq_lens))
        paged_decode_spec = None
        attention_page_size = 1
        page_index_count = token_index_count
        if self.use_page_attention_metadata:
            paged = build_flashinfer_paged_kv_metadata(
                full_slot_paths,
                page_size=self.page_size,
                validate_layout=self.validate_page_layout,
            )
            paged_decode_spec = AtlasPagedDecodeSpec(
                kv_indptr=paged.kv_indptr,
                kv_indices=paged.kv_page_indices,
                kv_last_page_len=paged.kv_last_page_len,
                page_size=int(paged.page_size),
            )
            attention_page_size = int(paged.page_size)
            page_index_count = int(paged.page_index_count)
        self.attention_metadata_builds += 1
        self.attention_metadata_token_indices += token_index_count
        self.attention_metadata_page_indices += page_index_count
        metadata = SGLangRouteKVMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_tensor,
            out_cache_loc=output_slot_ids,
            positions=torch.tensor(positions, device=self.device, dtype=torch.long),
            orig_seq_lens=seq_lens_tensor,
            seq_lens_sum=int(sum(seq_lens)),
            seq_lens_cpu=seq_lens_cpu,
            attention_page_size=attention_page_size,
            token_index_count=token_index_count,
            page_index_count=page_index_count,
            paged_decode_spec=paged_decode_spec,
        )
        return input_ids, metadata

    def _slot_path_for_route(self, route: RouteState) -> torch.Tensor:
        route_id = int(route.route_id)
        expected_length = (
            int(route.kv_view.prefix.committed_length)
            + len(route.kv_view.node_ids)
        )
        existing = self.route_slot_paths.get(route_id)
        if existing is not None:
            if int(existing.numel()) != expected_length:
                raise RuntimeError(
                    "route-specific physical path length does not match logical KV: "
                    f"route_id={route_id}, physical={int(existing.numel())}, "
                    f"logical={expected_length}"
                )
            return existing
        if route.parent_route_id is not None:
            parent_id = int(route.parent_route_id)
            parent_path = self.route_slot_paths.get(parent_id)
            if parent_path is not None:
                if int(parent_path.numel()) != expected_length:
                    raise RuntimeError(
                        "inherited physical path length does not match logical KV: "
                        f"route_id={route_id}, physical={int(parent_path.numel())}, "
                        f"logical={expected_length}"
                    )
                self.route_slot_paths[route_id] = parent_path
                parent_tail_key = self.route_tail_page_keys.get(parent_id)
                if parent_tail_key is not None:
                    self.route_tail_page_keys[route_id] = parent_tail_key
                return parent_path

        committed_length = int(route.kv_view.prefix.committed_length)
        if committed_length:
            if self.prefix_slot_ids is None:
                raise RuntimeError("prefix_slot_ids must be set before decoding non-empty prefixes")
            if int(self.prefix_slot_ids.numel()) < committed_length:
                raise RuntimeError(
                    f"prefix_slot_ids has length {int(self.prefix_slot_ids.numel())}, "
                    f"but route requires committed_length={committed_length}"
                )
            prefix = self.prefix_slot_ids[:committed_length]
        else:
            prefix = torch.empty((0,), device=self.device, dtype=torch.long)

        if route.kv_view.node_ids:
            raise RuntimeError(
                "route has materialized logical KV nodes but no route-specific "
                f"physical slot path: route_id={route_id}"
            )
        slot_path = prefix
        self.route_slot_paths[route_id] = slot_path
        if committed_length % self.page_size:
            tail_page_index = (committed_length - 1) // self.page_size
            tail_key = self.prefix_partial_tail_keys.get(tail_page_index)
            if tail_key is None:
                tail_key = self._new_tail_page_key()
                self.prefix_partial_tail_keys[tail_page_index] = tail_key
            if committed_length == self.prefix_slot_count:
                self.prefix_tail_page_key = tail_key
            self.route_tail_page_keys[route_id] = tail_key
        return slot_path

    def _write_req_token_row(self, req_pool_index: int, slot_path: torch.Tensor) -> None:
        self._write_req_token_slice(
            req_pool_index,
            0,
            slot_path,
            write_kind="full_init",
        )

    def _write_req_token_slice(
        self,
        req_pool_index: int,
        start: int,
        slot_ids: torch.Tensor,
        *,
        write_kind: str,
    ) -> None:
        if write_kind not in {"full_init", "append", "cow_patch"}:
            raise ValueError(f"unknown req-row write kind: {write_kind}")
        write_start = int(start)
        if write_start < 0:
            raise ValueError("req-row write start must be non-negative")
        values = _as_1d_long_tensor(slot_ids, device=self.device)
        write_count = int(values.numel())
        if write_count <= 0:
            raise ValueError("req-row write cannot be empty")
        write_end = write_start + write_count
        capacity = _req_to_token_pool_context_capacity(self.req_to_token_pool)
        if capacity is None:
            capacity = self.max_context_len
        if capacity is not None and write_end > int(capacity):
            raise RuntimeError(
                "SGLang req_to_token_pool row is too short for the requested prefix. "
                f"write_end={write_end}, req_pool_context_capacity={int(capacity)}, "
                f"req_pool_index={int(req_pool_index)}. Increase --context-length or recreate "
                "the runner with a larger ReqToTokenPool."
            )
        try:
            self.req_to_token_pool.write(
                (int(req_pool_index), slice(write_start, write_end)),
                values,
            )
        except Exception:
            positions = torch.arange(
                write_start,
                write_end,
                device=values.device,
                dtype=torch.long,
            )
            req_indices = torch.full_like(positions, int(req_pool_index))
            try:
                self.req_to_token_pool.write((req_indices, positions), values)
            except Exception as exc:
                raise RuntimeError("failed to write SGLang req_to_token_pool route row") from exc

        # Counters describe successful writes only.  The legacy full-rewrite
        # names remain as compatibility aliases for full row initialization;
        # they no longer count every decode step.
        self.req_row_write_calls += 1
        self.req_row_elements_written += write_count
        if write_kind == "full_init":
            self.req_row_full_init_calls += 1
            self.req_row_full_init_elements += write_count
            self.req_row_full_rewrite_calls += 1
            self.req_row_full_rewrite_elements += write_count
        elif write_kind == "append":
            self.req_row_append_calls += 1
            self.req_row_append_elements += write_count
        else:
            self.req_row_cow_patch_calls += 1
            self.req_row_cow_patch_elements += write_count

    def hot_path_counters(self) -> dict[str, int]:
        return {
            "attention_metadata_builds": int(self.attention_metadata_builds),
            "attention_metadata_token_indices": int(self.attention_metadata_token_indices),
            "attention_metadata_page_indices": int(self.attention_metadata_page_indices),
            "cow_pages_copied": int(self.cow_pages_copied),
            "cow_tokens_copied": int(self.cow_tokens_copied),
            "req_row_write_calls": int(self.req_row_write_calls),
            "req_row_elements_written": int(self.req_row_elements_written),
            "req_row_full_rewrite_calls": int(self.req_row_full_rewrite_calls),
            "req_row_full_rewrite_elements": int(self.req_row_full_rewrite_elements),
            "req_row_full_init_calls": int(self.req_row_full_init_calls),
            "req_row_full_init_elements": int(self.req_row_full_init_elements),
            "req_row_append_calls": int(self.req_row_append_calls),
            "req_row_append_elements": int(self.req_row_append_elements),
            "req_row_cow_patch_calls": int(self.req_row_cow_patch_calls),
            "req_row_cow_patch_elements": int(self.req_row_cow_patch_elements),
            "hot_path_host_transfer_batches": int(
                self.hot_path_host_transfer_batches
            ),
            "hot_path_host_transfer_elements": int(
                self.hot_path_host_transfer_elements
            ),
        }


class SGLangFlashInferFrontierModelBackend:
    """ATLAS frontier backend backed by SGLang ModelRunner + FlashInfer paged KV."""

    def __init__(
        self,
        *,
        route_store: KVTreeStore,
        executor: SGLangFlashInferPagedDecodeBackend,
        route_pool: SGLangRoutePoolBridge,
    ) -> None:
        self.route_store = route_store
        self.executor = executor
        self.route_pool = route_pool
        self.prefix_token_ids: tuple[int, ...] = ()
        self.prefix_next_token_logits: torch.Tensor | None = None
        add_hook = getattr(route_store, "add_route_release_hook", None)
        if callable(add_hook):
            add_hook(route_pool.reconcile_route_rows)

    @classmethod
    def from_runner(
        cls,
        *,
        route_store: KVTreeStore,
        model_runner: Any,
        page_size: int,
        device: str | torch.device = "cuda",
    ) -> "SGLangFlashInferFrontierModelBackend":
        ensure_sglang_runner_runtime_defaults(model_runner)
        assert_sglang_flashinfer_active(model_runner)
        executor = SGLangFlashInferPagedDecodeBackend(model_runner=model_runner, page_size=page_size)
        route_pool = SGLangRoutePoolBridge.from_runner(model_runner, page_size=page_size, device=device)
        return cls(route_store=route_store, executor=executor, route_pool=route_pool)

    def attach_prefilled_prefix(
        self,
        *,
        prompt_token_ids: Sequence[int],
        prefix_slot_ids: Sequence[int] | torch.Tensor,
        next_token_logits: torch.Tensor,
    ) -> DraftPrefixState:
        self.prefix_token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        self.prefix_next_token_logits = next_token_logits
        self.route_pool.set_prefix_slots(prefix_slot_ids)
        return DraftPrefixState(
            token_ids=torch.tensor(self.prefix_token_ids, device=next_token_logits.device, dtype=torch.long),
            prefix_kv_view=PrefixKVView(committed_length=len(self.prefix_token_ids)),
            next_token_logits=next_token_logits,
            committed_length=len(self.prefix_token_ids),
        )

    @torch.inference_mode()
    def append_known_tokens_as_prefix(
        self,
        token_ids: Sequence[int],
    ) -> torch.Tensor:
        """Append a known causal suffix with one SGLang EXTEND forward."""
        suffix = tuple(int(token_id) for token_id in token_ids)
        if not suffix:
            raise ValueError("known prefix extension cannot be empty")
        prefix_slots = self.route_pool.prefix_slot_ids
        if prefix_slots is None or int(prefix_slots.numel()) != len(self.prefix_token_ids):
            raise RuntimeError("persistent Drafter prefix slots are not initialized")

        ForwardBatch, ForwardMode = _sglang_forward_batch_classes()
        device = self.route_pool.device
        prefix_len = len(self.prefix_token_ids)
        suffix_len = len(suffix)
        seq_len = prefix_len + suffix_len
        input_ids = torch.tensor(suffix, device=device, dtype=torch.long)

        handle = _PoolReqHandle(rid="atlas-known-prefix-extend")
        indices = self.route_pool.req_to_token_pool.alloc([handle])
        if indices is None:
            raise RuntimeError("SGLang ReqToTokenPool.alloc returned None for known-token extend")
        req_pool_index = int(_int_list(indices)[0])
        handle.req_pool_idx = req_pool_index
        try:
            last_loc = (
                prefix_slots[-1:].to(dtype=torch.long)
                if prefix_len
                else torch.full((1,), -1, device=device, dtype=torch.long)
            )
            suffix_slots = self.route_pool.allocate_extend_slots(
                prefix_lens=[prefix_len],
                seq_lens=[seq_len],
                last_locs=last_loc,
            )
            full_slots = torch.cat([prefix_slots, suffix_slots], dim=0)
            self.route_pool._write_req_token_row(req_pool_index, full_slots)

            req_pool_indices = torch.tensor([req_pool_index], device=device, dtype=torch.long)
            seq_lens_cpu = torch.tensor([seq_len], dtype=torch.long)
            seq_lens = seq_lens_cpu.to(device=device)
            extend_seq_lens = torch.tensor([suffix_len], device=device, dtype=torch.long)
            extend_prefix_lens = torch.tensor([prefix_len], device=device, dtype=torch.long)
            forward_batch = ForwardBatch(
                forward_mode=_forward_mode(ForwardMode, ("EXTEND", "PREFILL")),
                batch_size=1,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=suffix_slots,
                seq_lens_sum=seq_len,
                orig_seq_lens=seq_lens,
                positions=torch.arange(prefix_len, seq_len, device=device, dtype=torch.long),
                seq_lens_cpu=seq_lens_cpu,
                capture_hidden_mode=_capture_hidden_mode_null(),
                is_extend_in_batch=True,
                all_extend_in_batch=True,
                extend_num_tokens=suffix_len,
                extend_seq_lens=extend_seq_lens,
                extend_prefix_lens=extend_prefix_lens,
                extend_start_loc=torch.tensor([0], device=device, dtype=torch.long),
                extend_seq_lens_cpu=[suffix_len],
                extend_prefix_lens_cpu=[prefix_len],
                extend_logprob_start_lens_cpu=[suffix_len - 1],
            )
            ensure_sglang_runner_runtime_defaults(self.executor.model_runner)
            ensure_sglang_eager_runner(self.executor.model_runner)
            output = self.executor.model_runner.forward(forward_batch)
            logits = _extract_next_token_logits(output)
            _sync_cuda_if_needed()
        finally:
            if hasattr(self.route_pool.req_to_token_pool, "free"):
                self.route_pool.req_to_token_pool.free(handle)

        self.route_pool.set_prefix_slots(full_slots)
        self.prefix_token_ids = (*self.prefix_token_ids, *suffix)
        if logits.ndim == 1:
            return logits
        return logits[-1]

    def set_prefix(self, token_ids: Sequence[int]) -> None:
        self.prefix_token_ids = tuple(int(token_id) for token_id in token_ids)

    def commit_stage1_and_promote(
        self,
        *,
        committed_route: RouteState,
        retained_routes: Sequence[RouteState],
    ) -> tuple[list[RouteState], dict[str, int]]:
        """Commit stage-1 KV and rebase selected stage-2 routes in place."""
        committed_node_ids = self.route_store.node_path_for_route(committed_route)
        if not committed_node_ids:
            raise ValueError("cannot commit a stage-1 route with no materialized nodes")

        committed_tokens = self.route_store.tokens_for_node_ids(committed_node_ids)
        self.route_pool.commit_route_as_prefix(committed_route)
        stored_tokens = self.route_store.commit_route(committed_route)
        if stored_tokens != committed_tokens:
            raise RuntimeError("logical committed tokens changed during route promotion")

        self.prefix_token_ids = (*self.prefix_token_ids, *committed_tokens)
        new_prefix = PrefixKVView(committed_length=len(self.prefix_token_ids))
        promoted = self.route_store.promote_routes_after_commit(
            committed_route,
            retained_routes,
            new_prefix,
        )

        released_rows = self.route_pool.retain_route_rows(
            [route.route_id for route in promoted]
        )
        live_node_ids = sorted(
            {
                int(node_id)
                for route in promoted
                for node_id in route.kv_view.node_ids
            }
        )
        released_pages = self.route_pool.release_unreferenced_node_slots(live_node_ids)
        return promoted, {
            "committed_kv_tokens": len(committed_node_ids),
            "retained_routes": len(promoted),
            "released_route_rows": released_rows,
            "released_kv_pages": released_pages,
        }

    @torch.inference_mode()
    def decode_frontier_one_token(
        self,
        active_routes: Sequence[RouteState],
        attention_backend: object = None,
    ) -> FrontierDecodeOutput:
        start = time.perf_counter()
        counters_before = self.route_pool.hot_path_counters()
        new_node_ids = self.route_store.reserve_node_ids(len(active_routes))
        input_ids, metadata = self.route_pool.prepare_frontier(active_routes)
        ensure_sglang_runner_runtime_defaults(self.executor.model_runner)
        ensure_sglang_eager_runner(self.executor.model_runner)
        logits = self.executor.forward_frontier(input_ids=input_ids, route_kv_metadata=metadata)
        self.route_pool.register_node_slots(new_node_ids, metadata.out_cache_loc)
        counters_after = self.route_pool.hot_path_counters()
        counter_delta = {
            key: int(counters_after[key] - counters_before.get(key, 0))
            for key in counters_after
        }
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return FrontierDecodeOutput(
            route_ids=torch.tensor(
                [int(route.route_id) for route in active_routes],
                device=input_ids.device,
                dtype=torch.long,
            ),
            next_token_logits=logits,
            new_node_ids=tuple(int(node_id) for node_id in new_node_ids),
            new_node_ids_cpu=tuple(int(node_id) for node_id in new_node_ids),
            model_ms=elapsed_ms,
            metadata={
                "backend": "sglang_flashinfer_paged_decode",
                "reference_only": False,
                "paged_kv": True,
                "cascade": False,
                "route_tail_page_cow": True,
                "branch_safe_page_size": True,
                "cow_pages_copied_total": int(self.route_pool.cow_pages_copied),
                "cow_tokens_copied_total": int(self.route_pool.cow_tokens_copied),
                "attention_metadata_page_size": int(metadata.attention_page_size),
                "attention_metadata_token_indices_step": int(metadata.token_index_count),
                "attention_metadata_page_indices_step": int(metadata.page_index_count),
                "hot_path_counter_delta": counter_delta,
                "hot_path_counters_total": counters_after,
                "attention_backend": str(attention_backend),
                **self.executor.runtime_metadata(cascade_level=0, verify_backend="none"),
            },
        )


def build_sglang_server_args(config: SGLangRunnerConfig) -> Any:
    ServerArgs = _import_attr("sglang.srt.server_args", "ServerArgs")
    kwargs = {
        "model_path": config.model_path,
        "tokenizer_path": config.tokenizer_path,
        "dtype": config.dtype,
        "kv_cache_dtype": config.kv_cache_dtype,
        "context_length": config.context_length,
        "page_size": config.page_size,
        "mem_fraction_static": config.mem_fraction_static,
        "max_running_requests": config.max_running_requests,
        "max_total_tokens": config.max_total_tokens,
        "trust_remote_code": config.trust_remote_code,
        "load_format": config.load_format,
        "model_impl": config.model_impl,
        "device": config.device,
        "base_gpu_id": config.gpu_id,
        "tp_size": config.tp_size,
        "pp_size": config.pp_size,
        "skip_tokenizer_init": config.skip_tokenizer_init,
        "disable_radix_cache": config.disable_radix_cache,
        "disable_cuda_graph": config.disable_cuda_graph,
        "skip_server_warmup": config.skip_server_warmup,
        "attention_backend": "flashinfer",
        "prefill_attention_backend": "flashinfer",
        "decode_attention_backend": "flashinfer",
    }
    return _construct_with_supported_kwargs(ServerArgs, kwargs)


def build_sglang_model_config(server_args: Any, config: SGLangRunnerConfig) -> Any:
    ModelConfig = _import_attr("sglang.srt.configs.model_config", "ModelConfig")
    if hasattr(ModelConfig, "from_server_args"):
        return ModelConfig.from_server_args(server_args)
    return _construct_with_supported_kwargs(
        ModelConfig,
        {
            "model_path": config.model_path,
            "trust_remote_code": config.trust_remote_code,
            "context_length": config.context_length,
            "dtype": config.dtype,
            "model_impl": config.model_impl,
        },
    )


def create_sglang_model_runner(
    config: SGLangRunnerConfig,
    *,
    initialize: bool = True,
) -> Any:
    ModelRunner = _import_attr("sglang.srt.model_executor.model_runner", "ModelRunner")
    server_args = build_sglang_server_args(config)
    model_config = build_sglang_model_config(server_args, config)
    pool_bundle = create_sglang_memory_pool_bundle(model_config, config)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=float(config.mem_fraction_static),
        gpu_id=int(config.gpu_id),
        tp_rank=0,
        tp_size=int(config.tp_size),
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=int(config.pp_size),
        nccl_port=int(config.nccl_port),
        server_args=server_args,
        req_to_token_pool=pool_bundle.req_to_token_pool,
        token_to_kv_pool_allocator=pool_bundle.token_to_kv_pool_allocator,
    )
    setattr(
        runner,
        "atlas_use_page_attention_metadata",
        bool(config.use_page_attention_metadata),
    )
    setattr(runner, "atlas_validate_page_layout", bool(config.validate_page_layout))
    attach_sglang_memory_pools(runner, pool_bundle)
    ensure_sglang_runner_runtime_defaults(runner)
    if initialize:
        attach_sglang_memory_pools(runner, pool_bundle)
        ensure_sglang_runner_runtime_defaults(runner)
        ensure_sglang_flashinfer_attention_backend(runner)
        ensure_sglang_eager_runner(runner)
        assert_sglang_flashinfer_active(runner)
    return runner


@torch.inference_mode()
def prefill_sglang_prefix(
    *,
    model_runner: Any,
    route_pool: SGLangRoutePoolBridge,
    prompt_token_ids: Sequence[int],
    rid: str = "atlas-prefix",
    chunk_size: int | None = 8192,
) -> SGLangPrefillResult:
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids cannot be empty")

    ForwardBatch, ForwardMode = _sglang_forward_batch_classes()
    device = route_pool.device
    all_input_ids = torch.tensor([int(token_id) for token_id in prompt_token_ids], device=device, dtype=torch.long)
    prompt_len = int(all_input_ids.numel())
    effective_chunk_size = int(chunk_size or prompt_len)
    if effective_chunk_size <= 0:
        effective_chunk_size = prompt_len

    req_handle = _PoolReqHandle(rid=rid)
    req_indices = route_pool.req_to_token_pool.alloc([req_handle])
    if req_indices is None:
        raise RuntimeError("SGLang ReqToTokenPool.alloc returned None for prefix prefill")
    req_pool_index = int(_int_list(req_indices)[0])
    req_handle.req_pool_idx = req_pool_index

    req_pool_indices = torch.tensor([req_pool_index], device=device, dtype=torch.long)
    extend_start_loc = torch.tensor([0], device=device, dtype=torch.long)
    prefix_slot_parts: list[torch.Tensor] = []
    logits: torch.Tensor | None = None
    chunk_count = 0

    ensure_sglang_runner_runtime_defaults(model_runner)
    ensure_sglang_eager_runner(model_runner)

    for chunk_start in range(0, prompt_len, effective_chunk_size):
        chunk_end = min(prompt_len, chunk_start + effective_chunk_size)
        chunk_input_ids = all_input_ids[chunk_start:chunk_end]
        chunk_len = int(chunk_input_ids.numel())
        if chunk_len <= 0:
            continue

        last_loc = (
            prefix_slot_parts[-1][-1:].to(dtype=torch.long)
            if prefix_slot_parts
            else torch.full((1,), -1, device=device, dtype=torch.long)
        )

        chunk_slot_ids = route_pool.allocate_extend_slots(
            prefix_lens=[chunk_start],
            seq_lens=[chunk_end],
            last_locs=last_loc,
        )
        prefix_slot_parts.append(chunk_slot_ids)
        prefix_slot_ids = torch.cat(prefix_slot_parts, dim=0)
        route_pool._write_req_token_row(req_pool_index, prefix_slot_ids)
        _sync_cuda_if_needed()

        seq_lens_cpu = torch.tensor([chunk_end], dtype=torch.long)
        seq_lens = seq_lens_cpu.to(device=device)
        extend_seq_lens = torch.tensor([chunk_len], device=device, dtype=torch.long)
        extend_prefix_lens = torch.tensor([chunk_start], device=device, dtype=torch.long)
        positions = torch.arange(chunk_start, chunk_end, device=device, dtype=torch.long)

        forward_batch = ForwardBatch(
            forward_mode=_forward_mode(ForwardMode, ("EXTEND", "PREFILL")),
            batch_size=1,
            input_ids=chunk_input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=chunk_slot_ids,
            seq_lens_sum=chunk_end,
            orig_seq_lens=seq_lens,
            positions=positions,
            seq_lens_cpu=seq_lens_cpu,
            capture_hidden_mode=_capture_hidden_mode_null(),
            is_extend_in_batch=True,
            all_extend_in_batch=True,
            extend_num_tokens=chunk_len,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=[chunk_len],
            extend_prefix_lens_cpu=[chunk_start],
            extend_logprob_start_lens_cpu=[chunk_len - 1],
        )
        output = model_runner.forward(forward_batch)
        _sync_cuda_if_needed()
        logits = _extract_next_token_logits(output)
        chunk_count += 1

    if logits is None:
        raise RuntimeError("SGLang prefix prefill did not run any chunks")

    if logits.ndim == 2 and int(logits.shape[0]) != 1:
        logits = logits[-1:, :]
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    prefix_slot_ids = torch.cat(prefix_slot_parts, dim=0)

    return SGLangPrefillResult(
        prompt_token_ids=tuple(int(token_id) for token_id in prompt_token_ids),
        prefix_slot_ids=prefix_slot_ids,
        next_token_logits=logits[0],
        req_pool_index=req_pool_index,
        forward_metadata={
            "forward_mode": str(_forward_mode(ForwardMode, ("EXTEND", "PREFILL"))),
            "prompt_len": prompt_len,
            "prefix_slot_count": int(prefix_slot_ids.numel()),
            "prefill_chunk_size": effective_chunk_size,
            "prefill_chunk_count": chunk_count,
        },
    )


def create_sglang_memory_pool_bundle(model_config: Any, config: SGLangRunnerConfig) -> SGLangMemoryPoolBundle:
    ReqToTokenPool = _import_attr("sglang.srt.mem_cache.memory_pool", "ReqToTokenPool")
    MHATokenToKVPool = _import_attr("sglang.srt.mem_cache.memory_pool", "MHATokenToKVPool")
    BaseTokenToKVPoolAllocator = _import_attr(
        "sglang.srt.mem_cache.allocator.base",
        "BaseTokenToKVPoolAllocator",
    )

    max_context_len = _max_context_len(model_config, config)
    max_running_requests = int(config.max_running_requests or 32)
    max_total_tokens = int(config.max_total_tokens or (max_context_len * max_running_requests))
    dtype = _torch_dtype(config.kv_cache_dtype if config.kv_cache_dtype != "auto" else config.dtype)
    device = _device_string(config)

    head_num = _num_kv_heads(model_config, config.tp_size)
    head_dim = _head_dim(model_config)
    layer_num = _layer_num(model_config)

    req_to_token_pool = ReqToTokenPool(
        size=max_running_requests,
        max_context_len=max_context_len,
        device=device,
        enable_memory_saver=False,
    )
    token_to_kv_pool = MHATokenToKVPool(
        size=max_total_tokens,
        page_size=int(config.page_size),
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=False,
        enable_kv_cache_copy=True,
    )
    token_to_kv_pool_allocator = _create_token_to_kv_pool_allocator(
        BaseTokenToKVPoolAllocator,
        {
            "size": max_total_tokens,
            "page_size": int(config.page_size),
            "dtype": dtype,
            "device": device,
            "kvcache": token_to_kv_pool,
            "need_sort": False,
        },
    )
    return SGLangMemoryPoolBundle(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        max_running_requests=max_running_requests,
        max_context_len=max_context_len,
        max_total_tokens=max_total_tokens,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        dtype=dtype,
    )


def attach_sglang_memory_pools(model_runner: Any, pool_bundle: SGLangMemoryPoolBundle) -> None:
    setattr(model_runner, "req_to_token_pool", pool_bundle.req_to_token_pool)
    setattr(model_runner, "token_to_kv_pool", pool_bundle.token_to_kv_pool)
    setattr(model_runner, "token_to_kv_pool_allocator", pool_bundle.token_to_kv_pool_allocator)
    setattr(model_runner, "atlas_max_context_len", pool_bundle.max_context_len)
    setattr(model_runner, "atlas_pool_page_size", getattr(pool_bundle.token_to_kv_pool, "page_size", None))
    setattr(model_runner, "max_running_requests", pool_bundle.max_running_requests)
    setattr(model_runner, "max_total_num_tokens", pool_bundle.max_total_tokens)
    setattr(model_runner, "full_max_total_num_tokens", pool_bundle.max_total_tokens)
    setattr(model_runner, "swa_max_total_num_tokens", 0)
    setattr(model_runner, "c4_max_total_num_tokens", 0)
    setattr(model_runner, "c128_max_total_num_tokens", 0)


def ensure_sglang_runner_runtime_defaults(model_runner: Any) -> None:
    default_factories = {
        "canary_manager": lambda: None,
        "decode_cuda_graph_runner": lambda: None,
        "prefill_cuda_graph_runner": lambda: None,
        "decode_attn_backend": lambda: None,
        "decode_attn_backend_group": list,
        "graph_mem_usage": lambda: 0,
        "eager_runner": lambda: None,
        "war_fastpath_read_done_event": lambda: None,
        "hisparse_coordinator": lambda: None,
        "lora_manager": lambda: None,
    }
    for name, factory in default_factories.items():
        if not hasattr(model_runner, name):
            setattr(model_runner, name, factory())


def ensure_sglang_eager_runner(model_runner: Any) -> Any:
    existing = getattr(model_runner, "eager_runner", None)
    if existing is not None:
        return existing

    ensure_sglang_runner_runtime_defaults(model_runner)
    ensure_sglang_flashinfer_attention_backend(model_runner)
    EagerRunner = _import_attr(
        "sglang.srt.model_executor.runner.eager_runner",
        "EagerRunner",
    )
    try:
        eager_runner = EagerRunner(model_runner)
    except Exception as exc:
        raise RuntimeError(f"failed to initialize SGLang EagerRunner: {exc!r}") from exc
    setattr(model_runner, "eager_runner", eager_runner)
    return eager_runner


def ensure_sglang_flashinfer_attention_backend(model_runner: Any) -> Any:
    existing = _first_existing_attr(
        model_runner,
        ("attn_backend", "attention_backend", "decode_attn_backend"),
    )
    if existing is not None and "flashinfer" in type(existing).__name__.lower():
        backend = existing
    else:
        FlashInferAttnBackend = _import_attr(
            "sglang.srt.layers.attention.flashinfer_backend",
            "FlashInferAttnBackend",
        )
        backend = FlashInferAttnBackend(model_runner)
        setattr(model_runner, "attn_backend", backend)

    page_size = int(getattr(model_runner, "atlas_pool_page_size", 1) or 1)
    if bool(getattr(model_runner, "atlas_use_page_attention_metadata", True)):
        install_atlas_paged_decode_attention(backend, page_size=page_size)
    return backend


def assert_sglang_flashinfer_active(model_runner: Any) -> None:
    report = sglang_runner_component_report(model_runner)
    if "flashinfer" not in str(report["attention_backend_class"]).lower():
        raise RuntimeError(
            "FlashInfer backend requested, but no instantiated SGLang FlashInfer attention backend was found."
        )
    if report["req_to_token_pool_class"] == "NoneType":
        raise RuntimeError("FlashInfer backend requested, but SGLang req_to_token_pool is missing.")
    if report["token_to_kv_pool_allocator_class"] == "NoneType":
        raise RuntimeError("FlashInfer backend requested, but SGLang token_to_kv_pool_allocator is missing.")


def sglang_runner_component_report(model_runner: Any) -> dict[str, Any]:
    server_args = getattr(model_runner, "server_args", None)
    attn_backend = _first_existing_attr(
        model_runner,
        ("attn_backend", "attention_backend", "decode_attn_backend"),
    )
    req_to_token_pool = _first_existing_attr(model_runner, ("req_to_token_pool",))
    token_to_kv_pool_allocator = _first_existing_attr(
        model_runner,
        ("token_to_kv_pool_allocator", "token_to_kv_pool_alloc", "kv_pool_allocator"),
    )
    token_to_kv_pool = _first_existing_attr(model_runner, ("token_to_kv_pool", "kv_pool", "kvcache"))
    eager_runner = _first_existing_attr(model_runner, ("eager_runner",))
    req_pool_shape = _req_to_token_pool_tensor_shape(req_to_token_pool)
    return {
        "model_runner_class": type(model_runner).__name__,
        "attention_backend_class": type(attn_backend).__name__,
        "eager_runner_class": type(eager_runner).__name__,
        "req_to_token_pool_class": type(req_to_token_pool).__name__,
        "req_to_token_pool_shape": req_pool_shape,
        "req_to_token_pool_context_capacity": _req_to_token_pool_context_capacity(req_to_token_pool),
        "token_to_kv_pool_allocator_class": type(token_to_kv_pool_allocator).__name__,
        "token_to_kv_pool_class": type(token_to_kv_pool).__name__,
        "atlas_max_context_len": getattr(model_runner, "atlas_max_context_len", None),
        "atlas_max_total_tokens": getattr(model_runner, "max_total_num_tokens", None),
        "server_attention_backend": getattr(server_args, "attention_backend", None),
        "server_prefill_attention_backend": getattr(server_args, "prefill_attention_backend", None),
        "server_decode_attention_backend": getattr(server_args, "decode_attention_backend", None),
        "atlas_use_page_attention_metadata": bool(
            getattr(model_runner, "atlas_use_page_attention_metadata", True)
        ),
        "atlas_validate_page_layout": bool(
            getattr(model_runner, "atlas_validate_page_layout", True)
        ),
        "atlas_attention_metadata_page_size": int(
            getattr(attn_backend, "atlas_paged_decode_page_size", 1)
        ),
        "atlas_paged_decode_enabled": bool(
            getattr(attn_backend, "atlas_paged_decode_enabled", False)
        ),
    }


def _construct_with_supported_kwargs(cls: Any, kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(cls)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return cls(**{key: value for key, value in kwargs.items() if value is not None})
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return cls(**supported)


def _sglang_forward_batch_classes() -> tuple[Any, Any]:
    ForwardBatch = _import_attr_any(
        [
            "sglang.srt.model_executor.forward_batch_info",
            "sglang.srt.managers.schedule_batch",
        ],
        "ForwardBatch",
    )
    ForwardMode = _import_attr_any(
        [
            "sglang.srt.model_executor.forward_batch_info",
            "sglang.srt.managers.schedule_batch",
        ],
        "ForwardMode",
    )
    return ForwardBatch, ForwardMode


def _forward_mode(ForwardMode: Any, preferred_names: Sequence[str]) -> Any:
    for preferred_name in preferred_names:
        for name in (preferred_name, preferred_name.lower(), preferred_name.capitalize()):
            if hasattr(ForwardMode, name):
                return getattr(ForwardMode, name)
    for item in ForwardMode:
        item_name = str(getattr(item, "name", "")).lower()
        item_value = str(getattr(item, "value", "")).lower()
        for preferred_name in preferred_names:
            if preferred_name.lower() in (item_name, item_value):
                return item
    raise RuntimeError(f"SGLang ForwardMode has none of: {list(preferred_names)}")


def _capture_hidden_mode_null() -> Any:
    CaptureHiddenMode = _import_attr_any(
        [
            "sglang.srt.model_executor.forward_batch_info",
            "sglang.srt.managers.schedule_batch",
        ],
        "CaptureHiddenMode",
    )
    return getattr(CaptureHiddenMode, "NULL")


def _allocate_kv_slots(allocator: Any, count: int, *, mode: str) -> Any:
    method_names: list[str] = []
    if mode == "extend":
        method_names.extend(["alloc_extend", "alloc"])
    elif mode == "decode":
        method_names.extend(["alloc_decode", "alloc"])
    else:
        method_names.append("alloc")

    errors: list[str] = []
    for method_name in method_names:
        method = getattr(allocator, method_name, None)
        if not callable(method):
            continue
        for args in ((int(count),),):
            try:
                result = method(*args)
            except Exception as exc:
                errors.append(f"{method_name}{args}: {exc!r}")
                continue
            if result is not None:
                return result
            errors.append(f"{method_name}{args}: returned None")

    raise RuntimeError(
        "SGLang token_to_kv_pool_allocator could not allocate slots. "
        + ("; ".join(errors) if errors else "no supported allocation method")
    )


def _slot_tensor_from_allocator_result(result: Any, *, device: str | torch.device) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return _as_1d_long_tensor(result, device=device)
    if isinstance(result, (list, tuple)):
        tensor_candidates = [item for item in result if isinstance(item, torch.Tensor)]
        if tensor_candidates:
            return _as_1d_long_tensor(tensor_candidates[0], device=device)
        flat_ints: list[int] = []
        for item in result:
            if isinstance(item, int):
                flat_ints.append(int(item))
            elif isinstance(item, (list, tuple)) and all(isinstance(value, int) for value in item):
                flat_ints.extend(int(value) for value in item)
        if flat_ints:
            return _as_1d_long_tensor(flat_ints, device=device)
    return _as_1d_long_tensor(result, device=device)


def _req_to_token_pool_tensor_shape(req_to_token_pool: Any) -> list[int] | None:
    if req_to_token_pool is None:
        return None
    try:
        attrs = vars(req_to_token_pool)
    except TypeError:
        attrs = {}
    preferred_names = (
        "req_to_token",
        "req_to_token_pool",
        "req_to_token_table",
        "pool",
        "data",
    )
    for name in preferred_names:
        value = getattr(req_to_token_pool, name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return [int(dim) for dim in value.shape]
    for value in attrs.values():
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            return [int(dim) for dim in value.shape]
    return None


def _req_to_token_pool_context_capacity(req_to_token_pool: Any) -> int | None:
    shape = _req_to_token_pool_tensor_shape(req_to_token_pool)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def _sync_cuda_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _create_token_to_kv_pool_allocator(base_cls: Any, kwargs: dict[str, Any]) -> Any:
    candidates = _discover_token_to_kv_pool_allocator_classes(base_cls)
    errors: list[str] = []
    for cls in candidates:
        try:
            allocator = _construct_with_supported_kwargs(cls, kwargs)
        except Exception as exc:
            errors.append(f"{cls.__module__}.{cls.__name__}: {exc!r}")
            continue
        missing_methods = [
            name
            for name in ("alloc", "free", "clear")
            if not callable(getattr(allocator, name, None))
        ]
        if missing_methods:
            errors.append(f"{cls.__module__}.{cls.__name__}: missing methods {missing_methods}")
            continue
        return allocator

    raise RuntimeError(
        "could not instantiate a concrete SGLang token-to-KV pool allocator. "
        "Discovered candidates failed: "
        + ("; ".join(errors) if errors else "none discovered")
    )


def _discover_token_to_kv_pool_allocator_classes(base_cls: Any) -> list[Any]:
    modules: list[Any] = []
    for module_name in (
        "sglang.srt.mem_cache.allocator.base",
        "sglang.srt.mem_cache.allocator",
    ):
        try:
            modules.append(importlib.import_module(module_name))
        except Exception:
            continue

    try:
        allocator_pkg = importlib.import_module("sglang.srt.mem_cache.allocator")
        for module_info in pkgutil.walk_packages(allocator_pkg.__path__, prefix=f"{allocator_pkg.__name__}."):
            try:
                modules.append(importlib.import_module(module_info.name))
            except Exception:
                continue
    except Exception:
        pass

    seen: set[type[Any]] = set()
    candidates: list[Any] = []
    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in seen or cls is base_cls:
                continue
            try:
                is_allocator = issubclass(cls, base_cls)
            except TypeError:
                is_allocator = False
            if not is_allocator or inspect.isabstract(cls):
                continue
            seen.add(cls)
            candidates.append(cls)

    def sort_key(cls: Any) -> tuple[int, str]:
        name = cls.__name__.lower()
        priority = 0
        if "base" in name or "abstract" in name:
            priority += 100
        if "token" not in name or "kv" not in name:
            priority += 10
        return priority, f"{cls.__module__}.{cls.__name__}"

    return sorted(candidates, key=sort_key)


def _import_attr(module_name: str, attr_name: str) -> Any:
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as exc:
        raise RuntimeError(f"failed to import {module_name}.{attr_name}: {exc!r}") from exc


def _first_existing_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _model_config_candidates(model_config: Any) -> list[Any]:
    candidates = [model_config]
    for name in ("hf_config", "config", "model_config"):
        value = getattr(model_config, name, None)
        if value is not None and value not in candidates:
            candidates.append(value)
    return candidates


def _first_int_config_attr(model_config: Any, names: Sequence[str]) -> int | None:
    for candidate in _model_config_candidates(model_config):
        for name in names:
            value = getattr(candidate, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def _max_context_len(model_config: Any, config: SGLangRunnerConfig) -> int:
    value = config.context_length or _first_int_config_attr(
        model_config,
        ("context_len", "context_length", "max_context_len", "max_position_embeddings"),
    )
    if value is None:
        raise RuntimeError("could not infer max_context_len from SGLang ModelConfig")
    return int(value)


def _num_kv_heads(model_config: Any, tp_size: int) -> int:
    if hasattr(model_config, "get_num_kv_heads"):
        return int(model_config.get_num_kv_heads(int(tp_size)))
    total = _first_int_config_attr(
        model_config,
        ("num_key_value_heads", "n_kv_heads", "multi_query_group_num", "num_attention_heads"),
    )
    if total is None:
        raise RuntimeError("could not infer num_key_value_heads from SGLang ModelConfig")
    return max(1, int(total) // int(tp_size))


def _head_dim(model_config: Any) -> int:
    value = _first_int_config_attr(model_config, ("head_dim", "kv_head_dim", "qk_head_dim"))
    if value is not None:
        return int(value)
    hidden_size = _first_int_config_attr(model_config, ("hidden_size", "n_embd"))
    num_heads = _first_int_config_attr(model_config, ("num_attention_heads", "n_head"))
    if hidden_size is None or num_heads is None:
        raise RuntimeError("could not infer attention head_dim from SGLang ModelConfig")
    return int(hidden_size) // int(num_heads)


def _layer_num(model_config: Any) -> int:
    value = _first_int_config_attr(model_config, ("num_hidden_layers", "n_layer", "num_layers"))
    if value is None:
        raise RuntimeError("could not infer num_hidden_layers from SGLang ModelConfig")
    return int(value)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).lower()
    if normalized in ("auto", "float16", "fp16", "half"):
        return torch.float16
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    if normalized in ("float32", "fp32", "float"):
        return torch.float32
    raise RuntimeError(f"manual SGLang KV pool creation does not support dtype={dtype_name!r} yet")


def _device_string(config: SGLangRunnerConfig) -> str:
    if config.device == "cuda" and int(config.gpu_id) != 0:
        return f"cuda:{int(config.gpu_id)}"
    return str(config.device)


def _as_1d_long_tensor(value: Sequence[int] | torch.Tensor, *, device: str | torch.device) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(list(value), dtype=torch.long)
    tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"expected a 1D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _assert_device_true(condition: torch.Tensor, message: str) -> None:
    """Validate a scalar tensor without synchronizing the CUDA hot path."""

    if condition.numel() != 1:
        raise ValueError("device assertion condition must be scalar")
    if condition.device.type == "cpu":
        if not bool(condition.item()):
            raise RuntimeError(message)
        return
    assert_async = getattr(torch, "_assert_async", None)
    if callable(assert_async):
        assert_async(condition, message)
        return
    raise RuntimeError(
        "CUDA validation requires torch._assert_async to avoid a "
        "device-to-host synchronization"
    )


def _int_list(value: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().view(-1).tolist()]
    return [int(item) for item in value]
