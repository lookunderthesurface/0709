"""Clean SGLang + FlashInfer paged batch decode path.

This package is the useful 0705 execution chain, separated from the reference
HF/SDPA controller and from the old synthetic/Cascade experiments. It provides
ordinary FlashInfer paged batch decode for ATLAS tree and forest construction.
It is not a Cascade/shared-prefix optimized backend.
"""

from .builders import (
    build_forest_depths,
    build_forest_one_depth,
    build_tree_depths,
    build_tree_one_depth,
    initialize_forest_routes,
    initialize_stage1_routes,
)
from .flashinfer_backends import SGLangFlashInferPagedDecodeBackend, SGLangRouteKVMetadata
from .kv import KVTreeNode, KVTreeStore
from .paged_metadata import FlashInferPagedKVMetadata, build_flashinfer_paged_kv_metadata
from .sampling import DrafterSamplingConfig, DrafterSamplingContext
from .sglang_runtime import (
    SGLangFlashInferFrontierModelBackend,
    SGLangMemoryPoolBundle,
    SGLangPrefillResult,
    SGLangRoutePoolBridge,
    SGLangRouteRow,
    SGLangRunnerConfig,
    assert_sglang_flashinfer_active,
    create_sglang_model_runner,
    prefill_sglang_prefix,
    sglang_runner_component_report,
)
from .sglang_page_attention import AtlasPagedDecodeSpec, reshape_token_kv_cache_as_pages
from .types import (
    BuildDepthsOutput,
    DecodePhase,
    DraftPrefixState,
    FrontierDecodeOutput,
    FrontierStepOutput,
    PendingCandidate,
    PrefixKVView,
    RouteKVView,
    RouteState,
)

__all__ = [
    "BuildDepthsOutput",
    "AtlasPagedDecodeSpec",
    "DecodePhase",
    "DraftPrefixState",
    "DrafterSamplingConfig",
    "DrafterSamplingContext",
    "FrontierDecodeOutput",
    "FrontierStepOutput",
    "FlashInferPagedKVMetadata",
    "KVTreeNode",
    "KVTreeStore",
    "PendingCandidate",
    "PrefixKVView",
    "RouteKVView",
    "RouteState",
    "SGLangFlashInferFrontierModelBackend",
    "SGLangFlashInferPagedDecodeBackend",
    "SGLangMemoryPoolBundle",
    "SGLangPrefillResult",
    "SGLangRoutePoolBridge",
    "SGLangRouteRow",
    "SGLangRouteKVMetadata",
    "SGLangRunnerConfig",
    "assert_sglang_flashinfer_active",
    "build_forest_depths",
    "build_forest_one_depth",
    "build_tree_depths",
    "build_tree_one_depth",
    "build_flashinfer_paged_kv_metadata",
    "create_sglang_model_runner",
    "initialize_forest_routes",
    "initialize_stage1_routes",
    "prefill_sglang_prefix",
    "reshape_token_kv_cache_as_pages",
    "sglang_runner_component_report",
]
