from .backends import DeterministicMockBackend
from .controller import AtlasCleanConfig, AtlasCleanGenerator, GenerationResult
from .hf_backend import HFRecomputeBatchBackend

_FLASHINFER_VERIFY_EXPORTS = {
    "FlashInferFullVerifyBenchmarkResult",
    "FlashInferFullVerifyConfig",
    "FlashInferFullVerifyRunner",
}

_DISTRIBUTED_EXPORTS = {
    "DirectFlashInferMaskedTreeVerifyBackend",
    "DistributedAtlasConfig",
    "DistributedGenerationResult",
    "PagedDistributedAtlasGenerator",
    "SerialAtlasGenerator",
    "RemoteTargetClient",
    "TargetServerApp",
}

__all__ = [
    "AtlasCleanConfig",
    "AtlasCleanGenerator",
    "DeterministicMockBackend",
    "FlashInferFullVerifyBenchmarkResult",
    "FlashInferFullVerifyConfig",
    "FlashInferFullVerifyRunner",
    "GenerationResult",
    "HFRecomputeBatchBackend",
    "DirectFlashInferMaskedTreeVerifyBackend",
    "DistributedAtlasConfig",
    "DistributedGenerationResult",
    "PagedDistributedAtlasGenerator",
    "SerialAtlasGenerator",
    "RemoteTargetClient",
    "TargetServerApp",
]


def __getattr__(name: str):
    if name in _FLASHINFER_VERIFY_EXPORTS:
        from . import flashinfer_full_verify

        return getattr(flashinfer_full_verify, name)
    if name in _DISTRIBUTED_EXPORTS:
        if name == "DirectFlashInferMaskedTreeVerifyBackend":
            from . import target_runtime

            return getattr(target_runtime, name)
        if name in {"RemoteTargetClient", "TargetServerApp"}:
            from . import rpc

            return getattr(rpc, name)
        if name == "SerialAtlasGenerator":
            from . import serial_system

            return getattr(serial_system, name)
        from . import distributed_system

        return getattr(distributed_system, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
