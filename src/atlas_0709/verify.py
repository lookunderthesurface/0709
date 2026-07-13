from __future__ import annotations

from typing import Sequence

from .backends import BatchVerifyBackend
from .types import RouteState, TargetVerifyResult


class BatchTargetVerifier:
    """Ordinary batch target verifier.

    The verifier receives the k completed stage-1 paths and asks the target
    backend to score them as one batch. It intentionally does not implement a
    custom masked tree attention path.
    """

    def __init__(self, backend: BatchVerifyBackend) -> None:
        self.backend = backend

    def verify(
        self,
        *,
        prefix_token_ids: Sequence[int],
        routes: Sequence[RouteState],
    ) -> TargetVerifyResult:
        if not routes:
            raise ValueError("verify requires at least one route")
        return self.backend.verify_batch(prefix_token_ids=prefix_token_ids, routes=routes)

