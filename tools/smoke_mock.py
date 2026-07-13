from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas_0709.backends import DeterministicMockBackend
from atlas_0709.controller import AtlasCleanConfig, AtlasCleanGenerator


def main() -> int:
    config = AtlasCleanConfig(k=3, d=4, max_new_tokens=16)
    generator = AtlasCleanGenerator(
        config=config,
        drafter=DeterministicMockBackend(name="drafter", salt=3),
        target=DeterministicMockBackend(name="target", salt=19),
    )
    result = generator.generate([1, 2, 3, 4])
    print(
        json.dumps(
            {
                "generated_token_ids": result.generated_token_ids,
                "round_count": len(result.rounds),
                "rounds": [
                    {
                        "round_index": trace.round_index,
                        "committed_tokens": list(trace.committed_tokens),
                        "forest_depth": trace.forest_depth,
                        "verify_returned_before_forest_done": trace.verify_returned_before_forest_done,
                    }
                    for trace in result.rounds
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

