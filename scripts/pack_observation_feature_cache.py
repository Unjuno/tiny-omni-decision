"""Create a new immutable packed snapshot of an observation feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.packed_cache import PackedObservationFeatureCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packed = PackedObservationFeatureCache.pack(args.source_cache, args.output)
    try:
        packed.verify()
        print(
            json.dumps(
                {
                    "status": "packed_cache_created_and_verified",
                    "output": str(args.output.resolve()),
                    **packed.manifest,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        packed.close()


if __name__ == "__main__":
    main()
