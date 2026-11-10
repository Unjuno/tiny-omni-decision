"""Create AVMIT source-group manifests without opening media or test targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.avmit_metadata import audit_avmit_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings-csv", type=Path, required=True)
    parser.add_argument("--test-set-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = audit_avmit_metadata(
        args.ratings_csv, args.test_set_csv, args.output_dir, seed=args.seed
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
