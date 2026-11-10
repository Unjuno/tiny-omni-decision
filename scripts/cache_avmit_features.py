"""Verify and cache only AVMIT train/calibration/validation pair features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.avmit_features import audit_and_cache_avmit_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-metadata-csv", type=Path, required=True)
    parser.add_argument("--pair-manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-source-metadata-sha256", required=True)
    args = parser.parse_args()
    report = audit_and_cache_avmit_features(
        args.archive,
        args.source_metadata_csv,
        args.pair_manifest_dir,
        args.output,
        expected_archive_sha256=args.expected_sha256,
        expected_source_metadata_sha256=args.expected_source_metadata_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
