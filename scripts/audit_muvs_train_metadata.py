"""Audit only MUVS publisher-train metadata and camera-selection labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_omni_decision.muvs_preflight import inspect_muvs_train_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output path must be new; preserve prior receipts")
    report = inspect_muvs_train_archive(args.archive)
    args.output.mkdir(parents=True)
    report["output_directory"] = str(args.output.resolve())
    output_file = args.output / "train-metadata-preflight.json"
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["task"], ensure_ascii=False, indent=2))
    print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
