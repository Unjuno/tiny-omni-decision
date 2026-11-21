"""Freeze the TallyQA-VG train-only grouped-CV assignments before model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.cv_tallyqa_vg_train_groups import _write_json, build_preflight
from tiny_omni_decision.dataset import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
    payload = build_preflight(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    print(json.dumps({
        "preflight": str(output),
        "sha256": sha256_file(output),
        "rows": payload["rows"],
        "unique_images": payload["unique_images"],
        "connected_groups": payload["connected_groups"],
        "folds": [
            {key: value for key, value in fold.items() if key != "fold"}
            for fold in payload["folds"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
