"""Assemble immutable task-scoped Path A runtimes in a new directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tiny_omni_decision.modular_bundle import assemble_modular_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-bundle", type=Path, required=True)
    parser.add_argument("--image-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout.strip()
    manifest = assemble_modular_bundle(
        {"audio": args.audio_bundle, "image": args.image_bundle},
        args.output_dir,
        source_commit=source_commit,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
