"""Build extraction inputs for the fixed Perception official-validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tiny_omni_decision.perception_visibility_validation import build_validation_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--media-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.selection.resolve(strict=True)
    media_path = args.media_validation.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    manifest = build_validation_manifest(
        json.loads(selection_path.read_text(encoding="utf-8")),
        json.loads(media_path.read_text(encoding="utf-8")),
        selection_sha256=sha256_file(selection_path),
        media_validation_sha256=sha256_file(media_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS_VALIDATION_TRANSFER_MANIFEST",
                "validation_rows": len(manifest["validation"]),
                "metadata": manifest["metadata"],
                "manifest_sha256": sha256_file(output),
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
