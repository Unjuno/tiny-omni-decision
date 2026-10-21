"""Build extractor inputs for the frozen train-only transfer cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tiny_omni_decision.perception_visibility_transfer import build_inference_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort_path = args.cohort.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite transfer manifest: {output}")
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("source_split") != "train" or cohort.get("fit_video_overlap") != 0:
        raise ValueError("cohort must be a frozen, video-disjoint train-only selection")
    manifest = build_inference_manifest(cohort["examples"])
    manifest["metadata"] = {
        "scope": "train-only unseen-video transfer diagnostic; nonblind, non-independent",
        "source_revision": cohort["source_revision"],
        "cohort_sha256": sha256_file(cohort_path),
        "fit_video_overlap": 0,
        "feature_window_seconds": 2.0,
        "video_frame_count": 8,
        "frame_sampling": "uniform inclusive endpoints; nearest decoded frame PTS",
        "labels_used_for_selection": True,
        "training_or_checkpoint_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS_TRANSFER_INFERENCE_MANIFEST",
                "cohort_sha256": manifest["metadata"]["cohort_sha256"],
                "train_rows": len(manifest["train"]),
                "validation_rows": len(manifest["validation"]),
                "manifest_sha256": sha256_file(output),
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
