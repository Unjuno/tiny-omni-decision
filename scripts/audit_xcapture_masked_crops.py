"""Create hash-verified masked crops for a bounded X-Capture sample audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tiny_omni_decision.xcapture_preprocess import masked_object_crop


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.sample_dir / "sample-audit.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_by_sample: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    partition_by_sample: dict[int, str] = {}
    for item in manifest["files"]:
        if item["partition"] not in {"train", "validation"}:
            raise SystemExit("sample manifest includes a non-development partition")
        sample_id = int(item["sample_id"])
        partition_by_sample[sample_id] = item["partition"]
        files_by_sample[sample_id][item["modality"]] = item

    rows = []
    for sample_id in sorted(files_by_sample):
        files = files_by_sample[sample_id]
        if not {"image", "object_mask"}.issubset(files):
            raise SystemExit(f"sample {sample_id} lacks image or object_mask")
        image_item, mask_item = files["image"], files["object_mask"]
        image_path = args.sample_dir / image_item["output_path"]
        mask_path = args.sample_dir / mask_item["output_path"]
        for path, item in ((image_path, image_item), (mask_path, mask_item)):
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise SystemExit(f"sample file missing or size mismatch: {path}")
            if sha256_file(path) != item["sha256"]:
                raise SystemExit(f"sample file SHA-256 mismatch: {path}")

        crop, measurements = masked_object_crop(image_path, mask_path)
        output_path = args.output_dir / f"sample-{sample_id:02d}-image-objectcrop.png"
        crop.save(output_path, format="PNG", optimize=False)
        rows.append(
            {
                "sample_id": sample_id,
                "partition": partition_by_sample[sample_id],
                "object_id": image_item["object_id"],
                "point_id": image_item["point_id"],
                "source_image_sha256": image_item["sha256"],
                "source_mask_sha256": mask_item["sha256"],
                "output_path": output_path.name,
                "output_bytes": output_path.stat().st_size,
                "output_sha256": sha256_file(output_path),
                **measurements,
            }
        )

    report = {
        "dataset_revision": manifest["dataset_revision"],
        "archive_sha256": manifest["archive_sha256"],
        "sample_manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(rows),
        "partitions": {row["partition"] for row in rows},
        "preprocessing": {
            "method": (
                "binary point-cloud mask; zero outside foreground; "
                "crop bbox with 8% padding"
            ),
            "evaluation_media_opened": False,
        },
        "rows": rows,
    }
    report["partitions"] = sorted(report["partitions"])
    (args.output_dir / "masked-crops.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
