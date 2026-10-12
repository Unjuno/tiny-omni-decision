"""Common-clock development windows, fail closed on incomplete acquisition."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_av_correspondence_windows import decode_window, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    manifests, rows = {}, []
    seen_ids, seen_media = set(), {}
    for split, count in (("train", 256), ("valid", 200)):
        path = root / f"perception-correspondence-{split}-media-01/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "PASS_ACQUISITION_NOT_DECODE":
            raise ValueError(f"{split} acquisition incomplete")
        if (
            manifest["plan_sha256"]
            != ("0c6a07048c393d2fddd54fb3e72883055bb43634f9e3a8cac5d8570edcc39931")
            or len(manifest["clips"]) != count
        ):
            raise ValueError("acquisition plan/count changed")
        manifests[str(path)] = sha(path)
        for row in manifest["clips"]:
            if row["metadata"]["split"] != split or row["video_id"] in seen_ids:
                raise ValueError("split/identity mismatch")
            seen_ids.add(row["video_id"])
            media = Path(row["path"])
            digest = sha(media)
            if digest != row["sha256"]:
                raise ValueError("media changed")
            if digest in seen_media and seen_media[digest] != split:
                raise ValueError("cross-split media duplicate")
            seen_media[digest] = split
            rows.append(row)
    args.output.mkdir(exist_ok=False)
    result = dict(
        status="RUNNING",
        manifests=manifests,
        windows=[],
        training=False,
        evaluation=False,
        content_check="exact prepared array bytes",
    )
    content_splits = {}
    try:
        for row in rows:
            payload, timing = decode_window(Path(row["path"]), row["cut_frame"])
            content = hashlib.sha256()
            for name in ("audio", "video"):
                array = payload[name]
                content.update(f"{name}:{array.dtype}:{array.shape}".encode())
                content.update(array.tobytes())
            digest = content.hexdigest()
            split = row["metadata"]["split"]
            if digest in content_splits and content_splits[digest] != split:
                raise ValueError("cross-split prepared-content duplicate")
            content_splits[digest] = split
            output = args.output / f"window-{len(result['windows']):03}.npz"
            np.savez_compressed(output, **payload)
            result["windows"].append(
                dict(
                    id=row["video_id"],
                    split=split,
                    status="prepared",
                    path=str(output),
                    sha256=sha(output),
                    media_sha256=row["sha256"],
                    content_sha256=digest,
                    metadata_sample_rate=row["metadata"]["audio_sample_rate"],
                    **timing,
                )
            )
            (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(dict(prepared=len(result["windows"]), total=len(rows))), flush=True)
        if any(sha(Path(path)) != digest for path, digest in manifests.items()):
            raise ValueError("manifest changed during preparation")
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
