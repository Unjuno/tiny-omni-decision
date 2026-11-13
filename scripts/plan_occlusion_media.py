"""Pinned ZIP index and local media audit for the fixed complete-pair family."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from scripts.audit_correspondence_validation import sha, write
from scripts.audit_mmfit import RemoteZip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    old = json.loads(
        (root / "perception-correspondence-acquisition-plan-01/plan.json").read_text(
            encoding="utf-8"
        )
    )
    pair_hashes = {
        "train": "8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7",
        "valid": "30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2",
    }
    report = dict(splits={}, training=False, downloaded_media=False)
    for split, digest in pair_hashes.items():
        path = root / f"perception-occlusion-pairs-01/{split}.json"
        if sha(path) != digest:
            raise ValueError("pair set changed")
        pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
        local = json.loads(
            (root / f"perception-correspondence-{split}-media-01/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        local = {r["video_id"]: r for r in local["clips"]}
        cutroot = (
            "perception-test-source-audit-01"
            if split == "train"
            else "perception-correspondence-valid-audit-01"
        )
        cuts_path = root / cutroot / f"cut_frame_mapping_{split}.json"
        cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
        pin = old["splits"][split]["pin"]
        rows = []
        with zipfile.ZipFile(RemoteZip(pin["url"], pin["bytes"], pin["etag"])) as archive:
            members = {}
            for info in archive.infolist():
                members.setdefault(Path(info.filename).stem, []).append(info)
            for pair in pairs:
                identity = pair["video_id"]
                entries = members.get(identity, [])
                if len(entries) != 1 or not entries[0].filename.endswith(".mp4"):
                    raise ValueError("missing/duplicate member")
                info = entries[0]
                existing = local.get(identity)
                if existing and sha(Path(existing["path"])) != existing["sha256"]:
                    raise ValueError("local media changed")
                rows.append(
                    dict(
                        video_id=identity,
                        member=info.filename,
                        bytes=info.file_size,
                        compressed_bytes=info.compress_size,
                        crc32=info.CRC,
                        cut_frame=cuts[identity],
                        existing=existing,
                    )
                )
        report["splits"][split] = dict(
            pin=pin,
            rows=rows,
            pair_sha256=digest,
            cuts_sha256=sha(cuts_path),
            missing=sum(r["existing"] is None for r in rows),
            additional_bytes=sum(r["bytes"] for r in rows if not r["existing"]),
        )
    report["free_bytes"] = shutil.disk_usage(root).free
    report["additional_bytes"] = sum(s["additional_bytes"] for s in report["splits"].values())
    report["reserve_8gb_after_media"] = (
        report["free_bytes"] - report["additional_bytes"] >= 8_000_000_000
    )
    write(args.output, report)
    print(
        json.dumps(
            {
                s: {k: v[k] for k in ("missing", "additional_bytes")}
                for s, v in report["splits"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
