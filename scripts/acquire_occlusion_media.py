"""Acquire the pinned complete occlusion family into a fresh directory."""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

from scripts.audit_correspondence_validation import sha, write
from scripts.audit_mmfit import RemoteZip

PLAN_SHA = "a0631ff6e69a1ba709ee9c8c502d7ea67896f04863f73a737ab46b7406fbb186"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    path = root / "perception-occlusion-media-plan-01.json"
    if sha(path) != PLAN_SHA:
        raise ValueError("plan changed")
    plan = json.loads(path.read_text(encoding="utf-8"))
    selected = plan["splits"][args.split]
    rows = selected["rows"]
    if len(rows) != (99 if args.split == "train" else 257):
        raise ValueError("selection changed")
    if any(not 0 < r["bytes"] <= 256_000_000 for r in rows):
        raise ValueError("member size cap")
    if shutil.disk_usage(root).free < plan["additional_bytes"] + 8_000_000_000:
        raise ValueError("insufficient disk reserve")
    args.output.mkdir(exist_ok=False)
    report = dict(
        status="RUNNING",
        split=args.split,
        plan_sha256=PLAN_SHA,
        started_unix=time.time(),
        script_sha256=sha(Path(__file__)),
        clips=[],
    )
    pin = selected["pin"]
    try:
        with zipfile.ZipFile(RemoteZip(pin["url"], pin["bytes"], pin["etag"])) as z:
            for row in rows:
                info = z.getinfo(row["member"])
                if (info.file_size, info.compress_size, info.CRC) != (
                    row["bytes"],
                    row["compressed_bytes"],
                    row["crc32"],
                ):
                    raise ValueError("ZIP entry changed")
                if row["existing"]:
                    saved = Path(row["existing"]["path"])
                    if sha(saved) != row["existing"]["sha256"]:
                        raise ValueError("existing media changed")
                else:
                    if shutil.disk_usage(root).free < row["bytes"] + 8_000_000_000:
                        raise ValueError("disk reserve exhausted")
                    saved = args.output / (row["video_id"] + ".mp4")
                    partial = saved.with_suffix(".mp4.partial")
                    with z.open(info) as reader, partial.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, 1024 * 1024)
                    if partial.stat().st_size != row["bytes"]:
                        raise ValueError("length mismatch")
                    partial.rename(saved)
                if saved.stat().st_size != row["bytes"]:
                    raise ValueError("size mismatch")
                report["clips"].append(
                    dict(
                        video_id=row["video_id"],
                        path=str(saved),
                        sha256=sha(saved),
                        cut_frame=row["cut_frame"],
                    )
                )
                write(args.output / "manifest.json", report)
                print(json.dumps(dict(complete=len(report["clips"]), total=len(rows))), flush=True)
        report["status"] = "PASS_ACQUISITION_NOT_DECODE"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_unix"] = time.time()
        write(args.output / "manifest.json", report)


if __name__ == "__main__":
    main()
