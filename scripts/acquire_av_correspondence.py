"""Bounded, immutable 256/200 development acquisition; no test archive access."""

import argparse
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from scripts.audit_correspondence_validation import sha, write
from scripts.audit_mmfit import RemoteZip

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
ROUTE_HASHES = {
    "train": "560f3bc7235c842746038e6e5a886471f4dc08cc515d9eba4d12968e608f393b",
    "valid": "03d389b0eda847b4d16e0205a7c285ad78ebecfb6405fde1c85c0ddfd569d395",
}


def checked_json(path, digest):
    if sha(path) != digest:
        raise ValueError(f"source changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_selection(rows, expected_count, split):
    if len(rows) != expected_count or len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("missing/duplicate selected records")
    if any(r["metadata"]["split"] != split for r in rows):
        raise ValueError("incorrect split")
    if any(not 0 < r["bytes"] <= 256_000_000 for r in rows):
        raise ValueError("member exceeds 256MB cap; do not replace")
    if sum(r["bytes"] for r in rows) > 8_000_000_000:
        raise ValueError("split exceeds 8GB cap")


def plan(output):
    audit = ROOT / "perception-correspondence-valid-audit-01"
    valid_route = checked_json(audit / "valid-routing.json", ROUTE_HASHES["valid"])
    train_route = checked_json(
        ROOT / "perception-correspondence-plan-01/train-routing.json", ROUTE_HASHES["train"]
    )
    pins = json.loads((audit / "pins.json").read_text(encoding="utf-8"))
    if pins["media"] != dict(
        url="https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip?generation=1685479088600941",
        bytes=75375143328,
        etag='"-CO3u6PLynf8CEAE="',
        generation="1685479088600941",
    ):
        raise ValueError("validation media pin changed")
    source = ROOT / "perception-test-source-audit-01"
    train_archive = source / "train_annotations.zip"
    if sha(train_archive) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("train annotations changed")
    with zipfile.ZipFile(train_archive) as z:
        annotations = json.loads(z.read("all_train.json"))
    cuts = checked_json(
        source / "cut_frame_mapping_train.json",
        "cbea6c721fd150f861ec6fe8e7ff005792f754df1359b1364349b5abcaab6cc4",
    )
    train_pin = dict(
        url="https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip?generation=1685446268139925",
        bytes=28404087974,
        etag='"-CJXz5tD4nP8CEAE="',
        generation="1685446268139925",
    )
    train_rows = []
    with zipfile.ZipFile(RemoteZip(train_pin["url"], train_pin["bytes"], train_pin["etag"])) as z:
        members = {}
        for entry in z.infolist():
            members.setdefault(Path(entry.filename).stem, []).append(entry)
        for row in train_route:
            video = row["audio_video_id"]
            entries = members.get(video, [])
            if len(entries) != 1 or not entries[0].filename.endswith(".mp4"):
                raise ValueError("missing/duplicate selected media")
            item = entries[0]
            train_rows.append(
                dict(
                    video_id=video,
                    member=item.filename,
                    bytes=item.file_size,
                    compressed_bytes=item.compress_size,
                    crc32=item.CRC,
                    metadata=annotations[video]["metadata"],
                    cut_frame=cuts[video],
                )
            )
    valid_rows = json.loads((audit / "selected-media.json").read_text(encoding="utf-8"))
    if [r["video_id"] for r in valid_rows] != [r["audio_video_id"] for r in valid_route]:
        raise ValueError("validation ordering changed")
    validate_selection(train_rows, 256, "train")
    validate_selection(valid_rows, 200, "valid")
    if {r["video_id"] for r in train_rows} & {r["video_id"] for r in valid_rows}:
        raise ValueError("cross-split identity overlap")
    result = dict(
        splits={
            "train": dict(pin=train_pin, rows=train_rows),
            "valid": dict(pin=pins["media"], rows=valid_rows),
        },
        total_bytes=sum(r["bytes"] for r in train_rows + valid_rows),
        route_hashes=ROUTE_HASHES,
        attribution="Perception Test, DeepMind Technologies Limited, CC-BY-4.0",
        source_revision="3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
    )
    output.mkdir(exist_ok=False)
    write(output / "plan.json", result)
    print(
        json.dumps(dict(total_bytes=result["total_bytes"], plan_sha256=sha(output / "plan.json")))
    )


def acquire(args):
    plan_path = args.plan
    source = checked_json(plan_path, args.plan_sha256)
    if source["route_hashes"] != ROUTE_HASHES:
        raise ValueError("routing pins changed")
    selected = source["splits"][args.split]
    rows = selected["rows"]
    validate_selection(rows, 256 if args.split == "train" else 200, args.split)
    # Reserve 8GB beyond both splits for preprocessing and the user's other work.
    if shutil.disk_usage(ROOT).free < source["total_bytes"] + 8_000_000_000:
        raise ValueError("insufficient disk reserve")
    args.output.mkdir(exist_ok=False)
    report = dict(
        split=args.split,
        plan_sha256=args.plan_sha256,
        clips=[],
        status="RUNNING",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=sha(Path(__file__)),
        started_unix=time.time(),
    )
    cache_dir = ROOT / "perception-media-probe-02"
    cache = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))["clips"]
    cache = {r["video_id"]: r for r in cache}
    pin = selected["pin"]
    try:
        with zipfile.ZipFile(RemoteZip(pin["url"], pin["bytes"], pin["etag"])) as z:
            for row in rows:
                entry = z.getinfo(row["member"])
                if (entry.file_size, entry.compress_size, entry.CRC) != (
                    row["bytes"],
                    row["compressed_bytes"],
                    row["crc32"],
                ):
                    raise ValueError("ZIP member changed")
                if shutil.disk_usage(ROOT).free < row["bytes"] + 8_000_000_000:
                    raise ValueError("disk reserve exhausted")
                saved = args.output / (row["video_id"] + ".mp4")
                prior = cache.get(row["video_id"]) if args.split == "train" else None
                if prior:
                    original = cache_dir / prior["file"]
                    if sha(original) != prior["sha256"] or original.stat().st_size != row["bytes"]:
                        raise ValueError("cached media changed")
                    # Reference verified immutable pilot bytes without duplicating media.
                    saved = original
                else:
                    partial = saved.with_suffix(".mp4.partial")
                    with z.open(entry) as reader, partial.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    if partial.stat().st_size != row["bytes"]:
                        raise ValueError("download length mismatch")
                    partial.rename(saved)
                report["clips"].append(
                    dict(**row, path=str(saved), sha256=sha(saved), reused=bool(prior))
                )
                write(args.output / "manifest.json", report)
                print(
                    json.dumps(
                        dict(split=args.split, complete=len(report["clips"]), total=len(rows))
                    ),
                    flush=True,
                )
        report["status"] = "PASS_ACQUISITION_NOT_DECODE"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_unix"] = time.time()
        write(args.output / "manifest.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--split", choices=("train", "valid"))
    args = parser.parse_args()
    if args.plan:
        if not args.plan_sha256 or not args.split:
            parser.error("acquisition requires plan hash and development split")
        acquire(args)
    else:
        plan(args.output)


if __name__ == "__main__":
    main()
