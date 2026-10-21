"""Pin official development validation and inspect media index, never test data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import requests

from scripts.audit_mmfit import RemoteZip
from tiny_omni_decision.av_correspondence import build_quartets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def pin(url):
    response = requests.head(url, timeout=30)
    response.raise_for_status()
    headers = response.headers
    generation = headers["x-goog-generation"]
    return dict(
        url=f"{url}?generation={generation}",
        bytes=int(headers["Content-Length"]),
        etag=headers["ETag"],
        generation=generation,
    )


def download(record, path, cap):
    if record["bytes"] > cap:
        raise ValueError("metadata exceeds fixed acquisition cap")
    with requests.get(
        record["url"], headers={"If-Match": record["etag"]}, stream=True, timeout=30
    ) as response:
        response.raise_for_status()
        total = 0
        with path.open("xb") as output:
            for chunk in response.iter_content(1024 * 1024):
                total += len(chunk)
                if total > record["bytes"]:
                    raise ValueError("download length exceeded")
                output.write(chunk)
        if total != record["bytes"]:
            raise ValueError("download length mismatch")
    record["sha256"] = sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    base = "https://storage.googleapis.com/dm-perception-test"
    records = {
        "annotations": pin(base + "/zip_data/valid_annotations.zip"),
        "cuts": pin(base + "/misc/cut_frame_mapping_valid.json"),
        "media": pin(base + "/zip_data/valid_videos.zip"),
    }
    write(args.output / "pins.json", records)
    archive = args.output / "valid_annotations.zip"
    cuts_path = args.output / "cut_frame_mapping_valid.json"
    download(records["annotations"], archive, 100_000_000)
    download(records["cuts"], cuts_path, 1_000_000)
    write(args.output / "pins.json", records)
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ["all_valid.json"] or z.infolist()[0].file_size > 600_000_000:
            raise ValueError("unexpected annotation member")
        annotations = json.loads(z.read("all_valid.json"))
    train = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip"
    )
    if sha(train) != "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78":
        raise ValueError("train revision changed")
    with zipfile.ZipFile(train) as z:
        train_ids = set(json.loads(z.read("all_train.json")))
    if train_ids.intersection(annotations):
        raise ValueError("official recording ID overlap")
    if any(r["metadata"]["split"] != "valid" for r in annotations.values()):
        raise ValueError("nonvalidation record")
    routing = build_quartets(annotations, split="valid", limit=200, seed=17)
    write(args.output / "valid-routing.json", routing)
    cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    media = records["media"]
    selected = []
    with zipfile.ZipFile(RemoteZip(media["url"], media["bytes"], media["etag"])) as z:
        members = {}
        for member in z.infolist():
            members.setdefault(Path(member.filename).stem, []).append(member)
        for row in routing:
            video = row["audio_video_id"]
            entries = members.get(video, [])
            if len(entries) != 1 or not entries[0].filename.endswith(".mp4"):
                raise ValueError("missing or duplicate selected media")
            item = entries[0]
            metadata = annotations[video]["metadata"]
            cut = cuts[video]
            if cut != -1 and not 0 <= cut < metadata["num_frames"]:
                raise ValueError("invalid cut")
            selected.append(
                dict(
                    video_id=video,
                    member=item.filename,
                    bytes=item.file_size,
                    compressed_bytes=item.compress_size,
                    crc32=item.CRC,
                    cut_frame=cut,
                    metadata=metadata,
                )
            )
    write(args.output / "selected-media.json", selected)
    result = dict(
        status="PASS_METADATA_ONLY",
        total_valid_records=len(annotations),
        selected_records=len(routing),
        quartets=len(routing) // 4,
        official_id_overlap=0,
        selected_media_bytes=sum(r["bytes"] for r in selected),
        maximum_member_bytes=max(r["bytes"] for r in selected),
        selected_cut_count=sum(r["cut_frame"] != -1 for r in selected),
        routing_sha256=sha(args.output / "valid-routing.json"),
        media_payloads_read=False,
        training=False,
        limitations=["No participant-disjoint guarantee", "Media content overlap not yet checked"],
    )
    write(args.output / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
