"""Pin public release metadata; never fetch media or outcome annotations."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import requests

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2/captaincook-release-audit-01")
REVISIONS = {
    "CaptainCook4D": "1137b884e038cc128589c18eab215024018fb642",
    "downloader": "cf248fe2139fc6e37f9737141ec719948fbd1747",
    "annotations": "a8a920a3293c4db27099a20ddbe3a3a9be1283e3",
}
FILES = {
    "CaptainCook4D": ["README.md"],
    "downloader": [
        "README.md", "LICENSE", "metadata/download_links.json",
        "download_gopro_data.py", "download_hololens_3d_data.py", "util.py",
    ],
    "annotations": [
        "README.md", "LICENSE", "ANNOTATIONS.md", "metadata/video_information.csv",
        "data_splits/person_data_split_combined.json",
    ],
}


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for repo, files in FILES.items():
        for name in files:
            url = (
                f"https://raw.githubusercontent.com/CaptainCook4D/{repo}/"
                f"{REVISIONS[repo]}/{name}"
            )
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path = ROOT / repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            hashes[f"{repo}/{name}"] = {
                "url": url, "sha256": hashlib.sha256(response.content).hexdigest(),
                "bytes": len(response.content),
            }
    links = json.loads((ROOT / "downloader/metadata/download_links.json").read_text())
    splits = json.loads(
        (ROOT / "annotations/data_splits/person_data_split_combined.json").read_text()
    )
    rows = list(csv.DictReader(io.StringIO(
        (ROOT / "annotations/metadata/video_information.csv").read_text()
    )))
    metadata = {row["recording_id"]: row for row in rows}
    stream_counts = Counter(key for row in links.values() for key, value in row.items()
                            if isinstance(value, str) and value.startswith("https://"))
    # Person identity comes from explicit metadata, never recording-ID prefixes.
    people = {name: {metadata[r]["person_id"] for r in ids} for name, ids in splits.items()}
    overlap = {}
    for a in splits:
        for b in splits:
            if a < b:
                overlap[f"{a}/{b}"] = {
                    "recordings": sorted(set(splits[a]) & set(splits[b])),
                    "persons": sorted(people[a] & people[b]),
                }
    groups = {
        "video_audio_accelerometer": [
            "hololens_sync_pv_video", "hololens_raw_mc_pkl",
            "hololens_sync_imu_accelerometer_pkl",
        ],
    }
    availability = {
        name: {split: [r for r in ids if all(links.get(r, {}).get(k) for k in keys)]
               for split, ids in splits.items()}
        for name, keys in groups.items()
    }
    result = {
        "revisions": REVISIONS, "files": hashes,
        "recordings_with_download_metadata": len(links),
        "recordings_with_identity_metadata": len(metadata),
        "stream_link_counts": dict(sorted(stream_counts.items())),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_overlap_from_video_information": overlap,
        "requested_stream_groups": groups, "availability": availability,
        "media_downloaded": False, "outcome_annotations_loaded": False,
        "limitation": "Link presence is not availability, synchronization, or task quality.",
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("files", "availability")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
