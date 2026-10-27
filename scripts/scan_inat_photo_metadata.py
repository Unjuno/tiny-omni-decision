"""Stream a pinned photo metadata object and retain only preselected photo IDs."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SNAPSHOT = ROOT / "inat-bird-development-snapshot-20261028-attempt-04"
PINS = {
    "private-snapshot.json": "1d97f5368855c80abdd98bfb95cdf0c6359b21bef7c8ab580318f3b3c914801d",
    "private-quarantine.json": "a83ddfe6369a46f2525d94173e2bb4ced63b52d1bd4f12e242cad8a993cac3e3",
}
URL = "https://inaturalist-open-data.s3.amazonaws.com/photos.csv.gz"
ETAG = '"c8ded8376313d0a25357209f8cafd0db-40"'
SIZE = 20668099997
REPO = Path(__file__).resolve().parents[1]


def select_photo_metadata(stream, target_ids):
    reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {"photo_id", "photo_uuid", "observation_uuid", "observer_id", "license"}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise ValueError("required metadata column missing")
    matches = defaultdict(list)
    count = 0
    for row in reader:
        count += 1
        raw_id = row.get("photo_id", "")
        if not raw_id.isdecimal():
            raise ValueError("malformed photo ID")
        photo_id = int(raw_id)
        if photo_id in target_ids:
            owner = row.get("observer_id", "")
            if not owner.isdecimal() or not row.get("observation_uuid"):
                raise ValueError("matched photo lacks identity")
            owner_key = hashlib.sha256(f"inat-bird-observer-v1:{int(owner)}".encode()).hexdigest()
            matches[photo_id].append({"photo_uuid": row["photo_uuid"],
                                      "observation_uuid": row["observation_uuid"],
                                      "photographer_key": owner_key, "license": row["license"]})
    return dict(matches), count


class DigestReader:
    def __init__(self, raw, output_parent):
        self.raw, self.output_parent = raw, output_parent
        self.digest = hashlib.sha256()
        self.count = self.last_report = 0

    def read(self, length=-1):
        data = self.raw.read(length, decode_content=False)
        self.count += len(data)
        if self.count > SIZE:
            raise ValueError("source exceeds pinned length")
        self.digest.update(data)
        if self.count // (512 * 1024**2) > self.last_report:
            self.last_report = self.count // (512 * 1024**2)
            if shutil.disk_usage(self.output_parent).free < 2 * 1024**3:
                raise OSError("disk reserve below 2 GiB")
            print(f"compressed bytes {self.count} / {SIZE}", flush=True)
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise OSError("disk reserve below 2 GiB")
    selected = []
    for name, expected in PINS.items():
        raw = (SNAPSHOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("candidate snapshot hash mismatch")
        selected.extend(json.loads(raw))
    targets = {row["photo_id"] for row in selected}
    if len(selected) != 1600 or len(targets) != 1598:
        raise ValueError("unexpected snapshot accounting")
    start = time.monotonic()
    with requests.get(URL, headers={"If-Match": ETAG, "Accept-Encoding": "identity"},
                      stream=True, timeout=(15, 120)) as response:
        if (response.status_code != 200 or response.headers.get("ETag") != ETAG
                or response.headers.get("Content-Length") != str(SIZE)):
            raise ValueError("photo source identity mismatch")
        response.raw.decode_content = False
        digest_reader = DigestReader(response.raw, output.parent)
        with gzip.GzipFile(fileobj=digest_reader) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text_stream:
                matches, count = select_photo_metadata(text_stream, targets)
        if digest_reader.count != SIZE:
            raise ValueError("incomplete compressed source")
    with requests.head(URL, timeout=30) as head:
        if (head.status_code != 200 or head.headers.get("ETag") != ETAG
                or head.headers.get("Content-Length") != str(SIZE)):
            raise ValueError("source changed during scan")
    output.mkdir(parents=True)
    payload = json.dumps(matches, sort_keys=True, indent=2) + "\n"
    (output / "private-photo-matches.json").write_text(payload, encoding="utf-8", newline="\n")
    summary = {"status": "SOURCE_SCAN_NOT_ADMITTED", "source_url": URL, "source_etag": ETAG,
               "source_bytes": SIZE, "observed_raw_sha256": digest_reader.digest.hexdigest(),
               "source_rows": count, "target_photo_ids": len(targets),
               "matched_photo_ids": len(matches),
               "missing_photo_ids": len(targets - matches.keys()),
               "photo_ids_multiple_rows": sum(len(v) > 1 for v in matches.values()),
               "matches_sha256": hashlib.sha256(payload.encode()).hexdigest(),
               "snapshot_pins": PINS, "elapsed_seconds": time.monotonic() - start,
               "limitations": ["observed SHA has no independent trusted reference",
                               "no observation-license reconciliation", "no media downloaded",
                               "no photo or ownership verification", "no split admission"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
