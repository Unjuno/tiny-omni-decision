"""One bounded retry of the two fixed quartet image failures."""

import argparse
import base64
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

from scripts.acquire_openimages_quartet_media import OLD, VAL1, VAL2
from scripts.acquire_openimages_train_candidate_media import compare_images, prepare_records
from scripts.download_openimages_pilot_media import MAX_MEDIA_BYTES, download_one
from scripts.train_source_unimodal_cv import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PRIOR = ROOT / "openimages-quartet-media-20261027-attempt-01"
PAGES = ROOT / "openimages-quartet-candidates-20261027-attempt-01/private-page-checks.jsonl"
PINNED = {
    PRIOR / "summary.json": "aedf53e3f2a30c6602b08843e688d720947df85be6807ee743cda83329d7d8a5",
    PRIOR / "private-media-manifest.jsonl": (
        "1ff2979c09722bd8554f6ac832d578d3483db1de2862a89d4f8958046a5c6c02"
    ),
    PAGES: "a937576c0951de6e006783f07622af26b4bbbfa4a40e007efd68df3a91ebcfb9",
    OLD: "7ff4b8bda635812ed404adf06b279e55a96838f18407e39ab80e9e659f59a9a7",
    VAL1: "81b225f38d311ffe94003b2419100b33f88ad484d7f0f4ab1f3f35877e048560",
    VAL2: "ab531846397629d454eebf5e36b1d69a5f5fc436a2d67861aac8e55ecb9e40b0",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_QUARTET_MEDIA_RETRY_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare_retries(prior, page_rows):
    failures = [row for row in prior if row["status"] != "saved"]
    if len(failures) != 2 or {row.get("error") for row in failures} != {
        "HTTP Error 410: Gone", "HTTP Error 429: Too Many Requests"
    }:
        raise ValueError("unexpected original failure set")
    page = {row["ImageID"]: row for row in page_rows}
    if len(page) != len(page_rows):
        raise ValueError("duplicate page receipt")
    selected = [page[row["image_id"]] for row in failures]
    candidates = [{"image_id": row["image_id"], "target_label": row["target_label"]}
                  for row in failures]
    records = {row["image_id"]: row for row in prepare_records(candidates, selected)}
    if len(records) != 2:
        raise ValueError("page declaration not matched")
    retries = []
    for old in sorted(failures, key=lambda row: "429" in row["error"]):
        receipt = page[old["image_id"]]
        original = receipt["OriginalURL"]
        checked = prepare_records(candidates=[{"image_id": old["image_id"],
                                               "target_label": old["target_label"]}],
                                  receipts=[receipt | {"Thumbnail300KURL": None}])[0]
        if checked["url"] != original:
            raise ValueError("original URL validation failed")
        if "429" in old["error"] and original != old["url"]:
            raise ValueError("429 retry must preserve the original URL")
        if "410" in old["error"] and old["url"] != receipt.get("Thumbnail300KURL"):
            raise ValueError("410 row is not the recorded thumbnail")
        if len(base64.b64decode(receipt["OriginalMD5"], validate=True)) != 16:
            raise ValueError("invalid original MD5")
        if not urlparse(original).path:
            raise ValueError("original URL missing path")
        retries.append(checked | {"original_md5_base64": receipt["OriginalMD5"],
                                  "prior_error": old["error"]})
    return retries


def verify_original(path, expected):
    expected_bytes = base64.b64decode(expected, validate=True)
    actual = hashlib.md5(path.read_bytes(), usedforsecurity=False).digest()
    return actual == expected_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != expected for path, expected in PINNED.items()):
        raise ValueError("pinned input hash mismatch")
    if time.time() - (PRIOR / "summary.json").stat().st_mtime < 300:
        raise ValueError("five-minute retry cooling period has not elapsed")
    prior = rows(PRIOR / "private-media-manifest.jsonl")
    records = prepare_retries(prior, rows(PAGES))
    output.mkdir(parents=True)
    results, stop = [], "completed"
    for record in records:
        if shutil.disk_usage(output).free < 1024**3 + MAX_MEDIA_BYTES:
            stop = "disk_reserve"
            break
        result = download_one(record, split_dir=output / "train")
        if result["status"] == "saved":
            path = output / "train" / result["local_path"]
            result["original_md5_verified"] = verify_original(path, record["original_md5_base64"])
            if not result["original_md5_verified"]:
                result["status"] = "rejected_original_md5_mismatch"
        results.append(result)
        if "429" in result.get("error", ""):
            stop = "rate_limited"
            break
    reference = [row for path in (OLD, VAL1, VAL2) for row in rows(path)] + prior
    matches = compare_images(results, reference)
    summary = {"attempted": len(results), "verified_saved": sum(
        row["status"] == "saved" for row in results), "stop_reason": stop,
        "duplicate_warning_pairs": len(matches), "individual_rights_verified": False,
        "source_hashes": {str(path): expected for path, expected in PINNED.items()},
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__))}
    (output / "private-retry-manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8")
    (output / "private-duplicate-warnings.json").write_text(
        json.dumps(matches, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
