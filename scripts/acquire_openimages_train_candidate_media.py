"""Bounded Flickr media acquisition for prechecked train candidates."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from scripts.download_openimages_pilot_media import ALLOWED_HOST, MAX_MEDIA_BYTES, download_one
from scripts.train_source_unimodal_cv import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SOURCE = ROOT / "openimages-in-scope-train-candidates-20261027-attempt-01"
INPUTS = {
    SOURCE / "candidates.json": "67a9b9a468b868c82ec43cc73b33fe477fb8b97fb079f1035a2c67e051a07d5a",
    SOURCE / "private-page-checks.jsonl": (
        "5f2b299dde5646711ab0a785643904aa0dd993f3bcd710cd9b76eadd568db9fc"
    ),
    ROOT / "openimages-v6-media-cache-20261026-attempt-01/validation-media-manifest.jsonl":
        "81b225f38d311ffe94003b2419100b33f88ad484d7f0f4ab1f3f35877e048560",
    ROOT / "openimages-freshval-20261026-attempt-01/media-cache/validation-media-manifest.jsonl":
        "ab531846397629d454eebf5e36b1d69a5f5fc436a2d67861aac8e55ecb9e40b0",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_TRAIN_CANDIDATE_MEDIA_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]
RESERVE = 1024**3
BUDGET = 128 * 1024**2


def prepare_records(candidates, receipts):
    by_id = {row["ImageID"]: row for row in receipts}
    if len(by_id) != len(receipts) or len(candidates) != len(by_id):
        raise ValueError("candidate and page receipt counts mismatch")
    records, seen = [], set()
    for candidate in candidates:
        image_id = candidate["image_id"]
        if not re.fullmatch(r"[0-9a-f]{16}", image_id) or image_id in seen:
            raise ValueError("invalid or duplicated image ID")
        seen.add(image_id)
        receipt = by_id[image_id]
        if receipt.get("landing_page_check_status") != "metadata_confirmed_on_landing_page":
            continue
        if receipt.get("page_license_and_attribution_match") is not True:
            raise ValueError("matching page status lacks matching declaration")
        url = receipt.get("Thumbnail300KURL") or receipt.get("OriginalURL") or ""
        parsed = urlparse(url)
        page_photo = urlparse(receipt["OriginalLandingURL"]).path.rstrip("/").split("/")[-1]
        photo = Path(parsed.path).name.split("_", 1)[0]
        if (parsed.scheme != "https" or not ALLOWED_HOST.fullmatch(parsed.hostname or "")
                or not page_photo.isdecimal() or photo != page_photo):
            raise ValueError("media CDN/photo identity mismatch")
        records.append({"image_id": image_id, "split": "train", "url": url,
                        "target_label": candidate["target_label"]})
    return records


def compare_images(train, validation):
    matches = []
    for a in train:
        if a.get("status") != "saved":
            continue
        for b in validation:
            if b.get("status") != "saved":
                continue
            exact = a["normalized_rgb_sha256"] == b["normalized_rgb_sha256"]
            distance = (int(a["perceptual_hash_64"], 16)
                        ^ int(b["perceptual_hash_64"], 16)).bit_count()
            if exact or distance <= 4:
                matches.append({"train_image_id": a["image_id"],
                                "validation_image_id": b["image_id"],
                                "exact_rgb": exact, "phash_distance": distance})
    return matches


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    for path, expected in INPUTS.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input mismatch: {path.name}")
    candidates = json.loads((SOURCE / "candidates.json").read_bytes())["candidates"]
    receipts = read_jsonl(SOURCE / "private-page-checks.jsonl")
    records = prepare_records(candidates, receipts)
    validation = [row for path in list(INPUTS)[2:] for row in read_jsonl(path)]
    output.mkdir(parents=True)
    results = []
    used = 0
    stop_reason = "completed"
    for record in records:
        if shutil.disk_usage(output).free < RESERVE + MAX_MEDIA_BYTES:
            stop_reason = "disk_reserve"
            break
        if used + MAX_MEDIA_BYTES > BUDGET:
            stop_reason = "media_budget"
            break
        result = download_one(record, split_dir=output / "train")
        results.append(result)
        used += result.get("bytes", 0)
        if len(results) % 25 == 0:
            print(f"checked {len(results)} / {len(records)}; saved bytes {used}", flush=True)
    matches = compare_images(results, validation)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    (output / "private-media-manifest.jsonl").write_text(payload, encoding="utf-8")
    (output / "private-duplicate-warnings.json").write_text(json.dumps(matches, indent=2) + "\n")
    summary = {
        "stop_reason": stop_reason, "eligible_requests": len(records),
        "attempted": len(results), "saved": sum(r.get("status") == "saved" for r in results),
        "bytes_saved": used, "exact_rgb_matches": sum(m["exact_rgb"] for m in matches),
        "phash_distance_at_most_4_pairs": len(matches), "individual_rights_verified": False,
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
        "manifest_text_lf_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_hashes": {str(path): expected for path, expected in INPUTS.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
