"""Acquire fixed page-matched train quartets with conservative disk limits."""

import argparse
import json
import shutil
from pathlib import Path

from scripts.acquire_openimages_train_candidate_media import compare_images, prepare_records
from scripts.download_openimages_pilot_media import MAX_MEDIA_BYTES, download_one
from scripts.train_source_unimodal_cv import digest

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SOURCE = ROOT / "openimages-quartet-candidates-20261027-attempt-01"
OLD = ROOT / "openimages-in-scope-train-media-20261027-attempt-01/private-media-manifest.jsonl"
VAL1 = ROOT / "openimages-v6-media-cache-20261026-attempt-01/validation-media-manifest.jsonl"
VAL2 = ROOT / "openimages-freshval-20261026-attempt-01/media-cache/validation-media-manifest.jsonl"
INPUTS = {
    SOURCE / "quartets.json": "c97478be225c79c9ee14928e2566dd8511a40f84c1375a96c624b78792c33ab5",
    SOURCE / "private-page-checks.jsonl": (
        "a937576c0951de6e006783f07622af26b4bbbfa4a40e007efd68df3a91ebcfb9"
    ),
    OLD: "7ff4b8bda635812ed404adf06b279e55a96838f18407e39ab80e9e659f59a9a7",
    VAL1: "81b225f38d311ffe94003b2419100b33f88ad484d7f0f4ab1f3f35877e048560",
    VAL2: "ab531846397629d454eebf5e36b1d69a5f5fc436a2d67861aac8e55ecb9e40b0",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_QUARTET_MEDIA_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]
RESERVE = 1024**3
BUDGET = 64 * 1024**2


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def page_matched_quartets(groups, receipts):
    indexed = {row["ImageID"]: row for row in receipts}
    if len(indexed) != len(receipts):
        raise ValueError("duplicate page receipts")
    seen, result = set(), []
    for group in groups:
        options, rows = group["options"], group["rows"]
        if len(options) != 4 or len(set(options)) != 4 or len(rows) != 4:
            raise ValueError("invalid quartet")
        if {row["target"] for row in rows} != set(options):
            raise ValueError("quartet targets are not symmetric")
        ids = [row["image_id"] for row in rows]
        if len(set(ids)) != 4 or seen.intersection(ids) or any(i not in indexed for i in ids):
            raise ValueError("missing or duplicate quartet image")
        seen.update(ids)
        if all(indexed[i].get("page_license_and_attribution_match") is True for i in ids):
            result.append(group)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != expected for path, expected in INPUTS.items()):
        raise ValueError("pinned input mismatch")
    groups = json.loads((SOURCE / "quartets.json").read_bytes())
    receipts = read_jsonl(SOURCE / "private-page-checks.jsonl")
    matched = page_matched_quartets(groups, receipts)
    candidates = [{"image_id": row["image_id"], "target_label": row["target"]}
                  for group in matched for row in group["rows"]]
    ids = {row["image_id"] for row in candidates}
    records = prepare_records(candidates, [row for row in receipts if row["ImageID"] in ids])
    if len(matched) != 12 or len(records) != 48:
        raise ValueError("unexpected fixed page-matched group count")
    old = read_jsonl(OLD)
    validation = read_jsonl(VAL1) + read_jsonl(VAL2)
    out.mkdir(parents=True)
    results, used, stop = [], 0, "completed"
    for record in records:
        if shutil.disk_usage(out).free < RESERVE + MAX_MEDIA_BYTES:
            stop = "disk_reserve"
            break
        if used + MAX_MEDIA_BYTES > BUDGET:
            stop = "media_budget"
            break
        row = download_one(record, split_dir=out / "train")
        results.append(row)
        used += row.get("bytes", 0)
        if len(results) % 10 == 0:
            print(f"processed {len(results)} / {len(records)}", flush=True)
    saved = {row["image_id"] for row in results if row.get("status") == "saved"}
    complete = sum(all(row["image_id"] in saved for row in group["rows"]) for group in matched)
    old_matches = compare_images(results, old)
    val_matches = compare_images(results, validation)
    self_matches = [m for m in compare_images(results, results)
                    if m["train_image_id"] < m["validation_image_id"]]
    report = {"stop_reason": stop, "fixed_groups": len(matched), "complete_groups": complete,
              "attempted": len(results), "saved": len(saved), "bytes_saved": used,
              "old_candidate_near_pairs": len(old_matches),
              "validation_near_pairs": len(val_matches),
              "new_internal_near_pairs": len(self_matches),
              "individual_rights_verified": False,
              "source_hashes": {str(k): v for k, v in INPUTS.items()},
              "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__))}
    (out / "private-media-manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8")
    (out / "private-duplicate-warnings.json").write_text(
        json.dumps({"old": old_matches, "validation": val_matches, "internal": self_matches},
                   indent=2) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
