"""Select source-declared train candidates for subsequent rights screening."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.audit_openimages_existing_train_labels import TRAIN, TRAIN_SHA
from scripts.audit_openimages_flickr_rights import _flickr_profile_identity
from scripts.audit_openimages_full_train_support import EXPECTED_SHA
from scripts.audit_openimages_symmetric_prefix import ETAG, TARGET_SHA, TARGETS, TOTAL, URL
from scripts.audit_openimages_train_metadata_join import (
    META_ETAG,
    META_SHA,
    META_SIZE,
    META_URL,
    open_rows,
)
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

VALIDATION = Path("C:/CodexArtifacts/pretrained-reuse-r2") / (
    "openimages-freshval-20261026-attempt-01/source/validation-images.csv"
)
VALIDATION_SHA = "ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17"
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_TRAIN_CANDIDATE_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def identity(row):
    profile = _flickr_profile_identity(row.get("AuthorProfileURL"))
    if profile is None:
        raise ValueError("unrecognized Flickr author identity")
    return profile


def select_candidates(eligible, metadata, excluded_ids, val_ids, val_md5, val_authors, targets):
    ranked = []
    for image_id, labels in eligible.items():
        row = metadata.get(image_id)
        if row is None or image_id in excluded_ids or image_id in val_ids:
            continue
        author = identity(row)
        if row["OriginalMD5"] in val_md5 or author in val_authors:
            continue
        for target in labels["positive"]:
            key = f"openimages-in-scope-candidate-v1|17|{image_id}|{target}"
            ranked.append((hashlib.sha256(key.encode()).hexdigest(), image_id, target, author))
    used_ids, used_md5, used_authors = set(), set(), set()
    counts = Counter()
    selected = []
    for _, image_id, target, author in sorted(ranked):
        row = metadata[image_id]
        md5 = row["OriginalMD5"]
        if (counts[target] >= 30 or image_id in used_ids or md5 in used_md5
                or author in used_authors):
            continue
        selected.append({"image_id": image_id, "target_label": target,
                         "positive_labels": sorted(eligible[image_id]["positive"]),
                         "negative_labels": sorted(eligible[image_id]["negative"])})
        counts[target] += 1
        used_ids.add(image_id)
        used_md5.add(md5)
        used_authors.add(author)
    return selected, {target: counts[target] for target in targets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError("output must be a new directory outside repository")
    if (digest(TRAIN) != TRAIN_SHA or digest(VALIDATION) != VALIDATION_SHA
            or digest(TARGETS) != TARGET_SHA):
        raise ValueError("pinned input hash mismatch")
    scope_data = json.loads(TARGETS.read_bytes())["selected_target_classes"]
    names = {item["label_id"]: item["display_name"] for item in scope_data}
    scope = set(names)
    old_ids = {row["image_id"] for row in json.loads(TRAIN.read_bytes())["examples"]}
    with VALIDATION.open(encoding="utf-8-sig", newline="") as file:
        val = list(csv.DictReader(file))
    val_ids = {row["ImageID"] for row in val}
    val_md5 = {row["OriginalMD5"] for row in val}
    val_authors = {identity(row) for row in val}

    def label_candidates(rows):
        found = {}
        for group in group_verification_labels(rows):
            positive = group["positive_labels"] & scope
            negative = group["negative_labels"] & scope
            if positive and len(negative) >= 3:
                found[group["image_id"]] = {"positive": positive, "negative": negative}
        return found

    eligible = open_rows(URL, ETAG, TOTAL, EXPECTED_SHA, label_candidates)

    def metadata_candidates(rows):
        found = {}
        required = ("ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
                    "AuthorProfileURL", "Author", "OriginalMD5")
        for row in rows:
            image_id = row.get("ImageID")
            if image_id not in eligible:
                continue
            if image_id in found or any(not row.get(key) for key in required):
                raise ValueError("duplicate or incomplete candidate metadata")
            if row["Subset"] != "train" or row["License"].rstrip("/") != (
                "https://creativecommons.org/licenses/by/2.0"
            ):
                raise ValueError("unexpected candidate subset or license")
            identity(row)
            found[image_id] = {key: row[key] for key in required} | {
                "Thumbnail300KURL": row.get("Thumbnail300KURL") or None,
                "individual_rights_verified": False,
            }
        return found

    metadata = open_rows(META_URL, META_ETAG, META_SIZE, META_SHA, metadata_candidates)
    selected, counts = select_candidates(eligible, metadata, old_ids, val_ids, val_md5,
                                         val_authors, sorted(scope))
    receipts = [metadata[row["image_id"]] for row in selected]
    result = {
        "candidate_count": len(selected), "by_target": {names[k]: counts[k] for k in sorted(names)},
        "label_sha256": EXPECTED_SHA, "metadata_sha256": META_SHA,
        "validation_metadata_sha256": VALIDATION_SHA, "prior_train_sha256": TRAIN_SHA,
        "scope_sha256": TARGET_SHA, "protocol_sha256": digest(PROTOCOL),
        "script_sha256": digest(Path(__file__)), "candidates": selected,
        "status": "CANDIDATES_ONLY_NOT_ADMITTED",
    }
    out.mkdir(parents=True)
    (out / "candidates.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "private-attribution.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in receipts), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
