"""Select unique-author train image quartets for subsequent rights screening."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.audit_openimages_balanced_quartets import build_quartets
from scripts.audit_openimages_existing_train_labels import TRAIN, TRAIN_SHA
from scripts.audit_openimages_full_train_support import EXPECTED_SHA
from scripts.audit_openimages_symmetric_prefix import ETAG, TARGET_SHA, TARGETS, TOTAL, URL
from scripts.audit_openimages_train_metadata_join import (
    META_ETAG,
    META_SHA,
    META_SIZE,
    META_URL,
    open_rows,
)
from scripts.audit_openimages_train_option_prior import hashed
from scripts.select_openimages_train_candidates import VALIDATION, VALIDATION_SHA, identity
from scripts.train_source_unimodal_cv import digest
from tiny_omni_decision.openimages_options import group_verification_labels

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
OLD_CANDIDATES = ROOT / (
    "openimages-in-scope-train-candidates-20261027-attempt-01/private-attribution.jsonl"
)
OLD_PILOT = ROOT / "openimages-v6-support-audit-20261026/private-train-page-checks.jsonl"
OLD_CANDIDATES_SHA = "e9db942ff0f81162586391f70789945d048749a5570d8686e5b60b8c8b924a46"
OLD_PILOT_SHA = "2b883c146494a62e1200fd06bd2ebf1d6a2eebd8e8248d6e2550c5298a9fd922"
PROTOCOL = Path("docs/pretrained_reuse/R2_OPENIMAGES_QUARTET_CANDIDATE_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_unique(metadata, excluded_ids, excluded_md5, excluded_authors):
    used_md5, used_authors, selected = set(), set(), {}
    for image_id in sorted(metadata, key=lambda value: hashed(f"quartet-unique-v1|17|{value}")):
        row = metadata[image_id]
        author, md5 = identity(row), row["OriginalMD5"]
        if (image_id in excluded_ids or md5 in excluded_md5 or author in excluded_authors
                or md5 in used_md5 or author in used_authors):
            continue
        selected[image_id] = row
        used_md5.add(md5)
        used_authors.add(author)
    return selected


def rank_quartets(quartets, limit=64):
    support = Counter(row["target"] for group in quartets for row in group["rows"])
    def rank(group):
        image_ids = sorted(row["image_id"] for row in group["rows"])
        return (min(support[row["target"]] for row in group["rows"]),
                hashed(f"quartet-screen-v1|17|{'|'.join(image_ids)}"))
    return sorted(quartets, key=rank)[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    pinned = {TRAIN: TRAIN_SHA, VALIDATION: VALIDATION_SHA, TARGETS: TARGET_SHA,
              OLD_CANDIDATES: OLD_CANDIDATES_SHA, OLD_PILOT: OLD_PILOT_SHA}
    if any(digest(path) != expected for path, expected in pinned.items()):
        raise ValueError("pinned source hash mismatch")
    scope_data = json.loads(TARGETS.read_bytes())["selected_target_classes"]
    names = {row["label_id"]: row["display_name"] for row in scope_data}
    scope = set(names)
    prior = read_jsonl(OLD_CANDIDATES) + read_jsonl(OLD_PILOT)
    with VALIDATION.open(encoding="utf-8-sig", newline="") as file:
        validation = list(csv.DictReader(file))
    excluded = prior + validation
    excluded_ids = {row["ImageID"] for row in excluded}
    excluded_ids.update(row["image_id"] for row in json.loads(TRAIN.read_bytes())["examples"])
    excluded_md5 = {row["OriginalMD5"] for row in excluded}
    excluded_authors = {identity(row) for row in excluded}

    def label_candidates(rows):
        found = {}
        for group in group_verification_labels(rows):
            positive = group["positive_labels"] & scope
            negative = group["negative_labels"] & scope
            if positive and len(negative) >= 3:
                found[group["image_id"]] = {"image_id": group["image_id"],
                    "positive_labels": sorted(positive), "negative_labels": sorted(negative)}
        return found

    labels = open_rows(URL, ETAG, TOTAL, EXPECTED_SHA, label_candidates)

    def metadata_candidates(rows):
        found = {}
        required = ("ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
                    "AuthorProfileURL", "Author", "OriginalMD5")
        for row in rows:
            image_id = row.get("ImageID")
            if image_id not in labels:
                continue
            if image_id in found or any(not row.get(key) for key in required):
                raise ValueError("duplicate or incomplete candidate metadata")
            if row["Subset"] != "train" or row["License"].rstrip("/") != (
                "https://creativecommons.org/licenses/by/2.0"
            ):
                raise ValueError("invalid subset or license")
            identity(row)
            found[image_id] = {key: row[key] for key in required} | {
                "Thumbnail300KURL": row.get("Thumbnail300KURL") or None,
                "individual_rights_verified": False,
            }
        return found

    metadata = open_rows(META_URL, META_ETAG, META_SIZE, META_SHA, metadata_candidates)
    selected_metadata = select_unique(metadata, excluded_ids, excluded_md5, excluded_authors)
    candidates = [labels[image_id] for image_id in selected_metadata]
    quartets, option_sets, complete, unused = build_quartets(
        candidates, set(selected_metadata), scope
    )
    screen = rank_quartets(quartets)
    image_ids = [row["image_id"] for group in screen for row in group["rows"]]
    counts = Counter(row["target"] for group in screen for row in group["rows"])
    result = {
        "status": "CANDIDATES_ONLY_NOT_ADMITTED", "unique_author_pool": len(candidates),
        "possible_option_sets": option_sets, "four_target_option_sets": complete,
        "all_disjoint_quartets": len(quartets), "unused_pool_images": unused,
        "screen_quartets": len(screen), "screen_images": len(image_ids),
        "target_counts": {names[k]: counts[k] for k in names},
        "source_hashes": {str(path): expected for path, expected in pinned.items()},
        "labels_sha256": EXPECTED_SHA, "metadata_sha256": META_SHA,
        "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(Path(__file__)),
    }
    out.mkdir(parents=True)
    (out / "quartets.json").write_text(json.dumps(screen, indent=2) + "\n", encoding="utf-8")
    (out / "private-attribution.jsonl").write_text(
        "".join(json.dumps(selected_metadata[x], ensure_ascii=False) + "\n" for x in image_ids),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
