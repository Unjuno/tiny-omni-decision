"""Capture a bounded, development-only iNaturalist bird metadata snapshot."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import requests

TAXA = (6930, 13858, 7089, 3017, 12727, 144455, 4956, 9083)
URL = "https://api.inaturalist.org/v1/observations"
REPO = Path(__file__).resolve().parents[1]


def clean_record(row, class_taxon_id):
    taxon = row.get("taxon") or {}
    actual = taxon.get("id")
    ancestors = str(taxon.get("ancestry") or "").split("/")
    if (actual != class_taxon_id and str(class_taxon_id) not in ancestors):
        return None
    user = (row.get("user") or {}).get("id")
    if (row.get("quality_grade") != "research" or row.get("license_code") not in
            ("cc0", "cc-by") or type(user) is not int or type(row.get("id")) is not int):
        return None
    photos = [p for p in row.get("photos", []) if type(p.get("id")) is int
              and p.get("license_code") in ("cc0", "cc-by")
              and isinstance(p.get("attribution"), str) and p["attribution"].strip()
              and p.get("hidden") is False and str(p.get("url", "")).startswith("https://")]
    if not photos:
        return None
    photo = min(photos, key=lambda p: p["id"])
    observer = hashlib.sha256(f"inat-bird-observer-v1:{user}".encode()).hexdigest()
    split = "train" if int(observer[:16], 16) % 100 < 80 else "validation"
    return {"observation_id": row["id"], "observer_key": observer, "split": split,
            "class_taxon_id": class_taxon_id, "observed_taxon_id": actual,
            "observation_license": row["license_code"], "photo_id": photo["id"],
            "photo_license": photo["license_code"], "attribution": photo["attribution"],
            "photo_url": photo["url"]}


def validate_snapshot(rows):
    ids, photos, observers = set(), set(), {}
    for row in rows:
        if row["observation_id"] in ids:
            raise ValueError("duplicate observation")
        if row["photo_id"] in photos:
            raise ValueError("duplicate photo")
        ids.add(row["observation_id"])
        photos.add(row["photo_id"])
        old = observers.setdefault(row["observer_key"], row["split"])
        if old != row["split"]:
            raise ValueError("observer crosses splits")
    if not rows:
        raise ValueError("empty snapshot")


def collision_summary(rows):
    grouped = defaultdict(list)
    obs = defaultdict(int)
    for row in rows:
        grouped[row["photo_id"]].append(row)
        obs[row["observation_id"]] += 1
    repeated = [group for group in grouped.values() if len(group) > 1]
    return {"duplicate_observation_ids": sum(count > 1 for count in obs.values()),
            "duplicate_photo_ids": len(repeated),
            "rows_in_duplicate_photo_groups": sum(len(group) for group in repeated),
            "photo_groups_cross_split": sum(len({r["split"] for r in g}) > 1 for g in repeated),
            "photo_groups_conflicting_class": sum(
                len({r["class_taxon_id"] for r in g}) > 1 for g in repeated),
            "photo_groups_multiple_observers": sum(
                len({r["observer_key"] for r in g}) > 1 for g in repeated),
            "photo_groups_conflicting_license_or_attribution": sum(
                len({(r["photo_license"], r["attribution"]) for r in g}) > 1 for g in repeated)}


def partition_photo_collisions(rows):
    observations = [r["observation_id"] for r in rows]
    if len(set(observations)) != len(observations):
        raise ValueError("duplicate observation")
    photo_counts = defaultdict(int)
    for row in rows:
        photo_counts[row["photo_id"]] += 1
    return ([row for row in rows if photo_counts[row["photo_id"]] == 1],
            [row for row in rows if photo_counts[row["photo_id"]] > 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new output outside repository required")
    rows, receipts = [], []
    for taxon in TAXA:
        for page in (1, 2):
            params = {"taxon_id": taxon, "quality_grade": "research", "photos": "true",
                      "license": "cc0,cc-by", "photo_license": "cc0,cc-by",
                      "per_page": 100, "page": page, "order_by": "id", "order": "asc"}
            response = requests.get(URL, params=params, timeout=(15, 45))
            print(f"taxon={taxon} page={page} status={response.status_code}", flush=True)
            response.raise_for_status()
            data = response.json()
            batch = data["results"]
            if len(batch) != 100:
                raise ValueError("expected complete page; snapshot not written")
            cleaned = [clean_record(row, taxon) for row in batch]
            if any(row is None for row in cleaned):
                raise ValueError("metadata gate failed; snapshot not written")
            rows.extend(cleaned)
            receipts.append({"taxon_id": taxon, "page": page,
                             "response_sha256": hashlib.sha256(response.content).hexdigest(),
                             "reported_total": data["total_results"]})
            time.sleep(1.1)
    if args.diagnose_only:
        output.mkdir(parents=True)
        result = {"status": "AGGREGATE_DIAGNOSTIC_NO_CORPUS", "count": len(rows),
                  "collisions": collision_summary(rows), "response_receipts": receipts,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    admitted, quarantine = partition_photo_collisions(rows)
    validate_snapshot(admitted)
    output.mkdir(parents=True)
    payload = json.dumps(admitted, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    quarantine_payload = json.dumps(quarantine, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (output / "private-snapshot.json").write_text(payload, encoding="utf-8", newline="\n")
    (output / "private-quarantine.json").write_text(
        quarantine_payload, encoding="utf-8", newline="\n"
    )
    summary = {"status": "DEVELOPMENT_METADATA_NOT_ADMITTED", "raw_count": len(rows),
               "count": len(admitted), "quarantined_rows": len(quarantine),
               "collision_summary": collision_summary(rows),
               "snapshot_sha256": hashlib.sha256(payload.encode()).hexdigest(),
               "quarantine_sha256": hashlib.sha256(quarantine_payload.encode()).hexdigest(),
               "response_receipts": receipts, "taxa": TAXA,
               "split_counts": {s: sum(row["split"] == s for row in admitted)
                                for s in ("train", "validation")},
               "class_split_counts": {str(t): {s: sum(row["split"] == s and
                                      row["class_taxon_id"] == t for row in admitted)
                                      for s in ("train", "validation")} for t in TAXA},
               "quarantine_class_split_counts": {str(t): {
                   s: sum(row["split"] == s and row["class_taxon_id"] == t for row in quarantine)
                   for s in ("train", "validation")} for t in TAXA},
               "limitations": ["nonrandom live API snapshot", "photo rights unverified",
                               "no image bytes", "no deduplication", "no sealed audit"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
