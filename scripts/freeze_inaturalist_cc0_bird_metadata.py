"""Freeze a privacy-minimized iNaturalist CC0 metadata cohort for a narrow image probe.

This script writes only train/validation manifests. It does not download images or
materialize audit IDs. Keep its output outside Git because it contains source IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "tiny-omni-decision-research-preflight/1.0"
PER_PAGE = 200
SPECIES = (
    ("Mallard", 6930),
    ("House Sparrow", 13858),
    ("Canada Goose", 7089),
    ("Rock Pigeon", 3017),
    ("American Robin", 12727),
    ("Great Egret", 144455),
    ("Great Blue Heron", 4956),
    ("Northern Cardinal", 9083),
)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def observer_group_hash(observer_id: str, seed: int) -> str:
    return _digest(f"seed{seed}|{observer_id}").hex()


def observer_split(observer_id: str, seed: int, train_percent: int) -> str:
    bucket = int.from_bytes(_digest(f"seed{seed}|{observer_id}")[:4], "little") % 100
    return "train" if bucket < train_percent else "validation"


def _photo_rank(photo_id: str, seed: int) -> str:
    return _digest(f"seed{seed}|photo|{photo_id}").hex()


def _record_rank(taxon_id: int, observer_hash: str, observation_id: str, seed: int) -> str:
    return _digest(f"seed{seed}|row|{taxon_id}|{observer_hash}|{observation_id}").hex()


def build_manifests(
    observations_by_taxon: list[tuple[str, int, dict]],
    *,
    seed: int = 17,
    train_percent: int = 70,
) -> tuple[dict[str, list[dict]], dict]:
    """Filter rights, remove exact identity conflicts, and keep one row per observer/class."""
    if not 1 <= train_percent <= 99:
        raise ValueError("train_percent must be between 1 and 99")

    photo_taxa: dict[str, set[int]] = defaultdict(set)
    observation_taxa: dict[str, set[int]] = defaultdict(set)
    eligible: list[tuple[str, int, dict, str, list[dict]]] = []
    counts = {
        "returned_observations": 0,
        "observation_license_rejected": 0,
        "photo_objects_non_cc0_or_missing": 0,
        "observations_without_eligible_photo": 0,
    }

    for label, requested_taxon_id, observation in observations_by_taxon:
        counts["returned_observations"] += 1
        observation_id = str(observation.get("id", ""))
        observer_id = str((observation.get("user") or {}).get("id", ""))
        if not observation_id or not observer_id:
            raise ValueError("observation ID and observer ID are required for integrity checks")
        observation_taxa[observation_id].add(requested_taxon_id)
        observation_licensed = str(observation.get("license_code", "")).casefold() == "cc0"
        if not observation_licensed:
            counts["observation_license_rejected"] += 1

        photos = []
        for photo in observation.get("photos") or []:
            photo_id = str(photo.get("id", ""))
            if photo_id:
                photo_taxa[photo_id].add(requested_taxon_id)
            url = str(photo.get("url", ""))
            photo_licensed = str(photo.get("license_code", "")).casefold() == "cc0"
            if not photo_id or not photo_licensed or not url.startswith("https://"):
                counts["photo_objects_non_cc0_or_missing"] += 1
                continue
            if observation_licensed:
                photos.append({"photo_id": photo_id, "photo_url": url})
        if not observation_licensed:
            continue
        if not photos:
            counts["observations_without_eligible_photo"] += 1
            continue
        eligible.append((label, requested_taxon_id, observation, observer_id, photos))

    cross_taxon_observation_ids = {
        observation_id for observation_id, taxa in observation_taxa.items() if len(taxa) > 1
    }
    cross_taxon_photo_ids = {
        photo_id for photo_id, taxa in photo_taxa.items() if len(taxa) > 1
    }

    candidates = []
    for label, taxon_id, observation, observer_id, photos in eligible:
        observation_id = str(observation["id"])
        if observation_id in cross_taxon_observation_ids:
            continue
        available_photos = [
            photo for photo in photos if photo["photo_id"] not in cross_taxon_photo_ids
        ]
        if not available_photos:
            continue
        chosen_photo = min(
            available_photos,
            key=lambda photo: _photo_rank(photo["photo_id"], seed),
        )
        group_hash = observer_group_hash(observer_id, seed)
        candidates.append(
            {
                "source": "inaturalist",
                "observation_id": observation_id,
                "taxon_id": taxon_id,
                "label": label,
                "observation_license": "cc0",
                "photo_id": chosen_photo["photo_id"],
                "photo_license": "cc0",
                "photo_url": chosen_photo["photo_url"],
                "observer_group_sha256": group_hash,
                "split": observer_split(observer_id, seed, train_percent),
                "_rank": _record_rank(taxon_id, group_hash, observation_id, seed),
            }
        )

    # If the same photo was attached to multiple observations in one taxon, keep one.
    candidates.sort(key=lambda row: row["_rank"])
    unique_photo_rows = []
    seen_photo_ids = set()
    for row in candidates:
        if row["photo_id"] in seen_photo_ids:
            continue
        seen_photo_ids.add(row["photo_id"])
        unique_photo_rows.append(row)

    # A prolific observer contributes at most one image to each class.
    best_by_group_class = {}
    for row in unique_photo_rows:
        key = (row["observer_group_sha256"], row["taxon_id"])
        previous = best_by_group_class.get(key)
        if previous is None or row["_rank"] < previous["_rank"]:
            best_by_group_class[key] = row

    manifests = {"train": [], "validation": []}
    for row in best_by_group_class.values():
        split = row["split"]
        manifests[split].append({key: value for key, value in row.items() if key != "_rank"})
    for split in manifests:
        manifests[split].sort(key=lambda row: (row["taxon_id"], row["observation_id"]))

    group_splits: dict[str, set[str]] = defaultdict(set)
    observation_splits: dict[str, set[str]] = defaultdict(set)
    photo_splits: dict[str, set[str]] = defaultdict(set)
    per_class = defaultdict(lambda: {"train": 0, "validation": 0})
    for split, rows in manifests.items():
        for row in rows:
            group_splits[row["observer_group_sha256"]].add(split)
            observation_splits[row["observation_id"]].add(split)
            photo_splits[row["photo_id"]].add(split)
            per_class[f"{row['label']} ({row['taxon_id']})"][split] += 1
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise AssertionError("observer group crosses train/validation")
    if any(len(splits) != 1 for splits in observation_splits.values()):
        raise AssertionError("observation ID crosses train/validation")
    if any(len(splits) != 1 for splits in photo_splits.values()):
        raise AssertionError("photo ID crosses train/validation")

    counts.update(
        {
            "cross_taxon_observation_ids": len(cross_taxon_observation_ids),
            "cross_taxon_photo_ids": len(cross_taxon_photo_ids),
            "selected_observer_class_groups": len(best_by_group_class),
            "train_records": len(manifests["train"]),
            "validation_records": len(manifests["validation"]),
            "per_class": dict(per_class),
            "sealed_audit_materialized": False,
        }
    )
    return manifests, counts


def _fetch_page(taxon_id: int, page: int, limit: int) -> tuple[int, list[dict]]:
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "license": "CC0",
        "photo_license": "CC0",
        "per_page": PER_PAGE,
        "page": page,
        "order_by": "date_added",
        "order": "desc",
    }
    request = Request(
        f"{API}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    expected = min(PER_PAGE, max(0, limit - (page - 1) * PER_PAGE))
    rows = payload.get("results") or []
    if len(rows) != expected:
        raise RuntimeError(
            f"taxon {taxon_id} page {page}: expected {expected} rows, got {len(rows)}"
        )
    return int(payload.get("total_results", 0)), rows


def fetch_observations(limit_per_taxon: int) -> tuple[list[tuple[str, int, dict]], dict]:
    if limit_per_taxon < 1 or limit_per_taxon > 2000 or limit_per_taxon % PER_PAGE:
        raise ValueError("limit_per_taxon must be a positive multiple of 200, up to 2000")
    pages = limit_per_taxon // PER_PAGE
    all_rows = []
    filtered_totals = {}
    last_request = 0.0
    for label, taxon_id in SPECIES:
        for page in range(1, pages + 1):
            elapsed = time.monotonic() - last_request
            if last_request and elapsed < 1.05:
                time.sleep(1.05 - elapsed)
            total, rows = _fetch_page(taxon_id, page, limit_per_taxon)
            last_request = time.monotonic()
            filtered_totals[str(taxon_id)] = total
            all_rows.extend((label, taxon_id, row) for row in rows)
            print(
                f"fetched {label} page {page}/{pages}: {len(rows)} observations",
                flush=True,
            )
    return all_rows, filtered_totals


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def freeze(output_dir: Path, *, limit_per_taxon: int, seed: int, train_percent: int) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    source_rows, live_totals = fetch_observations(limit_per_taxon)
    manifests, counts = build_manifests(
        source_rows,
        seed=seed,
        train_percent=train_percent,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    hashes = {
        split: _write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in manifests.items()
    }
    metadata = {
        "schema_version": "inat-cc0-bird-metadata-v1",
        "source": "iNaturalist public API; live response, not a dated dataset snapshot",
        "api_endpoint": API,
        "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "query": {
            "quality_grade": "research",
            "photos": True,
            "observation_license": "CC0",
            "photo_license": "CC0",
            "order_by": "date_added",
            "order": "desc",
            "limit_per_taxon": limit_per_taxon,
            "species": [{"label": label, "taxon_id": taxon_id} for label, taxon_id in SPECIES],
            "live_filtered_totals": live_totals,
        },
        "split": {
            "seed": seed,
            "train_percent": train_percent,
            "validation_percent": 100 - train_percent,
            "group": "observer SHA-256; first four digest bytes little-endian modulo 100",
            "max_observations_per_observer_per_taxon": 1,
            "sealed_audit_materialized": False,
        },
        "privacy": {
            "raw_api_rows_retained": False,
            "observer_ids_retained": False,
            "observer_names_retained": False,
            "coordinates_retained": False,
            "descriptions_retained": False,
        },
        "media_downloaded": False,
        "counts": counts,
        "sha256": hashes,
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    (output_dir / "metadata.json").write_bytes(metadata_bytes)
    metadata["sha256"]["metadata.json"] = hashlib.sha256(metadata_bytes).hexdigest()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-taxon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-percent", type=int, default=70)
    args = parser.parse_args()
    result = freeze(
        args.output_dir,
        limit_per_taxon=args.limit_per_taxon,
        seed=args.seed,
        train_percent=args.train_percent,
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
