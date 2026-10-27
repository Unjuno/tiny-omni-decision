"""Join fixed observation and photo receipts; never admit a quality dataset."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.inat_source_grouping import assign_components
from scripts.scan_inat_photo_metadata import ETAG, PINS, REPO, SIZE, SNAPSHOT

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
UUID_DIR = ROOT / "inat-bird-observation-reconcile-20261028-attempt-02"
UUID_SHA = "616afdb0c97d557e4e596fa8264af872001e7b0827f995fdcdbb23419a4ea6a0"


def reconcile_candidate(row, mapping, metadata):
    result = {"status": "QUARANTINE", "photographer_key": None, "reasons": []}
    if (mapping is None or mapping.get("observation_id") != row["observation_id"]
            or mapping.get("photo_id") != row["photo_id"] or mapping.get("status") != "MATCH"):
        result["reasons"].append("observation_mapping_unavailable")
        return result
    source = [item for item in metadata
              if item.get("observation_uuid") == mapping.get("observation_uuid")]
    if not source:
        result["reasons"].append("source_row_missing")
        return result
    if len(source) != 1:
        result["reasons"].append("source_row_ambiguous")
        return result
    item = source[0]
    if str(item.get("license", "")).lower() != row["photo_license"]:
        result["reasons"].append("license_mismatch")
    owner = item.get("photographer_key")
    if (not isinstance(owner, str) or len(owner) != 64
            or any(c not in "0123456789abcdef" for c in owner)):
        result["reasons"].append("photographer_missing")
    if not item.get("photo_uuid"):
        result["reasons"].append("photo_uuid_missing")
    if not result["reasons"]:
        result.update(status="MATCH", photographer_key=owner)
    return result


def load_pinned(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"hash mismatch for {path.name}")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True, type=Path)
    parser.add_argument("--scan-summary-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(REPO):
        raise ValueError("new output outside repository required")
    source = load_pinned(args.scan_dir / "summary.json", args.scan_summary_sha256)
    if (source.get("status") != "SOURCE_SCAN_NOT_ADMITTED" or source.get("source_etag") != ETAG
            or source.get("source_bytes") != SIZE or source.get("snapshot_pins") != PINS):
        raise ValueError("incomplete or incompatible source scan")
    photos = load_pinned(args.scan_dir / "private-photo-matches.json", source["matches_sha256"])
    mappings = load_pinned(UUID_DIR / "private-uuid-map.json", UUID_SHA)
    mapping_by_id = {r["observation_id"]: r for r in mappings}
    if len(mappings) != 1600 or len(mapping_by_id) != 1600:
        raise ValueError("observation mapping accounting mismatch")
    candidates = load_pinned(SNAPSHOT / "private-snapshot.json", PINS["private-snapshot.json"])
    ambiguous = load_pinned(SNAPSHOT / "private-quarantine.json", PINS["private-quarantine.json"])
    rows = [(r, False) for r in candidates] + [(r, True) for r in ambiguous]
    if len(rows) != 1600 or set(mapping_by_id) != {r["observation_id"] for r, _ in rows}:
        raise ValueError("snapshot and mapping identity mismatch")
    joined, quarantine, reasons = [], [], Counter()
    for row, originally_ambiguous in rows:
        outcome = reconcile_candidate(row, mapping_by_id.get(row["observation_id"]),
                                      photos.get(str(row["photo_id"]), []))
        if outcome["status"] != "MATCH":
            reasons.update(outcome["reasons"])
            quarantine.append(dict(row, quarantine_reasons=outcome["reasons"]))
        else:
            joined.append(dict(row, photographer_key=outcome["photographer_key"],
                               original_photo_quarantine=originally_ambiguous))
    grouped, missing = assign_components(joined)
    if missing:
        raise ValueError("joined identity unexpectedly missing")
    for row in grouped:
        if row["original_photo_quarantine"]:
            quarantine.append(dict(row, quarantine_reasons=["ambiguous_photo"]))
    candidates = [row for row in grouped if not row["original_photo_quarantine"]]
    if len(candidates) + len(quarantine) != 1600:
        raise ValueError("row accounting mismatch")
    groups = {s: {r["component_key"] for r in candidates if r["split"] == s}
              for s in ("train", "validation")}
    if groups["train"] & groups["validation"]:
        raise ValueError("connected component crosses splits")
    component_sizes = Counter(row["component_key"] for row in candidates)
    classes = sorted({row["class_taxon_id"] for row, _ in rows})
    per_class = {}
    for taxon in classes:
        subset = [row for row in candidates if row["class_taxon_id"] == taxon]
        per_class[str(taxon)] = {
            "rows": len(subset),
            "quarantined_rows": sum(row["class_taxon_id"] == taxon for row in quarantine),
            "components": len({row["component_key"] for row in subset}),
            "splits": {s: {
                "rows": sum(row["split"] == s for row in subset),
                "components": len({row["component_key"] for row in subset if row["split"] == s}),
            } for s in groups},
        }
    output.mkdir(parents=True)
    hashes = {}
    for name, payload in (("private-candidates.json", candidates),
                          ("private-quarantine.json", quarantine)):
        raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
        (output / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    summary = {"status": "CONNECTED_METADATA_NOT_ADMITTED", "raw_count": len(rows),
               "candidate_count": len(candidates), "quarantine_count": len(quarantine),
               "join_failure_reasons": dict(reasons),
               "split_counts": {s: sum(r["split"] == s for r in candidates) for s in groups},
               "split_components": {s: len(g) for s, g in groups.items()},
               "component_counts": {"total": len(component_sizes),
                                    "largest_rows": max(component_sizes.values(), default=0)},
               "per_class": per_class,
               "source_scan_summary_sha256": args.scan_summary_sha256,
               "uuid_map_sha256": UUID_SHA, "snapshot_pins": PINS, "output_hashes": hashes,
               "limitations": ["unverified source-version consistency", "no image hash checks",
                               "no ownership verification", "no admitted evaluation"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
