import hashlib
import json
import sys

import scripts.join_inat_photo_metadata as join
from scripts.join_inat_photo_metadata import reconcile_candidate


def candidate():
    return {"observation_id": 1, "photo_id": 7, "photo_license": "cc-by",
            "observer_key": "observer"}


def test_exact_uuid_license_and_photographer_are_required():
    mapping = {"observation_id": 1, "photo_id": 7, "status": "MATCH",
               "observation_uuid": "uuid"}
    metadata = [{"observation_uuid": "uuid", "photographer_key": "a" * 64,
                 "license": "CC-BY", "photo_uuid": "photo-uuid"}]
    result = reconcile_candidate(candidate(), mapping, metadata)
    assert result["status"] == "MATCH" and result["photographer_key"] == "a" * 64
    metadata[0]["license"] = "CC-BY-NC"
    assert "license_mismatch" in reconcile_candidate(candidate(), mapping, metadata)["reasons"]


def test_multiple_or_missing_source_rows_are_quarantined():
    mapping = {"observation_id": 1, "photo_id": 7, "status": "MATCH",
               "observation_uuid": "uuid"}
    row = {"observation_uuid": "uuid", "photographer_key": "x",
           "license": "CC-BY", "photo_uuid": "p"}
    outcome = reconcile_candidate(candidate(), mapping, [row, row])
    assert "source_row_ambiguous" in outcome["reasons"]
    assert "source_row_missing" in reconcile_candidate(candidate(), mapping, [])["reasons"]


def test_full_join_preserves_ambiguous_and_unmatched_rows(tmp_path, monkeypatch):
    def save(path, obj):
        raw = json.dumps(obj).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    snap, uuid_dir, scan = (tmp_path / x for x in ("snapshot", "uuid", "scan"))
    for folder in (snap, uuid_dir, scan):
        folder.mkdir()
    rows = [{"observation_id": i, "photo_id": i, "photo_license": "cc-by",
             "observer_key": f"observer-{i}"} for i in range(1596)]
    ambiguous = [dict(rows[0], observation_id=1600 + i) for i in range(4)]
    pins = {"private-snapshot.json": save(snap / "private-snapshot.json", rows),
            "private-quarantine.json": save(snap / "private-quarantine.json", ambiguous)}
    all_rows = rows + ambiguous
    maps = [{"observation_id": r["observation_id"], "photo_id": r["photo_id"],
             "observation_uuid": f"uuid-{r['observation_id']}", "status": "MATCH"}
            for r in all_rows]
    uuid_sha = save(uuid_dir / "private-uuid-map.json", maps)
    photos = {}
    for row in all_rows[:-1]:
        if row["observation_id"] == 1:
            continue
        photos.setdefault(str(row["photo_id"]), []).append({
            "observation_uuid": f"uuid-{row['observation_id']}",
            "photographer_key": "a" * 64, "license": "CC-BY", "photo_uuid": "p"})
    photo_sha = save(scan / "private-photo-matches.json", photos)
    summary_sha = save(scan / "summary.json", {
        "status": "SOURCE_SCAN_NOT_ADMITTED", "source_etag": join.ETAG,
        "source_bytes": join.SIZE, "snapshot_pins": pins, "matches_sha256": photo_sha})
    monkeypatch.setattr(join, "SNAPSHOT", snap)
    monkeypatch.setattr(join, "UUID_DIR", uuid_dir)
    monkeypatch.setattr(join, "UUID_SHA", uuid_sha)
    monkeypatch.setattr(join, "PINS", pins)
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["join", "--scan-dir", str(scan),
                                      "--scan-summary-sha256", summary_sha,
                                      "--output-dir", str(output)])
    join.main()
    result = json.loads((output / "summary.json").read_bytes())
    assert result["raw_count"] == 1600
    assert result["candidate_count"] == 1595
    assert result["quarantine_count"] == 5
    assert result["join_failure_reasons"]["source_row_missing"] == 2
