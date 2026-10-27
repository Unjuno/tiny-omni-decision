"""Recheck fixed bird observations and retain their UUID join keys privately."""

import argparse
import datetime
import hashlib
import json
import time
import uuid as uuid_lib
from pathlib import Path

import requests

from scripts.scan_inat_photo_metadata import PINS, REPO, SNAPSHOT


def observer_key(user_id):
    return hashlib.sha256(f"inat-bird-observer-v1:{user_id}".encode()).hexdigest()


def reconcile_row(frozen, current, key_fn=observer_key):
    result = {"observation_id": frozen["observation_id"], "photo_id": frozen["photo_id"],
              "observation_uuid": None, "status": "QUARANTINE", "reasons": []}
    if current is None:
        result["reasons"].append("missing_observation")
        return result
    if current.get("id") != frozen["observation_id"]:
        raise ValueError("response observation ID mismatch")
    uid = (current.get("user") or {}).get("id")
    if type(uid) is not int or key_fn(uid) != frozen["observer_key"]:
        result["reasons"].append("observer_changed")
    if current.get("quality_grade") != "research":
        result["reasons"].append("quality_changed")
    if current.get("license_code") != frozen["observation_license"]:
        result["reasons"].append("observation_license_changed")
    if (current.get("taxon") or {}).get("id") != frozen["observed_taxon_id"]:
        result["reasons"].append("taxon_changed")
    photo = [p for p in current.get("photos", []) if p.get("id") == frozen["photo_id"]]
    if len(photo) != 1:
        result["reasons"].append("photo_missing_or_duplicate")
    else:
        if photo[0].get("license_code") != frozen["photo_license"]:
            result["reasons"].append("photo_license_changed")
        if (photo[0].get("attribution") != frozen["attribution"]
                or photo[0].get("hidden") is not False):
            result["reasons"].append("photo_attribution_or_visibility_changed")
    uuid = current.get("uuid")
    if not isinstance(uuid, str) or not uuid.strip():
        result["reasons"].append("missing_uuid")
    else:
        try:
            parsed_uuid = uuid_lib.UUID(uuid)
        except ValueError:
            result["reasons"].append("invalid_uuid")
        else:
            if str(parsed_uuid) != uuid.lower():
                result["reasons"].append("invalid_uuid")
            else:
                result["observation_uuid"] = str(parsed_uuid)
    if not result["reasons"]:
        result["status"] = "MATCH"
    return result


def validate_batch(frozen, batch):
    digest = batch.get("response_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid response digest")
    results = batch.get("results", [])
    expected = [(r["observation_id"], r["photo_id"]) for r in frozen]
    actual = [(r["observation_id"], r["photo_id"]) for r in results]
    if actual != expected:
        raise ValueError("batch identity or order mismatch")
    return results


def validate_unique_uuids(rows):
    values = [row["observation_uuid"] for row in rows if row["observation_uuid"] is not None]
    if len(set(values)) != len(values):
        raise ValueError("duplicate UUID across observations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.is_relative_to(REPO) or (output.exists() != args.resume):
        raise ValueError("new output required, or explicit resume of existing output")
    rows = []
    for name, expected in PINS.items():
        raw = (SNAPSHOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("snapshot hash mismatch")
        rows.extend(json.loads(raw))
    ids = [row["observation_id"] for row in rows]
    if len(rows) != 1600 or len(set(ids)) != len(ids):
        raise ValueError("unexpected snapshot identity accounting")
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path = output / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("script_sha256") != script_sha or manifest.get("snapshot_pins") != PINS
                or (output / "summary.json").exists()):
            raise ValueError("resume provenance mismatch or run already complete")
    else:
        output.mkdir(parents=True)
        manifest = {"script_sha256": script_sha, "snapshot_pins": PINS, "batch_hashes": {}}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    results, receipts = [], []
    for offset in range(0, len(rows), 30):
        batch = rows[offset:offset + 30]
        batch_path = output / f"batch-{offset:04d}.json"
        if batch_path.name in manifest["batch_hashes"]:
            raw_batch = batch_path.read_bytes()
            if hashlib.sha256(raw_batch).hexdigest() != manifest["batch_hashes"][batch_path.name]:
                raise ValueError("saved batch hash mismatch")
            stored = json.loads(raw_batch)
            results.extend(validate_batch(batch, stored))
            receipts.append(stored["response_sha256"])
            continue
        if batch_path.exists():
            raise ValueError("unregistered batch found; refusing resume")
        url = "https://api.inaturalist.org/v1/observations/" + ",".join(
            str(row["observation_id"]) for row in batch
        )
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=(15, 45))
                break
            except (requests.exceptions.SSLError, requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        print(f"checked {offset}/{len(rows)} status={response.status_code}", flush=True)
        response.raise_for_status()
        source = response.json()["results"]
        by_id = {r["id"]: r for r in source}
        if len(by_id) != len(source) or not set(by_id) <= {r["observation_id"] for r in batch}:
            raise ValueError("unexpected or duplicate response IDs")
        stored = {"response_sha256": hashlib.sha256(response.content).hexdigest(),
                  "captured_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                  "results": [reconcile_row(row, by_id.get(row["observation_id"]))
                              for row in batch]}
        checked = validate_batch(batch, stored)
        payload = (json.dumps(stored, sort_keys=True) + "\n").encode()
        with batch_path.open("xb") as file:
            file.write(payload)
        manifest["batch_hashes"][batch_path.name] = hashlib.sha256(payload).hexdigest()
        temp = output / "manifest.next.json"
        with temp.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        temp.replace(manifest_path)
        results.extend(checked)
        receipts.append(stored["response_sha256"])
        time.sleep(1.1)
    validate_unique_uuids(results)
    payload = json.dumps(results, sort_keys=True, indent=2) + "\n"
    (output / "private-uuid-map.json").write_text(payload, encoding="utf-8", newline="\n")
    reasons = {}
    for row in results:
        for reason in row["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    summary = {"status": "CURRENT_API_RECONCILIATION_NOT_ADMITTED", "count": len(results),
               "matched": sum(r["status"] == "MATCH" for r in results), "reasons": reasons,
               "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
               "response_sha256": receipts, "snapshot_pins": PINS,
               "limitations": ["current API differs in time from monthly photo export",
                               "no photographer IDs verified", "no media or ownership verified",
                               "no model fitting or evaluation"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
