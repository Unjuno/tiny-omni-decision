"""Audit DROID raw episode metadata without downloading media or trajectories."""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BUCKET = "gresearch"
PREFIX = "robotics/droid_raw/1.0.1/"
SAMPLE_SEED = "droid-r2-metadata-preflight-20261128-v1"
SAMPLE_PER_LAB_STATUS = 20
PAGE_SIZE = 1000
MAX_WORKERS = 12
FIELDS = "items(name,generation,size,md5Hash),nextPageToken"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sample_rank(name: str, seed: str = SAMPLE_SEED) -> bytes:
    return hashlib.sha256(f"{seed}\0{name}".encode()).digest()


def _classify_path(name: str) -> tuple[str, str] | None:
    parts = name.split("/")
    if len(parts) < 7 or not name.endswith(".json") or "/metadata_" not in name:
        return None
    lab, status = parts[3], parts[4]
    if status not in {"success", "failure"}:
        return None
    return lab, status


def _list_metadata_objects(opener=urllib.request.urlopen) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {
            "prefix": PREFIX,
            "matchGlob": "**/metadata_*.json",
            "maxResults": str(PAGE_SIZE),
            "fields": FIELDS,
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "Codex-readonly-source-screen"}
        )
        with opener(request, timeout=60) as response:
            page = json.load(response)
        objects.extend(page.get("items", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    return objects


def _choose_sample(
    objects: list[dict[str, Any]], per_group: int = SAMPLE_PER_LAB_STATUS
) -> tuple[list[dict[str, Any]], collections.Counter[tuple[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    totals: collections.Counter[tuple[str, str]] = collections.Counter()
    for item in objects:
        group = _classify_path(str(item.get("name", "")))
        if group is None:
            continue
        totals[group] += 1
        groups[group].append(item)
    sample = [
        item
        for group in sorted(groups)
        for item in sorted(groups[group], key=lambda row: _sample_rank(row["name"]))[:per_group]
    ]
    return sample, totals


def _fetch_metadata(
    item: dict[str, Any], opener=urllib.request.urlopen
) -> tuple[dict[str, Any], bytes]:
    name = str(item["name"])
    encoded = urllib.parse.quote(name, safe="")
    params = urllib.parse.urlencode({"alt": "media", "generation": str(item["generation"])})
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{encoded}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-readonly-source-screen"})
    with opener(request, timeout=60) as response:
        data = response.read()
    expected_size = int(item["size"])
    expected_md5 = str(item["md5Hash"])
    if len(data) != expected_size:
        raise ValueError(f"metadata size mismatch: {len(data)} != {expected_size}")
    observed_md5 = base64.b64encode(hashlib.md5(data).digest()).decode("ascii")
    if observed_md5 != expected_md5:
        raise ValueError("metadata MD5 mismatch")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("episode metadata is not a JSON object")
    return parsed, data


def _group_fields(record: dict[str, Any]) -> tuple[bool, bool, bool]:
    building_ok = isinstance(record.get("building"), str) and record["building"] not in {"", "N/A"}
    scene = record.get("scene_id")
    scene_ok = isinstance(scene, int) and scene >= 0
    user_ok = isinstance(record.get("user_id"), str) and bool(record["user_id"])
    return building_ok, scene_ok, user_ok


def _audit_records(
    sample: list[dict[str, Any]], fetched: list[tuple[dict[str, Any], bytes]]
) -> dict[str, Any]:
    counters: collections.Counter[str] = collections.Counter()
    task_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    task_scene_groups: dict[str, dict[str, set[tuple[str, str, int]]]] = collections.defaultdict(
        lambda: {"success": set(), "failure": set()}
    )
    task_users: dict[str, dict[str, set[tuple[str, str]]]] = collections.defaultdict(
        lambda: {"success": set(), "failure": set()}
    )
    content_hashes: list[str] = []
    sample_groups: collections.Counter[tuple[str, str]] = collections.Counter()
    for item, (record, data) in zip(sample, fetched, strict=True):
        group = _classify_path(str(item["name"]))
        if group is None:
            raise ValueError("sample contains an unclassified path")
        lab, path_status = group
        sample_groups[group] += 1
        content_hashes.append(_sha256(data))
        label = record.get("success")
        if type(label) is bool:
            counters[f"explicit_{label}"] += 1
            counters[f"path_flag_match_{path_status}"] += int(label is (path_status == "success"))
            counters[f"path_flag_mismatch_{path_status}"] += int(
                label is not (path_status == "success")
            )
        else:
            counters["missing_or_nonboolean_success"] += 1
        building_ok, scene_ok, user_ok = _group_fields(record)
        counters["building_missing"] += int(not building_ok)
        counters["scene_id_missing"] += int(not scene_ok)
        counters["user_id_missing"] += int(not user_ok)
        task = record.get("current_task")
        counters["empty_task"] += int(not isinstance(task, str) or not task.strip())
        try:
            length = int(record.get("trajectory_length", 0))
        except (TypeError, ValueError):
            length = 0
        counters["positive_trajectory_length"] += int(length > 0)
        if isinstance(record.get("current_task"), str) and record["current_task"].strip():
            task_hash = _sha256(" ".join(record["current_task"].lower().split()).encode("utf-8"))
            task_counts[task_hash][path_status] += 1
            if building_ok and scene_ok:
                task_scene_groups[task_hash][path_status].add(
                    (lab, str(record["building"]), int(record["scene_id"]))
                )
            if user_ok:
                task_users[task_hash][path_status].add((lab, str(record["user_id"])))
    overlap = [
        task_hash
        for task_hash, status_counts in task_counts.items()
        if status_counts["success"] and status_counts["failure"]
    ]
    candidates = sorted(
        overlap,
        key=lambda key: (
            min(task_counts[key]["success"], task_counts[key]["failure"]),
            sum(task_counts[key].values()),
            key,
        ),
        reverse=True,
    )[:20]
    return {
        "sample_record_count": len(sample),
        "sample_records_by_lab_and_path_status": {
            f"{lab}/{status}": count for (lab, status), count in sorted(sample_groups.items())
        },
        "sample_record_integrity": dict(sorted(counters.items())),
        "task_hashes_present_in_both_path_statuses": len(overlap),
        "top_task_hash_candidates": [
            {
                "task_sha256": task_hash,
                "sampled_success_records": task_counts[task_hash]["success"],
                "sampled_failure_records": task_counts[task_hash]["failure"],
                "sampled_success_scene_groups": len(task_scene_groups[task_hash]["success"]),
                "sampled_failure_scene_groups": len(task_scene_groups[task_hash]["failure"]),
                "sampled_success_users": len(task_users[task_hash]["success"]),
                "sampled_failure_users": len(task_users[task_hash]["failure"]),
            }
            for task_hash in candidates
        ],
        "sample_record_content_hashes_sha256": _sha256(
            "\n".join(sorted(content_hashes)).encode("ascii")
        ),
    }


def run_audit(
    output_path: Path,
    *,
    opener=urllib.request.urlopen,
    sample_per_group: int = SAMPLE_PER_LAB_STATUS,
) -> dict[str, Any]:
    objects = _list_metadata_objects(opener=opener)
    sample, object_counts = _choose_sample(objects, sample_per_group)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        fetched = list(executor.map(lambda item: _fetch_metadata(item, opener=opener), sample))
    lab_names = {lab for lab, _status in object_counts}
    result = {
        "status": "droid_metadata_only_preflight",
        "bucket": BUCKET,
        "prefix": PREFIX,
        "sample_seed": SAMPLE_SEED,
        "sample_per_lab_and_path_status": sample_per_group,
        "metadata_object_count": sum(object_counts.values()),
        "metadata_index_inventory_sha256": _sha256(
            "\n".join(
                sorted(
                    f"{item.get('name', '')}\0{item.get('generation', '')}"
                    f"\0{item.get('md5Hash', '')}"
                    for item in objects
                    if _classify_path(str(item.get("name", ""))) is not None
                )
            ).encode()
        ),
        "labs_with_metadata": len(lab_names),
        "labs_with_success_and_failure_metadata": sum(
            object_counts[(lab, "success")] > 0 and object_counts[(lab, "failure")] > 0
            for lab in lab_names
        ),
        "metadata_object_counts_by_lab_and_path_status": {
            f"{lab}/{status}": count for (lab, status), count in sorted(object_counts.items())
        },
        "payload_scope": {
            "episode_metadata_json_fetched": len(sample),
            "video_or_svo_fetched": False,
            "trajectory_hdf5_fetched": False,
            "model_or_gpu_used": False,
            "sealed_or_test_split_accessed": False,
        },
    }
    result.update(_audit_records(sample, fetched))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-per-lab-status", type=int, default=SAMPLE_PER_LAB_STATUS)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.output, sample_per_group=args.sample_per_lab_status), indent=2))


if __name__ == "__main__":
    main()
