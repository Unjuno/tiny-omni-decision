"""Select and fetch content-disjoint CLEVRER reasoning-control scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.fetch_clevrer_probe import (
    REVISION,
    VALIDATION_BYTES,
    VALIDATION_ETAG,
    VALIDATION_URL,
    RangeFile,
)
from tiny_omni_decision.clevrer import (
    REASONING_TYPES,
    clevrer_binary_choice_content_key,
    expand_binary_reasoning_choices,
)
from tiny_omni_decision.dataset import sha256_file

SEED = 17
SCENE_COUNT = 200
QUESTION_HASHES = {
    "train": "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4",
    "validation": "fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d",
}
CONTENT_ROOT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
ORIGINAL_SAMPLE = CONTENT_ROOT / "clevrer-expanded-seed17-256train-200val"
TRAIN_DECISIONS = CONTENT_ROOT / "clevrer-binary-reasoning-v0" / "train-binary-decisions.jsonl"
TRAIN_QUESTIONS = CONTENT_ROOT / "clevrer-sample-seed17" / "train-questions.json"
VALIDATION_QUESTIONS = CONTENT_ROOT / "clevrer-sample-seed17" / "validation-questions.json"
EXCLUDED_SCENES = Path(
    "manifests/pretrained_reuse/clevrer-heldout-video-excluded-scenes-20261011.json"
)
PREFLIGHT_DIR = CONTENT_ROOT / "clevrer-binary-controls-preflight-v2"
SAMPLE_DIR = CONTENT_ROOT / "clevrer-binary-controls-sample-v2"


class RetryingRangeFile(RangeFile):
    """Range reader with bounded retries for idempotent byte-range GETs."""

    def __init__(self, url: str, expected_bytes: int, expected_etag: str):
        super().__init__(url, expected_bytes, expected_etag)
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        )


def _ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def _canonical_jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _discover_prior_rows() -> tuple[
    set[int], set[tuple[str, tuple[str, ...]]], list[dict[str, str]]
]:
    used_scenes: set[int] = set()
    used_content: set[tuple[str, tuple[str, ...]]] = set()
    input_files: list[dict[str, str]] = []
    for path in sorted(CONTENT_ROOT.glob("clevrer*")):
        if not path.is_dir() or path.name.startswith("clevrer-binary-controls-"):
            continue
        for sample_path in sorted(path.rglob("*.jsonl")):
            rows = _read_jsonl(sample_path)
            for row in rows:
                if "scene_index" in row:
                    used_scenes.add(int(row["scene_index"]))
                if "question" in row and "options" in row:
                    used_content.add(clevrer_binary_choice_content_key(row))
            input_files.append({"path": str(sample_path), "sha256": sha256_file(sample_path)})
    for source_path in (TRAIN_QUESTIONS, VALIDATION_QUESTIONS, TRAIN_DECISIONS, EXCLUDED_SCENES):
        if not source_path.is_file():
            raise FileNotFoundError(f"required exclusion input is missing: {source_path}")
    train_questions = json.loads(TRAIN_QUESTIONS.read_text(encoding="utf-8"))
    train_decisions = _read_jsonl(TRAIN_DECISIONS)
    for row in train_questions:
        used_scenes.add(int(row["scene_index"]))
    for row in train_decisions:
        used_scenes.add(int(row["scene_index"]))
        used_content.add(clevrer_binary_choice_content_key(row))
    exclusion = json.loads(EXCLUDED_SCENES.read_text(encoding="utf-8"))
    if set(exclusion) != {"train", "validation"}:
        raise ValueError("frozen CLEVRER scene exclusion manifest has unexpected keys")
    used_scenes.update(int(scene) for scene in exclusion["train"])
    used_scenes.update(int(scene) for scene in exclusion["validation"])
    for source_path in (TRAIN_QUESTIONS, VALIDATION_QUESTIONS, TRAIN_DECISIONS, EXCLUDED_SCENES):
        input_files.append({"path": str(source_path.resolve()), "sha256": sha256_file(source_path)})
    return used_scenes, used_content, input_files


def select_reasoning_rows(
    raw_rows: list[dict[str, Any]],
    *,
    excluded_scenes: set[int],
    excluded_content: set[tuple[str, tuple[str, ...]]],
    scene_count: int = SCENE_COUNT,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Select whole scenes with all three task types and unique input content."""
    scene_ids = [
        int(row["scene_index"])
        for row in raw_rows
        if int(row["scene_index"]) not in excluded_scenes
    ]
    by_scene = {int(row["scene_index"]): row for row in raw_rows}
    ordered = sorted(
        set(scene_ids),
        key=lambda scene: hashlib.sha256(f"clevrer-controls|{seed}|{scene}".encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    used_content = set(excluded_content)
    for scene in ordered:
        source_row = by_scene[scene]
        media = {
            "scene_index": scene,
            "video_filename": str(source_row["video_filename"]),
            "split": "validation",
        }
        expanded = expand_binary_reasoning_choices([source_row], [media], split="validation")
        unique_rows: list[dict[str, Any]] = []
        seen_in_scene: set[tuple[str, tuple[str, ...]]] = set()
        for row in expanded:
            key = clevrer_binary_choice_content_key(row)
            if key not in used_content and key not in seen_in_scene:
                unique_rows.append(row)
                seen_in_scene.add(key)
        if {row["question_type"] for row in unique_rows} != REASONING_TYPES:
            continue
        selected.extend(unique_rows)
        used_content.update(seen_in_scene)
        if len({int(row["scene_index"]) for row in selected}) == scene_count:
            break
    selected_scenes = {int(row["scene_index"]) for row in selected}
    if len(selected_scenes) != scene_count:
        raise ValueError(
            f"only {len(selected_scenes)} content-disjoint scenes retain all task types; "
            f"need {scene_count}"
        )
    selected.sort(key=lambda row: (row["scene_index"], row["question_id"], row["choice_id"]))
    return selected


def _preflight() -> dict[str, Any]:
    raw_train = TRAIN_QUESTIONS.read_bytes()
    raw_validation = VALIDATION_QUESTIONS.read_bytes()
    hashes = {
        "train": hashlib.sha256(raw_train).hexdigest(),
        "validation": hashlib.sha256(raw_validation).hexdigest(),
    }
    if hashes != QUESTION_HASHES:
        raise ValueError("CLEVRER question JSON hashes differ from the pinned revision")
    exclusion_bytes = EXCLUDED_SCENES.read_bytes()
    excluded = json.loads(exclusion_bytes)
    used_scenes, used_content, input_files = _discover_prior_rows()
    raw_rows = json.loads(raw_validation)
    selected = select_reasoning_rows(
        raw_rows,
        excluded_scenes=used_scenes,
        excluded_content=used_content,
    )
    counts = Counter(row["question_type"] for row in selected)
    scenes = sorted({int(row["scene_index"]) for row in selected})
    return {
        "schema_version": 1,
        "status": "preflight_pass_clevrer_binary_controls",
        "source_revision": REVISION,
        "seed": SEED,
        "requested_validation_scenes": SCENE_COUNT,
        "selected_validation_scenes": scenes,
        "selected_scene_ids_sha256": hashlib.sha256(
            "\n".join(map(str, scenes)).encode()
        ).hexdigest(),
        "validation_choice_count": len(selected),
        "validation_choice_id_order_sha256": _ordered_ids_sha256(selected),
        "selected_rows_sha256": _canonical_jsonl_sha256(selected),
        "question_type_counts": dict(counts),
        "validation_parent_question_count": len({row["parent_question_id"] for row in selected}),
        "prior_scene_count_excluded": len(used_scenes),
        "prior_content_key_count_excluded": len(used_content),
        "scene_overlap_with_prior": 0,
        "exact_content_overlap_with_prior": 0,
        "validation_question_sha256": hashes["validation"],
        "train_question_sha256": hashes["train"],
        "excluded_scene_manifest_sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
        "prior_sample_inputs": input_files,
        "prior_sample_inputs_manifest_sha256": hashlib.sha256(
            json.dumps(input_files, sort_keys=True).encode()
        ).hexdigest(),
        "sealed_audit_loaded": False,
        "feature_cache_extraction_required": True,
        "rows": selected,
        "excluded_scene_manifest": excluded,
    }


def preflight_only() -> None:
    if PREFLIGHT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {PREFLIGHT_DIR}")
    started = time.perf_counter()
    report = _preflight()
    rows = report.pop("rows")
    report.pop("excluded_scene_manifest")
    report["preflight_seconds"] = time.perf_counter() - started
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    _write_jsonl(PREFLIGHT_DIR / "selected-validation.jsonl", rows)
    _write_json(PREFLIGHT_DIR / "preflight.json", report)
    print(json.dumps(report, indent=2))


def fetch_sample(*, resume: bool = False) -> None:
    if SAMPLE_DIR.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite sample directory: {SAMPLE_DIR}")
    if resume and not SAMPLE_DIR.is_dir():
        raise FileNotFoundError(f"no partial sample exists to resume: {SAMPLE_DIR}")
    preflight_path = PREFLIGHT_DIR / "preflight.json"
    selected_path = PREFLIGHT_DIR / "selected-validation.jsonl"
    if not preflight_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError("run --preflight-only and inspect its report first")
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    for item in report["prior_sample_inputs"]:
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise ValueError(f"frozen preflight input changed: {source}")
    rows = _read_jsonl(selected_path)
    if _ordered_ids_sha256(rows) != report["validation_choice_id_order_sha256"]:
        raise ValueError("selected validation rows no longer match preflight ID-order hash")
    if sha256_file(selected_path) != report["selected_rows_sha256"]:
        raise ValueError("selected validation rows no longer match preflight content hash")

    source_train_rows = _read_jsonl(TRAIN_DECISIONS)
    source_train = next(iter({row["scene_index"]: row for row in source_train_rows}.values()))
    source_media = ORIGINAL_SAMPLE / source_train["media_path"]
    if sha256_file(source_media) != source_train["media_sha256"]:
        raise ValueError("training-split feature-extractor witness media hash mismatch")

    started = time.perf_counter()
    range_file = RetryingRangeFile(VALIDATION_URL, VALIDATION_BYTES, VALIDATION_ETAG)
    media_by_scene: dict[int, dict[str, Any]] = {}
    for row in rows:
        media_by_scene.setdefault(int(row["scene_index"]), row)
    expected_media_names = {str(row["video_filename"]) for row in media_by_scene.values()}
    SAMPLE_DIR.mkdir(parents=True, exist_ok=resume)
    train_media_dir = SAMPLE_DIR / "media" / "train"
    validation_media_dir = SAMPLE_DIR / "media" / "validation"
    train_media_dir.mkdir(parents=True, exist_ok=resume)
    validation_media_dir.mkdir(parents=True, exist_ok=resume)
    existing_media_names = {path.name for path in validation_media_dir.iterdir() if path.is_file()}
    if existing_media_names - expected_media_names:
        raise ValueError("partial sample contains unexpected media files")
    for manifest_name in ("train.jsonl", "validation.jsonl", "fetch-report.json"):
        if (SAMPLE_DIR / manifest_name).exists():
            raise FileExistsError(
                f"refusing to overwrite partially finalized output: {manifest_name}"
            )
    witness_path = train_media_dir / source_media.name
    if not witness_path.is_file():
        shutil.copyfile(source_media, witness_path)
    if sha256_file(witness_path) != source_train["media_sha256"]:
        raise ValueError("copied training-split witness hash mismatch")
    witness = {
        **source_train,
        "split": "train",
        "media_path": witness_path.relative_to(SAMPLE_DIR).as_posix(),
    }

    reused_media_count = 0
    downloaded_media_count = 0
    import zipfile

    with zipfile.ZipFile(range_file) as archive:
        names_by_basename: dict[str, list[str]] = {}
        for name in archive.namelist():
            if name.lower().endswith(".mp4"):
                names_by_basename.setdefault(Path(name).name, []).append(name)
        for scene, row in sorted(media_by_scene.items()):
            matches = names_by_basename.get(str(row["video_filename"]), [])
            if len(matches) != 1:
                raise ValueError(f"expected one video for new scene {scene}, found {len(matches)}")
            target = validation_media_dir / str(row["video_filename"])
            info = archive.getinfo(matches[0])
            if target.is_file():
                media = target.read_bytes()
                if len(media) != info.file_size or zlib.crc32(media) & 0xFFFFFFFF != info.CRC:
                    raise ValueError(f"existing media failed archive CRC/size validation: {target}")
                reused_media_count += 1
            else:
                media = archive.read(info)
                target.write_bytes(media)
                downloaded_media_count += 1
            if len(media) < 16 or b"ftyp" not in media[:16]:
                raise ValueError(f"invalid MP4 media for new CLEVRER scene {scene}")
            digest = hashlib.sha256(media).hexdigest()
            for choice_row in rows:
                if int(choice_row["scene_index"]) == scene:
                    choice_row.update(
                        {
                            "source": f"MIT-IBM/CLEVRER@{REVISION}",
                            "scene_group_id": f"clevrer:validation:{scene}",
                            "media_path": target.relative_to(SAMPLE_DIR).as_posix(),
                            "media_bytes": len(media),
                            "media_sha256": digest,
                            "archive_member": matches[0],
                        }
                    )
    train_scenes = {int(witness["scene_index"])}
    val_scenes = {int(row["scene_index"]) for row in rows}
    val_media = {str(row["media_sha256"]) for row in rows}
    if train_scenes & val_scenes or witness["media_sha256"] in val_media:
        raise ValueError("feature extraction witness overlaps the new validation split")
    _write_jsonl(SAMPLE_DIR / "train.jsonl", [witness])
    _write_jsonl(SAMPLE_DIR / "validation.jsonl", rows)
    fetch_report = {
        "status": "complete_clevrer_binary_control_sample",
        "revision": REVISION,
        "seed": SEED,
        "source_questions_sha256": QUESTION_HASHES,
        "preflight_sha256": sha256_file(preflight_path),
        "selected_validation_id_order_sha256": _ordered_ids_sha256(rows),
        "train_feature_extractor_witness_only": {
            "id": witness["id"],
            "scene_index": witness["scene_index"],
            "media_sha256": witness["media_sha256"],
            "excluded_from_readout_train_and_evaluation": True,
        },
        "train_witness_rows": 1,
        "validation_binary_choice_rows": len(rows),
        "validation_scenes": len(val_scenes),
        "validation_media_hashes": sorted(val_media),
        "validation_media_reused_from_partial_run": reused_media_count,
        "validation_media_downloaded_this_attempt": downloaded_media_count,
        "retry_policy": "up to 5 retries for transient idempotent HTTP range GET failures",
        "validation_scene_overlap_with_prior": 0,
        "validation_content_overlap_with_prior": 0,
        "train_witness_scene_overlap": 0,
        "validation_video_archive": {
            "url": VALIDATION_URL,
            "size_bytes": VALIDATION_BYTES,
            "etag": VALIDATION_ETAG,
            "range_bytes_read": range_file.bytes_read,
            "full_archive_sha256_verified": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_audit_loaded": False,
    }
    _write_json(SAMPLE_DIR / "fetch-report.json", fetch_report)
    print(json.dumps(fetch_report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--resume-fetch", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight_only()
    else:
        fetch_sample(resume=args.resume_fetch)


if __name__ == "__main__":
    main()
