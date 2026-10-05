from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.clevr4_quality import build_clevr4_quality_splits
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.media import materialize_clevr4_images
from tiny_omni_decision.schema import DatasetManifest, DecisionExample

ROOT = Path(".")
ANNOTATION_PATH = ROOT / "data/raw/clevr4-10k/clevr_4_annots.json"
MANIFEST_PATH = ROOT / "manifests/candidates/clevr4.yaml"
OUTPUT_DIR = ROOT / "data/processed/teacher-quality-next/clevr4"
ARCHIVE_PATH = ROOT / "data/raw/teacher-quality-next/clevr4/clevr_4_10k_v1.zip"
BLOCKED_PATH_COMPONENTS = {
    "sealed",
    "audit",
    "audits",
    "eval",
    "evaluation",
    "test",
    "tests",
    "heldout",
    "held-out",
    "heldout-candidates",
}
SAFE_CORPUS_NAMES = {"train.jsonl", "training.jsonl", "validation.jsonl"}


def prior_clevr4_image_ids() -> tuple[set[str], list[str]]:
    seen: set[str] = set()
    scanned: list[str] = []
    for base in (ROOT / "data/processed", ROOT / "artifacts"):
        if not base.exists():
            continue
        for path in base.rglob("*.jsonl"):
            relative = path.relative_to(ROOT).as_posix()
            parts = set(relative.lower().replace(".jsonl", "").replace(".", "/").split("/"))
            if parts & BLOCKED_PATH_COMPONENTS or "teacher-quality-next" in parts:
                continue
            if path.name not in SAFE_CORPUS_NAMES:
                continue
            read_any = False
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("source") != "sgvaze/clevr4":
                        continue
                    record_id = record.get("source_record_id")
                    if isinstance(record_id, str) and record_id:
                        seen.add(record_id)
                        read_any = True
            if read_any:
                scanned.append(relative)
    return seen, sorted(scanned)


def write_jsonl(path: Path, examples: list[DecisionExample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.model_dump_json() + "\n")
    return sha256_file(path)


def sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = DatasetManifest.model_validate(load_structured_file(MANIFEST_PATH))
    annotation_bytes = ANNOTATION_PATH.read_bytes()
    expected_annotation_hash = str(manifest.notes["annotation_sha256"])
    actual_annotation_hash = hashlib.sha256(annotation_bytes).hexdigest()
    if actual_annotation_hash != expected_annotation_hash:
        raise SystemExit("Clevr-4 annotation SHA-256 does not match the pinned manifest")
    if not ARCHIVE_PATH.is_file():
        raise SystemExit(
            "verified local Clevr-4 archive is missing; "
            "run scripts/download_clevr4_archive.py"
        )
    expected_archive_hash = str(manifest.notes["archive_sha512"])
    if sha512_file(ARCHIVE_PATH) != expected_archive_hash:
        raise SystemExit("Clevr-4 archive SHA-512 does not match the pinned manifest")
    annotations: dict[str, dict[str, Any]] = json.loads(annotation_bytes)
    previously_seen, scanned_paths = prior_clevr4_image_ids()
    training, validation, accounting = build_clevr4_quality_splits(
        annotations,
        previously_seen_image_ids=previously_seen,
        manifest=manifest,
        seed=17,
    )
    archive_url = str(manifest.notes["archive"])
    training, train_media = materialize_clevr4_images(
        training,
        data_root=ROOT / "data",
        archive_url=archive_url,
        archive_path=ARCHIVE_PATH,
    )
    validation, validation_media = materialize_clevr4_images(
        validation,
        data_root=ROOT / "data",
        archive_url=archive_url,
        archive_path=ARCHIVE_PATH,
    )
    train_hash = write_jsonl(OUTPUT_DIR / "train.jsonl", training)
    validation_hash = write_jsonl(OUTPUT_DIR / "validation.jsonl", validation)
    metadata = {
        "source": manifest.dataset_id,
        "source_revision": manifest.revision,
        "source_annotation_sha256": actual_annotation_hash,
        "source_archive_sha512": manifest.notes["archive_sha512"],
        "seed": 17,
        "prior_clevr4_image_ids_excluded": len(previously_seen),
        "prior_corpus_paths_read": scanned_paths,
        "accounting": accounting,
        "train_corpus_sha256": train_hash,
        "validation_corpus_sha256": validation_hash,
        "train_media": train_media,
        "validation_media": validation_media,
        "audit_reserve_policy": (
            "15% of fresh official-train image identities held out by the pinned seed; "
            "IDs and labels are not materialized into experiment corpora."
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
