from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.corpus import source_asset_identity  # noqa: E402
from tiny_omni_decision.dataset import (  # noqa: E402
    check_train_eval_splits,
    iter_local_rows,
    sha256_file,
)
from tiny_omni_decision.schema import DecisionExample  # noqa: E402

OUT = ROOT / "data/processed/teacher-quality-next/clean-dev-v2"
BASE_TRAIN = (
    ROOT
    / "artifacts/tiny-omni-decision-teacher-v2/candidate-e/"
    "corpus-v1-seed17-local-media-20261005/train.jsonl"
)
MASSIVE_TRAIN = ROOT / "data/processed/teacher-quality-next/massive/train.jsonl"
CLEVR4_TRAIN = ROOT / "data/processed/teacher-quality-next/clevr4/train.jsonl"
LIBRISPEECH_TRAIN = ROOT / "data/processed/teacher-quality-next/librispeech/train.jsonl"
LIBRISPEECH_DEV_OTHER = (
    ROOT / "data/processed/teacher-quality-next/librispeech-dev-other/validation.jsonl"
)
CLEVRER_ROOT = ROOT / "artifacts/teacher-quality-next/clevrer-fresh-scenes-v3"
SEED = 29
TEXT_GROUPS = 512
IMAGE_GROUPS = 512


def _read(path: Path) -> list[DecisionExample]:
    return [DecisionExample.model_validate(row) for row in iter_local_rows(path)]


def _group_holdout(
    rows: list[DecisionExample], *, source: str, count: int, key: Callable[[DecisionExample], str]
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, object]]:
    grouped: dict[str, list[DecisionExample]] = defaultdict(list)
    for row in rows:
        if row.source == source:
            group = key(row)
            if not group:
                raise ValueError(f"missing grouping identity for {row.id}")
            grouped[group].append(row)
    if len(grouped) < count + 1:
        raise ValueError(f"{source} has too few unique groups for the clean development split")
    ordered = sorted(
        grouped,
        key=lambda group: hashlib.sha256(f"{SEED}\0{source}\0{group}".encode()).digest(),
    )
    heldout = set(ordered[:count])
    validation = [
        row.model_copy(update={"split": "validation"})
        for row in rows
        if row.source == source and key(row) in heldout
    ]
    training = [row for row in rows if row.source != source or key(row) not in heldout]
    return training, validation, {
        "candidate_groups": len(grouped),
        "heldout_groups": len(heldout),
        "heldout_records": len(validation),
        "heldout_group_order_sha256": hashlib.sha256(
            "".join(f"{group}\n" for group in ordered[:count]).encode()
        ).hexdigest(),
    }


def _write(path: Path, rows: list[DecisionExample]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=True) + "\n")
    return sha256_file(path)


def _counts(rows: list[DecisionExample]) -> dict[str, object]:
    modality = Counter(row.modality for row in rows)
    source = Counter(row.source for row in rows)
    assets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        assets[row.modality].add(f"{row.source}:{source_asset_identity(row)}")
    return {
        "examples": len(rows),
        "by_modality": dict(sorted(modality.items())),
        "by_source": dict(sorted(source.items())),
        "unique_underlying_assets_by_modality": {
            key: len(value) for key, value in sorted(assets.items())
        },
        "unique_source_records": len({(row.source, row.source_record_id) for row in rows}),
    }


def _verify_media(rows: list[DecisionExample], *, full_hash: bool) -> None:
    checked: dict[str, str] = {}
    for row in rows:
        for media in row.media:
            if not media.path:
                raise ValueError(f"media path is unresolved: {row.id}")
            path = ROOT / "data" / media.path
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"media is missing or not a regular file: {path}")
            if full_hash and media.sha256 and str(path) not in checked:
                checked[str(path)] = sha256_file(path)
            if full_hash and media.sha256:
                digest = checked[str(path)]
                if digest != media.sha256:
                    raise ValueError(f"media SHA-256 differs from corpus record: {path}")


def build() -> dict[str, object]:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite clean development generation: {OUT}")
    for path in (
        BASE_TRAIN,
        MASSIVE_TRAIN,
        CLEVR4_TRAIN,
        LIBRISPEECH_TRAIN,
        LIBRISPEECH_DEV_OTHER,
        CLEVRER_ROOT / "train.jsonl",
        CLEVRER_ROOT / "validation.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_train = _read(BASE_TRAIN)
    massive_all = _read(MASSIVE_TRAIN)
    massive_train, massive_validation, massive_split = _group_holdout(
        massive_all,
        source="alexa/massive",
        count=TEXT_GROUPS,
        key=lambda row: row.task_group_id or row.source_record_id,
    )
    clevr4_all = _read(CLEVR4_TRAIN)
    clevr4_train, clevr4_validation, clevr4_split = _group_holdout(
        clevr4_all,
        source="sgvaze/clevr4",
        count=IMAGE_GROUPS,
        key=source_asset_identity,
    )
    librispeech_train = _read(LIBRISPEECH_TRAIN)
    audio_validation = _read(LIBRISPEECH_DEV_OTHER)
    clevrer_train = _read(CLEVRER_ROOT / "train.jsonl")
    video_validation = _read(CLEVRER_ROOT / "validation.jsonl")

    train = [*base_train, *massive_train, *clevr4_train, *librispeech_train, *clevrer_train]
    validation = [*massive_validation, *clevr4_validation, *audio_validation, *video_validation]
    if len({row.id for row in train}) != len(train):
        raise ValueError("combined training corpus has duplicate example IDs")
    if len({row.id for row in validation}) != len(validation):
        raise ValueError("combined development corpus has duplicate example IDs")
    integrity = check_train_eval_splits(train, validation)

    train_speakers = {
        row.task_group_id.removeprefix("speaker:")
        for row in train
        if row.source == "openslr/LibriSpeech" and row.task_group_id
    }
    validation_speakers = {
        row.task_group_id.removeprefix("speaker:")
        for row in validation
        if row.source == "openslr/LibriSpeech" and row.task_group_id
    }
    speaker_overlap = train_speakers & validation_speakers
    if speaker_overlap:
        raise ValueError(f"LibriSpeech speakers cross train/development: {len(speaker_overlap)}")
    train_scenes = {
        source_asset_identity(row)
        for row in train
        if row.source == "MIT-IBM/CLEVRER"
    }
    validation_scenes = {
        source_asset_identity(row)
        for row in validation
        if row.source == "MIT-IBM/CLEVRER"
    }
    if train_scenes & validation_scenes:
        raise ValueError("CLEVRER video media identity crosses train/development")

    _verify_media(train, full_hash=False)
    _verify_media(validation, full_hash=True)
    OUT.mkdir(parents=True, exist_ok=False)
    train_path = OUT / "train.jsonl"
    validation_path = OUT / "validation.jsonl"
    train_hash = _write(train_path, train)
    validation_hash = _write(validation_path, validation)
    id_order_hash = hashlib.sha256(
        "".join(f"{row.id}\n" for row in validation).encode()
    ).hexdigest()
    inputs = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            BASE_TRAIN,
            MASSIVE_TRAIN,
            CLEVR4_TRAIN,
            LIBRISPEECH_TRAIN,
            LIBRISPEECH_DEV_OTHER,
            CLEVRER_ROOT / "train.jsonl",
            CLEVRER_ROOT / "validation.jsonl",
            CLEVRER_ROOT / "corpus-manifest.json",
        )
    }
    result: dict[str, object] = {
        "generation_id": "teacher-quality-clean-dev-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "input_sha256": inputs,
        "group_selection": {"massive": massive_split, "clevr4": clevr4_split},
        "train": _counts(train),
        "validation": _counts(validation),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "validation_id_order_sha256": id_order_hash,
        "train_validation_integrity": integrity,
        "librispeech_speaker_overlap": 0,
        "clevrer_scene_overlap": 0,
        "sealed_audit_loaded": False,
        "test_partitions_loaded": False,
        "base_model_pretraining_overlap": "UNKNOWN; upstream pretraining corpus is not enumerable",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
