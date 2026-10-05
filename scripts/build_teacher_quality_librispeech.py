from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.librispeech import build_librispeech_decisions

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw/teacher-quality-next/librispeech"
AUDIO_ROOT = RAW_DIR / "audio"
OUTPUT_DIR = DATA_ROOT / "processed/teacher-quality-next/librispeech"
SOURCE_REVISION = "96519bc4ce8e8a57f84fd03b8833553f51e76dfc"
ARCHIVES = {
    "train": {
        "name": "train-clean-100.tar.gz",
        "partition": "train-clean-100",
        "md5": "2a93770f6d5c6c964bc36631d331a522",
        "sha256": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    },
    "validation": {
        "name": "dev-clean.tar.gz",
        "partition": "dev-clean",
        "md5": "42e2234ba48799c1f50f24a7926300a1",
        "sha256": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    },
}


def read_transcriptions(split: str, archive_path: Path) -> list[dict[str, Any]]:
    partition = str(ARCHIVES[split]["partition"])
    records: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".trans.txt"):
                continue
            member_path = PurePosixPath(member.name)
            if len(member_path.parts) != 5 or member_path.parts[:2] != ("LibriSpeech", partition):
                raise ValueError(f"unexpected LibriSpeech transcript path: {member.name}")
            speaker_id = member_path.parts[2]
            chapter_id = member_path.parts[3]
            transcription_file = archive.extractfile(member)
            if transcription_file is None:
                raise OSError(f"could not read transcript file {member.name}")
            for raw in transcription_file:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                utterance_id, transcript = line.split(" ", 1)
                expected_prefix = f"{speaker_id}-{member_path.parts[3]}-"
                if not utterance_id.startswith(expected_prefix):
                    raise ValueError(f"utterance ID does not match chapter: {utterance_id}")
                audio_member = "/".join(
                    (*member_path.parts[:-1], f"{utterance_id}.flac")
                )
                target = AUDIO_ROOT / partition / speaker_id / chapter_id / f"{utterance_id}.flac"
                records.append(
                    {
                        "split": split,
                        "speaker_id": speaker_id,
                        "chapter_id": member_path.parts[3],
                        "utterance_id": utterance_id,
                        "transcript": transcript,
                        "archive_member": audio_member,
                        "media_path": target.relative_to(DATA_ROOT).as_posix(),
                    }
                )
    return records


def extract_audio(split: str, archive_path: Path, records: list[dict[str, Any]]) -> None:
    wanted = {str(row["archive_member"]): row for row in records}
    found: set[str] = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            row = wanted.get(member.name)
            if row is None:
                continue
            if not member.isfile() or not member.name.endswith(".flac"):
                raise ValueError(f"expected a FLAC audio member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"could not extract LibriSpeech audio {member.name}")
            target = DATA_ROOT / str(row["media_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            digest = hashlib.sha256()
            with temporary.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            temporary.replace(target)
            row["media_sha256"] = digest.hexdigest()
            found.add(member.name)
            if len(found) % 2_000 == 0 or len(found) == len(wanted):
                print(f"{split}: extracted {len(found)}/{len(wanted)} audio clips", flush=True)
    if found != wanted.keys():
        raise ValueError(f"missing LibriSpeech audio members: {len(wanted.keys() - found)}")


def write_jsonl(path: Path, rows: list[Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
    return sha256_file(path)


def main() -> None:
    archives: dict[str, Path] = {}
    for split, source in ARCHIVES.items():
        path = RAW_DIR / str(source["name"])
        if sha256_file(path) != source["sha256"]:
            raise SystemExit(f"LibriSpeech {split} archive SHA-256 mismatch")
        archives[split] = path

    all_train = read_transcriptions("train", archives["train"])
    validation_records = read_transcriptions("validation", archives["validation"])
    extract_audio("train", archives["train"], all_train)
    extract_audio("validation", archives["validation"], validation_records)
    decisions_train, decisions_validation, accounting = build_librispeech_decisions(
        [*all_train, *validation_records], revision=SOURCE_REVISION, seed=17
    )
    train_sha256 = write_jsonl(OUTPUT_DIR / "train.jsonl", decisions_train)
    validation_sha256 = write_jsonl(OUTPUT_DIR / "validation.jsonl", decisions_validation)
    metadata = {
        "dataset": "OpenSLR SLR12 LibriSpeech",
        "source_revision": SOURCE_REVISION,
        "source_revision_basis": (
            "SHA-1 fingerprint of the pinned official SLR12 md5sum.txt snapshot; "
            "not a Git commit."
        ),
        "checksum_manifest_sha256": (
            "efe6522682d078fdcfd2c3b2d418a53a89683b342dd5147a769a41a7e021fe3f"
        ),
        "archives": {
            split: {
                "name": source["name"],
                "size_bytes": archives[split].stat().st_size,
                "md5": source["md5"],
                "sha256": source["sha256"],
            }
            for split, source in ARCHIVES.items()
        },
        "training_selection": {
            "source_train_utterances": len(all_train),
            "selected_train_utterances": len(all_train),
            "selection": "all official train-clean-100 utterances, without replacement",
            "selection_seed": 17,
        },
        "accounting": accounting,
        "train_corpus_sha256": train_sha256,
        "validation_corpus_sha256": validation_sha256,
        "test_archives_downloaded_or_read": False,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
