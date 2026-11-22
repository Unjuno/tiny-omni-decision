"""Prepare Japanese MASSIVE intent train/dev rows with template overlap removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_REVISION = "5092332a2835c4d29c617ef6d118123a1abecd19"
SOURCE_HASHES = {
    "train-00000-of-00001.parquet": (
        "f1404ede08f40ffe4fd64f72adb312304ab7cb5afa1d0bdf4fc61e518bc49de8"
    ),
    "validation-00000-of-00001.parquet": (
        "36283fc28102e6c232ce6928917fa31b898c5d678f18f90f411a086c008c1cc9"
    ),
}
INTENTS = (
    "datetime_query", "iot_hue_lightchange", "transport_ticket", "takeaway_query",
    "qa_stock", "general_greet", "recommendation_events", "music_dislikeness",
    "iot_wemo_off", "cooking_recipe", "qa_currency", "transport_traffic",
    "general_quirky", "weather_query", "audio_volume_up", "email_addcontact",
    "takeaway_order", "email_querycontact", "iot_hue_lightup", "recommendation_locations",
    "play_audiobook", "lists_createoradd", "news_query", "alarm_query", "iot_wemo_on",
    "general_joke", "qa_definition", "social_query", "music_settings", "audio_volume_other",
    "calendar_remove", "iot_hue_lightdim", "calendar_query", "email_sendemail",
    "iot_cleaning", "audio_volume_down", "play_radio", "cooking_query", "datetime_convert",
    "qa_maths", "iot_hue_lightoff", "iot_hue_lighton", "transport_query", "music_likeness",
    "email_query", "play_music", "audio_volume_mute", "social_post", "alarm_set",
    "qa_factoid", "calendar_set", "play_game", "alarm_remove", "lists_remove",
    "transport_taxi", "recommendation_movies", "iot_coffee", "music_query", "play_podcasts",
    "lists_query",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_utterance(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("utterance must be nonempty text")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prepare_rows(
    publisher_train: list[dict[str, Any]], publisher_dev: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(INTENTS) != 60 or len(set(INTENTS)) != 60:
        raise AssertionError("pinned intent taxonomy must contain 60 unique labels")
    for split, rows, partition in (
        ("train", publisher_train, "train"),
        ("validation", publisher_dev, "dev"),
    ):
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate upstream IDs in {split}")
        for row in rows:
            if row["locale"] != "ja-JP" or row["partition"] != partition:
                raise ValueError(f"unexpected locale or publisher partition in {split}")
            normalize_utterance(row["utt"])
            if int(row["intent"]) not in range(len(INTENTS)):
                raise ValueError("intent index outside pinned 60-label taxonomy")

    train_ids = {str(row["id"]) for row in publisher_train}
    dev_ids = {str(row["id"]) for row in publisher_dev}
    if train_ids & dev_ids:
        raise ValueError("underlying SLURP IDs overlap publisher train/dev")
    validation_texts = {normalize_utterance(row["utt"]) for row in publisher_dev}
    train = [
        row
        for row in publisher_train
        if normalize_utterance(row["utt"]) not in validation_texts
    ]
    validation = list(publisher_dev)

    def emit(row: dict[str, Any], split: str) -> dict[str, Any]:
        normalized = normalize_utterance(row["utt"])
        return {
            "sample_id": f"{split}:{row['id']}",
            "source_id": str(row["id"]),
            "content_group_id": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "utterance": row["utt"],
            "target_index": int(row["intent"]),
            "intent": INTENTS[int(row["intent"])],
        }

    train_out = [emit(row, "train") for row in train]
    validation_out = [emit(row, "validation") for row in validation]
    train_groups = {row["content_group_id"] for row in train_out}
    validation_groups = {row["content_group_id"] for row in validation_out}
    if train_groups & validation_groups:
        raise AssertionError("normalized utterance group overlap remains")
    if {row["source_id"] for row in train_out} & {
        row["source_id"] for row in validation_out
    }:
        raise AssertionError("underlying SLURP ID overlap remains")

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(row["intent"] for row in rows).items()))

    manifest = {
        "schema_version": 1,
        "protocol": "massive-ja-jp-publisher-train-dev-slurp-id-and-text-disjoint-v1",
        "source": {
            "repository": "https://huggingface.co/datasets/AmazonScience/massive",
            "revision": SOURCE_REVISION,
            "locale": "ja-JP",
            "license": "CC-BY-4.0",
            "task": "60-way Japanese voice-assistant intent classification",
        },
        "rows": {
            "publisher_train": len(publisher_train),
            "publisher_dev": len(publisher_dev),
            "train_after_overlap_removal": len(train_out),
            "validation": len(validation_out),
            "train_removed_for_normalized_text_overlap": len(publisher_train) - len(train_out),
        },
        "unique_slurp_id_overlap": 0,
        "normalized_utterance_overlap_after_removal": 0,
        "train_label_counts": counts(train_out),
        "validation_label_counts": counts(validation_out),
        "train_intent_coverage": len({row["target_index"] for row in train_out}),
        "validation_intent_coverage": len({row["target_index"] for row in validation_out}),
        "intent_names": list(INTENTS),
        "publisher_test_downloaded_or_opened": False,
        "annotated_utterances_or_worker_metadata_used": False,
    }
    return train_out, validation_out, manifest


def _read_parquet(path: Path, expected_name: str, expected_partition: str):
    if path.name != expected_name:
        raise ValueError(f"expected exact pinned input name {expected_name}")
    if sha256_file(path) != SOURCE_HASHES[expected_name]:
        raise ValueError(f"pinned source hash mismatch for {path}")
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["id", "locale", "partition", "intent", "utt"])
    rows = table.to_pylist()
    for row in rows:
        row["id"] = str(row["id"])
        row["intent"] = int(row["intent"])
        if row["partition"] != expected_partition or row["locale"] != "ja-JP":
            raise ValueError("unexpected source split or locale")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-train", type=Path, required=True)
    parser.add_argument("--publisher-dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    publisher_train = _read_parquet(
        args.publisher_train, "train-00000-of-00001.parquet", "train"
    )
    publisher_dev = _read_parquet(
        args.publisher_dev, "validation-00000-of-00001.parquet", "dev"
    )
    train, validation, manifest = prepare_rows(publisher_train, publisher_dev)
    manifest["source"]["train_parquet_sha256"] = sha256_file(args.publisher_train)
    manifest["source"]["dev_parquet_sha256"] = sha256_file(args.publisher_dev)
    args.output_dir.mkdir(parents=True)
    for name, rows in (("train.jsonl", train), ("validation.jsonl", validation)):
        path = args.output_dir / name
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        manifest[f"{name.removesuffix('.jsonl')}_sha256"] = sha256_file(path)
    manifest_path = args.output_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
