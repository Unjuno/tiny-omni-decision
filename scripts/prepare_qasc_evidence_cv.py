"""Prepare an evidence-context QASC TRAIN-only corpus for grouped cross-validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.qasc import _normalized

SOURCE_TRAIN_SHA256 = "b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1"
PREPARED_TRAIN_SHA256 = "d2e220510f98e19a2df5e44f89d29e1b1bd5b218397264b08ff1102ea3903e51"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_group(fact1: str, fact2: str) -> str:
    normalized = sorted((_normalized(fact1), _normalized(fact2)))
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def enrich_train_rows(
    prepared_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(prepared_rows) != len(source_rows):
        raise ValueError("prepared and source TRAIN row counts differ")
    if [row.get("id") for row in prepared_rows] != [row.get("id") for row in source_rows]:
        raise ValueError("source and prepared TRAIN ID order differs")
    source_by_id = {}
    for row in source_rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in source_by_id:
            raise ValueError("source TRAIN contains a missing or duplicate row ID")
        source_by_id[row_id] = row
    enriched = []
    targets_in_facts = 0
    unique_fact_strings: set[str] = set()
    fact_pair_ids: set[str] = set()
    for prepared in prepared_rows:
        row_id = prepared.get("id")
        source = source_by_id.get(row_id)
        if source is None:
            raise ValueError(f"source row ID is missing: {row_id}")
        choices = source.get("choices")
        if not isinstance(choices, dict):
            raise ValueError(f"source choices are malformed: {row_id}")
        labels, texts = choices.get("label"), choices.get("text")
        if (
            not isinstance(labels, list)
            or not isinstance(texts, list)
            or len(labels) != 8
            or len(texts) != 8
            or set(labels) != set("ABCDEFGH")
            or len(set(texts)) != 8
        ):
            raise ValueError(f"source options are malformed: {row_id}")
        target_label = source.get("answerKey")
        if target_label not in labels:
            raise ValueError(f"source answer label is malformed: {row_id}")
        target = texts[labels.index(target_label)]
        if (
            prepared.get("question") != source.get("question")
            or prepared.get("options") != texts
            or prepared.get("target") != target
            or prepared.get("split") != "train"
        ):
            raise ValueError(f"prepared/source question-choice-target mismatch: {row_id}")
        fact1, fact2 = source.get("fact1"), source.get("fact2")
        if (
            not isinstance(fact1, str)
            or not fact1.strip()
            or not isinstance(fact2, str)
            or not fact2.strip()
        ):
            raise ValueError(f"source support facts are missing: {row_id}")
        group = str(prepared.get("source_record_id", "")).split(":", 1)[0]
        expected_group = _source_group(fact1, fact2)
        if group != expected_group:
            raise ValueError(f"fact-pair identity mismatch: {row_id}")
        facts = f"Evidence:\n{fact1}\n{fact2}"
        target_norm = _normalized(target)
        normalized_facts = (_normalized(fact1), _normalized(fact2))
        targets_in_facts += int(
            any(target_norm and target_norm in fact for fact in normalized_facts)
        )
        unique_fact_strings.update(normalized_facts)
        fact_pair_ids.add(group)
        row = dict(prepared)
        row["state"] = facts
        enriched.append(row)
    if len(source_by_id) != len(prepared_rows):
        raise ValueError("source TRAIN contains unmatched rows")
    report = {
        "rows": len(enriched),
        "unique_fact_pair_groups": len(fact_pair_ids),
        "unique_normalized_fact_strings": len(unique_fact_strings),
        "target_option_exact_normalized_substring_in_either_fact_rows": targets_in_facts,
        "source_id_order_preserved": True,
        "validation_or_test_opened": False,
    }
    return enriched, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-train", type=Path, required=True)
    parser.add_argument("--source-train-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.prepared_train.resolve(strict=True)
    parquet_path = args.source_train_parquet.resolve(strict=True)
    if _sha256(train_path) != PREPARED_TRAIN_SHA256:
        raise ValueError("prepared QASC TRAIN hash changed")
    if _sha256(parquet_path) != SOURCE_TRAIN_SHA256:
        raise ValueError("pinned QASC source TRAIN Parquet hash changed")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence-context corpus: {output}")
    import pyarrow.parquet as parquet

    prepared_rows = [
        json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
    ]
    source_rows = parquet.read_table(parquet_path).to_pylist()
    if len(prepared_rows) != 8134 or len(source_rows) != 8134:
        raise ValueError("expected exactly 8,134 QASC TRAIN rows")
    enriched, audit = enrich_train_rows(prepared_rows, source_rows)
    output.mkdir(parents=True, exist_ok=False)
    enriched_path = output / "evidence-train.jsonl"
    enriched_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in enriched),
        encoding="utf-8",
    )
    audit.update(
        {
            "source_train_sha256": SOURCE_TRAIN_SHA256,
            "prepared_train_sha256": PREPARED_TRAIN_SHA256,
            "evidence_train_sha256": _sha256(enriched_path),
            "source_revision": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
            "license": "CC-BY-4.0",
            "only_train_shard_opened": True,
        }
    )
    (output / "preparation-report.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
