"""Prepare pinned QASC train/validation rows for the generic frozen text probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from tiny_omni_decision.qasc import prepare_qasc_split_rows

DATASET_REVISION = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
OFFICIAL_SOURCE_REVISION = "c44a320c2bb682d614a675d2ba4be072150db4eb"
TRAIN_SHA256 = "b9a297b5ab55f1605c7682ffbb7042c26d7ecb9ff1e1aa5a820d4e791c8302d1"
VALIDATION_SHA256 = "d1ae34ae13c5fce2c55305372c203c6cdb789728d0d7e5ea2956d55bc33f40ae"
EXPECTED_ROWS = {"train": 8134, "validation": 926}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_order_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(f"{row['id']}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shard", type=Path, required=True)
    parser.add_argument("--validation-shard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if output == repo_root or output.is_relative_to(repo_root):
        raise ValueError("QASC artifacts must be written outside the repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")

    inputs = {"train": args.train_shard.resolve(), "validation": args.validation_shard.resolve()}
    expected_hashes = {"train": TRAIN_SHA256, "validation": VALIDATION_SHA256}
    for split, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned QASC {split} shard: {path}")
        actual = _sha256(path)
        if actual != expected_hashes[split]:
            raise ValueError(f"QASC {split} shard hash mismatch: {actual}")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("preparation requires the optional PyArrow data dependency") from exc

    source_rows = {
        split: parquet.read_table(path).to_pylist() for split, path in inputs.items()
    }
    for split, rows in source_rows.items():
        if len(rows) != EXPECTED_ROWS[split]:
            raise ValueError(f"QASC {split} row count mismatch: {len(rows)}")
    train_rows, validation_rows, split_audit = prepare_qasc_split_rows(
        source_rows["train"], source_rows["validation"], revision=DATASET_REVISION
    )

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "validation.jsonl", validation_rows)
    report = {
        "schema_version": 1,
        "status": "prepared_train_validation_only",
        "source": "allenai/qasc",
        "source_revision": DATASET_REVISION,
        "official_repository_revision": OFFICIAL_SOURCE_REVISION,
        "license": "CC-BY-4.0",
        "train_shard_sha256": expected_hashes["train"],
        "validation_shard_sha256": expected_hashes["validation"],
        "train_sample_id_order_sha256": _id_order_sha256(source_rows["train"]),
        "validation_sample_id_order_sha256": _id_order_sha256(source_rows["validation"]),
        **split_audit,
        "train_answer_label_counts": dict(
            sorted(Counter(row["answerKey"] for row in source_rows["train"]).items())
        ),
        "validation_answer_label_counts": dict(
            sorted(Counter(row["answerKey"] for row in source_rows["validation"]).items())
        ),
        "train_normalized_question_duplicates": len(source_rows["train"])
        - len({_normalized_question(row["question"]) for row in source_rows["train"]}),
        "validation_normalized_question_duplicates": len(source_rows["validation"])
        - len({_normalized_question(row["question"]) for row in source_rows["validation"]}),
        "facts_in_model_inputs": False,
        "test_shard_read_or_written": False,
        "prepared_train_sha256": _sha256(output / "train.jsonl"),
        "prepared_validation_sha256": _sha256(output / "validation.jsonl"),
    }
    (output / "preparation-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
