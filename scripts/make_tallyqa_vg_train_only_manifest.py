"""Materialize only the frozen TallyQA-VG TRAIN rows for grouped CV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_train_only_manifest(source: Path) -> dict[str, object]:
    source = source.resolve(strict=True)
    task = json.loads(source.read_text(encoding="utf-8"))
    rows = task.get("train")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source task manifest has no TRAIN rows")
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("TRAIN question IDs are not unique")
    return {
        "schema_version": 1,
        "split_scope": "train_only",
        "source_task_manifest_sha256": sha256_file(source),
        "experiment_id": "tallyqa_vg_amt_four_choice_image_question_v1",
        "train": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    payload = make_train_only_manifest(args.source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output),
        "rows": len(payload["train"]),
        "source_task_manifest_sha256": payload["source_task_manifest_sha256"],
        "split_scope": payload["split_scope"],
    }, indent=2))


if __name__ == "__main__":
    main()
