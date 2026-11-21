"""Read-only TRAIN coverage audit for the frozen native common corpus.

Only train.jsonl is opened. Validation and sealed-audit inputs are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts.native_fit import sample_order

DEFAULT_TRAIN = Path(r"C:\CodexArtifacts\pretrained-reuse-r2\native-common-corpus-02\train.jsonl")
DEFAULT_SHA256 = "d279ca44c6ff8e619529805be21675a8b4088fdcc476c7bb04649abdb5e60ebb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_summary(rows: list[dict], *, steps: int = 2048, seed: int = 17) -> dict:
    if not rows or any(row.get("split") != "train" for row in rows):
        raise ValueError("TRAIN rows required; no other split is accepted")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("unique TRAIN question IDs required")

    order = sample_order(rows, steps, seed)
    draws = [index for batch in order for index in batch]
    used_ids = {rows[index]["id"] for index in draws}
    used = [rows[index] for index in draws]
    sample_id_order = "\n".join(rows[index]["id"] for index in draws).encode("utf-8")
    per_modality = {}
    for modality in ("text", "image", "audio", "video"):
        all_rows = [row for row in rows if row["modality"] == modality]
        draw_rows = [row for row in used if row["modality"] == modality]
        used_rows = [row for row in all_rows if row["id"] in used_ids]
        unseen_rows = [row for row in all_rows if row["id"] not in used_ids]

        def observations(values):
            return {row["observation"] for row in values}

        def groups(values):
            return {row["group"] for row in values}

        def assets(values):
            return {
                row["observation"].rsplit(":", 1)[0]
                if row["source"] == "starss"
                else row["observation"]
                for row in values
                if row["modality"] != "text"
            }

        per_modality[modality] = {
            "corpus_questions": len(all_rows),
            "unique_questions_consumed": len(used_rows),
            "unique_questions_unseen": len(unseen_rows),
            "sample_draws": len(draw_rows),
            "repeated_question_draws": len(draw_rows) - len(used_rows),
            "unique_observations_total": len(observations(all_rows)),
            "unique_observations_consumed": len(observations(used_rows)),
            "unique_observations_unseen": len(observations(all_rows) - observations(used_rows)),
            "unique_groups_total": len(groups(all_rows)),
            "unique_groups_consumed": len(groups(used_rows)),
            "unique_groups_unseen": len(groups(all_rows) - groups(used_rows)),
            "unique_media_assets_total": len(assets(all_rows)),
            "unique_media_assets_consumed": len(assets(used_rows)),
            "unique_media_assets_unseen": len(assets(all_rows) - assets(used_rows)),
            "sources_total": dict(sorted(Counter(row["source"] for row in all_rows).items())),
            "sources_consumed": dict(sorted(Counter(row["source"] for row in draw_rows).items())),
        }
    if any(per_modality[modality]["sample_draws"] != steps for modality in per_modality):
        raise ValueError("sampler no longer emits one draw per modality per update")
    return {
        "status": "PASS_NATIVE_COMMON_TRAIN_COVERAGE_AUDIT",
        "scope": "read-only TRAIN question/observation/group/media coverage",
        "steps": steps,
        "seed": seed,
        "consumed_draws": len(draws),
        "unique_questions_consumed": len(used_ids),
        "repeated_question_draws": len(draws) - len(used_ids),
        "sample_id_order_sha256": hashlib.sha256(sample_id_order).hexdigest(),
        "corpus_questions": len(rows),
        "unique_questions_unseen": len(rows) - len(used_ids),
        "modalities": per_modality,
        "validation_loaded": False,
        "sealed_audit_loaded": False,
        "training_performed": False,
        "optimizer_updates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--expected-sha256", default=DEFAULT_SHA256)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"C:\CodexArtifacts\pretrained-reuse-r2\native-common-train-coverage-audit-01"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit output must be a new path")
    start_tick = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    before = sha256(args.train)
    if before != args.expected_sha256:
        raise ValueError("frozen TRAIN JSONL hash differs")
    rows = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines()]
    summary = coverage_summary(rows, steps=args.steps, seed=args.seed)
    after = sha256(args.train)
    if after != before:
        raise ValueError("frozen TRAIN JSONL changed during audit")
    finished = datetime.now(UTC).isoformat()
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "coverage.json"
    result_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "status": summary["status"],
        "input_train_path": str(args.train),
        "input_train_sha256_before": before,
        "input_train_sha256_after": after,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "script_sha256": sha256(Path(__file__)),
        "coverage_json_sha256": sha256(result_path),
        "started_utc": started,
        "finished_utc": finished,
        "wall_seconds": time.perf_counter() - start_tick,
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
