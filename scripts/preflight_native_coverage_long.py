"""CPU-only integrity and sampler preflight for the long fresh-coverage run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.native_coverage import make_pool
from scripts.native_fit import accounting
from scripts.native_long_budget import (
    EVALUATION_STEPS,
    MAX_STEPS,
    extend_plan,
    fixed_parent_probe_indices,
    learning_rate,
    verify_pins,
)
from scripts.native_matched_batches import build_plan

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/pretrained_reuse/native_unique_coverage_long_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify(pins: dict[str, str]) -> None:
    verify_pins(pins, repo_root=REPO)


def merge_pins(target: dict[str, str], source: dict[str, str]) -> None:
    for path, digest in source.items():
        if path in target and target[path] != digest:
            raise ValueError(f"conflicting hash pin for {path}")
        target[path] = digest


def id_order_hash(order, train) -> str:
    digest = hashlib.sha256()
    for batch in order:
        for index in batch:
            digest.update(train[index]["id"].encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def fresh_bucket_flow(rows, parent_pool, coverage):
    parent = set(parent_pool)
    available = {}
    for index, row in enumerate(rows):
        key = (row["modality"], row["source"], row["task"], len(row["options"]))
        if index not in parent:
            available[key] = available.get(key, 0) + 1
    additions = {}
    for index in coverage["pool"][len(parent_pool) :]:
        row = rows[index]
        key = (row["modality"], row["source"], row["task"], len(row["options"]))
        item = additions.setdefault(key, {"fresh": 0, "repeat": 0})
        item["repeat" if index in parent else "fresh"] += 1
    result = []
    for key in sorted(set(available) | set(additions)):
        counts = additions.get(key, {"fresh": 0, "repeat": 0})
        result.append(
            {
                "modality": key[0],
                "source": key[1],
                "task": key[2],
                "option_count": key[3],
                "available_unused_questions": available.get(key, 0),
                "new_unique_consumed": counts["fresh"],
                "repeated_parent_consumed": counts["repeat"],
                "bucket_exhausted": counts["repeat"] > 0,
            }
        )
    return result


def empty_accounting():
    return {
        "consumed": 0,
        "unique_questions": 0,
        "repeated_questions": 0,
        "unique_observations": 0,
        "unique_media_assets": 0,
        "unique_groups": 0,
        "modalities": {},
        "sources": {},
        "tasks": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse preflight output: {output}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("committed clean source required for run preflight")
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    config = read_json(config_path)
    expected_training = {
        "seed": 17,
        "max_steps": MAX_STEPS,
        "microbatches_per_update": 8,
        "lr": 0.001,
        "warmup_updates": 62,
        "scheduler": "cosine",
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "evaluation_steps": list(EVALUATION_STEPS),
        "performance_early_stopping": False,
    }
    if config.get("id") != "native-unique-coverage-long-v1":
        raise ValueError("unexpected experiment config identity")
    if config.get("training") != expected_training:
        raise ValueError("long-run conditions differ from preregistered schedule")
    if "fresh" not in config.get("pools", {}) or "fresh" not in config.get("plans", {}):
        raise ValueError("fresh coverage pool and plan must be pinned")

    pins = dict(config["input_pins"])
    merge_pins(pins, {str(config_path): sha256_file(config_path)})
    verify(pins)

    corpus = Path(config["corpus"])
    query_root = Path(config["queries"])
    corpus_receipt = read_json(corpus / "result.json")
    query_receipt = read_json(query_root / "result.json")
    if (
        corpus_receipt.get("status") != "PASS_COMPLETE_NATIVE_COMMON_CORPUS"
        or corpus_receipt.get("sealed_audit_loaded") is not False
        or query_receipt.get("status") != "PASS_NATIVE_COMMON_QUERIES_CACHE"
    ):
        raise ValueError("complete approved development corpus/query cache required")
    merge_pins(pins, corpus_receipt["validated_hashes"])
    merge_pins(
        pins,
        {str(corpus / name): digest for name, digest in corpus_receipt["artifacts"].items()},
    )
    merge_pins(pins, query_receipt["validated_hashes"])
    merge_pins(
        pins,
        {str(query_root / name): digest for name, digest in query_receipt["artifacts"].items()},
    )
    pin_evidence = verify_pins(
        pins,
        source_commits=[corpus_receipt["source_commit"], query_receipt["source_commit"]],
        repo_root=REPO,
    )

    baseline_path = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/native-long-budget-fit-01/result.json"
    )
    baseline = read_json(baseline_path)
    if (
        baseline.get("status") != "PASS_NATIVE_LONG_BUDGET_TRAIN"
        or baseline.get("policy") != "repeat"
        or baseline.get("step") != MAX_STEPS
        or baseline.get("best_step") != 8192
    ):
        raise ValueError("pinned full repeat control does not match the planned comparison")

    train = rows_jsonl(corpus / "train.jsonl")
    validation = rows_jsonl(corpus / "validation.jsonl")
    if (
        len(train) != 100580
        or len(validation) != 24346
        or any(row.get("split") != "train" for row in train)
        or any(row.get("split") != "validation" for row in validation)
        or len({row["id"] for row in train}) != len(train)
        or len({row["id"] for row in validation}) != len(validation)
        or {row["id"] for row in train} & {row["id"] for row in validation}
    ):
        raise ValueError("full unique TRAIN and disjoint development validation required")
    validation_ids = [
        dict(id=row["id"], option_ids=row["option_ids"], target_index=row["target_index"])
        for row in validation
    ]
    if validation_ids != read_json(baseline_path.parent / "validation-ids.json"):
        raise ValueError("validation IDs, option order or targets differ from repeat control")

    original = read_json(Path(config["parent_sampling"]))["order"]
    if len(original) != 2048 or any(len(batch) != 4 for batch in original):
        raise ValueError("pinned original 8,192-occurrence pool required")
    parent_pool = [index for batch in original for index in batch]
    coverage = make_pool(train, parent_pool, "fresh")
    if coverage != read_json(Path(config["pools"]["fresh"])):
        raise ValueError("regenerated fresh coverage pool differs from pinned pool")
    plan = build_plan(train, coverage["pool"], "independent")
    if plan != read_json(Path(config["plans"]["fresh"])):
        raise ValueError("regenerated fresh occurrence plan differs from pinned plan")
    long_plan = extend_plan(coverage["pool"], plan["occurrences"])
    if len(long_plan["order"]) != MAX_STEPS or any(len(batch) != 8 for batch in long_plan["order"]):
        raise ValueError("complete 32,768-update sampler plan required")

    probe_indices = fixed_parent_probe_indices(long_plan, parent_pool)
    probe_ids = [train[index]["id"] for index in probe_indices]
    baseline_probe = read_json(baseline_path.parent / "train-probe-ids.json")
    if probe_ids != baseline_probe:
        raise ValueError("fixed common TRAIN probe differs from repeat control")

    lr_values = [learning_rate(step) for step in range(1, MAX_STEPS + 1)]
    baseline_lr = read_json(baseline_path.parent / "lr-plan.json")
    if lr_values != baseline_lr or not (lr_values[512] > 0 and lr_values[-1] == 0):
        raise ValueError("full-horizon cosine schedule differs from repeat control")

    checkpoints = {}
    for step in EVALUATION_STEPS:
        consumed = [train[index] for batch in long_plan["order"][:step] for index in batch]
        checkpoints[str(step)] = accounting(consumed) if consumed else empty_accounting()
    bucket_flow = fresh_bucket_flow(train, parent_pool, coverage)

    sampling_details = {
        "coverage": {
            "policy": coverage["policy"],
            "seed": coverage["seed"],
            "parent_occurrences": len(parent_pool),
            "pool_occurrences": len(coverage["pool"]),
            "unique_questions": coverage["accounting"]["unique_questions"],
            "repeated_presentations": coverage["accounting"]["repeated_questions"],
            "new_unique_by_modality": coverage["new_unique_by_modality"],
            "extra_repeated_occurrences": coverage["extra_repeated_occurrences"],
            "accounting": coverage["accounting"],
            "bucket_flow": bucket_flow,
        },
        "long_run": {
            "steps": len(long_plan["order"]),
            "cycles": long_plan["cycles"],
            "occurrence_plan_sha256": long_plan["order_sha256"],
            "sample_id_order_sha256": id_order_hash(long_plan["order"], train),
            "full_run_accounting": accounting(
                [train[index] for batch in long_plan["order"] for index in batch]
            ),
            "checkpoint_accounting": checkpoints,
        },
    }
    sampling_path = output / "sampling-accounting.json"
    output.mkdir(parents=True)
    sampling_path.write_text(
        json.dumps(sampling_details, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lr_path = output / "lr-plan.json"
    lr_path.write_text(
        json.dumps(lr_values, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    (output / "effective-config.json").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    report = {
        "status": "PASS_NATIVE_COVERAGE_LONG_CPU_PREFLIGHT",
        "experiment_id": config["id"],
        "source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "started_utc": started_at.isoformat(),
        "python": sys.version,
        "device": "CPU only; no torch or CUDA imported",
        "sealed_audit_loaded": False,
        "training_performed": False,
        "baseline": {
            "result_sha256": sha256_file(baseline_path),
            "source_commit": baseline["source_commit"],
            "best_step": baseline["best_step"],
            "steps": baseline["step"],
            "best_checkpoint_sha256": baseline["artifacts"]["best.pt"],
            "final_checkpoint_sha256": baseline["artifacts"]["final.pt"],
        },
        "verified_input_file_count": len(pins),
        "verified_historical_source_blob_count": sum(
            item["method"] == "pinned_git_blob_sha256" for item in pin_evidence.values()
        ),
        "source_commits": [corpus_receipt["source_commit"], query_receipt["source_commit"]],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_ids_sha256": sha256_file(baseline_path.parent / "validation-ids.json"),
        "train_probe_ids_sha256": sha256_file(baseline_path.parent / "train-probe-ids.json"),
        "coverage": {
            "pool_sha256": sha256_file(Path(config["pools"]["fresh"])),
            "unique_questions": coverage["accounting"]["unique_questions"],
            "new_unique_by_modality": coverage["new_unique_by_modality"],
            "extra_repeated_occurrences": coverage["extra_repeated_occurrences"],
            "bucket_count": len(bucket_flow),
            "exhausted_bucket_count": sum(item["bucket_exhausted"] for item in bucket_flow),
        },
        "long_run": {
            "steps": len(long_plan["order"]),
            "microbatches_per_update": 8,
            "cycles": long_plan["cycles"],
            "occurrence_plan_sha256": long_plan["order_sha256"],
            "sample_id_order_sha256": id_order_hash(long_plan["order"], train),
            "consumed_examples": len(long_plan["order"]) * 8,
            "unique_questions": sampling_details["long_run"]["full_run_accounting"][
                "unique_questions"
            ],
            "repeated_presentations": sampling_details["long_run"]["full_run_accounting"][
                "repeated_questions"
            ],
            "modalities": sampling_details["long_run"]["full_run_accounting"]["modalities"],
            "sources": sampling_details["long_run"]["full_run_accounting"]["sources"],
        },
        "checkpoint_accounting_steps": list(map(str, EVALUATION_STEPS)),
        "sampling_accounting_path": str(sampling_path),
        "sampling_accounting_sha256": sha256_file(sampling_path),
        "lr_plan_path": str(lr_path),
        "lr_plan_sha256": sha256_file(lr_path),
        "lr_schedule": {
            "kind": "cosine",
            "count": len(lr_values),
            "warmup_updates": 62,
            "step_1": lr_values[0],
            "step_62": lr_values[61],
            "step_512": lr_values[511],
            "step_2048": lr_values[2047],
            "step_32768": lr_values[-1],
            "sha256": hashlib.sha256(
                json.dumps(lr_values, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
        },
        "finished_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
