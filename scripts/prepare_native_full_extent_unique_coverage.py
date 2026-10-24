"""CPU-only, hash-pinned preflight for the full-extent/fresh-coverage fit."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.native_coverage import make_pool
from scripts.native_fit import accounting
from scripts.native_full_extent import parent_probe
from scripts.native_long_budget import EVALUATION_STEPS, MAX_STEPS, extend_plan, learning_rate
from scripts.native_matched_batches import build_plan
from scripts.native_video_cache import sha

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/pretrained_reuse/native_full_extent_unique_coverage_long_v1.json"
FRESH_POLICY = "fresh"
PERMUTATION = "seed17:pool-occurrence-index:original-row-id SHA256-seeded Python shuffle"


def write(path, value):
    path.write_text(serialize(value), encoding="utf-8")


def serialize(value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return text.replace("\n", "\r\n") if os.name == "nt" else text


def serialized_sha(value):
    return hashlib.sha256(serialize(value).encode("utf-8")).hexdigest()


def verify(pins):
    for path, digest in pins.items():
        if sha(Path(path)) != digest:
            raise ValueError("changed or missing input: " + path)


def build_sampling(train, original, config):
    original_pool = [i for batch in original for i in batch]
    if len(original) != 2048 or any(len(batch) != 4 for batch in original):
        raise ValueError("fixed original 8,192-occurrence source pool required")
    coverage = make_pool(train, original_pool, FRESH_POLICY)
    if coverage != json.loads(Path(config["pools"][FRESH_POLICY]).read_text(encoding="utf-8")):
        raise ValueError("fresh coverage pool differs from pinned plan")
    plan = build_plan(train, coverage["pool"], "independent")
    if plan != json.loads(Path(config["plans"][FRESH_POLICY]).read_text(encoding="utf-8")):
        raise ValueError("fresh occurrence plan differs from pinned plan")
    long_plan = extend_plan(coverage["pool"], plan["occurrences"])
    sampling = dict(
        order=long_plan["order"],
        occurrence_plan=long_plan,
        parent_occurrence_plan=plan,
        coverage_pool=coverage,
        accounting=accounting([train[i] for batch in long_plan["order"] for i in batch]),
        option_permutation=PERMUTATION,
    )
    return original_pool, coverage, plan, long_plan, sampling


def run(config_path, output):
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("preflight output must be new: " + str(output))
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["id"] != "native-full-extent-unique-coverage-long-v1":
        raise ValueError("unexpected frozen preflight configuration")
    if config["common_train_probe_scope"] != "parent-6656":
        raise ValueError("common parent probe must remain fixed")
    verify(config["input_pins"])
    pins = dict(config["input_pins"])
    pins[str(config_path)] = sha(config_path)
    py_files = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO
    ).decode().split("\0")
    code_pins = {str(REPO / name): sha(REPO / name) for name in py_files if name}

    corpus = Path(config["corpus"])
    train = [
        json.loads(line)
        for line in (corpus / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (corpus / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(train) != 100580
        or len(validation) != 24346
        or any(row["split"] != "train" for row in train)
        or any(row["split"] != "validation" for row in validation)
    ):
        raise ValueError("full fixed TRAIN/validation corpus required")
    original = json.loads(Path(config["parent_sampling"]).read_text(encoding="utf-8"))["order"]
    original_pool, coverage, plan, long_plan, sampling = build_sampling(train, original, config)

    fresh_fit = Path(config["fresh_control_fit"])
    fresh_result = json.loads((fresh_fit / "result.json").read_text(encoding="utf-8"))
    if (
        fresh_result["status"] != "PASS_NATIVE_LONG_BUDGET_TRAIN"
        or fresh_result["step"] != MAX_STEPS
    ):
        raise ValueError("completed 32,768-step fresh-coverage comparator required")
    repeat_result = json.loads(
        (Path(config["control_fit"]) / "result.json").read_text(encoding="utf-8")
    )
    if (
        repeat_result["status"] != "PASS_NATIVE_LONG_BUDGET_TRAIN"
        or repeat_result["step"] != MAX_STEPS
    ):
        raise ValueError("completed 32,768-step repeat/full-extent comparator required")
    lr_plan = [learning_rate(step) for step in range(1, MAX_STEPS + 1)]
    validation_ids = [
        dict(id=row["id"], option_ids=row["option_ids"], target_index=row["target_index"])
        for row in validation
    ]
    probe = parent_probe(train, original_pool)
    probe_ids = [row["id"] for row in probe]
    sample_digest = serialized_sha(sampling)
    lr_digest = serialized_sha(lr_plan)
    validation_digest = serialized_sha(validation_ids)
    probe_digest = serialized_sha(probe_ids)
    if (
        sample_digest != fresh_result["artifacts"]["sampling.json"]
        or lr_digest != fresh_result["artifacts"]["lr-plan.json"]
        or validation_digest != repeat_result["artifacts"]["validation-ids.json"]
        or validation_digest != fresh_result["artifacts"]["validation-ids.json"]
        or probe_digest != repeat_result["artifacts"]["train-probe-ids.json"]
    ):
        raise ValueError("reconstructed plans differ from pinned comparator artifacts")
    if len(set(coverage["pool"])) != 12800 or len(coverage["pool"]) != 16384:
        raise ValueError("fixed 12,800-unique / 16,384-occurrence pool required")
    if coverage["new_unique_by_modality"] != dict(text=2048, image=2048, audio=2048, video=0):
        raise ValueError("unexpected unique coverage change")
    output.mkdir(parents=True)
    for name, value in (
        ("sampling.json", sampling),
        ("lr-plan.json", lr_plan),
        ("validation-ids.json", validation_ids),
        ("common-parent-train-probe-ids.json", probe_ids),
        ("effective-config.json", config),
    ):
        write(output / name, value)

    result = dict(
        status="PASS_CPU_FRESH_FULL_EXTENT_PREFLIGHT",
        config_path=str(config_path),
        config_sha256=sha(config_path),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        code_pins=code_pins,
        input_pins=pins,
        fresh_control_result_sha256=sha(fresh_fit / "result.json"),
        repeat_control_result_sha256=sha(Path(config["control_fit"]) / "result.json"),
        completed_utc=datetime.now(UTC).isoformat(),
        environment={
            "python": sys.version,
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
        },
        train_rows=len(train),
        validation_rows=len(validation),
        original_occurrences=len(original_pool),
        pool_occurrences=len(coverage["pool"]),
        unique_questions=len(set(coverage["pool"])),
        unique_questions_by_modality={
            modality: sum(
                1 for index in set(coverage["pool"]) if train[index]["modality"] == modality
            )
            for modality in ("text", "image", "audio", "video")
        },
        new_unique_by_modality=coverage["new_unique_by_modality"],
        repeated_occurrences_in_pool=coverage["extra_repeated_occurrences"],
        updates=MAX_STEPS,
        examples=MAX_STEPS * 8,
        evaluation_steps=list(EVALUATION_STEPS),
        common_parent_probe_rows=len(probe),
        sample_order_sha256=sample_digest,
        lr_plan_sha256=lr_digest,
        validation_ids_sha256=validation_digest,
        common_parent_probe_sha256=probe_digest,
        matches_center_crop_fresh_order=True,
        matches_repeat_and_fresh_validation=True,
        matches_repeat_common_train_probe=True,
        sealed_audit_loaded=False,
        model_loaded=False,
        gpu_used=False,
        artifacts={},
    )
    result["artifacts"] = {
        path.name: sha(path)
        for path in (
            output / "sampling.json",
            output / "lr-plan.json",
            output / "validation-ids.json",
            output / "common-parent-train-probe-ids.json",
            output / "effective-config.json",
        )
    }
    write(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
