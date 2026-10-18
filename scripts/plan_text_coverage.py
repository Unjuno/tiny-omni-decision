"""Dry-run original-text coverage for a possible subsequent training run."""

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def coverage_order(original, curriculum, already_consumed, steps=2048, seed=17):
    if steps < 1 or len(set(original)) != len(original) or len(set(curriculum)) != len(curriculum):
        raise ValueError("invalid budget or duplicate source IDs")
    if set(original) & set(curriculum):
        raise ValueError("cross-source duplicate IDs")
    pools = [[x for x in original if x not in already_consumed], list(curriculum)]
    rng = random.Random(seed)
    for pool in pools:
        if len(pool) < 4 * steps:
            raise ValueError("insufficient unique examples")
        rng.shuffle(pool)
    return [x for step in range(steps) for pool in pools for x in pool[4 * step : 4 * (step + 1)]]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import yaml

    from scripts.train_frozen_text_probe import _assert_disjoint, _load_examples
    from scripts.train_numeric_deterministic import verify_splits
    from tiny_omni_decision.schema import DatasetManifest

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    train_path = root / "text/train.jsonl"
    validation_path = root / "text/validation.jsonl"
    curriculum_path = root / "comparison-vocabulary-curriculum-01/train.jsonl"
    cv_path = root / "conditional-curriculum-01/validation.jsonl"
    expected = {
        train_path: "1f3e0f751b6fdd1115cf6af4ea3ebc13545a1148614eda4f9f56ae8492c80675",
        validation_path: "7be1d1624752938553e13e1eecef5363ca2c56dd264da2915b97ddae8d0b60a8",
        curriculum_path: "41b35f3c2b1161d21ebf123d8ba39b29aa17dc9dc82c5f03ff7f291b9d368c76",
        cv_path: "6ff01e9af03cf481e875dec643e7824ef9fc61718337c7c18774484dea30a34c",
    }
    for path, value in expected.items():
        if sha(path) != value:
            raise ValueError(f"hash mismatch {path}")
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )
    train = _load_examples(train_path, manifest.model_copy(update={"split": "train"}), True)
    validation = _load_examples(
        validation_path, manifest.model_copy(update={"split": "validation"}), True
    )
    _assert_disjoint(train, validation)
    curriculum = [json.loads(x) for x in curriculum_path.read_text(encoding="utf-8").splitlines()]
    cv = [json.loads(x) for x in cv_path.read_text(encoding="utf-8").splitlines()]
    verify_splits(
        [dict(id=e.id, state=e.state) for e in train] + curriculum,
        [dict(id=e.id, state=e.state) for e in validation] + cv,
    )
    original_ids = [e.id for e in train]
    history_paths = [
        root / name / "config.json"
        for name in ["conditional-repair-encoder-coverage-01", "numeric-deterministic-control-01"]
    ]
    consumed = set()
    for path in history_paths:
        config = json.loads(path.read_text(encoding="utf-8"))
        if config["trainable_policy"] != "all_encoder_blocks":
            raise ValueError("incorrect inherited layer policy")
        consumed.update(set(config["train_ids"]) & set(original_ids))
    order = coverage_order(original_ids, [x["id"] for x in curriculum], consumed)
    metadata = {}
    for line in train_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key, definition in json.loads(row["questions"]).items():
            metadata[f"n4ze3m/typed-decisions-synth:{row['state_id']}:{key}"] = (
                row["state_id"],
                definition["type"],
                row["domain"],
            )
    original_selected = [x for x in order if x in metadata]
    output = root / "text-coverage-preflight-01"
    output.mkdir(exist_ok=False)
    (output / "sample-order.json").write_text(json.dumps(order) + "\n", encoding="utf-8")
    # Two42.62MB weights plus best atomic replacement, logs, probabilities and reserve.
    allowance = 160 * 1024**2
    reserve = 512 * 1024**2
    free = shutil.disk_usage(root).free
    result = dict(
        status="DRY_RUN_PASS_STORAGE_BLOCKED" if free < allowance + reserve else "DRY_RUN_PASS",
        training_started=False,
        planned_steps=2048,
        seed=17,
        batch=8,
        original_available=len(original_ids),
        original_previous_all_block_ids=len(consumed),
        original_remaining=len(set(original_ids) - consumed),
        encounters=len(order),
        unique_ids=len(set(order)),
        repeats=0,
        source_counts=dict(
            original=len(original_selected), curriculum=len(order) - len(original_selected)
        ),
        original_states=len({metadata[x][0] for x in original_selected}),
        original_types=dict(Counter(metadata[x][1] for x in original_selected)),
        original_domains=dict(Counter(metadata[x][2] for x in original_selected)),
        order_sha256=sha(output / "sample-order.json"),
        input_hashes={str(p): sha(p) for p in [*expected, *history_paths]},
        disk_free=free,
        output_allowance=allowance,
        reserve=reserve,
        required_free=allowance + reserve,
        limitations=[
            "All original questions were used by inherited last-block adaptation",
            "Unused means absent from prior all-block runs, not lifetime unseen",
            "Curriculum reuses previously trained data; this is not all-new data",
            "This preflight does not authorize execution or claim a learning result",
            "No legacy files, model settings or checkpoints changed",
        ],
    )
    text = json.dumps(result, indent=2) + "\n"
    (output / "result.json").write_text(text, encoding="utf-8")
    Path("docs/pretrained_reuse/R2_TEXT_COVERAGE_PREFLIGHT_20261012.json").write_text(
        text, encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["input_hashes", "original_domains"]}
        )
    )


if __name__ == "__main__":
    main()
