"""One fixed development-transfer comparison after completed coverage training."""

import json
import subprocess
from pathlib import Path

from scripts.train_numeric_deterministic import verify_splits


def validate_transfer_disjoint(train, rows):
    verify_splits(train, rows)
    if {r["group"] for r in train} & {r["group"] for r in rows}:
        raise ValueError("latent state overlap")


def main():
    import yaml

    from scripts.build_conditional_curriculum import verify
    from scripts.build_vocabulary_transfer import VARIANTS
    from scripts.evaluate_vocabulary_transfer import opposing_pairs
    from scripts.range_generalization import evaluate, sha, write
    from scripts.report_text_coverage import verify_finished
    from scripts.train_frozen_text_probe import _load_examples
    from tiny_omni_decision.schema import DatasetManifest

    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    run = root / "text-coverage-training-01"
    data = root / "vocabulary-transfer-inputs-01"
    out = root / "coverage-transfer-eval-01"
    selection = root / "coverage-transfer-eval-01-selection.json"
    if out.exists() or selection.exists():
        raise FileExistsError("preserve prior output")

    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    result = read(run / "result.json")
    history = read(run / "history.json")
    config = read(run / "config.json")
    verify_finished(result, history)
    if (
        result["best_step"] != 1792
        or result["best_sha256"]
        != "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5"
    ):
        raise ValueError("fixed selection changed")
    for p, h in config["input_hashes"].items():
        if sha(Path(p)) != h:
            raise ValueError("training inputs changed")
    input_sha = "1a825b413e32f319dead47ccbef07ef3b36f2c391ca2c197a694aaeb4ed54dd2"
    if sha(data / "inputs.jsonl") != input_sha:
        raise ValueError("transfer inputs changed")
    rows = [json.loads(x) for x in (data / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        verify(row)
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text(encoding="utf-8"))
    )
    train = [
        dict(id=e.id, state=e.state, group=e.source_record_id.split(":")[0])
        for e in _load_examples(
            root / "text/train.jsonl", manifest.model_copy(update={"split": "train"}), True
        )
    ]
    train += [
        json.loads(x)
        for x in (root / "comparison-vocabulary-curriculum-01/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    validate_transfer_disjoint(train, rows)
    checkpoints = {
        "parent": (
            "numeric-deterministic-control-01",
            "0b785581bc5db5db01a9b39494af41b32d26fc6958d2c6191ccd74edcf016882",
        ),
        "candidate": (run.name, result["best_sha256"]),
    }
    write(
        selection,
        dict(
            checkpoints=checkpoints,
            input_sha256=input_sha,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            wrapper_sha256=sha(Path(__file__)),
            selected_step=1792,
            training=False,
            training_state_overlap=0,
            training_id_overlap=0,
            targets_verified=len(rows),
            selection_evidence={
                p: sha(run / p) for p in ["result.json", "history.json", "config.json"]
            },
        ),
    )
    evaluate(
        root,
        Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model"),
        data,
        input_sha,
        out,
        checkpoints,
        VARIANTS,
        10240,
    )
    pairs = {}
    for name in checkpoints:
        pairs[name] = {
            variant: opposing_pairs(
                [r for r in rows if r["variant"] == variant],
                read(out / f"{name}-{variant}-predictions.json"),
            )
            for variant in VARIANTS
        }
    write(out / "opposing-pairs.json", pairs)
    for directory, h in checkpoints.values():
        if sha(root / directory / "best-delta.pt") != h:
            raise ValueError("source weight changed")
    if sha(data / "inputs.jsonl") != input_sha:
        raise ValueError("input changed during evaluation")
    write(
        out / "completion.json",
        dict(status="PASS", frozen_sources_verified=True, opposing_pairs_verified=True),
    )


if __name__ == "__main__":
    main()
