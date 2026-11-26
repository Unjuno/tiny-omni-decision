"""Compare the completed training-order intervention without further tuning."""
import json
from pathlib import Path

import yaml

from scripts.compare_numeric_embedding_predictions import compare
from scripts.range_generalization import sha, write
from scripts.shuffle_training_options import shuffled_options
from scripts.train_frozen_text_probe import _load_examples
from scripts.verify_wording_result import verify_run
from tiny_omni_decision.schema import DatasetManifest


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")

    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    old, new = root / "joint-readout-training-01", root / "joint-shuffled-training-01"
    triples = [tuple(read(p / f"{n}.json") for n in ("config", "result", "history"))
               for p in (old, new)]
    for triple in triples:
        verify_run(*triple)
    a, b = [t[0] for t in triples]
    for key in ("train_ids", "parent_ids", "validation_ids", "seed", "steps", "lr", "batch",
                "weight_decay", "clip", "dropout", "deterministic", "checkpoint_sha256",
                "tokenization", "input_hashes", "selector", "trainable_parameters"):
        if a[key] != b[key]:
            raise ValueError(f"uncontrolled difference {key}")
    for source in ("original", "curriculum"):
        if triples[0][2][0][source] != triples[1][2][0][source]:
            raise ValueError("initial metric mismatch")
    manifest = DatasetManifest.model_validate(yaml.safe_load(
        Path("manifests/dataset.example.yaml").read_text(encoding="utf-8")))
    lookup = {e.id: dict(options=e.options, target=e.target) for e in _load_examples(
        root / "text/train.jsonl", manifest.model_copy(update={"split": "train"}), True)}
    for line in (root / "paired-wording-training-views-02/train-views.jsonl").read_text(
        encoding="utf-8").splitlines():
        row = json.loads(line)
        lookup[row["id"]] = row
    parents = [lookup[k] for k in a["train_ids"]]
    expected = [r["options"] for r in shuffled_options(parents)]
    assert expected == b["training_options"]
    changed = sum(p["options"] != v for p,v in zip(parents, expected, strict=True))
    for p,t in zip((old,new),triples,strict=True):
        for which in ("best", "final"):
            assert sha(p / f"{which}-delta.pt") == t[1][f"{which}_sha256"]
    comparisons = {s:compare(
        read(old / f"predictions-{triples[0][1]['best_step']}-{s}.json"),
        read(new / f"predictions-{triples[1][1]['best_step']}-{s}.json"))
        for s in ("original", "curriculum")}
    diagnostic = root / "joint-shuffled-order-diagnostic-01"
    diag = read(diagnostic / "result.json")
    assert diag["status"] == "PASS_DIAGNOSTIC_ONLY"
    output = Path("docs/pretrained_reuse/R2_JOINT_SHUFFLED_RESULTS_20261012.json")
    if output.exists():
        raise FileExistsError(output)
    write(output, dict(
        result=triples[1][1], history=triples[1][2],
        config={k:v for k,v in b.items() if k not in (
            "train_ids", "parent_ids", "validation_ids", "training_options")},
        training_permutation_verified=True, changed_encounters=changed,
        unchanged_encounters=8192-changed, comparisons=comparisons, diagnostic=diag,
        hashes={str(p):sha(p) for folder in (new,diagnostic) for p in folder.iterdir()
                if p.is_file()},
    ))
    print("Verified completed run, all realized permutations, hashes, and comparison.")


if __name__ == "__main__":
    main()
