"""CPU-only token-budget audit for a proposed joint-readout comparison."""
import json
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from scripts.range_generalization import sha, write
from scripts.train_frozen_text_probe import _load_examples
from tiny_omni_decision.joint_candidates import encode_joint, encode_matched
from tiny_omni_decision.schema import DatasetManifest


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    output = root / "joint-context-preflight-02"
    if output.exists():
        raise FileExistsError(output)
    config_path = root / "paired-wording-control-01/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parent_ids = set(config["parent_ids"])
    manifest = DatasetManifest.model_validate(yaml.safe_load(
        Path("manifests/dataset.example.yaml").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(
        "C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model",
        local_files_only=True,
    )
    groups, hashes = {}, {str(config_path): sha(config_path)}
    for split in ("train", "validation"):
        p = root / f"text/{split}.jsonl"
        hashes[str(p)] = sha(p)
        rows = [dict(id=e.id, question=e.question, state=e.state, options=e.options)
                for e in _load_examples(p, manifest.model_copy(update={"split": split}), True)]
        if split == "train":
            rows = [r for r in rows if r["id"] in parent_ids]
        else:
            assert [r["id"] for r in rows] == config["validation_ids"][0]
        groups["original-"+split] = rows
        p = root / ("comparison-vocabulary-curriculum-01/train.jsonl" if split == "train"
                    else "conditional-curriculum-01/validation.jsonl")
        hashes[str(p)] = sha(p)
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        if split == "train":
            rows = [r for r in rows if r["id"] in parent_ids]
        else:
            assert [r["id"] for r in rows] == config["validation_ids"][1]
        groups["curriculum-"+split] = rows
    results = {}
    for name, rows in groups.items():
        counts = Counter()
        failures, truncations = [], []
        for row in rows:
            counts["questions"] += 1
            state_length = len(tokenizer("State: " + row["state"],
                                         add_special_tokens=False)["input_ids"])
            first = [f"Question: {row['question']}\nCandidate answer: {o}" for o in row["options"]]
            pair_rooms = [512 - len(v) - 3 for v in tokenizer(
                first, add_special_tokens=False)["input_ids"]]
            counts["independent_state_truncated"] += int(min(pair_rooms) < state_length)
            try:
                batch, markers, meta = encode_joint(
                    tokenizer, row["question"], row["state"], row["options"])
            except ValueError as error:
                failures.append(dict(id=row["id"], error=str(error)))
                continue
            assert len(markers) == len(row["options"])
            (joint, _), pairs, matched_meta = encode_matched(
                tokenizer, row["question"], row["state"], row["options"])
            evidence = []
            for encoded in [joint, *pairs]:
                evidence.append([v for v, t in zip(encoded["input_ids"],
                    encoded["token_type_ids"], strict=True) if t == 1][:-1])
                assert len(encoded["input_ids"]) <= 512
            assert all(v == evidence[0] for v in evidence)
            counts["matched_evidence_verified"] += 1
            counts["matched_state_truncated"] += int(matched_meta["state_truncated"])
            counts["joint_state_truncated"] += int(meta["state_truncated"])
            counts["joint_unpadded_tokens"] += len(batch["input_ids"])
            if meta["state_truncated"]:
                truncations.append(dict(id=row["id"], **meta,
                                        independent_min_kept=min(state_length, min(pair_rooms))))
        results[name] = dict(counts=counts, failures=failures, truncations=truncations)
    output.mkdir()
    write(output / "result.json", dict(groups=results, hashes=hashes,
          source_sha256=sha(Path(__file__)), model_loaded=False, training_started=False))
    print(json.dumps({k: dict(v["counts"], failures=len(v["failures"]))
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
