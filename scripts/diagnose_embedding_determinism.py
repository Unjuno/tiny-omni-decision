"""Isolate repeated selected-embedding CUDA backward using real training token IDs."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from scripts.range_generalization import sha, write
from scripts.train_frozen_text_probe import _load_examples
from tiny_omni_decision.schema import DatasetManifest
from tiny_omni_decision.selected_embedding import SelectedEmbeddingDelta


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    out = root / "numeric-embedding-determinism-01"
    if out.exists():
        raise FileExistsError(out)
    config = json.loads((root / "numeric-embedding-training-01/config.json").read_text())
    wanted = config["train_ids"][:8]
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text())
    )
    original = _load_examples(
        root / "text/train.jsonl", manifest.model_copy(update={"split": "train"}), True
    )
    by_id = {
        e.id: dict(id=e.id, state=e.state, question=e.question, options=e.options)
        for e in original
        if e.id in wanted
    }
    for line in (root / "comparison-vocabulary-curriculum-01/train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in wanted:
            by_id[row["id"]] = row
    rows = [by_id[i] for i in wanted]
    tokenizer = AutoTokenizer.from_pretrained(
        "C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model",
        local_files_only=True,
    )
    first = []
    second = []
    for row in rows:
        first.extend(
            [f"Question: {row['question']}\nCandidate answer: {o}" for o in row["options"]]
        )
        second.extend([f"State: {row['state']}"] * len(row["options"]))
    ids = tokenizer(
        first, second, max_length=512, truncation="only_second", padding=True, return_tensors="pt"
    )["input_ids"].cuda()
    coverage = json.loads((root / "numeric-embedding-coverage-01/result.json").read_text())
    torch.manual_seed(17)
    module = SelectedEmbeddingDelta(
        torch.nn.Embedding(30522, 384, device="cuda"), [r["id"] for r in coverage["numeric_tokens"]]
    )
    upstream = torch.randn(*ids.shape, 384, device="cuda")
    results = {}
    for deterministic in [False, True]:
        torch.use_deterministic_algorithms(deterministic)
        gradients = []
        for _ in range(30):
            module.zero_grad(set_to_none=True)
            (module(ids) * upstream).sum().backward()
            gradients.append(module.delta.grad.detach().clone())
        torch.cuda.synchronize()
        results[str(deterministic)] = dict(
            unique_gradient_hashes=len({digest(g) for g in gradients}),
            max_abs=float(max((g - gradients[0]).abs().max() for g in gradients)),
            repeats=30,
            first_gradient_sha256=digest(gradients[0]),
        )
    report = dict(
        results=results,
        input_shape=list(ids.shape),
        input_sha256=digest(ids),
        upstream_sha256=digest(upstream),
        training_ids=wanted,
        synthetic_upstream_gradient=True,
        optimizer_updates=0,
        model_quality_evaluated=False,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        script_sha256=sha(Path(__file__)),
    )
    out.mkdir()
    write(out / "result.json", report)
    write(Path("docs/pretrained_reuse/R2_EMBEDDING_DETERMINISM_20261012.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
