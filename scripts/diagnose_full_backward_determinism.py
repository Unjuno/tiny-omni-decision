"""Isolate repeated selected-embedding CUDA backward using real training token IDs."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.range_generalization import sha, write
from scripts.train_conditional_repair import configure_repair_policy
from scripts.train_frozen_text_probe import _load_examples
from scripts.train_text_crossencoder import choice_loss
from tiny_omni_decision.schema import DatasetManifest
from tiny_omni_decision.selected_embedding import SelectedEmbeddingDelta


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    out = root / "numeric-full-backward-determinism-01"
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
        e.id: dict(id=e.id, state=e.state, question=e.question, options=e.options, target=e.target)
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
    torch.set_num_threads(4)
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        .cuda()
        .eval()
    )
    names = configure_repair_policy(model, "all_encoder_blocks")
    weights = root / "conditional-repair-encoder-coverage-01/best-delta.pt"
    assert sha(weights) == "4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318"
    saved = torch.load(weights, weights_only=True, map_location="cpu")
    assert set(names) == set(saved)
    model.load_state_dict(saved, strict=False)
    model.set_input_embeddings(
        SelectedEmbeddingDelta(
            model.get_input_embeddings(), [r["id"] for r in coverage["numeric_tokens"]]
        )
    )
    inputs = tokenizer(
        first, second, max_length=512, truncation="only_second", padding=True, return_tensors="pt"
    ).to("cuda")
    targets = torch.tensor([r["options"].index(r["target"]) for r in rows], device="cuda")
    widths = [len(r["options"]) for r in rows]
    results = {}
    for deterministic in [False, True]:
        torch.use_deterministic_algorithms(deterministic)
        reference = None
        changed = {}
        logits_hashes = []
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            logits = model(**inputs).logits[:, 0]
            logits_hashes.append(digest(logits))
            choice_loss(logits, widths, targets).backward()
            gradients = {
                n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad
            }
            if reference is None:
                reference = gradients
            else:
                for n, g in gradients.items():
                    difference = float((g - reference[n]).abs().max())
                    if difference:
                        changed[n] = max(changed.get(n, 0), difference)
        results[str(deterministic)] = dict(
            repeats=5,
            unique_logits_hashes=len(set(logits_hashes)),
            varying_gradient_max_abs=changed,
        )
    report = dict(
        results=results,
        input_shape=list(ids.shape),
        input_sha256=digest(ids),
        attention_backend=model.config._attn_implementation,
        training_ids=wanted,
        synthetic_upstream_gradient=False,
        optimizer_updates=0,
        model_quality_evaluated=False,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        script_sha256=sha(Path(__file__)),
    )
    out.mkdir()
    write(out / "result.json", report)
    write(Path("docs/pretrained_reuse/R2_FULL_BACKWARD_DETERMINISM_20261012.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
