"""Sixteen-update training trace with512 versus2048-step materialized input plans."""

import argparse
import hashlib
import json
import os
import random
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


def tensor_hashes(model, gradients=False):
    result = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            value = p.grad if gradients else p
            if value is None:
                raise ValueError("missing gradient")
            result[name] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, choices=[512, 2048], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    source = json.loads(
        (root / "conditional-repair-comparison-vocabulary-01/config.json").read_text()
    )
    for path, expected in source["paths"].items():
        assert sha(Path(path)) == expected, path
    manifest = DatasetManifest.model_validate(
        yaml.safe_load(Path("manifests/dataset.example.yaml").read_text())
    )
    all_rows = {}
    for split in ["train", "validation"]:
        examples = _load_examples(
            root / f"text/{split}.jsonl", manifest.model_copy(update={"split": split}), True
        )
        all_rows.update(
            {
                e.id: dict(
                    id=e.id, state=e.state, question=e.question, options=e.options, target=e.target
                )
                for e in examples
            }
        )
        path = root / (
            "comparison-vocabulary-curriculum-01/train.jsonl"
            if split == "train"
            else "conditional-curriculum-01/validation.jsonl"
        )
        all_rows.update({r["id"]: r for r in map(json.loads, path.read_text().splitlines())})
    rows = [all_rows[i] for i in source["train_ids"][: args.budget * 8]]
    validation = [[all_rows[i] for i in ids] for ids in source["validation_ids"]]
    random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.set_num_threads(4)
    path = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    def encode(data):
        output = []
        for row in data:
            first = [f"Question: {row['question']}\nCandidate answer: {o}" for o in row["options"]]
            batch = tokenizer(
                first,
                [f"State: {row['state']}"] * len(first),
                max_length=512,
                truncation="only_second",
            )
            output.append([{k: v[i] for k, v in batch.items()} for i in range(len(first))])
        return output

    encoded = encode(rows)
    val_encoded = [encode(data) for data in validation]
    model = (
        AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        .cuda()
        .eval()
    )
    names = configure_repair_policy(model, "all_encoder_blocks")
    checkpoint = root / "conditional-repair-encoder-coverage-01/best-delta.pt"
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert set(names) == set(saved)
    model.load_state_dict(saved, strict=False)
    coverage = json.loads((root / "numeric-embedding-coverage-01/result.json").read_text())
    model.set_input_embeddings(
        SelectedEmbeddingDelta(
            model.get_input_embeddings(), [r["id"] for r in coverage["numeric_tokens"]]
        )
    )
    if args.control:
        model.get_input_embeddings().delta.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.01)
    args.output.mkdir()
    config = dict(
        budget=args.budget,
        numeric_delta_trainable=not args.control,
        actual_updates=16,
        seed=17,
        first128_input_sha256=hashlib.sha256(
            json.dumps(encoded[:128], sort_keys=True).encode()
        ).hexdigest(),
        initial_parameter_hashes=tensor_hashes(model),
        validation_tokenized=[len(v) for v in val_encoded],
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        attention_backend=model.config._attn_implementation,
        torch=torch.__version__,
        script_sha256=sha(Path(__file__)),
    )
    write(args.output / "config.json", config)
    history = []
    for step in range(1, 17):
        offset = (step - 1) * 8
        batch = encoded[offset : offset + 8]
        inputs = tokenizer.pad(
            [v for group in batch for v in group], padding=True, return_tensors="pt"
        ).to("cuda")
        logits = model(**inputs).logits[:, 0]
        target = torch.tensor(
            [r["options"].index(r["target"]) for r in rows[offset : offset + 8]], device="cuda"
        )
        loss = choice_loss(logits, [len(group) for group in batch], target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        gradients = tensor_hashes(model, True)
        optimizer.step()
        history.append(
            dict(
                step=step,
                loss=float(loss.detach()),
                norm=float(norm),
                gradient_hashes=gradients,
                parameter_hashes=tensor_hashes(model),
            )
        )
    write(args.output / "trace.json", history)
    print(
        json.dumps(
            dict(
                status="PASS_DIAGNOSTIC_ONLY",
                budget=args.budget,
                numeric_delta_trainable=not args.control,
                updates=16,
                loss=history[-1]["loss"],
            )
        )
    )


if __name__ == "__main__":
    main()
