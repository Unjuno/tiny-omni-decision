"""CPU-only real-checkpoint sharing smoke; no training or quality evaluation."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from scripts.train_text_crossencoder import frozen_hash
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file
    from tiny_omni_decision.shared_text import shared_text_features

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model_path = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    checkpoint = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/conditional-repair-encoder-coverage-01/best-delta.pt"
    )
    expected = {
        model_path
        / "model.safetensors": "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae",
        checkpoint: "4024788b1a9d17735612a98da6acb1a65af5d82a9d440b4fddb4ab62efb9d318",
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"source changed: {path}")
    started = time.perf_counter()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    delta = torch.load(checkpoint, weights_only=True, map_location="cpu")
    parameters = dict(model.named_parameters())
    if not set(delta) <= parameters.keys():
        raise ValueError("checkpoint scope mismatch")
    model.load_state_dict(delta, strict=False)
    model.eval().requires_grad_(False)
    identities = [id(p) for p in model.parameters()]
    before = frozen_hash(model)
    prompts = [
        f"Image attribute: {taxonomy}. Candidate class: {option}."
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    ]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[1] > 64:
        raise ValueError("unexpected prompt length")
    with torch.inference_mode():
        logits_before = model(**encoded).logits.clone()
        features = shared_text_features(model, encoded)
        repeated = shared_text_features(model, encoded)
        logits_after = model(**encoded).logits
    assert torch.equal(features, repeated)
    assert torch.equal(logits_before, logits_after)
    assert identities == [id(p) for p in model.parameters()]
    assert before == frozen_hash(model)
    assert features.shape == (40, 384)
    result = dict(
        status="PASS_IMPLEMENTATION_ONLY",
        device=str(next(model.parameters()).device),
        parameters=sum(p.numel() for p in model.parameters()),
        added_parameters=0,
        feature_shape=list(features.shape),
        logits_unchanged=True,
        repeated_features_exact=True,
        frozen_sha256=before,
        max_tokens=encoded["input_ids"].shape[1],
        seconds=time.perf_counter() - started,
        source_hashes={str(p): h for p, h in expected.items()},
        checkpoint_role="existing initialization, not current experiment selection",
        training=False,
        quality_evaluation=False,
    )
    args.output.mkdir(parents=True)
    torch.save(features, args.output / "features.pt")
    result["features_sha256"] = sha256_file(args.output / "features.pt")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
