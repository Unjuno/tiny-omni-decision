"""Pinned real-model plumbing check; no quality inference or training."""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.range_generalization import sha, write
from tiny_omni_decision.decision import DecisionOutput
from tiny_omni_decision.joint_candidates import encode_joint, joint_logits


def main():
    base = Path("C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model")
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    delta = root / "text-coverage-training-01/best-delta.pt"
    output = root / "joint-candidate-smoke-01"
    expected = "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5"
    if sha(delta) != expected:
        raise ValueError("delta changed")
    if sha(base / "model.safetensors") != (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    ):
        raise ValueError("base changed")
    output.mkdir(exist_ok=False)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(base, local_files_only=True)
    model.load_state_dict(torch.load(delta, weights_only=True, map_location="cpu"), strict=False)
    model = model.eval().to("cuda")
    count = sum(p.numel() for p in model.parameters())
    assert count == 22713601
    cases = [("Which item costs less?", ["A", "B"]),
             ("Which item has the greatest cost?", ["A", "B", "C"])]
    state = json.dumps([dict(id="A", cost=3), dict(id="B", cost=7), dict(id="C", cost=9)])
    encoded = [encode_joint(tokenizer, q, state, options) for q, options in cases]
    batch = tokenizer.pad([r[0] for r in encoded], padding=True, return_tensors="pt").to("cuda")
    positions = [r[1] for r in encoded]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        first = joint_logits(model, batch, positions)
        torch.cuda.synchronize()
        started = time.perf_counter()
        second = joint_logits(model, batch, positions)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    responses = []
    for i, logits in enumerate(second):
        assert torch.isfinite(logits).all()
        probabilities = logits.softmax(-1)
        prediction = int(probabilities.argmax())
        result = DecisionOutput(logits, probabilities, prediction, float(probabilities[prediction]))
        responses.append(json.loads(result.to_json(f"question_{i}", cases[i][1])))
    assert sha(delta) == expected
    write(output / "result.json", dict(
        status="PASS_PLUMBING_ONLY", parameters=count, added_parameters=0,
        repeated_logits_exact=True, warm_batch_seconds=seconds,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        lengths=[len(r[0]["input_ids"]) for r in encoded],
        truncation=[r[2] for r in encoded], responses=responses,
        training=False, quality_claim=False, source_delta_sha256=expected,
        runner_sha256=sha(Path(__file__)),
    ))
    print((output / "result.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
