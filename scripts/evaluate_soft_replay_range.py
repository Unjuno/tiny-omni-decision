import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scripts.range_generalization import sha, write
from scripts.train_frozen_text_probe import _metrics

root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
parser = argparse.ArgumentParser()
parser.add_argument("--parent", action="store_true")
args = parser.parse_args()
out = root / ("coverage-parent-range-01" if args.parent else "soft-replay-range-01")
out.mkdir(exist_ok=False)
inputs = root / "range-generalization-inputs-01/inputs.jsonl"
assert sha(inputs) == "7be0e5e6398a313051527122c95b928629b774080336c4a03e628b11c188a2a6"
rows = [json.loads(line) for line in inputs.read_text().splitlines()]
checkpoint = root / "context-soft-replay-context-run-01-all-encoder/final-encoder.pt"
expected = "f42842256697769d03ec5a35f7127b36bd852405408aaba23d8baa6a1962c457"
if args.parent:
    checkpoint = root / "text-coverage-training-01/best-delta.pt"
    expected = "b063f124e59c8a4c1b09c7bf2a8423846f73489d7ce8cd6649735745bc6cebf5"
assert sha(checkpoint) == expected
write(
    out / "protocol.json",
    {
        "input_sha256": sha(inputs),
        "checkpoint_sha256": expected,
        "scope": "fixed observed-development transfer; no training or selection",
        "script_sha256": sha(Path(__file__)),
    },
)


torch.set_num_threads(4)
start = time.perf_counter()
model_path = "C:/CodexArtifacts/pretrained-reuse-r1/crossencoder-minilm-l6-audit-01/model"
assert sha(Path(model_path) / "model.safetensors") == (
    "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
m = (
    AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    .cuda()
    .eval()
    .requires_grad_(False)
)
delta = torch.load(checkpoint, weights_only=True, map_location="cpu")
assert set(delta) == {
    n for n, _ in m.named_parameters() if n.startswith(("bert.encoder.layer.", "classifier."))
}
m.load_state_dict(delta, strict=False)
results = {}
with torch.inference_mode():
    for variant in sorted({r["variant"] for r in rows}):
        selected = [r for r in rows if r["variant"] == variant]
        scores = []
        for off in range(0, len(selected), 8):
            batch = selected[off : off + 8]
            first = [
                f"Question: {r['question']}\nCandidate answer: {o}"
                for r in batch
                for o in r["options"]
            ]
            second = [f"State: {r['state']}" for r in batch for _ in r["options"]]
            token = tokenizer(first, second, padding=True, truncation=False, return_tensors="pt")
            assert token["input_ids"].shape[1] <= 512
            logits = m(**token.to("cuda")).logits[:, 0].cpu()
            assert torch.isfinite(logits).all()
            scores.extend(logits.split([len(r["options"]) for r in batch]))
        target = [r["options"].index(r["target"]) for r in selected]
        results[variant] = _metrics(scores, target)
        write(
            out / (variant + "-predictions.json"),
            [
                dict(
                    id=r["id"],
                    target=r["target"],
                    options=r["options"],
                    probabilities=s.softmax(-1).tolist(),
                )
                for r, s in zip(selected, scores, strict=True)
            ],
        )
        print(variant, results[variant], flush=True)
assert sha(checkpoint) == expected
write(
    out / "result.json",
    {
        "status": "COMPLETE_FIXED_FACTORIAL",
        "metrics": results,
        "seconds": time.perf_counter() - start,
        "peak_vram": torch.cuda.max_memory_allocated(),
        "source_checkpoint_unchanged": True,
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    },
)
