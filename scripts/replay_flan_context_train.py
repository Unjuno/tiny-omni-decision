"""Fixed-budget FLAN final decoder block context adaptation."""

import hashlib
import json
import time
from pathlib import Path

import torch

loader = Path("C:/CodexArtifacts/pretrained-reuse-r1/audit_flan_untied_runtime.py")
assert (
    hashlib.sha256(loader.read_bytes()).hexdigest()
    == "6286765156f246f915604e7260e62699f52477a2f4b7c94776f34d87f191658f"
)
namespace = {}
exec(compile(loader.read_text().split("m=build();tokenizer=")[0], str(loader), "exec"), namespace)
corpus = Path("C:/CodexArtifacts/pretrained-reuse-r2/context-robustness-corpus-01/train.jsonl")
assert (
    hashlib.sha256(corpus.read_bytes()).hexdigest()
    == "a0116e4e2c5024fe39e113461c3b286da3d76b1f8b11940dc195fd5594c2982b"
)
rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
out = Path("C:/CodexArtifacts/pretrained-reuse-r2/flan-last-block-train-replay-01")
out.mkdir(exist_ok=False)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)
model = namespace["build"]().cuda()
block = model.decoder.block[-1]
block.requires_grad_(True)
# Eval mode deliberately disables dropout while allowing gradients.
model.eval()
selected = {n: p for n, p in model.named_parameters() if p.requires_grad}
assert selected and all(n.startswith("decoder.block.7.") for n in selected)
initial = {n: p.detach().cpu().clone() for n, p in selected.items()}
tokenizer = namespace["AutoTokenizer"].from_pretrained(namespace["root"], local_files_only=True)
optimizer = torch.optim.AdamW(selected.values(), lr=2e-5, weight_decay=0.01)

import random  # noqa: E402

import numpy as np  # noqa: E402

from scripts.train_sensor_subjects import metrics  # noqa: E402

dev_path = corpus.with_name("development.jsonl")
assert (
    hashlib.sha256(dev_path.read_bytes()).hexdigest()
    == "310a0c7d3daad92dddb807a58d9e1483d06fd1b14bca67b4b29e9293520d8e4c"
)
dev = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()]
assert not {r["group"] for r in rows} & {r["group"] for r in dev}
random.Random(17).shuffle(rows)
rows = rows[:3072]
assert len({r["id"] for r in rows}) == 3072
order = [r["id"] for r in rows]
(out / "order.json").write_text(json.dumps(order))
protocol = dict(
    seed=17,
    steps=384,
    batch_size=8,
    lr=2e-5,
    weight_decay=0.01,
    dropout=False,
    clip=1.0,
    trainable="decoder.block.7",
    selector="earliest minimum pooled development NLL including step0",
    validation_steps=[0, 128, 256, 384],
    order_sha256=hashlib.sha256((out / "order.json").read_bytes()).hexdigest(),
    corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
    development_sha256=hashlib.sha256(dev_path.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
(out / "protocol.json").write_text(json.dumps(protocol, indent=2))
maximum = max(len(r["options"]) for r in rows + dev)
label_tokens = [tokenizer.encode(chr(65 + i), add_special_tokens=False) for i in range(maximum)]
assert all(len(t) == 1 for t in label_tokens)
label_ids = [t[0] for t in label_tokens]


def prepare(items):
    texts = []
    for row in items:
        texts.append(
            "State: "
            + row["state"]
            + "\nCondition: "
            + row["question"]
            + "\n"
            + "\n".join(f"{chr(65 + i)}: {o}" for i, o in enumerate(row["options"]))
            + "\nAnswer with the option letter."
        )
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
    assert inputs["input_ids"].shape[1] <= 512
    mask = torch.tensor(
        [[i < len(r["options"]) for i in range(maximum)] for r in items], device="cuda"
    )
    target = torch.tensor([r["options"].index(r["target"]) for r in items], device="cuda")
    return inputs, mask, target


def forward(items):
    inputs, mask, target = prepare(items)
    logits = model(
        **inputs,
        decoder_input_ids=torch.zeros((len(items), 1), dtype=torch.long, device="cuda"),
        use_cache=False,
    ).logits[:, 0, label_ids]
    assert torch.isfinite(logits).all()
    return logits.masked_fill(~mask, float("-inf")), target


original = out.parent / "flan-last-block-context-01"
original_protocol = json.loads((original / "protocol.json").read_text())
assert protocol["order_sha256"] == original_protocol["order_sha256"]
weights = torch.load(original / "final.pt", map_location="cpu", weights_only=True)
assert set(weights) == set(selected)
with torch.no_grad():
    for name, parameter in selected.items():
        parameter.copy_(weights[name])
model.requires_grad_(False)
start = time.perf_counter()
probabilities = []
targets = []
with torch.inference_mode():
    for offset in range(0, len(rows), 16):
        logits, target = forward(rows[offset : offset + 16])
        probabilities.extend(logits.softmax(-1).cpu().tolist())
        targets.extend(target.cpu().tolist())
p = np.array(probabilities)
y = np.array(targets)
result = dict(
    status="FIXED_CONSUMED_TRAIN_REPLAY",
    metrics=metrics(p, y),
    contexts={},
    seconds=time.perf_counter() - start,
    checkpoint_sha256=hashlib.sha256((original / "final.pt").read_bytes()).hexdigest(),
    order_sha256=protocol["order_sha256"],
)
for context in sorted({r["context"] for r in rows}):
    mask = np.array([r["context"] == context for r in rows])
    result["contexts"][context] = metrics(p[mask], y[mask])
with torch.inference_mode():
    logits, _ = forward(dev[:16])
    assert np.array_equal(
        logits.softmax(-1).cpu().numpy(),
        np.load(original / "validation-0384.npz")["probabilities"][:16],
    )
result["validation_replay_first16_exact"] = True
np.savez(out / "predictions.npz", probabilities=p, targets=y)
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
