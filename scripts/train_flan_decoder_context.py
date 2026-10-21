"""Fixed-budget FLAN decoder-block context adaptation; encoder frozen."""

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
out = Path("C:/CodexArtifacts/pretrained-reuse-r2/flan-decoder-context-01")
out.mkdir(exist_ok=False)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)
model = namespace["build"]().cuda()
block = model.decoder.block
block.requires_grad_(True)
# Eval mode deliberately disables dropout while allowing gradients.
model.eval()
selected = {n: p for n, p in model.named_parameters() if p.requires_grad}
assert selected and all(n.startswith("decoder.block.") for n in selected)
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
    trainable="decoder.block.*",
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


history = []
best = float("inf")
best_step = None
start = time.perf_counter()
torch.cuda.reset_peak_memory_stats()


def evaluate(step):
    global best, best_step
    ps = []
    ys = []
    with torch.inference_mode():
        for offset in range(0, len(dev), 16):
            logits, target = forward(dev[offset : offset + 16])
            ps.extend(logits.softmax(-1).cpu().tolist())
            ys.extend(target.cpu().tolist())
    p = np.array(ps)
    y = np.array(ys)
    report = dict(step=step, metrics=metrics(p, y), contexts={})
    for context in sorted({r["context"] for r in dev}):
        index = np.array([r["context"] == context for r in dev])
        report["contexts"][context] = metrics(p[index], y[index])
    np.savez(out / f"validation-{step:04}.npz", probabilities=p, targets=y)
    value = report["metrics"]["nll"]
    if value < best:
        best = value
        best_step = step
        torch.save({n: p.detach().cpu() for n, p in selected.items()}, out / "best.pt")
    report["best_step"] = best_step
    history.append(report)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(
        json.dumps(
            dict(step=step, accuracy=report["metrics"]["accuracy"], nll=value, best_step=best_step)
        ),
        flush=True,
    )


evaluate(0)
training = []
for step in range(1, 385):
    optimizer.zero_grad()
    logits, target = forward(rows[(step - 1) * 8 : step * 8])
    loss = torch.nn.functional.cross_entropy(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(selected.values(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    training.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm)))
    if step % 128 == 0:
        evaluate(step)
torch.save({n: p.detach().cpu() for n, p in selected.items()}, out / "final.pt")
assert all(
    torch.equal(p.detach().cpu(), namespace["state"][n])
    for n, p in model.named_parameters()
    if not p.requires_grad
)
best_weights = torch.load(out / "best.pt", weights_only=True)
with torch.no_grad():
    for n, p in selected.items():
        p.copy_(best_weights[n])
with torch.inference_mode():
    logits, _ = forward(dev[:16])
    saved = np.load(out / f"validation-{best_step:04}.npz")["probabilities"][:16]
    assert np.array_equal(logits.softmax(-1).cpu().numpy(), saved)
(out / "training.json").write_text(json.dumps(training))
report = dict(
    status="COMPLETE_DEVELOPMENT_NOT_PRODUCT",
    best_step=best_step,
    steps=384,
    examples=3072,
    unique_groups=len({r["group"] for r in rows}),
    trainable_parameters=sum(p.numel() for p in selected.values()),
    parameters=sum(p.numel() for p in model.parameters()),
    frozen_unchanged=True,
    best_reload_exact_first16=True,
    seconds=time.perf_counter() - start,
    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    history=history,
)
(out / "result.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)
