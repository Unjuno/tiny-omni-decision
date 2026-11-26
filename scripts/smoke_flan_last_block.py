"""Two-update FLAN last-decoder-block training mechanics, no quality claim."""

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
rows = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()][:8]
out = Path("C:/CodexArtifacts/pretrained-reuse-r2/flan-last-block-smoke-01")
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
losses = []
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
for step in range(2):
    optimizer.zero_grad()
    values = []
    for row in rows[step * 4 : (step + 1) * 4]:
        options = row["options"]
        labels = [chr(65 + i) for i in range(len(options))]
        text = (
            "State: "
            + row["state"]
            + "\nCondition: "
            + row["question"]
            + "\n"
            + "\n".join(f"{label}: {o}" for label, o in zip(labels, options, strict=True))
            + "\nAnswer with the option letter."
        )
        tokens = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        assert all(len(t) == 1 for t in tokens)
        inputs = tokenizer(text, return_tensors="pt", truncation=False).to("cuda")
        assert inputs["input_ids"].shape[1] <= 512
        logits = model(
            **inputs,
            decoder_input_ids=torch.zeros((1, 1), dtype=torch.long, device="cuda"),
            use_cache=False,
        ).logits[0, 0, [t[0] for t in tokens]]
        target = torch.tensor([options.index(row["target"])], device="cuda")
        loss = torch.nn.functional.cross_entropy(logits[None], target)
        assert torch.isfinite(loss)
        (loss / 4).backward()
        values.append(loss.item())
    norm = torch.nn.utils.clip_grad_norm_(selected.values(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    losses.append(dict(step=step + 1, loss=sum(values) / 4, gradient_norm=float(norm)))
state = namespace["state"]
assert all(
    torch.equal(p.detach().cpu(), state[n])
    for n, p in model.named_parameters()
    if not p.requires_grad
)
changed = [n for n, p in selected.items() if not torch.equal(p.detach().cpu(), initial[n])]
assert changed
report = dict(
    status="PASS_TRAINING_MECHANICS_ONLY",
    updates=2,
    examples=8,
    seed=17,
    history=losses,
    trainable_parameters=sum(p.numel() for p in selected.values()),
    total_parameters=sum(p.numel() for p in model.parameters()),
    frozen_parameters_unchanged=True,
    changed_names=changed,
    seconds=time.perf_counter() - start,
    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    sample_ids=[r["id"] for r in rows],
    validation_used=False,
    weights_saved=False,
)
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
