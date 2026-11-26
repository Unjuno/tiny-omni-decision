"""Pinned compact NLI CPU load/count audit; not a task-quality result."""

import hashlib
import json
import platform
import struct
import time
from collections import Counter
from pathlib import Path


def serializable_loading_info(info):
    return {k: sorted(v) if isinstance(v, set) else v for k, v in info.items()}


def main():
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    root = Path("C:/CodexArtifacts/pretrained-reuse-r1/deberta-xsmall-zeroshot-audit-01")
    path = root / "model"
    output = root / "load-audit.json"
    if output.exists():
        raise FileExistsError("preserve existing audit")
    p = path / "model.safetensors"
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != "56b89067a4e60362f498f39b7067020911618fad37d64fa23a5f959a04fd4080":
        raise ValueError("weight hash mismatch")
    with p.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    count = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        size = 1
        for d in entry["shape"]:
            size *= d
        count += size
    torch.set_num_threads(2)
    start = time.perf_counter()
    model, info = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(info.values()):
        raise ValueError(f"loading mismatch {info}")
    model.cpu().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    inputs = tokenizer(
        ["The box is red.", "The box is red."],
        ["The box has a red color.", "The box is blue."],
        return_tensors="pt",
        padding=True,
    )
    with torch.inference_mode():
        a = model(**inputs).logits
        b = model(**inputs).logits
    if not torch.isfinite(a).all() or not torch.equal(a, b):
        raise ValueError("nonfinite or nonrepeatable smoke")
    actual = sum(p.numel() for p in model.parameters())
    if actual != count or actual != 70830722:
        raise ValueError("parameter reconciliation")
    groups = Counter()
    for name, param in model.named_parameters():
        group = (
            "embeddings"
            if name.startswith("deberta.embeddings.")
            else "encoder"
            if name.startswith("deberta.encoder.")
            else "pooler_classifier"
        )
        groups[group] += param.numel()
    report = dict(
        status="PASS_LOAD_SMOKE_ONLY",
        repository="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        revision="262ae02f29173eec1c250f90804dc7edc677dcff",
        parameters=actual,
        header_elements=count,
        parameters_by_group=dict(groups),
        resident_fp32_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        checkpoint_bytes=p.stat().st_size
        if isinstance(p, Path)
        else (path / "model.safetensors").stat().st_size,
        logits=a.tolist(),
        id2label=model.config.id2label,
        repeat_exact=True,
        loading_info=serializable_loading_info(info),
        seconds=time.perf_counter() - start,
        device="cpu",
        threads=2,
        torch=torch.__version__,
        transformers=transformers.__version__,
        python=platform.python_version(),
        dtype=str(next(model.parameters()).dtype),
        tokenizer_class=type(tokenizer).__name__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        weights_sha256=digest,
        limitation="Two synthetic sentence pairs only; no benchmark or numeric-reasoning claim",
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
