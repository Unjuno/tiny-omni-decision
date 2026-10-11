"""Pinned ViLT checkpoint load/forward audit; no quality evaluation or training."""

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import requests

REPO = "dandelin/vilt-b32-finetuned-vqa"
REVISION = "d0a1f6ab88522427a7ae76ceb6e1e1e7b68a1d08"
WEIGHTS_SHA = "4d5f3409947b0369487ece7c5868f0040ceb67d25735dbb4ac5e99e03bab3a19"


def main():
    root = Path("C:/CodexArtifacts/pretrained-reuse-r1/vilt-vqa-audit-01")
    if root.exists():
        raise FileExistsError(root)
    if shutil.disk_usage(root.parent).free < 470435927 + 1024**3:
        raise ValueError("insufficient disk reserve")
    root.mkdir()
    model_dir = root / "model"
    model_dir.mkdir()
    started = time.perf_counter()
    api = requests.get(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=60
    )
    api.raise_for_status()
    metadata = api.json()
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "apache-2.0":
        raise ValueError("revision/license differs")
    (root / "hub-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    names = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "README.md",
        "pytorch_model.bin",
    ]
    hashes = {}
    for name in names:
        response = requests.get(
            f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
            timeout=(30, 60),
            stream=True,
        )
        response.raise_for_status()
        path = model_dir / name
        digest = hashlib.sha256()
        with path.open("xb") as target:
            for chunk in response.iter_content(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
        expected = next(r for r in metadata["siblings"] if r["rfilename"] == name)
        if path.stat().st_size != expected["size"]:
            raise ValueError("download size mismatch")
        print("downloaded", name, path.stat().st_size, flush=True)
    if hashes["pytorch_model.bin"] != WEIGHTS_SHA:
        raise ValueError("weights hash mismatch")
    download_seconds = time.perf_counter() - started
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    config = ViltConfig.from_pretrained(model_dir, local_files_only=True)
    model = ViltForQuestionAnswering(config)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    processor = ViltProcessor.from_pretrained(model_dir, local_files_only=True)
    # A deterministic synthetic image is only a runtime probe, not a labeled task.
    array = np.zeros((224, 224, 3), dtype=np.uint8)
    array[40:180, 60:160, 0] = 255
    inputs = processor(
        images=Image.fromarray(array), text="What color is the object?", return_tensors="pt"
    ).to("cuda")
    load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    with torch.inference_mode():
        first = model(**inputs).logits
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        second = model(**inputs).logits
        torch.cuda.synchronize()
    if not torch.isfinite(first).all() or first.shape != (1, len(config.id2label)):
        raise ValueError("invalid output")
    result = dict(
        status="PASS_RUNTIME_ONLY",
        repo=REPO,
        revision=REVISION,
        hashes=hashes,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        parameters=parameters,
        components={n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()},
        input_shapes={k: list(v.shape) for k, v in inputs.items()},
        output_shape=list(first.shape),
        first_forward_seconds=first_seconds,
        repeat_forward_seconds=time.perf_counter() - tick,
        repeat_max_abs=float((first - second).abs().max()),
        load_seconds=load_seconds,
        download_seconds=download_seconds,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        dtype=str(next(model.parameters()).dtype),
        fixed_answer_vocabulary=True,
        quality_evaluation=False,
        cost_usd=0,
        limitation="Synthetic image forward only; no image/video/general decision quality claim.",
    )
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
