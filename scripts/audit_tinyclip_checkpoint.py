"""CPU-only pinned TinyCLIP loading/counting smoke, not a quality benchmark."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path


def main():
    import torch
    import transformers
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
              for f in args.model.iterdir() if f.is_file()}
    if hashes["model.safetensors"] != (
        "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587"
    ):
        raise ValueError("weight hash mismatch")
    torch.set_num_threads(2)
    start = time.perf_counter()
    model, info = CLIPModel.from_pretrained(args.model, local_files_only=True,
                                            output_loading_info=True)
    if any(info.values()):
        raise ValueError(f"loading mismatch: {info}")
    model.cpu().eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    inputs = processor(text=["a red square", "a blue square"],
                       images=[Image.new("RGB", (256, 256), color) for color in ("red", "blue")],
                       padding=True, return_tensors="pt")
    with torch.inference_mode():
        a = model(**inputs)
        b = model(**inputs)
    if not torch.isfinite(a.logits_per_image).all() or not torch.equal(
        a.logits_per_image, b.logits_per_image
    ):
        raise ValueError("nonfinite or nonrepeatable output")
    counts = Counter()
    for name, parameter in model.named_parameters():
        counts[name.split(".")[0]] += parameter.numel()
    report = dict(status="PASS_LOAD_SMOKE_ONLY", parameters=sum(counts.values()),
                  parameters_by_top_level=dict(counts), input_sha256=hashes,
                  parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                  image_shape=list(a.image_embeds.shape), text_shape=list(a.text_embeds.shape),
                  logits=a.logits_per_image.tolist(), seconds=time.perf_counter()-start,
                  device="cpu", threads=2, torch=torch.__version__,
                  transformers=transformers.__version__, python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation="Solid-color load smoke only, not task quality or temporal reasoning.")
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding="utf-8")


if __name__ == "__main__":
    main()
