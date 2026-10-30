"""Frozen synthetic image grounding with a fixed image-permutation control."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def prompts(taxonomy, options):
    patterns = {
        "color": "An image of {} objects.",
        "texture": "An image of objects with a {} texture.",
        "count": "An image containing {} objects.",
        "shape": "An image of {}-shaped objects.",
    }
    return [patterns[taxonomy].format(option) for option in options]


def rotated_indices(ids):
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("need unique image IDs")
    return [(i + 1) % len(ids) for i in range(len(ids))]


def main():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    from scripts.train_frozen_text_probe import _metrics
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "sample", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    expected = {
        args.model
        / "model.safetensors": "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587",
        args.sample
        / "val-labels.jsonl": "3c213f13ef2214baf1130cbe49c1d981b0125bbff3813e4ac5b4e341e443d4a1",
        args.sample
        / "val-image-manifest.jsonl": (
            "77faf433844be6b350cd0205fd3834cd8b"
            "028a6491ddd9eae0f6e9b7610cd38a"
        ),
    }
    for path, digest in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"hash mismatch: {path}")
    rows = [
        json.loads(line) for line in (args.sample / "val-labels.jsonl").read_text().splitlines()
    ]
    media = [
        json.loads(line)
        for line in (args.sample / "val-image-manifest.jsonl").read_text().splitlines()
    ]
    ids = [r["id"] for r in rows]
    rotation = rotated_indices(ids)
    if len(rows) != 512 or ids != [r["id"] for r in media]:
        raise ValueError("incomplete or mismatched IDs")
    paths = [args.sample / "val/images" / (r["id"] + ".png") for r in rows]
    for path, item in zip(paths, media, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("media hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    write(
        "protocol.json",
        dict(
            paths={str(p): h for p, h in expected.items()},
            ids=ids,
            rotation=rotation,
            prompts={k: prompts(k, v) for k, v in CLEVR4_TAXONOMIES.items()},
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    model, info = CLIPModel.from_pretrained(
        args.model, local_files_only=True, output_loading_info=True
    )
    if any(info.values()):
        raise ValueError("model load mismatch")
    model.to(device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    features = []
    with torch.inference_mode():
        for offset in range(0, len(paths), 8):
            images = []
            for path in paths[offset : offset + 8]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            batch = processor(images=images, return_tensors="pt").to(device)
            output = model.vision_model(**batch).pooler_output
            output = model.visual_projection(output)
            features.append(torch.nn.functional.normalize(output, dim=-1))
        image_features = torch.cat(features)
        reports = {c: {} for c in ("correct_image", "rotated_image", "uniform")}
        predictions = []
        for taxonomy, options in CLEVR4_TAXONOMIES.items():
            tokens = processor(
                text=prompts(taxonomy, options), padding=True, truncation=False, return_tensors="pt"
            ).to(device)
            if tokens["input_ids"].shape[1] > model.config.text_config.max_position_embeddings:
                raise ValueError("text too long")
            text = model.text_projection(model.text_model(**tokens).pooler_output)
            text = torch.nn.functional.normalize(text, dim=-1)
            scores = (model.logit_scale.exp() * image_features @ text.T).cpu()
            if not torch.isfinite(scores).all():
                raise ValueError("nonfinite logits")
            targets = [options.index(str(r[taxonomy])) for r in rows]
            for condition, logits in (
                ("correct_image", scores),
                ("rotated_image", scores[rotation]),
                ("uniform", torch.zeros_like(scores)),
            ):
                reports[condition][taxonomy] = _metrics(list(logits), targets)
                for i, (row, probs) in enumerate(zip(rows, logits.softmax(-1), strict=True)):
                    assigned = ids[rotation[i]] if condition == "rotated_image" else row["id"]
                    predictions.append(
                        dict(
                            id=row["id"],
                            taxonomy=taxonomy,
                            condition=condition,
                            image_id=None if condition == "uniform" else assigned,
                            options=options,
                            target=str(row[taxonomy]),
                            probabilities=probs.tolist(),
                        )
                    )
    write("predictions.json", predictions)
    write(
        "result.json",
        dict(
            status="PASS_EXECUTION_ONLY",
            metrics=reports,
            macro={
                c: {
                    m: sum(t[m] for t in r.values()) / 4
                    for m in ("accuracy", "nll", "brier", "ece_15_bins")
                }
                for c, r in reports.items()
            },
            seconds=time.perf_counter() - start,
            device=device,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_vram_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
            limitation="Reused synthetic development; no general question or temporal claim.",
        ),
    )


if __name__ == "__main__":
    main()
