"""Run a pinned PE-Core zero-shot comparison on the frozen Open Images pilot."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_openimages_tinyclip_zero_shot import (
    PROMPT_TEMPLATE,
    gather_choice_logits,
    probabilities_from_logits,
    quality_metrics,
    read_jsonl,
    sha256_file,
    validate_rows,
)

MODEL_ID = "timm/PE-Core-T-16-384"
MODEL_REVISION = "7fe539ed578ac49a1c2b4f946e4b0747704c825a"
WEIGHTS_SHA256 = "477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5"
TASK_SHA256 = "6fbd56f0caebc77b1067dfffaadb444aaf51b4b01eabff3855c11a245f86f964"
DERANGEMENT_SEED = 20261026
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _safe_image_path(row: dict[str, Any]) -> Path:
    relative = Path(str(row["local_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("media local_path must be a safe relative path")
    root = (Path(str(row["_artifact_root"])).resolve(strict=True) / "validation").resolve(
        strict=True
    )
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("resolved image must be inside the validation media directory")
    if sha256_file(path) != row.get("media_sha256"):
        raise ValueError(f"media SHA-256 mismatch for {row.get('image_id')}")
    return path


def load_inputs(tasks_path: Path, feature_dir: Path, media_dirs: list[Path]):
    task_document = json.loads(tasks_path.read_text(encoding="utf-8"))
    all_tasks = task_document["examples"]
    features = read_jsonl(feature_dir / "validation-rows.jsonl")
    media_by_id: dict[str, dict[str, Any]] = {}
    media_hashes = {}
    for directory in media_dirs:
        manifest = directory / "validation-media-manifest.jsonl"
        media_hashes[str(manifest)] = sha256_file(manifest)
        for row in read_jsonl(manifest):
            image_id = str(row["image_id"])
            prior = media_by_id.get(image_id)
            if (
                prior is not None
                and prior.get("media_sha256") is not None
                and row.get("media_sha256") is not None
                and prior.get("media_sha256") != row.get("media_sha256")
            ):
                raise ValueError("retried media manifests disagree on content hash")
            media_by_id[image_id] = row | {"_artifact_root": str(directory)}
    task_by_id = {str(row["image_id"]): row for row in all_tasks}
    if len(task_by_id) != len(all_tasks):
        raise ValueError("task manifest contains duplicate image IDs")
    task_rows = [task_by_id[str(row["image_id"])] for row in features]
    media_rows = [media_by_id[str(row["image_id"])] for row in features]
    validate_rows(task_rows, features, media_rows)
    if any(
        feature.get("media_sha256") != media.get("media_sha256")
        for feature, media in zip(features, media_rows, strict=True)
    ):
        raise ValueError("feature/media content hashes differ")
    if len(task_rows) != 168:
        raise ValueError(f"expected the frozen 168-image cohort, found {len(task_rows)}")
    return task_rows, features, media_rows, media_hashes


def score_conditions(
    model,
    transform,
    tokenizer,
    labels: list[str],
    paths: list[Path],
    *,
    controls: bool,
    batch_size: int,
):
    import torch
    from PIL import Image

    text = tokenizer([PROMPT_TEMPLATE.format(label=label.lower()) for label in labels])
    with torch.inference_mode():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def run(image_paths: list[Path] | None):
        chunks = []
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0, len(paths), batch_size):
                count = min(batch_size, len(paths) - offset)
                if image_paths is None:
                    # The transform's normalization maps this neutral color to zero.
                    mean = (0.48145466, 0.4578275, 0.40821073)
                    fill = tuple(round(x * 255) for x in mean)
                    images = [Image.new("RGB", (384, 384), fill) for _ in range(count)]
                else:
                    images = []
                    for path in image_paths[offset : offset + count]:
                        with Image.open(path) as image:
                            images.append(image.convert("RGB"))
                pixels = torch.stack([transform(image) for image in images])
                image_features = model.encode_image(pixels)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = model.logit_scale.exp() * image_features @ text_features.T
                chunks.append(logits.cpu().numpy())
                for image in images:
                    image.close()
        return np.concatenate(chunks), time.perf_counter() - started

    aligned, elapsed = run(paths)
    output = {"aligned": (aligned, elapsed)}
    if controls:
        order = list(range(len(paths)))
        rng = random.Random(DERANGEMENT_SEED)
        for _ in range(1000):
            rng.shuffle(order)
            if all(i != j for i, j in enumerate(order)):
                break
        else:
            raise RuntimeError("could not create deterministic image derangement")
        deranged, deranged_time = run([paths[i] for i in order])
        neutral, neutral_time = run(None)
        output["deranged"] = (deranged, deranged_time)
        output["neutral"] = (neutral, neutral_time)
        output["control_elapsed"] = deranged_time + neutral_time
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, action="append", required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-controls", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("batch size must be in [1, 8]")
    output = args.output_dir.resolve()
    if output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("model outputs must remain outside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256_file(args.tasks_manifest) != TASK_SHA256:
        raise ValueError("task manifest hash differs from the fixed cohort")
    if sha256_file(args.model_file) != WEIGHTS_SHA256:
        raise ValueError("PE-Core checkpoint hash mismatch")
    task_rows, feature_rows, media_rows, media_hashes = load_inputs(
        args.tasks_manifest.resolve(strict=True),
        args.features_dir.resolve(strict=True),
        [path.resolve(strict=True) for path in args.media_dir],
    )
    paths = [_safe_image_path(row) for row in media_rows]
    labels = sorted({str(option) for row in task_rows for option in row["options"]})

    import open_clip
    import torch

    torch.set_num_threads(4)
    model, transform = open_clip.create_model_from_pretrained(
        "PE-Core-T-16-384",
        pretrained=str(args.model_file.resolve(strict=True)),
        device="cpu",
        cache_dir=None,
    )
    model.eval().requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 69_546_113:
        raise ValueError(f"unexpected PE-Core parameter count: {parameter_count}")
    tokenizer = open_clip.get_tokenizer("PE-Core-T-16-384")
    scored = score_conditions(
        model,
        transform,
        tokenizer,
        labels,
        paths,
        controls=args.image_controls,
        batch_size=args.batch_size,
    )
    target_indices = np.asarray(
        [row["options"].index(row["target"]) for row in task_rows], dtype=np.int64
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    condition_rows = {}
    prediction_rows = []
    base_probabilities = None
    for name in ("aligned", "neutral", "deranged"):
        if name not in scored:
            continue
        logits, duration = scored[name]
        choice_logits = gather_choice_logits(logits, task_rows, label_to_index)
        probabilities = probabilities_from_logits(choice_logits)
        if name == "aligned":
            base_probabilities = probabilities
        condition_rows[name] = {
            "metrics": quality_metrics(choice_logits, target_indices),
            "elapsed_seconds": duration,
        }
        if name != "aligned":
            condition_rows[name]["top1_agreement_with_aligned"] = float(
                np.mean(probabilities.argmax(1) == base_probabilities.argmax(1))
            )
        for i, row in enumerate(task_rows):
            prediction_rows.append(
                {
                    "condition": name,
                    "sample_id": row["sample_id"],
                    "options": row["options"],
                    "target": row["target"],
                    "probabilities": probabilities[i].tolist(),
                    "prediction": row["options"][int(probabilities[i].argmax())],
                }
            )
    result = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_sha256": sha256_file(args.model_file),
        "runtime_parameters": parameter_count,
        "validation_count": len(task_rows),
        "prompt_template": PROMPT_TEMPLATE,
        "task_manifest_sha256": TASK_SHA256,
        "feature_rows_sha256": sha256_file(args.features_dir / "validation-rows.jsonl"),
        "media_manifest_sha256": media_hashes,
        "batch_size": args.batch_size,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "image_controls": args.image_controls,
        "derangement_seed": DERANGEMENT_SEED if args.image_controls else None,
        "conditions": condition_rows,
        "parameter_bytes": args.model_file.stat().st_size,
        "python_version": __import__("platform").python_version(),
        "torch_version": torch.__version__,
        "open_clip_version": open_clip.__version__,
    }
    output.mkdir(parents=True)
    result_path = output / "result.json"
    predictions_path = output / "predictions.jsonl"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    receipt = {
        "result_sha256": sha256_file(result_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
