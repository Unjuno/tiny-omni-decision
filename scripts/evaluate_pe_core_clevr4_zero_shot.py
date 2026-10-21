"""One-shot CPU zero-shot evaluation of PE-Core T16 on CLEVR-4 reporting rows.

This is a development diagnostic on an existing, previously observed validation
subset. It does not calibrate, fit, select prompts, or read sealed audit data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TAXONOMIES = ("color", "texture", "shape", "count")
PROMPTS = {
    "color": "In this synthetic 3D image, an object's color is {option}.",
    "texture": "In this synthetic 3D image, an object's surface texture is {option}.",
    "shape": "In this synthetic 3D image, an object's shape is {option}.",
    "count": "In this synthetic 3D image, there are {option} objects.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_image(path: Path, preprocess) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def read_examples(
    path: Path,
    image_dir: Path,
    manifest_path: Path,
    labels_path: Path,
    selection_rows_path: Path,
) -> tuple[list[str], dict, dict, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    manifest = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    labels = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    selected_ids = {
        str(json.loads(line)["image_id"])
        for line in selection_rows_path.read_text(encoding="utf-8").splitlines()
        if line
    }
    by_image: dict[str, dict] = defaultdict(dict)
    options_by_taxonomy: dict[str, list[str]] = {}
    ids: list[str] = []
    for row in rows:
        image_id = str(row["image_id"])
        taxonomy = str(row["taxonomy"])
        if taxonomy not in TAXONOMIES or row["id"] != f"{image_id}:{taxonomy}":
            raise ValueError(f"Unexpected decision identity: {row.get('id')!r}")
        if taxonomy in by_image[image_id]:
            raise ValueError(f"Duplicate taxonomy decision for {image_id}/{taxonomy}")
        options = [str(option) for option in row["options"]]
        if len(options) != 10 or len(set(options)) != 10:
            raise ValueError(f"Expected 10 unique options for {row['id']}")
        if taxonomy in options_by_taxonomy and options_by_taxonomy[taxonomy] != options:
            raise ValueError(f"Inconsistent option order for {taxonomy}")
        options_by_taxonomy[taxonomy] = options
        target = str(row["target"])
        if target not in options:
            raise ValueError(f"Target is absent from options: {row['id']}")
        by_image[image_id][taxonomy] = {"id": row["id"], "target": target}
        ids.append(str(row["id"]))
    image_ids = list(dict.fromkeys(str(row["image_id"]) for row in rows))
    if len(image_ids) != 1595 or len(rows) != 6380:
        raise ValueError(
            "Expected fixed 1,595-image/6,380-row reporting slice, "
            f"got {len(image_ids)}/{len(rows)}"
        )
    if any(set(by_image[image_id]) != set(TAXONOMIES) for image_id in image_ids):
        raise ValueError("Every image must have exactly the four fixed taxonomies")
    media_hashes = {}
    for image_id in image_ids:
        if image_id in selected_ids:
            raise ValueError(f"Reporting image overlaps checkpoint-selection image: {image_id}")
        if image_id not in manifest or manifest[image_id].get("split") != "val":
            raise ValueError(f"Missing validation image manifest row: {image_id}")
        if image_id not in labels or labels[image_id].get("split") != "val":
            raise ValueError(f"Missing validation label row: {image_id}")
        media = image_dir / f"{image_id}.png"
        if not media.is_file():
            raise FileNotFoundError(media)
        actual_hash = sha256_file(media)
        if (
            actual_hash != manifest[image_id]["sha256"]
            or media.stat().st_size != manifest[image_id]["bytes"]
        ):
            raise ValueError(f"Validation media does not match pinned manifest: {image_id}")
        media_hashes[image_id] = actual_hash
        for taxonomy in TAXONOMIES:
            if str(labels[image_id][taxonomy]) != by_image[image_id][taxonomy]["target"]:
                raise ValueError(
                    f"Target mismatch against validation labels: {image_id}/{taxonomy}"
                )
    if len(set(media_hashes.values())) != len(media_hashes):
        raise ValueError("Duplicate image bytes detected in fixed reporting slice")
    return image_ids, by_image, options_by_taxonomy, media_hashes


def metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    selected = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    ece = 0.0
    for bin_index in range(15):
        lower = bin_index / 15
        upper = (bin_index + 1) / 15
        in_bin = (confidence >= lower) & (
            confidence < upper if bin_index < 14 else confidence <= upper
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.clip(selected, 1e-12, 1.0)).mean()),
        "brier": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
    }


def group_bootstrap(
    probabilities: np.ndarray, targets: np.ndarray, seed: int = 17, draws: int = 10_000
) -> dict:
    rng = np.random.default_rng(seed)
    correct = (probabilities.argmax(axis=2) == targets).astype(np.float64)
    nll = -np.log(
        np.clip(np.take_along_axis(probabilities, targets[..., None], axis=2)[..., 0], 1e-12, 1.0)
    )
    one_hot = np.eye(probabilities.shape[2], dtype=np.float64)[targets]
    brier = ((probabilities - one_hot) ** 2).sum(axis=2)
    samples = {"accuracy": [], "nll": [], "brier": []}
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        index = rng.integers(0, len(targets), size=(size, len(targets)))
        for key, values in (("accuracy", correct), ("nll", nll), ("brier", brier)):
            per_taxonomy = values[index].mean(axis=1)
            samples[key].extend(per_taxonomy.mean(axis=1).tolist())
    return {
        key: [float(value) for value in np.percentile(values, [2.5, 97.5], method="linear")]
        for key, values in samples.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        required=True,
        help="Existing reporting JSONL; prediction columns are ignored",
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Pinned validation image manifest"
    )
    parser.add_argument("--labels", type=Path, required=True, help="Pinned validation labels")
    parser.add_argument(
        "--selection-rows",
        type=Path,
        required=True,
        help="Rows used for prior checkpoint selection; IDs only",
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="Pinned OpenCLIP-converted PE-Core safetensors"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    image_ids, by_image, options, media_hashes = read_examples(
        args.rows, args.images, args.manifest, args.labels, args.selection_rows
    )
    row_ids = [
        by_image[image_id][taxonomy]["id"] for image_id in image_ids for taxonomy in TAXONOMIES
    ]
    targets = np.zeros((len(image_ids), len(TAXONOMIES)), dtype=np.int64)
    text_options: dict[str, list[str]] = {}
    for taxonomy in TAXONOMIES:
        taxonomy_options = options[taxonomy]
        text_options[taxonomy] = [
            PROMPTS[taxonomy].format(option=option) for option in taxonomy_options
        ]

    import open_clip

    started = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "PE-Core-T-16-384", pretrained=str(args.weights), device="cpu"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("PE-Core-T-16-384")
    with torch.inference_mode():
        text_features = {
            taxonomy: torch.nn.functional.normalize(
                model.encode_text(tokenizer(text_options[taxonomy])), dim=-1
            )
            for taxonomy in TAXONOMIES
        }
        scale = float(model.logit_scale.exp().item())
        all_probabilities = np.zeros((len(image_ids), len(TAXONOMIES), 10), dtype=np.float64)
        predictions = []
        for start in range(0, len(image_ids), args.batch_size):
            current_ids = image_ids[start : start + args.batch_size]
            batch = torch.stack(
                [
                    load_image(args.images / f"{image_id}.png", preprocess)
                    for image_id in current_ids
                ]
            )
            image_features = torch.nn.functional.normalize(model.encode_image(batch), dim=-1)
            for taxonomy_index, taxonomy in enumerate(TAXONOMIES):
                scores = scale * image_features @ text_features[taxonomy].T
                probs = torch.softmax(scores.float(), dim=-1).cpu().numpy().astype(np.float64)
                all_probabilities[start : start + len(current_ids), taxonomy_index] = probs
            for offset, image_id in enumerate(current_ids):
                for taxonomy_index, taxonomy in enumerate(TAXONOMIES):
                    options_for_task = options[taxonomy]
                    target = by_image[image_id][taxonomy]["target"]
                    targets[start + offset, taxonomy_index] = options_for_task.index(target)
                    prob = all_probabilities[start + offset, taxonomy_index]
                    prediction = options_for_task[int(prob.argmax())]
                    predictions.append(
                        {
                            "id": by_image[image_id][taxonomy]["id"],
                            "image_id": image_id,
                            "taxonomy": taxonomy,
                            "target": target,
                            "prediction": prediction,
                            "options": options_for_task,
                            "probabilities": [float(value) for value in prob],
                        }
                    )
            completed = start + len(current_ids)
            if completed == len(image_ids) or completed % (args.batch_size * 50) == 0:
                print(f"encoded_images={completed}/{len(image_ids)}", flush=True)
    elapsed = time.perf_counter() - started

    per_taxonomy = {
        taxonomy: metrics(all_probabilities[:, index], targets[:, index])
        for index, taxonomy in enumerate(TAXONOMIES)
    }
    macro = {
        key: float(np.mean([per_taxonomy[taxonomy][key] for taxonomy in TAXONOMIES]))
        for key in ("accuracy", "nll", "brier", "ece_15", "mean_confidence")
    }
    result = {
        "status": "CPU_ZERO_SHOT_DEV_EVALUATION_COMPLETE",
        "evaluation_scope": (
            "Previously observed CLEVR-4 development reporting subset; not a blind audit."
        ),
        "model": {
            "hub_repo": "timm/PE-Core-T-16-384",
            "hub_revision": "7fe539ed578ac49a1c2b4f946e4b0747704c825a",
            "weights_sha256": sha256_file(args.weights),
            "runtime": "open_clip_torch 3.3.0",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "logit_scale": scale,
        },
        "data": {
            "source": "CLEVR-4 reporting-validation subset from "
            "image-videoprism-clevr4-full-seed17",
            "image_count": len(image_ids),
            "decision_count": len(predictions),
            "image_id_order_sha256": ordered_hash(image_ids),
            "decision_id_order_sha256": ordered_hash(row_ids),
            "image_hashes_sha256": hashlib.sha256(
                json.dumps(media_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "input_file_sha256": {
                "reporting_rows": sha256_file(args.rows),
                "selection_rows_ids_only": sha256_file(args.selection_rows),
                "validation_media_manifest": sha256_file(args.manifest),
                "validation_labels": sha256_file(args.labels),
            },
            "per_image_hashes": media_hashes,
        },
        "protocol": {
            "prompt_templates": PROMPTS,
            "selection": "No prompt search, calibration, training or checkpoint selection.",
            "probabilities": "softmax(exp(logit_scale) * cosine(image,text)); 10 supplied options.",
            "bootstrap": {
                "unit": "image group; all four taxonomies stay together",
                "seed": 17,
                "draws": 10_000,
                "interval": "linear percentile 95%",
            },
        },
        "metrics_by_taxonomy": per_taxonomy,
        "macro_metrics": macro,
        "macro_group_bootstrap_95ci": group_bootstrap(all_probabilities, targets),
        "runtime": {
            "device": "cpu",
            "torch_threads": 2,
            "elapsed_model_and_inference_seconds": elapsed,
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "model",
                    "data",
                    "metrics_by_taxonomy",
                    "macro_metrics",
                    "macro_group_bootstrap_95ci",
                    "runtime",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
