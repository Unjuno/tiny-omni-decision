"""One-time fixed-checkpoint evaluation on the official public Clevr-4 val split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.clevr4_data import (  # noqa: E402
    read_annotations_for_split,
    require_disjoint_media,
)
from tiny_omni_decision.clevr4_readout import (  # noqa: E402
    build_candidate_features,
    classification_quality,
)
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES  # noqa: E402
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer  # noqa: E402
from tiny_omni_decision.dinov2_features import cls_patch_mean  # noqa: E402

ARCHIVE_BYTES = 36_057_238_315
ARCHIVE_SHA512 = (
    "7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b"
    "64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c"
)
DINO_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
TEXT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
READOUT_SHA256 = "6f08e88014ca043570888264e857821eb71b13106d9a3673ea81240a192a0439"
EXPECTED_VAL_IMAGES = 19_988
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_ordered_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_label_rows(rows: list[dict]) -> None:
    ids: set[str] = set()
    for row in rows:
        image_id = row.get("id")
        if not isinstance(image_id, str) or not image_id or image_id in ids:
            raise ValueError("official val has an invalid or duplicate image id")
        ids.add(image_id)
        if row.get("split") != "val":
            raise ValueError("requested annotation is not from the official val split")
        for taxonomy, options in CLEVR4_TAXONOMIES.items():
            if str(row.get(taxonomy)) not in options:
                raise ValueError(f"unsupported {taxonomy} label for {image_id}")


def build_option_embeddings(model_path: Path, device: torch.device):
    if sha256_file(model_path / "model.safetensors") != TEXT_SHA256:
        raise ValueError("pinned MiniLM weights changed")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    encoder.requires_grad_(False)
    prompts = [
        (taxonomy, option, f"Image attribute: {taxonomy}. Candidate class: {option}.")
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    ]
    encoded = tokenizer(
        [prompt for _, _, prompt in prompts],
        max_length=64,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden = encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu()
    if not torch.isfinite(pooled).all():
        raise ValueError("non-finite option embeddings")
    embeddings = {
        f"{taxonomy}|{option}": pooled[index]
        for index, (taxonomy, option, _) in enumerate(prompts)
    }
    option_features = {
        taxonomy: torch.stack([embeddings[f"{taxonomy}|{option}"] for option in options]).to(
            device
        )
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
    }
    embeddings_sha256 = sha256_bytes(pooled.contiguous().numpy().tobytes())
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    del encoder
    torch.cuda.empty_cache()
    return option_features, embeddings_sha256, parameter_count


def score_public_val(
    features: np.ndarray,
    labels: list[dict],
    option_features: dict[str, torch.Tensor],
    scorer: torch.nn.Module,
    output_path: Path,
    device: torch.device,
) -> dict:
    probabilities_by_taxonomy = {key: [] for key in CLEVR4_TAXONOMIES}
    targets_by_taxonomy = {key: [] for key in CLEVR4_TAXONOMIES}
    correct_by_taxonomy: Counter[str] = Counter()
    decisions = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        scorer.eval()
        with torch.inference_mode():
            for start in range(0, len(labels), BATCH_SIZE):
                batch_rows = labels[start : start + BATCH_SIZE]
                image_batch = torch.from_numpy(
                    np.asarray(features[start : start + len(batch_rows)]).copy()
                ).to(device)
                candidates = build_candidate_features(
                    image_batch, option_features, list(CLEVR4_TAXONOMIES)
                )
                logits = scorer(candidates).float()
                probs = torch.softmax(logits, dim=-1).cpu()
                if not torch.isfinite(probs).all():
                    raise ValueError("non-finite prediction probabilities")
                for row_index, row in enumerate(batch_rows):
                    for taxonomy_index, (taxonomy, options) in enumerate(
                        CLEVR4_TAXONOMIES.items()
                    ):
                        target_index = options.index(str(row[taxonomy]))
                        row_probs = probs[row_index, taxonomy_index]
                        prediction_index = int(row_probs.argmax())
                        probabilities_by_taxonomy[taxonomy].append(row_probs)
                        targets_by_taxonomy[taxonomy].append(target_index)
                        correct_by_taxonomy[taxonomy] += int(prediction_index == target_index)
                        decisions += 1
                        record = {
                            "id": f"{row['id']}:{taxonomy}",
                            "image_id": row["id"],
                            "taxonomy": taxonomy,
                            "target": options[target_index],
                            "prediction": options[prediction_index],
                            "options": options,
                            "probabilities": row_probs.tolist(),
                        }
                        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                        stream.write("\n")

    by_taxonomy = {
        taxonomy: classification_quality(
            torch.stack(probabilities_by_taxonomy[taxonomy]),
            torch.tensor(targets_by_taxonomy[taxonomy]),
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }
    macro = {
        metric: float(np.mean([metrics[metric] for metrics in by_taxonomy.values()]))
        for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
    }
    macro["examples_per_taxonomy"] = len(labels)
    macro["total_decisions"] = decisions
    macro["pooled_accuracy"] = sum(correct_by_taxonomy.values()) / decisions
    return {"macro": macro, "by_taxonomy": by_taxonomy}


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("pinned official Clevr-4 archive size mismatch")
    if sha256_file(args.reference_readout) != READOUT_SHA256:
        raise ValueError("fixed selected readout checkpoint SHA-256 mismatch")
    dino_weights = args.dino_model / "model.safetensors"
    if sha256_file(dino_weights) != DINO_SHA256:
        raise ValueError("pinned DINOv2 weights changed")
    previous_media = load_jsonl(args.media_hashed_dir / "train-media-manifest.jsonl")
    previous_media += load_jsonl(args.media_hashed_dir / "validation-media-manifest.jsonl")
    previous_sha = {row["media_sha256"] for row in previous_media}
    if len(previous_sha) != len(previous_media):
        raise ValueError("prior official-train media inventory contains duplicates")
    previous_ids = {row["id"] for row in previous_media}

    with ZipFile(args.archive) as archive:
        infos = {
            PurePosixPath(info.filename).stem: info
            for info in archive.infolist()
            if PurePosixPath(info.filename).parent == PurePosixPath("images")
            and PurePosixPath(info.filename).suffix.lower() == ".png"
        }
        if len(infos) != 100_000:
            raise ValueError("official image member inventory must contain 100,000 unique IDs")
        annotation_bytes = archive.read("clevr_4_annots.json")
        labels, split_counts = read_annotations_for_split(
            BytesIO(annotation_bytes), requested_split="val"
        )
        if split_counts != {"train": 80_012, "val": EXPECTED_VAL_IMAGES}:
            raise ValueError(f"unexpected official split counts: {split_counts}")
        if len(labels) != EXPECTED_VAL_IMAGES:
            raise ValueError("official public validation count mismatch")
        validate_label_rows(labels)
        label_ids = [row["id"] for row in labels]
        if len(set(label_ids)) != len(label_ids) or set(label_ids) - infos.keys():
            raise ValueError("official val annotations do not map one-to-one to archive images")
        official_train_ids = previous_ids
        official_val_ids = set(label_ids)
        if official_train_ids & official_val_ids:
            raise ValueError("official train and val identity overlap")
        if official_train_ids | official_val_ids != infos.keys():
            raise ValueError("official split IDs do not cover the image archive")

        archive_manifest = []
        labels_for_output = []
        val_media_hashes: dict[str, str] = {}
        for row in labels:
            member = f"images/{row['id']}.png"
            raw = archive.read(member)
            digest = sha256_bytes(raw)
            val_media_hashes[row["id"]] = digest
            archive_manifest.append(
                {
                    "id": row["id"],
                    "asset_id": row["id"],
                    "archive_member": member,
                    "bytes": len(raw),
                    "media_sha256": digest,
                }
            )
            labels_for_output.append(
                {
                    "id": row["id"],
                    "asset_id": row["id"],
                    "source_split": "val",
                    **{taxonomy: str(row[taxonomy]) for taxonomy in CLEVR4_TAXONOMIES},
                }
            )
    val_records = archive_manifest
    prior_records = [
        {"id": row["id"], "media_sha256": row["media_sha256"]} for row in previous_media
    ]
    require_disjoint_media(prior_records, val_records)
    if len(set(val_media_hashes.values())) != EXPECTED_VAL_IMAGES:
        raise ValueError("official val contains duplicate media bytes")
    historical_validation = load_jsonl(args.historical_validation_manifest)
    historical_ids = {row["id"] for row in historical_validation}
    historical_hashes = {row["sha256"] for row in historical_validation}
    historical_id_overlap = len(historical_ids & set(label_ids))
    historical_media_overlap = len(historical_hashes & set(val_media_hashes.values()))

    args.output.mkdir(parents=True)
    label_sha = write_jsonl(args.output / "public-val-labels.jsonl", labels_for_output)
    manifest_sha = write_jsonl(args.output / "public-val-media-manifest.jsonl", archive_manifest)
    (args.output / "official-val-feature-order.json").write_text(
        json.dumps(label_ids, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    order_sha = sha256_ordered_ids(label_ids)
    annotation_sha = sha256_bytes(annotation_bytes)
    result = {
        "status": "RUNNING",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dataset": "Oxford Clevr-4 100k official public validation split",
        "archive_url": "https://thor.robots.ox.ac.uk/clevr4/clevr_4_100k_v1.zip",
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha512_pinned_from_prior_verified_receipt": ARCHIVE_SHA512,
        "archive_full_sha512_recomputed_this_run": False,
        "annotation_member_sha256": annotation_sha,
        "official_split_counts": split_counts,
        "validation_examples": len(labels),
        "unique_validation_media": len(set(val_media_hashes.values())),
        "official_train_media_overlap": 0,
        "official_train_asset_overlap": 0,
        "prior_10k_validation_identity_overlap": historical_id_overlap,
        "prior_10k_validation_media_overlap": historical_media_overlap,
        "validation_id_order_sha256": order_sha,
        "validation_label_manifest_sha256": label_sha,
        "validation_media_manifest_sha256": manifest_sha,
        "evaluation_only": True,
        "checkpoint_selector_used": False,
        "training_performed": False,
        "sealed_audit": False,
        "model": {
            "dino_revision": "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "dino_weights_sha256": DINO_SHA256,
            "text_weights_sha256": TEXT_SHA256,
            "readout_checkpoint_sha256": READOUT_SHA256,
            "readout_checkpoint_role": "fixed best selected on internal validation; no updates",
            "representation": "DINOv2 CLS token concatenated with mean of 256 patch tokens",
            "resize": [224, 224],
            "pre_quantization_parameter_count": 39_594_113,
            "trainable_readout_parameter_count": 147_713,
        },
        "hotgpu_reservation": {
            "reserved_mib": int(os.environ["TINY_OMNI_GPU_GATE_RESERVED_MIB"]),
            "free_mib_at_child_start": int(os.environ["TINY_OMNI_GPU_GATE_FREE_MIB"]),
            "total_mib": int(os.environ["TINY_OMNI_GPU_GATE_TOTAL_MIB"]),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow")
        },
    }
    (args.output / "run-config.json").write_text(json.dumps(result, indent=2) + "\n")

    torch.set_num_threads(2)
    device = torch.device("cuda")
    model, load_info = AutoModel.from_pretrained(
        args.dino_model, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    if load_info.get("missing_keys") or load_info.get("unexpected_keys"):
        raise ValueError("pinned DINOv2 checkpoint did not load exactly")
    model.eval().requires_grad_(False).to(device)
    processor = AutoImageProcessor.from_pretrained(args.dino_model, local_files_only=True)
    option_features, option_sha, text_parameters = build_option_embeddings(args.text_model, device)
    scorer = FrozenFeatureCandidateScorer(1152, hidden_size=128).to(device)
    if sum(parameter.numel() for parameter in scorer.parameters()) != 147_713:
        raise ValueError("unexpected fixed readout architecture")
    state = torch.load(args.reference_readout, weights_only=True, map_location=device)
    scorer.load_state_dict(state)
    torch.cuda.reset_peak_memory_stats()
    features_path = args.output / "public-val-features.npy"
    features = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float32, shape=(len(labels), 768)
    )
    started = time.perf_counter()
    with ZipFile(args.archive) as archive:
        for offset in range(0, len(labels), BATCH_SIZE):
            batch = labels[offset : offset + BATCH_SIZE]
            images = []
            for row in batch:
                raw = archive.read(f"images/{row['id']}.png")
                if sha256_bytes(raw) != val_media_hashes[row["id"]]:
                    raise ValueError(f"validation media changed during extraction: {row['id']}")
                with Image.open(BytesIO(raw)) as image:
                    images.append(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC))
            encoded = processor(
                images=images,
                do_resize=False,
                do_center_crop=False,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                tokens = model(**encoded).last_hidden_state
            batch_features = cls_patch_mean(tokens.float().cpu().numpy())
            if not np.isfinite(batch_features).all():
                raise ValueError("non-finite DINOv2 validation features")
            features[offset : offset + len(batch)] = batch_features
            if offset % 1024 == 0 or offset + len(batch) == len(labels):
                print(
                    json.dumps(
                        {
                            "completed": offset + len(batch),
                            "total": len(labels),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    features.flush()
    feature_sha = sha256_file(features_path)
    result["option_embedding_sha256"] = option_sha
    result["frozen_text_parameter_count"] = text_parameters
    result["feature_sha256"] = feature_sha
    result["feature_shape"] = list(features.shape)
    result["feature_extraction_seconds"] = time.perf_counter() - started
    result["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
    metrics = score_public_val(
        features,
        labels_for_output,
        option_features,
        scorer,
        args.output / "predictions.jsonl",
        device,
    )
    result["metrics"] = metrics
    result["prediction_sha256"] = sha256_file(args.output / "predictions.jsonl")
    result["finished_utc"] = datetime.now(UTC).isoformat()
    result["status"] = "COMPLETE"
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--media-hashed-dir", type=Path, required=True)
    parser.add_argument("--historical-validation-manifest", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--reference-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        if args.output.exists():
            (args.output / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "FAILED",
                        "finished_utc": datetime.now(UTC).isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
