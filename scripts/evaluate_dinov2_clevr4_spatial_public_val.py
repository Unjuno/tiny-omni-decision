"""One-time fixed paired evaluation of global/spatial DINOv2 readouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.hotgpu_preflight import validate_gpu_preflight  # noqa: E402
from tiny_omni_decision.cellwise_readout import CellwiseCandidateScorer  # noqa: E402
from tiny_omni_decision.clevr4_data import (  # noqa: E402
    read_annotations_for_split,
    require_disjoint_media,
)
from tiny_omni_decision.clevr4_readout import classification_quality  # noqa: E402
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES  # noqa: E402
from tiny_omni_decision.dinov2_features import readout_features, spatial_cells  # noqa: E402

ARCHIVE_BYTES = 36_057_238_315
ARCHIVE_SHA512 = (
    "7e8454f9f44d7b527eb20252a687b15b4cd28f3e9c79cf6147317971034c367b"
    "64bf6bba98d2a66f64b26f1e7eea2105a8acf221198342e89a844fd0007cc94c"
)
DINO_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
TEXT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"
HEAD_SHA256 = {
    "global": "3c109e2c027ea20e8177d6f5daf30cd282a9d44e5e73356423bdc62046a41a8f",
    "spatial": "22fdd1e72e2ffac1efe31cc4e172358fb78431aa06aac23390604bd8e0f16134",
}
EXPECTED_VAL_IMAGES = 19_988
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_ordered_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def require_gpu_slot() -> dict[str, int | str]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, utilization, total = (int(value.strip()) for value in query.split(","))
    pid_output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    pids = {
        int(line.strip())
        for line in pid_output.splitlines()
        if line.strip() and line.strip().casefold() != "no running processes found"
    }
    return validate_gpu_preflight(
        memory_used_mib=used,
        utilization_pct=utilization,
        total_mib=total,
        active_pids=pids,
        own_pid=os.getpid(),
        environment=os.environ,
    )


def embed_options(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], str, int]:
    weights = path / "model.safetensors"
    if sha256_file(weights) != TEXT_SHA256:
        raise ValueError("pinned MiniLM weights changed")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
    encoder.requires_grad_(False)
    prompts = [
        f"Image attribute: {task}. Candidate class: {option}."
        for task, options in CLEVR4_TAXONOMIES.items()
        for option in options
    ]
    encoded = tokenizer(
        prompts, max_length=64, truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        hidden = encoder(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vectors = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    count = sum(parameter.numel() for parameter in encoder.parameters())
    del encoder
    torch.cuda.empty_cache()
    by_task: dict[str, torch.Tensor] = {}
    cursor = 0
    for task, options in CLEVR4_TAXONOMIES.items():
        by_task[task] = vectors[cursor : cursor + len(options)]
        cursor += len(options)
    embedding_hash = hashlib.sha256(vectors.detach().cpu().numpy().tobytes()).hexdigest()
    return by_task, embedding_hash, count


def score_feature_batch(
    scorer: torch.nn.Module,
    cells: np.ndarray,
    options: dict[str, torch.Tensor],
    condition: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    visual = torch.from_numpy(readout_features(cells, condition)).to(device)
    output: dict[str, torch.Tensor] = {}
    for task, candidates in options.items():
        paired = torch.cat(
            [
                visual[:, None, :].expand(-1, len(candidates), -1),
                candidates[None, :, :].expand(len(visual), -1, -1),
            ],
            dim=-1,
        )
        probabilities = torch.softmax(scorer(paired).float(), dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("non-finite candidate probabilities")
        output[task] = probabilities.cpu()
    return output


def bootstrap_differences(
    per_image_differences: dict[str, np.ndarray], *, seed: int = 17, replicates: int = 5000
) -> dict:
    rng = np.random.default_rng(seed)
    count = len(next(iter(per_image_differences.values())))
    if count == 0 or any(len(values) != count for values in per_image_differences.values()):
        raise ValueError("paired bootstrap requires a nonempty common image cohort")
    output = {
        "seed": seed,
        "replicates": replicates,
        "unit": "official validation image; four taxonomy decisions stay grouped",
        "method": "paired percentile interval; numpy linear quantile",
        "direction": "spatial minus global",
        "metrics": {},
    }
    for metric, differences in per_image_differences.items():
        samples = np.empty(replicates, dtype=np.float64)
        cursor = 0
        while cursor < replicates:
            size = min(100, replicates - cursor)
            indexes = rng.integers(0, count, size=(size, count), dtype=np.int32)
            samples[cursor : cursor + size] = differences[indexes].mean(axis=1)
            cursor += size
        output["metrics"][metric] = {
            "difference": float(differences.mean()),
            "percentile_95_interval": np.quantile(
                samples, [0.025, 0.975], method="linear"
            ).tolist(),
        }
    return output


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("pinned official Clevr-4 archive size mismatch")
    if sha256_file(args.dino_model / "model.safetensors") != DINO_SHA256:
        raise ValueError("pinned DINOv2 weights changed")

    readout_config = json.loads((args.readout_dir / "config.json").read_text(encoding="utf-8"))
    readout_result = json.loads((args.readout_dir / "result.json").read_text(encoding="utf-8"))
    if readout_config.get("conditions") != ["global", "spatial"]:
        raise ValueError("unexpected matched readout arms")
    if readout_result.get("status") != "PASS_EXECUTION_ONLY":
        raise ValueError("selected readout run did not pass its integrity gate")
    for condition, expected_hash in HEAD_SHA256.items():
        head_path = args.readout_dir / condition / "best.pt"
        if sha256_file(head_path) != expected_hash:
            raise ValueError(f"selected {condition} checkpoint hash mismatch")
        primary = (args.readout_dir / condition / "best-predictions.json").read_bytes()
        reloaded = (args.readout_dir / condition / "best-predictions-reloaded.json").read_bytes()
        if primary != reloaded:
            raise ValueError(f"selected {condition} best-checkpoint reload mismatch")

    prior_result = json.loads((args.prior_public_val_dir / "result.json").read_text())
    if prior_result.get("status") != "COMPLETE":
        raise ValueError("prior public validation record is incomplete")
    prior_media = load_jsonl(args.prior_public_val_dir / "public-val-media-manifest.jsonl")
    if len(prior_media) != EXPECTED_VAL_IMAGES:
        raise ValueError("prior public validation media manifest has the wrong size")
    previous_train_media = load_jsonl(args.media_hashed_dir / "train-media-manifest.jsonl")
    previous_train_media += load_jsonl(args.media_hashed_dir / "validation-media-manifest.jsonl")
    historical_validation = load_jsonl(args.historical_validation_manifest)
    historical_ids = {row["id"] for row in historical_validation}
    historical_hashes = {row["sha256"] for row in historical_validation}
    if len(historical_ids) != len(historical_validation) or len(historical_hashes) != len(
        historical_validation
    ):
        raise ValueError("historical validation inventory contains duplicate IDs or media hashes")
    previous_ids = {row["id"] for row in previous_train_media}
    previous_hashes = {row["media_sha256"] for row in previous_train_media}
    if len(previous_ids) != len(previous_train_media) or len(previous_hashes) != len(
        previous_train_media
    ):
        raise ValueError("prior official-train media inventory contains duplicates")

    with ZipFile(args.archive) as archive:
        image_members = {
            PurePosixPath(info.filename).stem
            for info in archive.infolist()
            if PurePosixPath(info.filename).parent == PurePosixPath("images")
            and PurePosixPath(info.filename).suffix.lower() == ".png"
        }
        if len(image_members) != 100_000:
            raise ValueError("expected 100,000 unique official image members")
        annotation_bytes = archive.read("clevr_4_annots.json")
        labels, split_counts = read_annotations_for_split(
            BytesIO(annotation_bytes), requested_split="val"
        )
        if split_counts != {"train": 80_012, "val": EXPECTED_VAL_IMAGES}:
            raise ValueError(f"unexpected official split counts: {split_counts}")
        ids = [row["id"] for row in labels]
        if len(ids) != EXPECTED_VAL_IMAGES or len(set(ids)) != EXPECTED_VAL_IMAGES:
            raise ValueError("invalid or duplicate official public-val labels")
        if set(ids) & historical_ids:
            raise ValueError("official public-val IDs overlap historical validation")
        if set(ids) - image_members or set(ids) & previous_ids:
            raise ValueError("official public-val ID overlap or missing media")
        if set(ids) | previous_ids != image_members:
            raise ValueError("official train/validation IDs do not partition the archive")
        if sha256_ordered_ids(ids) != prior_result["validation_id_order_sha256"]:
            raise ValueError("public validation order differs from the prior fixed replay")
        prior_by_id = {row["id"]: row["media_sha256"] for row in prior_media}
        if set(prior_by_id) != set(ids):
            raise ValueError("prior public-val manifest does not match the official IDs")

    gpu_slot = require_gpu_slot()
    args.output.mkdir(parents=True)
    label_rows = [
        {
            "id": row["id"],
            "source_split": "official-public-val",
            **{task: str(row[task]) for task in CLEVR4_TAXONOMIES},
        }
        for row in labels
    ]
    labels_sha = write_jsonl(args.output / "public-val-labels.jsonl", label_rows)
    id_order_sha = sha256_ordered_ids(ids)
    annotation_sha = sha256_bytes(annotation_bytes)
    config = {
        "status": "RUNNING",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "dataset": "Oxford Clevr-4 100k official public validation",
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha512_pinned_from_prior_verified_receipt": ARCHIVE_SHA512,
        "archive_full_sha512_recomputed_this_run": False,
        "official_split_counts": split_counts,
        "validation_images": len(labels),
        "validation_id_order_sha256": id_order_sha,
        "validation_label_manifest_sha256": labels_sha,
        "annotation_member_sha256": annotation_sha,
        "official_train_id_media_overlap": 0,
        "prior_public_val_replay": {
            "same_ids_and_order": True,
            "already_exposed": True,
            "macro_accuracy": prior_result["metrics"]["macro"]["accuracy"],
            "prior_model": "fixed DINOv2+MiniLM global readout; no spatial-cell head",
        },
        "evaluation_only": True,
        "training_performed": False,
        "checkpoint_selection": False,
        "sealed_audit": False,
        "no_metrics_used_for_tuning_or_promotion": True,
        "model": {
            "dino_revision": "facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "dino_weights_sha256": DINO_SHA256,
            "text_weights_sha256": TEXT_SHA256,
            "readout_checkpoints_sha256": HEAD_SHA256,
            "readout_training_config_sha256": sha256_file(args.readout_dir / "config.json"),
            "readout_result_sha256": sha256_file(args.readout_dir / "result.json"),
            "readout_selected_epochs": {
                arm: readout_result["conditions"][arm]["best_epoch"]
                for arm in ("global", "spatial")
            },
            "representation": (
                "DINOv2 CLS + 4x4 spatial patch means; global control repeats patch mean"
            ),
            "frozen_parameters": 39_446_400,
            "trainable_readout_parameters": 180_736,
            "total_unique_prequant_parameters": 39_627_136,
        },
        "hotgpu_reservation": gpu_slot,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "Pillow")
        },
        "exact_argv": sys.argv,
    }
    (args.output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")

    torch.set_num_threads(2)
    device = torch.device("cuda")
    dino, load_info = AutoModel.from_pretrained(
        args.dino_model, local_files_only=True, dtype=torch.float32, output_loading_info=True
    )
    if load_info.get("missing_keys") or load_info.get("unexpected_keys"):
        raise ValueError("pinned DINOv2 checkpoint did not load exactly")
    dino.eval().requires_grad_(False).to(device)
    processor = AutoImageProcessor.from_pretrained(args.dino_model, local_files_only=True)
    options, option_hash, text_parameter_count = embed_options(args.text_model, device)
    scorers: dict[str, CellwiseCandidateScorer] = {}
    for condition in ("global", "spatial"):
        scorer = CellwiseCandidateScorer().to(device)
        scorer.load_state_dict(
            torch.load(
                args.readout_dir / condition / "best.pt", weights_only=True, map_location=device
            )
        )
        scorer.eval().requires_grad_(False)
        scorers[condition] = scorer
    if sum(parameter.numel() for parameter in scorers["spatial"].parameters()) != 180_736:
        raise ValueError("unexpected cellwise scorer parameter count")

    torch.cuda.reset_peak_memory_stats()
    image_differences = {
        metric: np.zeros(len(labels), dtype=np.float64) for metric in ("accuracy", "nll", "brier")
    }
    prediction_path = args.output / "predictions.jsonl"
    macro_metrics: dict[str, dict] = {}
    per_task_probabilities: dict[str, dict[str, list[torch.Tensor]]] = {
        arm: {task: [] for task in CLEVR4_TAXONOMIES} for arm in ("global", "spatial")
    }
    per_task_targets: dict[str, list[int]] = {task: [] for task in CLEVR4_TAXONOMIES}
    media_rows: list[dict] = []
    started = time.perf_counter()
    with (
        ZipFile(args.archive) as archive,
        prediction_path.open("w", encoding="utf-8", newline="\n") as predictions,
    ):
        for offset in range(0, len(labels), BATCH_SIZE):
            batch_rows = labels[offset : offset + BATCH_SIZE]
            images, batch_hashes = [], []
            for row in batch_rows:
                raw = archive.read(f"images/{row['id']}.png")
                digest = sha256_bytes(raw)
                if digest != prior_by_id[row["id"]]:
                    raise ValueError(f"official public-val asset changed: {row['id']}")
                if digest in historical_hashes:
                    raise ValueError("official public-val bytes overlap historical validation")
                batch_hashes.append(digest)
                with Image.open(BytesIO(raw)) as image:
                    images.append(image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC))
            encoded = processor(
                images=images, do_resize=False, do_center_crop=False, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                tokens = dino(**encoded).last_hidden_state.float()
            cells = spatial_cells(tokens.cpu().numpy())
            if cells.shape != (len(batch_rows), 17, 384) or not np.isfinite(cells).all():
                raise ValueError("invalid frozen DINOv2 spatial features")
            scored = {
                arm: score_feature_batch(scorers[arm], cells, options, arm, device)
                for arm in ("global", "spatial")
            }
            for task, choices in CLEVR4_TAXONOMIES.items():
                targets = torch.tensor(
                    [choices.index(str(row[task])) for row in batch_rows], dtype=torch.long
                )
                per_task_targets[task].extend(targets.tolist())
                for arm in ("global", "spatial"):
                    per_task_probabilities[arm][task].extend(scored[arm][task].unbind(0))
                one_hot = torch.nn.functional.one_hot(targets, len(choices)).float()
                global_probs, spatial_probs = scored["global"][task], scored["spatial"][task]
                image_slice = slice(offset, offset + len(batch_rows))
                image_differences["accuracy"][image_slice] += (
                    (spatial_probs.argmax(-1) == targets).float().numpy()
                    - (global_probs.argmax(-1) == targets).float().numpy()
                ) / len(CLEVR4_TAXONOMIES)
                image_differences["nll"][image_slice] += (
                    -spatial_probs.gather(1, targets[:, None])
                    .squeeze(1)
                    .clamp_min(1e-12)
                    .log()
                    .numpy()
                    + global_probs.gather(1, targets[:, None])
                    .squeeze(1)
                    .clamp_min(1e-12)
                    .log()
                    .numpy()
                ) / len(CLEVR4_TAXONOMIES)
                image_differences["brier"][image_slice] += (
                    (spatial_probs - one_hot).square().sum(-1).numpy()
                    - (global_probs - one_hot).square().sum(-1).numpy()
                ) / len(CLEVR4_TAXONOMIES)
            for index, (row, digest) in enumerate(zip(batch_rows, batch_hashes, strict=True)):
                media_rows.append(
                    {
                        "id": row["id"],
                        "asset_id": row["id"],
                        "bytes_sha256": digest,
                        "source_split": "official-public-val",
                    }
                )
                for task, choices in CLEVR4_TAXONOMIES.items():
                    target = choices.index(str(row[task]))
                    predictions.write(
                        json.dumps(
                            {
                                "id": f"{row['id']}:{task}",
                                "image_id": row["id"],
                                "taxonomy": task,
                                "options": choices,
                                "target": choices[target],
                                "global_probabilities": scored["global"][task][index].tolist(),
                                "spatial_probabilities": scored["spatial"][task][index].tolist(),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
            completed = offset + len(batch_rows)
            if completed % 1024 == 0 or completed == len(labels):
                print(
                    json.dumps(
                        {
                            "completed_images": completed,
                            "total_images": len(labels),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    media_sha = write_jsonl(args.output / "public-val-media-manifest.jsonl", media_rows)
    if len({row["bytes_sha256"] for row in media_rows}) != len(labels):
        raise ValueError("duplicate official public-val image bytes")
    require_disjoint_media(previous_train_media, media_rows)
    for arm in ("global", "spatial"):
        per_task = {}
        for task, _choices in CLEVR4_TAXONOMIES.items():
            probs = torch.stack(per_task_probabilities[arm][task])
            targets = torch.tensor(per_task_targets[task])
            per_task[task] = classification_quality(probs, targets)
        macro = {
            metric: float(np.mean([row[metric] for row in per_task.values()]))
            for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
        }
        macro["images_per_taxonomy"] = len(labels)
        macro["decisions"] = len(labels) * len(CLEVR4_TAXONOMIES)
        macro["by_taxonomy"] = per_task
        macro_metrics[arm] = macro

    config.update(
        {
            "option_embedding_sha256": option_hash,
            "frozen_text_parameter_count": text_parameter_count,
            "validation_media_manifest_sha256": media_sha,
            "prediction_sha256": sha256_file(prediction_path),
            "feature_extraction_and_scoring_seconds": time.perf_counter() - started,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "status": "COMPLETE",
            "finished_utc": datetime.now(UTC).isoformat(),
        }
    )
    config_path = args.output / "run-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    config["run_config_sha256"] = sha256_file(config_path)
    spatial_minus_global = {
        key: macro_metrics["spatial"][key] - macro_metrics["global"][key]
        for key in ("accuracy", "nll", "brier", "ece", "mean_confidence")
    }
    result = {
        **config,
        "status": "COMPLETE",
        "metrics": macro_metrics,
        "spatial_minus_global_macro": spatial_minus_global,
        "paired_image_bootstrap": bootstrap_differences(image_differences),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "archive",
        "media-hashed-dir",
        "historical-validation-manifest",
        "prior-public-val-dir",
        "dino-model",
        "text-model",
        "readout-dir",
        "output",
    ):
        parser.add_argument("--" + arg, type=Path, required=True)
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": result["metrics"],
                "delta": result["spatial_minus_global_macro"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
