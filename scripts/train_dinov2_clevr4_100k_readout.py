"""Train one fixed small candidate scorer on the fresh Clevr-4 100k split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_omni_decision.clevr4_readout import build_candidate_features, classification_quality
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer

SEED = 17
EPOCHS = 8
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.0001
DINO_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
TEXT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_jsonl_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _option_embeddings(
    model_path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], int]:
    weight_path = model_path / "model.safetensors"
    if sha256_file(weight_path) != TEXT_SHA256:
        raise ValueError("pinned MiniLM model weight SHA-256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    encoder.requires_grad_(False)
    prompts = {
        f"{taxonomy}|{option}": f"Image attribute: {taxonomy}. Candidate class: {option}."
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    }
    keys = list(prompts)
    encoded = tokenizer(
        [prompts[key] for key in keys],
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
        raise ValueError("non-finite frozen option embedding")
    count = sum(parameter.numel() for parameter in encoder.parameters())
    del encoder
    torch.cuda.empty_cache()
    return dict(zip(keys, pooled, strict=True)), count


def _candidate_options(embeddings: dict[str, torch.Tensor], device: torch.device):
    return {
        taxonomy: torch.stack([embeddings[f"{taxonomy}|{option}"] for option in options]).to(device)
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
    }


def _evaluate(scorer, features, rows, option_features, device, *, save_predictions=False):
    scorer.eval()
    all_probs = {name: [] for name in CLEVR4_TAXONOMIES}
    all_targets = {name: [] for name in CLEVR4_TAXONOMIES}
    prediction_rows = []
    with torch.inference_mode():
        for offset in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[offset : offset + BATCH_SIZE]
            image_batch = torch.from_numpy(
                np.asarray(features[offset : offset + len(batch_rows)]).copy()
            ).to(device)
            candidates = build_candidate_features(
                image_batch, option_features, list(CLEVR4_TAXONOMIES)
            )
            logits = scorer(candidates).float()
            probabilities = torch.softmax(logits, dim=-1).cpu()
            for taxonomy_index, (taxonomy, options) in enumerate(CLEVR4_TAXONOMIES.items()):
                for row_index, row in enumerate(batch_rows):
                    target = options.index(str(row[taxonomy]))
                    all_probs[taxonomy].append(probabilities[row_index, taxonomy_index])
                    all_targets[taxonomy].append(target)
                    if save_predictions:
                        probs = probabilities[row_index, taxonomy_index]
                        prediction_rows.append(
                            {
                                "id": f"{row['id']}:{taxonomy}",
                                "image_id": row["id"],
                                "taxonomy": taxonomy,
                                "target": options[target],
                                "prediction": options[int(probs.argmax())],
                                "options": options,
                                "probabilities": probs.tolist(),
                            }
                        )
    by_taxonomy = {
        taxonomy: classification_quality(
            torch.stack(all_probs[taxonomy]), torch.tensor(all_targets[taxonomy])
        )
        for taxonomy in CLEVR4_TAXONOMIES
    }
    macro = {
        metric: float(np.mean([value[metric] for value in by_taxonomy.values()]))
        for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
    }
    macro["examples_per_taxonomy"] = len(rows)
    macro["total_decisions"] = len(rows) * len(CLEVR4_TAXONOMIES)
    return {"macro": macro, "by_taxonomy": by_taxonomy}, prediction_rows


def _verify_inputs(feature_dir: Path, index_dir: Path, media_dir: Path):
    feature_report = json.loads((feature_dir / "result.json").read_text(encoding="utf-8"))
    index_report = json.loads((index_dir / "index-report.json").read_text(encoding="utf-8"))
    media_report = json.loads((media_dir / "media-hash-report.json").read_text(encoding="utf-8"))
    if feature_report["archive_sha512"] != index_report["archive_sha512"]:
        raise ValueError("feature and index source archives differ")
    if (
        media_report["train_validation_media_overlap"]
        or media_report["overlap_with_previous_train_media"]
    ):
        raise ValueError("media overlap gate failed")
    if feature_report["official_val_media_opened"] != 0:
        raise ValueError("official validation media must remain untouched")

    rows, manifests = {}, {}
    feature_media = json.loads(
        (feature_dir / "feature-source-media-sha256.json").read_text(encoding="utf-8")
    )
    for split in ("train", "validation"):
        rows[split] = load_jsonl(index_dir / f"{split}-labels.jsonl")
        manifests[split] = load_jsonl(media_dir / f"{split}-media-manifest.jsonl")
        label_name = f"{split}-labels.jsonl"
        if canonical_jsonl_sha256(rows[split]) != index_report["manifest_sha256"][label_name]:
            raise ValueError(f"{split} labels differ from the frozen index")
        media_name = f"{split}-media-manifest.jsonl"
        if sha256_file(media_dir / media_name) != media_report["media_manifest_sha256"][media_name]:
            raise ValueError(f"{split} hashed media manifest changed")
        if [row["id"] for row in rows[split]] != [row["id"] for row in manifests[split]]:
            raise ValueError(f"{split} labels/media IDs differ")
        if any(feature_media.get(row["id"]) != row["media_sha256"] for row in manifests[split]):
            raise ValueError(f"{split} features do not bind to indexed media bytes")
        ids = json.loads((feature_dir / f"{split}-ids.json").read_text(encoding="utf-8"))
        if ids != [row["id"] for row in rows[split]]:
            raise ValueError(f"{split} feature IDs/order differ from frozen labels")
        data = np.load(feature_dir / f"{split}-features.npy", mmap_mode="r")
        if data.shape != (len(rows[split]), 768) or not np.isfinite(data).all():
            raise ValueError(f"invalid {split} feature matrix")
        if (
            sha256_file(feature_dir / f"{split}-features.npy")
            != feature_report["feature_sha256"][f"{split}-features.npy"]
        ):
            raise ValueError(f"{split} feature hash mismatch")
    return feature_report, index_report, media_report, rows, manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--media-hashed-dir", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--reference-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-mode", choices=("image_grouped", "question"), default="image_grouped"
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not np.isfinite(args.gradient_clip_norm) or args.gradient_clip_norm < 0:
        raise ValueError("gradient-clip-norm must be finite and nonnegative")
    feature_report, index_report, media_report, rows, manifests = _verify_inputs(
        args.features, args.index, args.media_hashed_dir
    )
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    option_embeddings, text_parameters = _option_embeddings(args.text_model, device)
    option_features = _candidate_options(option_embeddings, device)
    train_features = np.load(args.features / "train-features.npy", mmap_mode="r")
    validation_features = np.load(args.features / "validation-features.npy", mmap_mode="r")
    train_rows, validation_rows = rows["train"], rows["validation"]

    scorer = FrozenFeatureCandidateScorer(1152, hidden_size=128).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    if trainable_parameters != 147713:
        raise ValueError(f"unexpected readout parameter count: {trainable_parameters}")
    old_state = torch.load(args.reference_readout, weights_only=True, map_location=device)
    reference = FrozenFeatureCandidateScorer(1152, hidden_size=128).to(device)
    reference.load_state_dict(old_state)
    reference_metrics, _ = _evaluate(
        reference, validation_features, validation_rows, option_features, device
    )
    del reference
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    run_config = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "feature_result_sha256": sha256_file(args.features / "result.json"),
        "index_report_sha256": sha256_file(args.index / "index-report.json"),
        "media_hash_report_sha256": sha256_file(args.media_hashed_dir / "media-hash-report.json"),
        "validation_id_order_sha256": index_report["validation_id_order_sha256"],
        "train_id_order_sha256": index_report["train_id_order_sha256"],
        "reference_readout_sha256": sha256_file(args.reference_readout),
        "dino_weight_sha256": feature_report["model_weight_sha256"],
        "text_weight_sha256": TEXT_SHA256,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_images": BATCH_SIZE,
        "batch_mode": args.batch_mode,
        "gradient_clip_norm": args.gradient_clip_norm,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "selector": "earliest minimum macro validation NLL across four taxonomies",
        "representation": feature_report["representation"],
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "unique_train_media": len({row["media_sha256"] for row in manifests["train"]}),
        "unique_validation_media": len({row["media_sha256"] for row in manifests["validation"]}),
        "previous_train_media_overlap": media_report["overlap_with_previous_train_media"],
        "trainable_readout_parameters": trainable_parameters,
        "frozen_dinov2_parameters": 22056576,
        "frozen_text_parameters": text_parameters,
        "total_encoder_plus_readout_parameters": 22056576 + text_parameters + trainable_parameters,
        "gpu": torch.cuda.get_device_name(),
        "reference_same_validation": reference_metrics,
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")
        },
    }
    (args.output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    history = []
    best_nll = float("inf")
    best_epoch = None
    rng = random.Random(SEED)
    training_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, EPOCHS + 1):
        taxonomy_count = len(CLEVR4_TAXONOMIES)
        order = list(
            range(len(train_rows) * (taxonomy_count if args.batch_mode == "question" else 1))
        )
        rng.shuffle(order)
        scorer.train()
        losses, online_correct, online_total = [], 0, 0
        for offset in range(0, len(order), BATCH_SIZE):
            indices = order[offset : offset + BATCH_SIZE]
            if args.batch_mode == "image_grouped":
                image_batch = torch.from_numpy(np.asarray(train_features[indices]).copy()).to(
                    device
                )
                batch_rows = [train_rows[i] for i in indices]
                targets = torch.tensor(
                    [
                        [
                            list(options).index(str(row[taxonomy]))
                            for taxonomy, options in CLEVR4_TAXONOMIES.items()
                        ]
                        for row in batch_rows
                    ],
                    dtype=torch.long,
                    device=device,
                )
                candidate_batch = build_candidate_features(
                    image_batch, option_features, list(CLEVR4_TAXONOMIES)
                )
                logits = scorer(candidate_batch).float()
                loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
            else:
                image_ids = [index // taxonomy_count for index in indices]
                taxonomy_ids = [index % taxonomy_count for index in indices]
                taxonomy_names = list(CLEVR4_TAXONOMIES)
                task_names = [taxonomy_names[index] for index in taxonomy_ids]
                image_batch = torch.from_numpy(np.asarray(train_features[image_ids]).copy()).to(
                    device
                )
                task_options = torch.stack([option_features[name] for name in task_names])
                candidate_batch = torch.cat(
                    [
                        image_batch[:, None, :].expand(-1, task_options.shape[1], -1),
                        task_options,
                    ],
                    dim=-1,
                )
                targets = torch.tensor(
                    [
                        list(CLEVR4_TAXONOMIES[name]).index(str(train_rows[image_id][name]))
                        for image_id, name in zip(image_ids, task_names, strict=True)
                    ],
                    dtype=torch.long,
                    device=device,
                )
                logits = scorer(candidate_batch).float()
                loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    scorer.parameters(), args.gradient_clip_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            losses.append(float(loss.detach()))
            online_correct += int(logits.argmax(-1).eq(targets).sum())
            online_total += targets.numel()
            if offset % 8192 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "examples_seen": offset + len(indices),
                            "train_examples": len(order),
                            "batch_unit": "questions"
                            if args.batch_mode == "question"
                            else "images",
                        }
                    ),
                    flush=True,
                )

        validation, predictions = _evaluate(
            scorer,
            validation_features,
            validation_rows,
            option_features,
            device,
            save_predictions=True,
        )
        record = {
            "epoch": epoch,
            "online_train_ce": float(np.mean(losses)),
            "online_train_accuracy": online_correct / online_total,
            "validation": validation,
            "train_accuracy_is_online_minibatch_metric": True,
        }
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if validation["macro"]["nll"] < best_nll:
            best_nll = validation["macro"]["nll"]
            best_epoch = epoch
            torch.save(scorer.state_dict(), args.output / "best-readout.pt")
            (args.output / "best-predictions.json").write_text(
                json.dumps(predictions, indent=2) + "\n"
            )
        torch.save(scorer.state_dict(), args.output / "final-readout.pt")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "online_train_ce": record["online_train_ce"],
                    "validation": validation["macro"],
                    "best_epoch": best_epoch,
                }
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - training_started
    final_metrics, _ = _evaluate(
        scorer, validation_features, validation_rows, option_features, device
    )
    best = FrozenFeatureCandidateScorer(1152, hidden_size=128).to(device)
    best.load_state_dict(
        torch.load(args.output / "best-readout.pt", weights_only=True, map_location=device)
    )
    selected_metrics, selected_predictions = _evaluate(
        best,
        validation_features,
        validation_rows,
        option_features,
        device,
        save_predictions=True,
    )
    if selected_metrics != history[best_epoch - 1]["validation"]:
        raise AssertionError("best-checkpoint reload metrics differ from selected epoch")
    (args.output / "best-predictions-reloaded.json").write_text(
        json.dumps(selected_predictions, indent=2) + "\n"
    )
    if sha256_file(args.features / "result.json") != run_config["feature_result_sha256"]:
        raise ValueError("feature result metadata changed during training")
    result = {
        "status": "PASS_EXECUTION_ONLY",
        "best_epoch": best_epoch,
        "selected": selected_metrics,
        "final": final_metrics,
        "reference_readout_same_validation": reference_metrics,
        "history": history,
        "train_images": len(train_rows),
        "validation_images": len(validation_rows),
        "consumed_image_presentations": len(train_rows)
        * EPOCHS
        * (len(CLEVR4_TAXONOMIES) if args.batch_mode == "question" else 1),
        "consumed_decision_examples": len(train_rows) * EPOCHS * len(CLEVR4_TAXONOMIES),
        "optimizer_updates": int(
            np.ceil(
                len(train_rows)
                * (len(CLEVR4_TAXONOMIES) if args.batch_mode == "question" else 1)
                / BATCH_SIZE
            )
            * EPOCHS
        ),
        "trainable_readout_parameters": trainable_parameters,
        "total_encoder_plus_readout_parameters": run_config[
            "total_encoder_plus_readout_parameters"
        ],
        "training_seconds": elapsed,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "best_readout_sha256": sha256_file(args.output / "best-readout.pt"),
        "final_readout_sha256": sha256_file(args.output / "final-readout.pt"),
        "best_reload_exact_metrics": True,
        "official_val_used": False,
        "product_promotion": False,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
