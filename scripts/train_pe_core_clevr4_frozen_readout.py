"""Train a fixed small readout on frozen PE-Core CLEVR-4 features.

Development-only probe. The encoder and prompt protocol are frozen, and the
reporting subset is opened only after checkpoint selection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from scripts.evaluate_pe_core_clevr4_zero_shot import PROMPTS, TAXONOMIES, load_image
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES, sha256_file

WEIGHT_SHA256 = "477b7507c79e3484195f4e389af6d044b5d56f42ce503d6b0eb7e99439b545e5"
REVISION = "7fe539ed578ac49a1c2b4f946e4b0747704c825a"


class CandidateScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_rows(labels: Path, ids: list[str] | None = None) -> list[dict]:
    rows = read_jsonl(labels)
    if ids is None:
        return rows
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or set(ids) - set(by_id):
        raise ValueError("requested validation IDs are missing or duplicated")
    return [by_id[key] for key in ids]


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def encode_images(model, preprocess, image_dir: Path, rows: list[dict], device, batch_size: int):
    ids = [str(row["id"]) for row in rows]
    results = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = torch.stack(
                [
                    load_image(image_dir / f"{image_id}.png", preprocess)
                    for image_id in ids[start : start + batch_size]
                ]
            ).to(device)
            results.append(F.normalize(model.encode_image(batch).float(), dim=-1).cpu())
            if (start + len(batch)) % 512 == 0 or start + len(batch) == len(ids):
                print(f"encoded_images={start + len(batch)}/{len(ids)}", flush=True)
    features = torch.cat(results)
    if features.shape != (len(ids), 512) or not torch.isfinite(features).all():
        raise ValueError("invalid PE-Core image embeddings")
    return dict(zip(ids, features, strict=True))


def build_tasks(rows, image_features, text_features):
    image_indices, taxonomy_indices, targets = [], [], []
    for row in rows:
        image_id = str(row["id"])
        for taxonomy_index, taxonomy in enumerate(TAXONOMIES):
            options = CLEVR4_TAXONOMIES[taxonomy]
            target = str(row[taxonomy])
            if target not in options:
                raise ValueError(f"invalid {taxonomy} target for {image_id}")
            image_indices.append(image_id)
            taxonomy_indices.append(taxonomy_index)
            targets.append(options.index(target))
    images = torch.stack([image_features[key] for key in image_indices])
    texts = torch.stack(text_features)
    x = torch.cat([images[:, None, :].expand(-1, 10, -1), texts[taxonomy_indices]], dim=-1)
    return x, torch.tensor(targets, dtype=torch.long), image_indices, taxonomy_indices


def score(scorer, x, device, batch_size=256):
    scorer.eval()
    with torch.inference_mode():
        return torch.cat([scorer(batch.to(device)).cpu() for batch in x.split(batch_size)])


def metrics(logits, targets):
    p = torch.softmax(logits.float(), dim=-1)
    pred = p.argmax(-1)
    one_hot = F.one_hot(targets, p.shape[-1]).float()
    confidence = p.max(-1).values
    correct = pred.eq(targets).float()
    ece = 0.0
    for index in range(15):
        lo, hi = index / 15, (index + 1) / 15
        mask = (confidence >= lo) & ((confidence < hi) if index < 14 else (confidence <= hi))
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(correct[mask].mean() - confidence[mask].mean())
            )
    return {
        "accuracy": float(correct.mean()),
        "nll": float(F.cross_entropy(logits.float(), targets)),
        "brier": float(((p - one_hot) ** 2).sum(-1).mean()),
        "ece_15": ece,
        "mean_confidence": float(confidence.mean()),
    }


def predictions(rows, logits, targets):
    output, offset = [], 0
    for row in rows:
        for taxonomy in TAXONOMIES:
            options = CLEVR4_TAXONOMIES[taxonomy]
            p = torch.softmax(logits[offset].float(), dim=-1)
            target = int(targets[offset])
            output.append(
                {
                    "id": f"{row['id']}:{taxonomy}",
                    "image_id": str(row["id"]),
                    "taxonomy": taxonomy,
                    "target": options[target],
                    "prediction": options[int(p.argmax())],
                    "options": options,
                    "probabilities": p.tolist(),
                }
            )
            offset += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--selection-rows", type=Path, required=True)
    parser.add_argument("--reporting-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrained_reuse/pe_core_clevr4_frozen_readout.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if sha256_file(args.weights) != WEIGHT_SHA256:
        raise ValueError("PE-Core weight hash mismatch")
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("experiment_id") != "pe-core-t16-clevr4-frozen-readout-seed17"
        or config.get("seed") != 17
        or config.get("encoder", {}).get("revision") != REVISION
        or config.get("encoder", {}).get("converted_weights_sha256") != WEIGHT_SHA256
        or config.get("readout", {}).get("trainable_parameters") != 131329
        or config.get("readout", {}).get("epochs") != 8
        or config.get("readout", {}).get("batch_questions") != 64
        or config.get("readout", {}).get("image_batch_size") != args.batch_size
    ):
        raise ValueError("effective config differs from the preregistered experiment")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = args.sample_dir.resolve(strict=True)
    selection_rows_path = args.selection_rows.resolve(strict=True)
    train_rows = load_rows(sample / "train-labels.jsonl")
    validation_rows_all = load_rows(sample / "val-labels.jsonl")
    train_manifest = read_jsonl(sample / "train-image-manifest.jsonl")
    validation_manifest = read_jsonl(sample / "val-image-manifest.jsonl")
    if [str(row["id"]) for row in train_manifest] != [str(row["id"]) for row in train_rows]:
        raise ValueError("training labels and media manifest order differ")
    if [str(row["id"]) for row in validation_manifest] != [
        str(row["id"]) for row in validation_rows_all
    ]:
        raise ValueError("validation labels and media manifest order differ")
    prior_selection_rows = read_jsonl(selection_rows_path)
    selection_ids = list(dict.fromkeys(str(row["image_id"]) for row in prior_selection_rows))
    validation_by_id = {str(row["id"]): row for row in validation_rows_all}
    selection_rows = [validation_by_id[key] for key in selection_ids]
    reporting_rows = read_jsonl(args.reporting_rows.resolve(strict=True))
    reporting_ids = list(dict.fromkeys(str(row["image_id"]) for row in reporting_rows))
    if (len(train_rows), len(selection_rows), len(reporting_ids), len(reporting_rows)) != (
        8424,
        512,
        1595,
        6380,
    ):
        raise ValueError("CLEVR-4 split counts differ from the preregistered protocol")
    if set(selection_ids) & set(reporting_ids) or {str(r["id"]) for r in train_rows} & set(
        selection_ids + reporting_ids
    ):
        raise ValueError("train/selection/reporting identity overlap")
    expected_reporting_ids = set(validation_by_id) - set(selection_ids)
    if set(reporting_ids) != expected_reporting_ids:
        raise ValueError("reporting IDs differ from the frozen validation remainder")
    report_by_id = defaultdict(dict)
    for row in reporting_rows:
        if (
            row["taxonomy"] not in TAXONOMIES
            or row["taxonomy"] in report_by_id[str(row["image_id"])]
        ):
            raise ValueError("duplicate or unexpected reporting task")
        report_by_id[str(row["image_id"])][row["taxonomy"]] = row
    if any(set(report_by_id[key]) != set(TAXONOMIES) for key in reporting_ids):
        raise ValueError("reporting images must each have all four tasks")
    report_rows = [
        {"id": key, **{t: report_by_id[key][t]["target"] for t in TAXONOMIES}}
        for key in reporting_ids
    ]
    for image_id in reporting_ids:
        for taxonomy in TAXONOMIES:
            source = report_by_id[image_id][taxonomy]
            if str(source["target"]) != str(validation_by_id[image_id][taxonomy]):
                raise ValueError(
                    f"reporting label differs from pinned validation label: {image_id}"
                )
            if source["options"] != CLEVR4_TAXONOMIES[taxonomy]:
                raise ValueError(f"reporting option order differs from frozen task: {taxonomy}")
    image_dir = sample / "train" / "images"
    val_dir = sample / "val" / "images"
    train_hashes, validation_hashes = [], []
    for manifest_row, folder in [
        *((row, image_dir) for row in train_manifest),
        *((row, val_dir) for row in validation_manifest),
    ]:
        media = folder / f"{manifest_row['id']}.png"
        if not media.is_file():
            raise FileNotFoundError(media)
        digest = sha256_file(media)
        if digest != manifest_row["sha256"] or media.stat().st_size != manifest_row["bytes"]:
            raise ValueError(f"media differs from frozen manifest: {media.name}")
        (train_hashes if folder == image_dir else validation_hashes).append(digest)
    if len(set(train_hashes)) != len(train_hashes) or len(set(validation_hashes)) != len(
        validation_hashes
    ):
        raise ValueError("duplicate media identities within a split")
    if set(train_hashes) & set(validation_hashes):
        raise ValueError("train/validation media identity overlap")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed_no_model_loaded",
                    "training_images": len(train_rows),
                    "selection_images": len(selection_rows),
                    "reporting_images": len(reporting_rows) // len(TAXONOMIES),
                    "train_selection_reporting_overlap": 0,
                    "train_validation_media_hash_overlap": 0,
                    "train_manifest_sha256": sha256_file(sample / "train-image-manifest.jsonl"),
                    "validation_manifest_sha256": sha256_file(sample / "val-image-manifest.jsonl"),
                    "weight_sha256": WEIGHT_SHA256,
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
                indent=2,
            )
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "effective-config.yaml").write_bytes(config_path.read_bytes())
    metadata = {
        "status": "started",
        "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.argv,
        "effective_config_sha256": sha256_file(config_path),
        "effective_config_path": str(config_path),
        "weight_sha256": WEIGHT_SHA256,
        "preflight": {
            "training_images": len(train_rows),
            "selection_images": len(selection_rows),
            "reporting_images": len(report_rows),
            "train_validation_media_identity_overlap": 0,
            "sealed_audit_loaded": False,
        },
    }
    (args.output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    torch.manual_seed(17)
    random.seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "PE-Core-T-16-384", pretrained=str(args.weights), device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    started = time.perf_counter()
    train_features = encode_images(
        model, preprocess, image_dir, train_rows, device, args.batch_size
    )
    validation_all = encode_images(
        model, preprocess, val_dir, validation_rows_all, device, args.batch_size
    )
    tokenizer = open_clip.get_tokenizer("PE-Core-T-16-384")
    prompts = {t: [PROMPTS[t].format(option=o) for o in CLEVR4_TAXONOMIES[t]] for t in TAXONOMIES}
    with torch.inference_mode():
        text_features = [
            F.normalize(model.encode_text(tokenizer(prompts[t]).to(device)).float(), dim=-1).cpu()
            for t in TAXONOMIES
        ]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    x_train, y_train, _, _ = build_tasks(train_rows, train_features, text_features)
    x_val, y_val, _, _ = build_tasks(selection_rows, validation_all, text_features)
    scorer = CandidateScorer().to(device)
    trainable = sum(p.numel() for p in scorer.parameters())
    if trainable != 131329:
        raise ValueError(f"unexpected trainable parameter count: {trainable}")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    order_rng = random.Random(17)
    history_path = args.output_dir / "training-history.jsonl"
    history, best_state, best_nll, best_epoch = [], None, float("inf"), 0
    for epoch in range(1, 9):
        order = list(range(len(y_train)))
        order_rng.shuffle(order)
        scorer.train()
        losses = []
        for start in range(0, len(order), 64):
            ids = order[start : start + 64]
            logits = scorer(x_train[ids].to(device))
            loss = F.cross_entropy(logits, y_train[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = score(scorer, x_val, device)
        val_metrics = metrics(val_logits, y_val)
        per_task = {}
        for index, taxonomy in enumerate(TAXONOMIES):
            take = slice(index, len(y_val), len(TAXONOMIES))
            per_task[taxonomy] = metrics(val_logits[take], y_val[take])
        history.append(
            {
                "epoch": epoch,
                "train_ce": float(np.mean(losses)),
                "validation": val_metrics,
                "validation_by_task": per_task,
            }
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1]) + "\n")
            handle.flush()
        if val_metrics["nll"] < best_nll:
            best_nll, best_epoch = val_metrics["nll"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
            torch.save(best_state, args.output_dir / "best-readout.pt")
        print(json.dumps(history[-1]), flush=True)
    if best_state is None:
        raise RuntimeError("no selected readout")
    readout_path = args.output_dir / "best-readout.pt"
    torch.save(scorer.state_dict(), args.output_dir / "last-epoch-readout.pt")
    scorer.load_state_dict(best_state)
    before_reload_logits = score(scorer, x_val, device)
    scorer.load_state_dict(torch.load(readout_path, map_location=device, weights_only=True))
    best_val_logits = score(scorer, x_val, device)
    if not torch.equal(before_reload_logits, best_val_logits):
        raise ValueError("saved/reloaded best readout changed selection predictions")
    selection_pred = predictions(selection_rows, best_val_logits, y_val)
    selection_path = args.output_dir / "checkpoint-selection-predictions.jsonl"
    selection_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selection_pred), encoding="utf-8"
    )
    x_report, y_report, _, _ = build_tasks(report_rows, validation_all, text_features)
    report_logits = score(scorer, x_report, device)
    report_pred = predictions(report_rows, report_logits, y_report)
    feature_path = args.output_dir / "frozen-features.npz"
    np.savez_compressed(
        feature_path,
        train_ids=np.array(list(train_features)),
        train_features=torch.stack(list(train_features.values())).numpy().astype("float16"),
        validation_ids=np.array(list(validation_all)),
        validation_features=torch.stack(list(validation_all.values())).numpy().astype("float16"),
        text_features=torch.stack(text_features).numpy().astype("float16"),
    )
    (args.output_dir / "reporting-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in report_pred), encoding="utf-8"
    )
    report = {
        "status": "complete_development_frozen_readout_probe",
        "seed": 17,
        "model_revision": REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "effective_config_sha256": sha256_file(config_path),
        "frozen_encoder_parameters": 69546113,
        "trainable_readout_parameters": trainable,
        "training_images": len(train_rows),
        "selection_images": len(selection_rows),
        "reporting_images": len(report_rows),
        "reporting_decisions": len(report_pred),
        "training_ids_sha256": ordered_hash([str(r["id"]) for r in train_rows]),
        "selection_ids_sha256": ordered_hash(selection_ids),
        "reporting_ids_sha256": ordered_hash(reporting_ids),
        "train_media_hashes_sha256": hashlib.sha256(
            "\n".join(train_hashes).encode("utf-8")
        ).hexdigest(),
        "validation_media_hashes_sha256": hashlib.sha256(
            "\n".join(validation_hashes).encode("utf-8")
        ).hexdigest(),
        "train_labels_sha256": sha256_file(sample / "train-labels.jsonl"),
        "validation_labels_sha256": sha256_file(sample / "val-labels.jsonl"),
        "train_manifest_sha256": sha256_file(sample / "train-image-manifest.jsonl"),
        "validation_manifest_sha256": sha256_file(sample / "val-image-manifest.jsonl"),
        "selection_rows_sha256": sha256_file(selection_rows_path),
        "reporting_rows_sha256": sha256_file(args.reporting_rows),
        "frozen_features_sha256": sha256_file(feature_path),
        "selected_readout_sha256": sha256_file(readout_path),
        "best_epoch": best_epoch,
        "validation": metrics(best_val_logits, y_val),
        "history": history,
        "reporting": metrics(report_logits, y_report),
        "reporting_by_task": {
            t: metrics(report_logits[i::4], y_report[i::4]) for i, t in enumerate(TAXONOMIES)
        },
        "training_and_feature_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "open_clip_version": importlib.metadata.version("open_clip_torch"),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "reporting_scope": "previously observed development subset; not blind audit",
        "sealed_audit_loaded": False,
    }
    (args.output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    metadata.update(
        {
            "status": "complete",
            "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "best_epoch": best_epoch,
            "run_report_sha256": sha256_file(args.output_dir / "run-report.json"),
        }
    )
    (args.output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
