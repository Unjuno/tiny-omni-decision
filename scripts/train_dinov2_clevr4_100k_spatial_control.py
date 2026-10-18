"""One paired global-versus-spatial DINOv2 cellwise readout diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
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

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.hotgpu_preflight import validate_gpu_preflight  # noqa: E402
from tiny_omni_decision.cellwise_readout import (  # noqa: E402
    CellwiseCandidateScorer,
    trainable_parameter_count,
)
from tiny_omni_decision.clevr4_readout import classification_quality  # noqa: E402
from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES  # noqa: E402
from tiny_omni_decision.dinov2_features import readout_features  # noqa: E402

SEED, EPOCHS, BATCH, LR, WD = 17, 8, 64, 1e-3, 1e-4
TEXT_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require_gpu_slot() -> dict[str, int | str]:
    usage = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory, utilization, total = [int(value.strip()) for value in usage.split(",")]
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
        memory_used_mib=memory,
        utilization_pct=utilization,
        total_mib=total,
        active_pids=pids,
        own_pid=os.getpid(),
        environment=os.environ,
    )


def state_hash(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def text_options(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    weights = path / "model.safetensors"
    if sha(weights) != TEXT_SHA256:
        raise ValueError("pinned MiniLM hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
    encoder.requires_grad_(False)
    prompts = [
        f"Image attribute: {task}. Candidate class: {option}."
        for task, options in CLEVR4_TAXONOMIES.items()
        for option in options
    ]
    keys = [(task, option) for task, options in CLEVR4_TAXONOMIES.items() for option in options]
    encoded = tokenizer(
        prompts, max_length=64, truncation=True, padding=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        hidden = encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vectors = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float()
    result = {key: vector for key, vector in zip(keys, vectors, strict=True)}
    del encoder
    torch.cuda.empty_cache()
    return result


def evaluate(scorer, cells, rows, options, condition, device, predictions=False):
    scorer.eval()
    task_names, result_probs, result_targets = (
        list(CLEVR4_TAXONOMIES),
        {x: [] for x in CLEVR4_TAXONOMIES},
        {x: [] for x in CLEVR4_TAXONOMIES},
    )
    pred_rows = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch_rows = rows[start : start + BATCH]
            raw = np.asarray(cells[start : start + len(batch_rows)]).copy()
            visual = torch.from_numpy(readout_features(raw, condition)).to(device)
            candidates = []
            for task in task_names:
                text = torch.stack([options[(task, option)] for option in CLEVR4_TAXONOMIES[task]])
                candidates.append(
                    torch.cat(
                        [
                            visual[:, None, :].expand(-1, len(text), -1),
                            text[None].expand(len(visual), -1, -1),
                        ],
                        -1,
                    )
                )
            probs = torch.softmax(scorer(torch.stack(candidates, 1)).float(), -1).cpu()
            for ti, task in enumerate(task_names):
                choices = CLEVR4_TAXONOMIES[task]
                for ri, row in enumerate(batch_rows):
                    target = choices.index(str(row[task]))
                    p = probs[ri, ti]
                    result_probs[task].append(p)
                    result_targets[task].append(target)
                    if predictions:
                        pred_rows.append(
                            {
                                "id": f"{row['id']}:{task}",
                                "image_id": row["id"],
                                "taxonomy": task,
                                "target": choices[target],
                                "prediction": choices[int(p.argmax())],
                                "probabilities": p.tolist(),
                            }
                        )
    by_task = {
        task: classification_quality(
            torch.stack(result_probs[task]), torch.tensor(result_targets[task])
        )
        for task in task_names
    }
    macro = {
        metric: float(np.mean([v[metric] for v in by_task.values()]))
        for metric in ("accuracy", "nll", "brier", "ece", "mean_confidence")
    }
    return {"macro": macro, "by_taxonomy": by_task}, pred_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "index", "text-model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cache = json.loads((args.features / "result.json").read_text(encoding="utf-8"))
    rows = {s: load_jsonl(args.index / f"{s}-labels.jsonl") for s in ("train", "validation")}
    if bool(cache.get("smoke_subset")) != args.smoke:
        raise ValueError("smoke and full-data feature caches cannot be mixed")
    if args.smoke:
        rows = {split: items[:256] for split, items in rows.items()}
    for split in rows:
        ids = json.loads((args.features / f"{split}-ids.json").read_text(encoding="utf-8"))
        if ids != [row["id"] for row in rows[split]]:
            raise ValueError(f"{split} feature IDs/order mismatch")
        array = np.load(args.features / f"{split}-cells.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (len(rows[split]), 17, 384) or not np.isfinite(array).all():
            raise ValueError(f"invalid {split} spatial cache")
        if sha(args.features / f"{split}-cells.npy") != cache["sha256"][f"{split}-cells.npy"]:
            raise ValueError(f"{split} feature hash mismatch")
    gpu_slot = require_gpu_slot()
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    opts = text_options(args.text_model, device)
    train_rows, val_rows = rows["train"], rows["validation"]
    train_cells = np.load(args.features / "train-cells.npy", mmap_mode="r", allow_pickle=False)[
        : len(train_rows)
    ]
    val_cells = np.load(args.features / "validation-cells.npy", mmap_mode="r", allow_pickle=False)[
        : len(val_rows)
    ]
    config = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "epochs": 1 if args.smoke else EPOCHS,
        "batch_questions": BATCH,
        "optimizer": "AdamW lr=0.001 weight_decay=0.0001",
        "selection": "earliest minimum macro validation NLL",
        "conditions": ["global", "spatial"],
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "feature_result_sha256": sha(args.features / "result.json"),
        "validation_status": "previously exposed development split; not independent evaluation",
        "official_public_validation_opened": False,
        "trainable_parameters": 180736,
        "gpu_preflight": gpu_slot,
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "numpy": importlib.metadata.version("numpy"),
        "text_model_weights_sha256": TEXT_SHA256,
        "train_label_sha256": sha(args.index / "train-labels.jsonl"),
        "validation_label_sha256": sha(args.index / "validation-labels.jsonl"),
        "exact_argv": sys.argv,
        "optimizer_updates_per_epoch": int(np.ceil(len(train_rows) * 4 / BATCH)),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    labels = {
        s: np.asarray(
            [
                [list(CLEVR4_TAXONOMIES[t]).index(str(r[t])) for t in CLEVR4_TAXONOMIES]
                for r in data
            ],
            dtype=np.int64,
        )
        for s, data in (("train", train_rows), ("validation", val_rows))
    }
    order_digest_ref, init_ref, results = None, None, {}
    for condition in ("global", "spatial"):
        torch.cuda.reset_peak_memory_stats()
        out = args.output / condition
        out.mkdir()
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        scorer = CellwiseCandidateScorer().to(device)
        if trainable_parameter_count(scorer) != 180736:
            raise ValueError("unexpected cellwise scorer parameter count")
        initial_hash = state_hash(scorer)
        if init_ref is not None and initial_hash != init_ref:
            raise ValueError("matched arms did not start from identical weights")
        init_ref = initial_hash
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=LR, weight_decay=WD)
        rng, order_digest, history, best_nll, best_epoch = (
            random.Random(SEED),
            hashlib.sha256(),
            [],
            float("inf"),
            None,
        )
        epochs = 1 if args.smoke else EPOCHS
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            order = list(range(len(train_rows) * len(CLEVR4_TAXONOMIES)))
            rng.shuffle(order)
            order_digest.update(np.asarray(order, dtype="<i8").tobytes())
            scorer.train()
            losses = []
            online_correct = 0
            online_seen = 0
            for offset in range(0, len(order), BATCH):
                ids = order[offset : offset + BATCH]
                image_ids, task_ids = np.asarray(ids) // 4, np.asarray(ids) % 4
                feature = torch.from_numpy(
                    readout_features(np.asarray(train_cells[image_ids]).copy(), condition)
                ).to(device)
                candidates = []
                for ti, task in enumerate(CLEVR4_TAXONOMIES):
                    mask = task_ids == ti
                    if not mask.any():
                        candidates.append(None)
                        continue
                    text = torch.stack([opts[(task, option)] for option in CLEVR4_TAXONOMIES[task]])
                    selected = feature[torch.as_tensor(mask, device=device)]
                    candidates.append(
                        (
                            mask,
                            torch.cat(
                                [
                                    selected[:, None].expand(-1, len(text), -1),
                                    text[None].expand(len(selected), -1, -1),
                                ],
                                -1,
                            ),
                        )
                    )
                optimizer.zero_grad(set_to_none=True)
                loss_parts, correct, seen = [], 0, 0
                for ti, item in enumerate(candidates):
                    if item is None:
                        continue
                    mask, candidate = item
                    logits = scorer(candidate).float()
                    targets = torch.as_tensor(labels["train"][image_ids[mask], ti], device=device)
                    loss_parts.append(F.cross_entropy(logits, targets, reduction="sum"))
                    correct += int(logits.argmax(-1).eq(targets).sum())
                    seen += len(targets)
                loss = torch.stack(loss_parts).sum() / seen
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), float("inf"))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
                online_correct += correct
                online_seen += seen
            metrics, preds = evaluate(
                scorer, val_cells, val_rows, opts, condition, device, predictions=True
            )
            rec = {
                "epoch": epoch,
                "online_train_ce": float(np.mean(losses)),
                "online_train_accuracy": online_correct / online_seen,
                "validation": metrics,
                "train_accuracy_is_online": True,
            }
            history.append(rec)
            (out / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            if metrics["macro"]["nll"] < best_nll:
                best_nll, best_epoch = metrics["macro"]["nll"], epoch
                torch.save(scorer.state_dict(), out / "best.pt")
                (out / "best-predictions.json").write_text(
                    json.dumps(preds, indent=2) + "\n", encoding="utf-8"
                )
            torch.save(scorer.state_dict(), out / "final.pt")
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "validation": metrics["macro"],
                        "best_epoch": best_epoch,
                    }
                ),
                flush=True,
            )
        order_hash = order_digest.hexdigest()
        if order_digest_ref is not None and order_hash != order_digest_ref:
            raise ValueError("paired arms used different example order")
        order_digest_ref = order_hash
        selected = CellwiseCandidateScorer().to(device)
        selected.load_state_dict(
            torch.load(out / "best.pt", weights_only=True, map_location=device)
        )
        reloaded, reloaded_preds = evaluate(
            selected, val_cells, val_rows, opts, condition, device, predictions=True
        )
        if reloaded != history[best_epoch - 1]["validation"]:
            raise ValueError("best checkpoint replay mismatch")
        (out / "best-predictions-reloaded.json").write_text(
            json.dumps(reloaded_preds, indent=2) + "\n", encoding="utf-8"
        )
        results[condition] = {
            "best_epoch": best_epoch,
            "selected": reloaded,
            "final": history[-1]["validation"],
            "initial_tensor_sha256": initial_hash,
            "sample_order_sha256": order_hash,
            "best_sha256": sha(out / "best.pt"),
            "final_sha256": sha(out / "final.pt"),
            "training_seconds": time.perf_counter() - start,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
            "unique_train_images": len(train_rows),
            "unique_train_decision_examples": len(train_rows) * 4,
            "consumed_train_decision_examples": len(train_rows) * 4 * epochs,
            "selected_checkpoint_consumed_decision_examples": len(train_rows) * 4 * best_epoch,
            "optimizer_updates": int(np.ceil(len(train_rows) * 4 / BATCH)) * epochs,
            "history": history,
        }
        del scorer, selected, optimizer
        torch.cuda.empty_cache()
    result = {
        "status": "PASS_EXECUTION_ONLY",
        "conditions": results,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "official_public_validation_opened": False,
        "independent_evaluation": False,
        "note": (
            "Same exposed internal development validation split; paired readout diagnostic only."
        ),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
