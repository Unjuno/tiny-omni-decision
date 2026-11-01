"""Run a fixed, nested room-held-out CV of the frozen DCASE feature readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tiny_omni_decision.dcase2025_readout import DCASE2025Readout, binary_probability_metrics

MODES = ("audio", "video", "joint")
DEVTRAIN_PROTOCOL = "all official dev-train rooms; fixed budget; no validation selection"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def room_folds(rows: list[dict[str, object]], *, seed: int = 17):
    """One outer test room and one distinct inner validation room per fold."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rooms = sorted({str(row["room_id"]) for row in rows})
    if len(rooms) < 3:
        raise ValueError("nested room CV requires at least three rooms")
    row_rooms = np.asarray([str(row["room_id"]) for row in rows], dtype=object)
    folds = []
    row_parents = np.asarray([str(row["parent_recording"]) for row in rows], dtype=object)
    for test_room in rooms:
        remaining = [room for room in rooms if room != test_room]
        validation_room = min(
            remaining,
            key=lambda room: hashlib.sha256(f"{seed}\0{test_room}\0{room}".encode()).digest(),
        )
        train_rooms = set(remaining) - {validation_room}
        train_indices = np.flatnonzero(np.isin(row_rooms, list(train_rooms)))
        validation_indices = np.flatnonzero(row_rooms == validation_room)
        test_indices = np.flatnonzero(row_rooms == test_room)
        if not len(train_indices) or not len(validation_indices) or not len(test_indices):
            raise ValueError("room fold produced an empty partition")
        partition_parents = [
            set(row_parents[index].tolist())
            for index in (train_indices, validation_indices, test_indices)
        ]
        if any(
            partition_parents[left] & partition_parents[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError("parent recording overlaps nested room partitions")
        folds.append(
            {
                "test_room": test_room,
                "validation_room": validation_room,
                "train_rooms": sorted(train_rooms),
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "test_indices": test_indices,
            }
        )
    test_seen = np.concatenate([fold["test_indices"] for fold in folds])
    if not np.array_equal(np.sort(test_seen), np.arange(len(rows))):
        raise AssertionError("each sample must be an outer test example exactly once")
    return folds


def _metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float | int | None]:
    values = binary_probability_metrics(probabilities, targets)
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in values.items()
    }


def _standardize(train: np.ndarray, other: np.ndarray):
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    return (train - mean) / scale, (other - mean) / scale


def _event_prior(train_events: np.ndarray, train_targets: np.ndarray, test_events: np.ndarray):
    counts = np.full((13, 2), 1.0, dtype=np.float64)
    for event, target in zip(train_events, train_targets, strict=True):
        counts[int(event), int(target)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs[test_events]


def fit_fold_mode(
    mode: str,
    train_audio: np.ndarray,
    train_video: np.ndarray,
    train_events: np.ndarray,
    train_targets: np.ndarray,
    validation_audio: np.ndarray,
    validation_video: np.ndarray,
    validation_events: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
):
    """Fit one CPU model; validation chooses the epoch, outer test is untouched."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = DCASE2025Readout(mode, hidden_width=64, event_classes=13)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    xa = torch.from_numpy(train_audio)
    xv = torch.from_numpy(train_video)
    xe = torch.as_tensor(train_events, dtype=torch.long)
    y = torch.as_tensor(train_targets, dtype=torch.long)
    va = torch.from_numpy(validation_audio)
    vv = torch.from_numpy(validation_video)
    ve = torch.as_tensor(validation_events, dtype=torch.long)
    vy = torch.as_tensor(validation_targets, dtype=torch.long)
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        train_loss = 0.0
        for start in range(0, len(y), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xa[indices], xv[indices], xe[indices])
            loss = F.cross_entropy(logits, y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {mode} training loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            validation_logits = model(va, vv, ve)
            validation_loss = float(F.cross_entropy(validation_logits, vy))
        if not math.isfinite(validation_loss):
            raise FloatingPointError(f"non-finite {mode} validation loss at epoch {epoch + 1}")
        history.append(
            {
                "epoch": epoch + 1,
                "train_ce": train_loss / len(y),
                "validation_nll": validation_loss,
            }
        )
        if validation_loss < best_nll - 0.0001:
            best_nll = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise AssertionError("validation selector did not save a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        train_probs = model(xa, xv, xe).softmax(-1).numpy()
        validation_probs = model(va, vv, ve).softmax(-1).numpy()
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_nll": best_nll,
        "selected_train_metrics": _metrics(train_probs, train_targets),
        "selected_validation_metrics": _metrics(validation_probs, validation_targets),
        "history": history,
    }


def run(features: Path, output: Path, *, seed: int = 17, max_epochs: int = 30, patience: int = 5):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite nested room-CV output: {output}")
    progress_path = features / "progress.json"
    preflight_path = features / "preflight.json"
    manifest_path = features / "query-manifest.jsonl"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        progress.get("status") != "PASS"
        or progress.get("valid_rows") != progress.get("total_queries")
        or preflight.get("status") != "PASS"
        or preflight.get("training_protocol") != DEVTRAIN_PROTOCOL
        or preflight.get("devtest_label_content_opened") is not False
        or progress.get("identity", {}).get("query_ids_sha256")
        != preflight.get("selected_sample_id_order_sha256")
    ):
        raise ValueError("features must be a complete DCASE devtrain cache")
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    ids_hash = hashlib.sha256("".join(f"{row['sample_id']}\n" for row in rows).encode()).hexdigest()
    if (
        len(rows) != progress["total_queries"]
        or ids_hash != progress["identity"]["query_ids_sha256"]
    ):
        raise ValueError("devtrain rows differ from their frozen ID order")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate devtrain sample IDs")
    room_ids = np.asarray([str(row["room_id"]) for row in rows], dtype=object)
    parent_ids = np.asarray([str(row["parent_recording"]) for row in rows], dtype=object)
    events = np.asarray([row["event_class"] for row in rows], dtype=np.int64)
    targets = np.asarray([row["target_onscreen"] for row in rows], dtype=np.int64)
    audio = np.load(features / "features_audio.npy", mmap_mode="r")
    video = np.load(features / "features_video.npy", mmap_mode="r")
    if audio.shape != (len(rows), 768) or video.shape != audio.shape:
        raise ValueError("feature shape mismatch")
    valid = np.load(features / "valid_mask.npy", mmap_mode="r")
    processed = np.load(features / "processed_mask.npy", mmap_mode="r")
    if not valid.all() or not processed.all():
        raise ValueError("feature cache contains invalid or unprocessed rows")
    if not np.isin(targets, (0, 1)).all() or not np.isin(events, np.arange(13)).all():
        raise ValueError("unexpected target or event class")
    folds = room_folds(rows, seed=seed)
    output.mkdir(parents=True)
    split_report = [
        {
            "test_room": fold["test_room"],
            "validation_room": fold["validation_room"],
            "train_rooms": fold["train_rooms"],
            "train_rows": len(fold["train_indices"]),
            "validation_rows": len(fold["validation_indices"]),
            "test_rows": len(fold["test_indices"]),
        }
        for fold in folds
    ]
    config = {
        "source_split": "DCASE official dev-train only; no devtest/audit access",
        "feature_identity": progress["identity"],
        "query_manifest_sha256": sha256_file(manifest_path),
        "audio_features_sha256": sha256_file(features / "features_audio.npy"),
        "video_features_sha256": sha256_file(features / "features_video.npy"),
        "valid_mask_sha256": sha256_file(features / "valid_mask.npy"),
        "processed_mask_sha256": sha256_file(features / "processed_mask.npy"),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "readout_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1]
            / "src"
            / "tiny_omni_decision"
            / "dcase2025_readout.py"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "outer_protocol": "leave-one-room-out across 9 rooms; every sample tested once",
        "inner_validation": (
            "one different deterministic room per outer fold; group-disjoint from train and test"
        ),
        "split_seed": seed,
        "model_seed": seed,
        "modes": list(MODES),
        "architecture": "existing DCASE2025Readout; hidden width 64; frozen encoders/features",
        "optimizer": "AdamW lr=0.003 weight_decay=0.0001, batch_size=256, CPU",
        "standardization": "fit mean/std on the 7 inner-train rooms only",
        "checkpoint_rule": (
            "minimum inner-validation NLL; min_delta=0.0001; patience=5; max_epochs=30"
        ),
        "parent_groups": int(len(set(parent_ids.tolist()))),
        "rooms": sorted(set(room_ids.tolist())),
        "folds": split_report,
        "sealed_audit_accessed": False,
        "development_test_accessed": False,
    }
    (output / "effective-config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output / "status.json").write_text(
        json.dumps(
            {"status": "RUNNING", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    oof = {mode: np.full((len(rows), 2), np.nan, dtype=np.float64) for mode in MODES}
    oof["event_prior"] = np.full((len(rows), 2), np.nan, dtype=np.float64)
    fold_results = []
    started = time.perf_counter()
    for fold in folds:
        train_idx = fold["train_indices"]
        val_idx = fold["validation_indices"]
        test_idx = fold["test_indices"]
        xa_train = np.asarray(audio[train_idx], dtype=np.float32)
        xv_train = np.asarray(video[train_idx], dtype=np.float32)
        xa_val_raw = np.asarray(audio[val_idx], dtype=np.float32)
        xv_val_raw = np.asarray(video[val_idx], dtype=np.float32)
        xa_test_raw = np.asarray(audio[test_idx], dtype=np.float32)
        xv_test_raw = np.asarray(video[test_idx], dtype=np.float32)
        xa_train_norm, xa_val_norm = _standardize(xa_train, xa_val_raw)
        _, xa_test_norm = _standardize(xa_train, xa_test_raw)
        xv_train_norm, xv_val_norm = _standardize(xv_train, xv_val_raw)
        _, xv_test_norm = _standardize(xv_train, xv_test_raw)
        oof["event_prior"][test_idx] = _event_prior(
            events[train_idx], targets[train_idx], events[test_idx]
        )
        fold_result = {
            "test_room": fold["test_room"],
            "validation_room": fold["validation_room"],
            "modes": {},
        }
        for mode in MODES:
            model, fit = fit_fold_mode(
                mode,
                xa_train_norm,
                xv_train_norm,
                events[train_idx],
                targets[train_idx],
                xa_val_norm,
                xv_val_norm,
                events[val_idx],
                targets[val_idx],
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=256,
            )
            with torch.inference_mode():
                probs = (
                    model(
                        torch.from_numpy(xa_test_norm),
                        torch.from_numpy(xv_test_norm),
                        torch.as_tensor(events[test_idx], dtype=torch.long),
                    )
                    .softmax(-1)
                    .numpy()
                )
            if not np.isfinite(probs).all():
                raise FloatingPointError(f"non-finite outer-test probabilities for {mode}")
            oof[mode][test_idx] = probs
            fit["outer_test_metrics"] = _metrics(probs, targets[test_idx])
            fit["train_validation_nll_gap"] = (
                fit["selected_validation_metrics"]["nll"] - fit["selected_train_metrics"]["nll"]
            )
            fold_result["modes"][mode] = fit
        fold_results.append(fold_result)
        (output / "fold-history.json").write_text(
            json.dumps(fold_results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise AssertionError("out-of-fold predictions are incomplete")
    predictions_path = output / "oof-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            item = {
                "sample_id": row["sample_id"],
                "parent_recording": str(parent_ids[index]),
                "room_id": str(room_ids[index]),
                "event_class": int(events[index]),
                "target_onscreen": int(targets[index]),
                "probabilities": {name: values[index].tolist() for name, values in oof.items()},
            }
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    overall = {name: _metrics(values, targets) for name, values in oof.items()}
    per_room = {}
    for room in sorted(set(room_ids.tolist())):
        mask = room_ids == room
        per_room[room] = {
            name: _metrics(values[mask], targets[mask]) for name, values in oof.items()
        }
    room_macro = {
        name: {
            key: float(np.mean([per_room[room][name][key] for room in per_room]))
            for key in ("accuracy", "nll", "brier", "ece_15")
        }
        for name in [*MODES, "event_prior"]
    }
    result = {
        "status": "PASS",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(rows),
        "rooms": len(per_room),
        "parent_recordings": int(len(set(parent_ids.tolist()))),
        "target_counts": {str(label): int((targets == label).sum()) for label in (0, 1)},
        "overall_oof_metrics": overall,
        "room_macro_metrics": room_macro,
        "per_room_metrics": per_room,
        "fold_results": fold_results,
        "predictions_sha256": sha256_file(predictions_path),
        "selection_used_outer_test": False,
        "sealed_audit_accessed": False,
        "development_test_accessed": False,
        "interpretation_limit": (
            "Nine exposed dev-train rooms only; development CV, not independent benchmark "
            "or product evidence."
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "status.json").write_text(
        json.dumps({"status": "PASS", "completed_at": result["completed_at"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    result = run(
        args.features,
        args.output,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
