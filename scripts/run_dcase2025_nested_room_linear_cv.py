"""Evaluate an affine frozen-feature readout on nested held-out DCASE rooms."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_dcase2025_nested_room_cv import (
    MODES,
    _event_prior,
    _metrics,
    _standardize,
    room_folds,
    sha256_file,
)
from tiny_omni_decision.dcase2025_readout import (
    DCASE2025LinearReadout,
)


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
):
    """Train and choose by inner-room NLL; outer-room labels stay untouched."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = DCASE2025LinearReadout(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
    train = (
        torch.from_numpy(train_audio),
        torch.from_numpy(train_video),
        torch.as_tensor(train_events, dtype=torch.long),
        torch.as_tensor(train_targets, dtype=torch.long),
    )
    valid = (
        torch.from_numpy(validation_audio),
        torch.from_numpy(validation_video),
        torch.as_tensor(validation_events, dtype=torch.long),
        torch.as_tensor(validation_targets, dtype=torch.long),
    )
    best_state = None
    best_nll = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train[3]), generator=generator)
        train_losses = []
        for indices in order.split(256):
            logits = model(*(value[indices] for value in train[:3]))
            loss = F.cross_entropy(logits, train[3][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite linear train loss ({mode}, epoch={epoch + 1})"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            valid_logits = model(*valid[:3])
            valid_nll = float(F.cross_entropy(valid_logits, valid[3]))
            train_logits = model(*train[:3])
            train_nll = float(F.cross_entropy(train_logits, train[3]))
        if not np.isfinite(valid_nll) or not np.isfinite(train_nll):
            raise FloatingPointError(
                f"non-finite linear evaluation loss ({mode}, epoch={epoch + 1})"
            )
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_batch_ce": float(np.mean(train_losses)),
                "train_nll": train_nll,
                "validation_nll": valid_nll,
            }
        )
        if valid_nll < best_nll - 0.0001:
            best_nll = valid_nll
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise AssertionError("no finite inner-validation checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        valid_probs = model(*valid[:3]).softmax(-1).numpy()
        train_probs = model(*train[:3]).softmax(-1).numpy()
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
        "selected_train_metrics": _metrics(train_probs, train_targets),
        "selected_validation_metrics": _metrics(valid_probs, validation_targets),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def run(features: Path, output: Path, *, seed: int = 17, max_epochs: int = 30, patience: int = 5):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    progress = json.loads((features / "progress.json").read_text(encoding="utf-8"))
    preflight = json.loads((features / "preflight.json").read_text(encoding="utf-8"))
    if (
        progress.get("status") != "PASS"
        or progress.get("valid_rows") != progress.get("total_queries")
        or preflight.get("status") != "PASS"
        or preflight.get("training_protocol")
        != "all official dev-train rooms; fixed budget; no validation selection"
        or preflight.get("devtest_label_content_opened") is not False
        or progress.get("identity", {}).get("query_ids_sha256")
        != preflight.get("selected_sample_id_order_sha256")
    ):
        raise ValueError("features must be a complete frozen DCASE dev-train cache")
    manifest_path = features / "query-manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    query_ids_hash = hashlib.sha256(
        "".join(f"{row['sample_id']}\n" for row in rows).encode()
    ).hexdigest()
    if (
        len(rows) != progress["total_queries"]
        or len({row["sample_id"] for row in rows}) != len(rows)
        or query_ids_hash != progress.get("identity", {}).get("query_ids_sha256")
        or query_ids_hash != preflight.get("selected_sample_id_order_sha256")
    ):
        raise ValueError("dev-train manifest row count differs from feature progress")
    ids_hash = sha256_file(manifest_path)
    audio = np.load(features / "features_audio.npy", mmap_mode="r")
    video = np.load(features / "features_video.npy", mmap_mode="r")
    if audio.shape != (len(rows), 768) or video.shape != audio.shape:
        raise ValueError("feature shape mismatch")
    if not np.load(features / "valid_mask.npy", mmap_mode="r").all():
        raise ValueError("invalid feature rows")
    if not np.load(features / "processed_mask.npy", mmap_mode="r").all():
        raise ValueError("unprocessed feature rows")
    room_ids = np.asarray([str(row["room_id"]) for row in rows], dtype=object)
    parent_ids = np.asarray([str(row["parent_recording"]) for row in rows], dtype=object)
    events = np.asarray([row["event_class"] for row in rows], dtype=np.int64)
    targets = np.asarray([row["target_onscreen"] for row in rows], dtype=np.int64)
    if not np.isin(targets, (0, 1)).all() or not np.isin(events, np.arange(13)).all():
        raise ValueError("unexpected target or event class")
    folds = room_folds(rows, seed=seed)
    output.mkdir(parents=True)
    config = {
        "protocol": (
            "DCASE official dev-train only; nested leave-one-room-out; no devtest/audit access"
        ),
        "feature_identity": progress["identity"],
        "query_manifest_sha256": ids_hash,
        "query_id_order_sha256": query_ids_hash,
        "audio_features_sha256": sha256_file(features / "features_audio.npy"),
        "video_features_sha256": sha256_file(features / "features_video.npy"),
        "preflight_sha256": sha256_file(features / "preflight.json"),
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
        "seed": seed,
        "model": "single affine layer over frozen modality features plus 13-way event one-hot",
        "modes": list(MODES),
        "optimizer": "AdamW lr=0.003 weight_decay=0.0001, batch_size=256, CPU",
        "selection": "inner-room NLL; min_delta=0.0001; patience=5; max_epochs=30",
        "outer_protocol": "leave-one-room-out across 9 rooms; every sample tested once",
        "development_test_accessed": False,
        "sealed_audit_accessed": False,
    }
    (output / "effective-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    probabilities = {mode: np.full((len(rows), 2), np.nan) for mode in MODES}
    probabilities["event_prior"] = np.full((len(rows), 2), np.nan)
    fold_records = []
    started = time.perf_counter()
    for fold in folds:
        train_idx, valid_idx, test_idx = (
            fold["train_indices"],
            fold["validation_indices"],
            fold["test_indices"],
        )
        train_a, valid_a = _standardize(np.asarray(audio[train_idx]), np.asarray(audio[valid_idx]))
        train_v, valid_v = _standardize(np.asarray(video[train_idx]), np.asarray(video[valid_idx]))
        _, test_a = _standardize(np.asarray(audio[train_idx]), np.asarray(audio[test_idx]))
        _, test_v = _standardize(np.asarray(video[train_idx]), np.asarray(video[test_idx]))
        probabilities["event_prior"][test_idx] = _event_prior(
            events[train_idx], targets[train_idx], events[test_idx]
        )
        record = {
            "test_room": fold["test_room"],
            "validation_room": fold["validation_room"],
            "modes": {},
        }
        for mode in MODES:
            model, fit = fit_fold_mode(
                mode,
                train_a,
                train_v,
                events[train_idx],
                targets[train_idx],
                valid_a,
                valid_v,
                events[valid_idx],
                targets[valid_idx],
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
            )
            fit["train_validation_nll_gap"] = (
                fit["selected_validation_metrics"]["nll"] - fit["selected_train_metrics"]["nll"]
            )
            with torch.inference_mode():
                probabilities[mode][test_idx] = (
                    model(
                        torch.from_numpy(test_a),
                        torch.from_numpy(test_v),
                        torch.as_tensor(events[test_idx], dtype=torch.long),
                    )
                    .softmax(-1)
                    .numpy()
                )
            fit["outer_test_metrics"] = _metrics(probabilities[mode][test_idx], targets[test_idx])
            record["modes"][mode] = fit
        fold_records.append(record)
        (output / "fold-history.json").write_text(
            json.dumps(fold_records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise AssertionError("incomplete or non-finite OOF probabilities")
    predictions_path = output / "oof-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            stream.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "room_id": room_ids[index],
                        "parent_recording": parent_ids[index],
                        "target_onscreen": int(targets[index]),
                        "probabilities": {
                            name: value[index].tolist() for name, value in probabilities.items()
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    room_metrics = {}
    for room in sorted(set(room_ids.tolist())):
        mask = room_ids == room
        room_metrics[room] = {
            name: _metrics(value[mask], targets[mask]) for name, value in probabilities.items()
        }
    room_macro = {
        name: {
            key: float(np.mean([room_metrics[room][name][key] for room in room_metrics]))
            for key in ("accuracy", "nll", "brier", "ece_15")
        }
        for name in probabilities
    }
    result = {
        "status": "PASS",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": len(rows),
        "rooms": len(set(room_ids)),
        "parent_recordings": len(set(parent_ids)),
        "overall_oof_metrics": {
            name: _metrics(value, targets) for name, value in probabilities.items()
        },
        "room_macro_metrics": room_macro,
        "per_room_metrics": room_metrics,
        "fold_results": fold_records,
        "predictions_sha256": sha256_file(predictions_path),
        "development_test_accessed": False,
        "sealed_audit_accessed": False,
        "selection_used_outer_test": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
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
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
