"""Train-only grouped CV for one lower-capacity RoboMNIST AV readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import fit_robomnist_av_match_readout as reference_fit  # noqa: E402
from tiny_omni_decision.robomnist_av_match_evaluation import (  # noqa: E402
    classification_metrics,
    paired_bootstrap_delta,
)
from tiny_omni_decision.robomnist_av_match_readout import (  # noqa: E402
    BilinearAVMatchScorer,
)
from tiny_omni_decision.robomnist_capacity_cv import (  # noqa: E402
    make_pair_folds,
    select_capacity_rank,
)

MANIFEST_SHA256 = "3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f"
REFERENCE_EVAL = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-readout-20261104-frozen-validation-attempt-01"
)
REFERENCE_TRAIN_FIT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-readout-20261103-attempt-02-train-only"
)
SEED = 17
FOLD_COUNT = 5
STEPS = 400
CANDIDATE_RANK = 8
REFERENCE_RANK = 32
RANKS = (CANDIDATE_RANK, REFERENCE_RANK)
MIN_NLL_GAIN = 0.02
MAX_ACCURACY_LOSS = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _load_group_features(cache: Path, group_ids: set[str]):
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hashes: dict[str, str] = {}
    for group_id in sorted(group_ids):
        group_dir = cache / "groups" / group_id
        receipt = json.loads((group_dir / "receipt.json").read_text(encoding="utf-8"))
        feature_path = group_dir / "features.npz"
        if (
            receipt.get("status") != "COMPLETE"
            or receipt.get("group_id") != group_id
            or _sha256(feature_path) != receipt.get("feature_sha256")
        ):
            raise ValueError(f"training feature receipt/hash invalid for {group_id}")
        with np.load(feature_path, allow_pickle=False) as data:
            if str(data["group_id"].item()) != group_id:
                raise ValueError(f"training feature group mismatch for {group_id}")
            audio = np.asarray(data["audio_features"], dtype=np.float32)
            video = np.asarray(data["video_features"], dtype=np.float32)
        if (
            audio.shape != (8, 384)
            or video.shape != (8, 768)
            or not np.isfinite(audio).all()
            or not np.isfinite(video).all()
        ):
            raise ValueError(f"training feature shape/value mismatch for {group_id}")
        features[group_id] = audio, video
        hashes[group_id] = receipt["feature_sha256"]
    return features, hashes


def _train_arrays(rows, summaries):
    audio = np.stack([summaries[row["audio_group"]][0] for row in rows])
    video = np.stack([summaries[row["video_group"]][1] for row in rows])
    target = np.asarray([row["target"] for row in rows], dtype=np.int64)
    return torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(target)


def _row_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    metrics = classification_metrics(probabilities, targets)
    return {key: float(value) for key, value in metrics.items()}


def _prediction(model, audio: torch.Tensor, video: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(audio, video).float(), dim=-1).cpu().numpy()


def _load_training_manifest(manifest_path: Path, cache: Path):
    manifest_sha = _sha256(manifest_path)
    if manifest_sha != MANIFEST_SHA256:
        raise ValueError("frozen RoboMNIST manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows = manifest.get("examples", {}).get("train")
    pair_units = manifest.get("pair_units", {}).get("train")
    if not isinstance(train_rows, list) or not isinstance(pair_units, list):
        raise ValueError("frozen training examples/pair units are missing")
    if len(train_rows) != 1560 or len(pair_units) != 390:
        raise ValueError("frozen training rows or pair-unit count changed")
    pair_by_id = {unit["pair_unit_id"]: unit for unit in pair_units}
    if len(pair_by_id) != len(pair_units):
        raise ValueError("duplicate training pair-unit IDs")
    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        pair_id = row.get("pair_unit_id")
        unit = pair_by_id.get(pair_id)
        if unit is None or row.get("target") not in (0, 1):
            raise ValueError("invalid training row identity or target")
        if row.get("condition") != unit.get("condition"):
            raise ValueError("training row/pair condition mismatch")
        if row.get("audio_group") not in (unit.get("group_a"), unit.get("group_b")):
            raise ValueError("training audio group is outside its pair unit")
        if row.get("video_group") not in (unit.get("group_a"), unit.get("group_b")):
            raise ValueError("training video group is outside its pair unit")
        rows_by_pair.setdefault(pair_id, []).append(row)
    if set(rows_by_pair) != set(pair_by_id):
        raise ValueError("training examples do not cover all pair units")
    for pair_id, rows in rows_by_pair.items():
        targets = [row["target"] for row in rows]
        if len(rows) != 4 or targets.count(0) != 2 or targets.count(1) != 2:
            raise ValueError(f"pair unit {pair_id} is not a balanced four-row unit")
    run_meta = json.loads((cache / "run.json").read_text(encoding="utf-8"))
    if (
        run_meta.get("status") != "COMPLETE"
        or run_meta.get("effective", {}).get("manifest_sha256") != manifest_sha
        or run_meta.get("effective", {}).get("group_count") != 1780
        or run_meta.get("training") is not False
        or run_meta.get("sealed_audit_opened") is not False
    ):
        raise ValueError("feature cache is incomplete or mismatched")
    fold_by_pair = make_pair_folds(pair_units, fold_count=FOLD_COUNT, seed=SEED)
    return manifest, train_rows, pair_units, fold_by_pair, run_meta


def _check_reference_artifacts():
    fit_meta = json.loads((REFERENCE_TRAIN_FIT / "run.json").read_text(encoding="utf-8"))
    checkpoint = REFERENCE_TRAIN_FIT / "readouts.pt"
    eval_meta = json.loads((REFERENCE_EVAL / "result.json").read_text(encoding="utf-8"))
    predictions = REFERENCE_EVAL / "validation-predictions.jsonl"
    if (
        fit_meta.get("status") != "TRAIN_COMPLETE_VALIDATION_PENDING"
        or fit_meta.get("manifest_sha256") != MANIFEST_SHA256
        or fit_meta.get("readout_checkpoint_sha256") != _sha256(checkpoint)
        or eval_meta.get("frozen_fit_checkpoint_sha256") != _sha256(checkpoint)
        or eval_meta.get("manifest_sha256") != MANIFEST_SHA256
        or eval_meta.get("validation_count") != 2000
        or eval_meta.get("sealed_audit_opened") is not False
        or _sha256(predictions) != eval_meta.get("prediction_file_sha256")
    ):
        raise ValueError("preserved reference fit/evaluation integrity check failed")
    return fit_meta, eval_meta, predictions, checkpoint


def _baseline_probabilities(prediction_path: Path, validation_rows):
    parsed = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    if len(parsed) != len(validation_rows):
        raise ValueError("reference predictions do not cover the frozen validation rows")
    audio, video = [], []
    for index, (record, row) in enumerate(zip(parsed, validation_rows, strict=True)):
        if (
            record.get("example_index") != index
            or record.get("pair_unit_id") != row.get("pair_unit_id")
            or record.get("audio_group") != row.get("audio_group")
            or record.get("video_group") != row.get("video_group")
            or record.get("target_option_id") != ("same" if row["target"] else "different")
        ):
            raise ValueError("reference prediction identity does not match validation order")
        audio.append(record["predictions"]["audio_only"]["probabilities"])
        video.append(record["predictions"]["video_only"]["probabilities"])
    return np.asarray(audio, dtype=np.float64), np.asarray(video, dtype=np.float64)


def run(manifest_path: Path, cache: Path, output: Path) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    if not output.is_absolute():
        raise ValueError("experiment output must be an absolute path")
    if output.drive.upper() != "D:":
        raise ValueError("this experiment's isolated outputs must be on D:")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise OSError("require at least 2 GiB free on the output volume")

    manifest, train_rows, pair_units, fold_by_pair, cache_meta = _load_training_manifest(
        manifest_path, cache
    )
    reference_fit_meta, reference_eval_meta, reference_predictions, reference_checkpoint = (
        _check_reference_artifacts()
    )
    train_pair_ids = {unit["pair_unit_id"] for unit in pair_units}
    if set(fold_by_pair) != train_pair_ids:
        raise ValueError("fold assignment does not cover exactly the training pair units")
    train_group_ids = {
        row[key] for row in train_rows for key in ("audio_group", "video_group")
    }
    if len(train_group_ids) != 780:
        raise ValueError("frozen training feature groups changed")
    features, feature_hashes = _load_group_features(cache, train_group_ids)

    torch.set_num_threads(4)
    oof_outputs: dict[int, np.ndarray] = {
        rank: np.full((len(train_rows), 2), np.nan, dtype=np.float64) for rank in RANKS
    }
    row_fold = [fold_by_pair[row["pair_unit_id"]] for row in train_rows]
    fold_histories: dict[str, list[dict[str, Any]]] = {str(rank): [] for rank in RANKS}
    started = time.perf_counter()
    for fold in range(FOLD_COUNT):
        fold_train = [row for row in train_rows if fold_by_pair[row["pair_unit_id"]] != fold]
        fold_holdout_indices = [index for index, value in enumerate(row_fold) if value == fold]
        fold_holdout = [train_rows[index] for index in fold_holdout_indices]
        fold_train_ids = {row["pair_unit_id"] for row in fold_train}
        fold_holdout_ids = {row["pair_unit_id"] for row in fold_holdout}
        if fold_train_ids & fold_holdout_ids or not fold_holdout_ids:
            raise ValueError("pair unit crossed or is missing from a CV fold")
        fold_features, normalizer_hashes = reference_fit._pool_features(features, fold_train)
        train_audio, train_video, train_target, _ = reference_fit._arrays(
            fold_train, fold_features
        )
        hold_audio, hold_video, hold_target, _ = reference_fit._arrays(
            fold_holdout, fold_features
        )
        for rank in RANKS:
            torch.manual_seed(SEED)
            model = BilinearAVMatchScorer(rank=rank)
            fit_started = time.perf_counter()
            history = reference_fit._fit(
                model, (train_audio, train_video), train_target, steps=STEPS
            )
            fold_seconds = time.perf_counter() - fit_started
            probabilities = reference_fit._predict(
                model, hold_audio, hold_video, mode="joint"
            ).astype(np.float64)
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("non-finite out-of-fold probabilities")
            oof_outputs[rank][fold_holdout_indices] = probabilities
            fold_histories[str(rank)].append(
                {
                    "fold": fold,
                    "train_pair_units": len(fold_train_ids),
                    "holdout_pair_units": len(fold_holdout_ids),
                    "train_rows": len(fold_train),
                    "holdout_rows": len(fold_holdout),
                    "normalizer_hashes": normalizer_hashes,
                    "final_train_ce": history[-1]["train_ce"],
                    "fit_seconds": fold_seconds,
                }
            )
    cv_seconds = time.perf_counter() - started
    targets = np.asarray([row["target"] for row in train_rows], dtype=np.int64)
    if any(not np.isfinite(values).all() for values in oof_outputs.values()):
        raise RuntimeError("OOF predictions are incomplete")
    cv_metrics = {str(rank): _row_metrics(values, targets) for rank, values in oof_outputs.items()}
    selected_rank = select_capacity_rank(
        {int(rank): metrics for rank, metrics in cv_metrics.items()},
        candidate_rank=CANDIDATE_RANK,
        reference_rank=REFERENCE_RANK,
        min_nll_gain=MIN_NLL_GAIN,
        max_accuracy_loss=MAX_ACCURACY_LOSS,
    )
    run_id = output.name
    output.mkdir()
    train_hash = hashlib.sha256(
        "\n".join(f"{key}:{feature_hashes[key]}" for key in sorted(train_group_ids)).encode()
    ).hexdigest()
    config = {
        "experiment_id": run_id,
        "seed": SEED,
        "fold_count": FOLD_COUNT,
        "fold_unit": "pair_unit_id; source groups are required to appear in one pair only",
        "candidate_ranks": list(RANKS),
        "optimizer": "AdamW",
        "learning_rate": reference_fit.LEARNING_RATE,
        "weight_decay": reference_fit.WEIGHT_DECAY,
        "steps_per_fold_fit": STEPS,
        "full_train_refit_steps": STEPS,
        "selection_rule": {
            "primary": (
                f"rank {CANDIDATE_RANK} OOF NLL at least {MIN_NLL_GAIN} below "
                f"rank {REFERENCE_RANK}"
            ),
            "guardrail": (
                f"rank {CANDIDATE_RANK} OOF Accuracy no more than "
                f"{MAX_ACCURACY_LOSS} below rank {REFERENCE_RANK}"
            ),
            "otherwise": (
                f"retain reference rank {REFERENCE_RANK}; do not run another "
                "validation evaluation"
            ),
        },
        "validation_used_for_training_or_selection": False,
        "sealed_audit_opened": False,
    }
    config_path = output / "config.json"
    _write_json(config_path, config)
    oof_path = output / "training-oof-predictions.jsonl"
    with oof_path.open("x", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(train_rows):
            handle.write(
                json.dumps(
                    {
                        "row_index": index,
                        "pair_unit_id": row["pair_unit_id"],
                        "audio_group": row["audio_group"],
                        "video_group": row["video_group"],
                        "fold": row_fold[index],
                        "target": int(row["target"]),
                        "rank_8_probabilities": oof_outputs[CANDIDATE_RANK][index].tolist(),
                        "rank_32_probabilities": oof_outputs[REFERENCE_RANK][index].tolist(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())

    selected_checkpoint_sha = None
    train_seconds = 0.0
    selected_train_metrics: dict[str, float] | None = None
    validation_results: dict[str, Any] | None = None
    validation_predictions_sha = None
    if selected_rank == REFERENCE_RANK:
        selection_status = "REFERENCE_RETAINED_NO_VALIDATION_REPLAY"
    else:
        train_features = features
        validation_rows = manifest.get("examples", {}).get("validation")
        if not isinstance(validation_rows, list) or len(validation_rows) != 2000:
            raise ValueError("frozen development validation rows are unavailable")
        validation_groups = {
            row[key] for row in validation_rows for key in ("audio_group", "video_group")
        }
        if validation_groups & train_group_ids:
            raise ValueError("training/validation source groups overlap")
        # This is reached only after rank is selected using train-only OOF NLL.
        val_features, val_hashes = _load_group_features(cache, validation_groups)
        all_features = {**train_features, **val_features}
        summaries, normalizer_hashes = reference_fit._pool_features(all_features, train_rows)
        full_audio, full_video, full_target, _ = reference_fit._arrays(train_rows, summaries)
        val_audio, val_video, val_target, val_pair_ids = reference_fit._arrays(
            validation_rows, summaries
        )
        torch.manual_seed(SEED)
        selected_model = BilinearAVMatchScorer(rank=selected_rank)
        train_started = time.perf_counter()
        history = reference_fit._fit(
            selected_model, (full_audio, full_video), full_target, steps=STEPS
        )
        train_seconds = time.perf_counter() - train_started
        selected_train_metrics = _row_metrics(
            reference_fit._predict(selected_model, full_audio, full_video, mode="joint"),
            full_target.numpy(),
        )
        candidate_probabilities = _prediction(selected_model, val_audio, val_video)
        targets_val = val_target.numpy()
        control_audio, control_video = _baseline_probabilities(
            reference_predictions, validation_rows
        )
        validation_results = {
            "rank_8_candidate": _row_metrics(candidate_probabilities, targets_val),
            "rank_32_preserved_reference": reference_eval_meta["metrics"]["joint"],
            "audio_only_preserved_reference": _row_metrics(control_audio, targets_val),
            "video_only_preserved_reference": _row_metrics(control_video, targets_val),
            "paired_bootstrap_vs_audio_only": paired_bootstrap_delta(
                candidate_probabilities.argmax(-1) == targets_val,
                control_audio.argmax(-1) == targets_val,
                val_pair_ids,
                seed=SEED,
                replicates=2000,
            ),
            "paired_bootstrap_vs_video_only": paired_bootstrap_delta(
                candidate_probabilities.argmax(-1) == targets_val,
                control_video.argmax(-1) == targets_val,
                val_pair_ids,
                seed=SEED,
                replicates=2000,
            ),
            "validation_ids_sha256": hashlib.sha256(
                "\n".join(
                    f"{row['pair_unit_id']}:{row['audio_group']}:{row['video_group']}:{row['target']}"
                    for row in validation_rows
                ).encode()
            ).hexdigest(),
            "validation_feature_hashes_sha256": hashlib.sha256(
                "\n".join(f"{key}:{val_hashes[key]}" for key in sorted(val_hashes)).encode()
            ).hexdigest(),
            "normalizer_hashes": normalizer_hashes,
            "checkpoint_selection_or_calibration_used_validation": False,
        }
        checkpoint_path = output / "selected-readout.pt"
        torch.save(selected_model.state_dict(), checkpoint_path)
        selected_checkpoint_sha = _sha256(checkpoint_path)
        reloaded = BilinearAVMatchScorer(rank=selected_rank)
        reloaded.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True
        )
        if not np.array_equal(_prediction(reloaded, val_audio, val_video), candidate_probabilities):
            raise RuntimeError("save/reload changed candidate probabilities")
        predictions_path = output / "validation-predictions.jsonl"
        with predictions_path.open("x", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(validation_rows):
                handle.write(
                    json.dumps(
                        {
                            "example_index": index,
                            "pair_unit_id": row["pair_unit_id"],
                            "audio_group": row["audio_group"],
                            "video_group": row["video_group"],
                            "target": int(row["target"]),
                            "probabilities": candidate_probabilities[index].tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        validation_predictions_sha = _sha256(predictions_path)
        selection_status = "LOWER_RANK_SELECTED_AND_EVALUATED_ON_DEVELOPMENT_VALIDATION"

    result = {
        "status": selection_status,
        "experiment_id": run_id,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": MANIFEST_SHA256,
        "feature_cache_run": str(cache),
        "feature_cache_run_sha256": _sha256(cache / "run.json"),
        "train_feature_groups": len(train_group_ids),
        "train_feature_hashes_sha256": train_hash,
        "reference_fit_checkpoint_sha256": _sha256(reference_checkpoint),
        "reference_validation_predictions_sha256": _sha256(reference_predictions),
        "reference_fit_status": reference_fit_meta.get("status"),
        "cv": {
            "fold_count": FOLD_COUNT,
            "pair_units": len(pair_units),
            "rows": len(train_rows),
            "all_groups_source_disjoint_across_pair_units": True,
            "metrics_by_rank": cv_metrics,
            "fold_histories": fold_histories,
            "selected_rank": selected_rank,
            "selection_thresholds": {
                "minimum_nll_gain": MIN_NLL_GAIN,
                "maximum_accuracy_loss": MAX_ACCURACY_LOSS,
            },
            "wall_seconds": cv_seconds,
        },
        "full_train_refit": {
            "rank": selected_rank if selected_rank == CANDIDATE_RANK else None,
            "steps": STEPS if selected_rank == CANDIDATE_RANK else 0,
            "wall_seconds": train_seconds,
            "metrics": selected_train_metrics,
            "checkpoint_sha256": selected_checkpoint_sha,
        },
        "development_validation": validation_results,
        "validation_predictions_sha256": validation_predictions_sha,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "CPU; CUDA_VISIBLE_DEVICES empty",
            "cuda_initialized": torch.cuda.is_initialized(),
        },
        "sealed_audit_opened": False,
        "raw_media_loaded": False,
        "limits": [
            "one seed and one train-only five-fold capacity comparison",
            "training pair groups are disjoint but rooms/devices/scenes are not held out",
            "development validation was already observed by the preserved rank-32 evaluation",
            "not a blind estimate or product-quality pass",
        ],
    }
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.cache, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
