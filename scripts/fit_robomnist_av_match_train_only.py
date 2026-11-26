"""Fit the preregistered readouts from verified TRAIN features only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import fit_robomnist_av_match_readout as fit  # noqa: E402
from scripts.extract_robomnist_av_match_feature_cache import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    group_order,
    sha256_file,
)
from tiny_omni_decision.robomnist_av_match_evaluation import validate_examples  # noqa: E402
from tiny_omni_decision.robomnist_av_match_readout import (  # noqa: E402
    BilinearAVMatchScorer,
    SingleModalityMatchScorer,
)

EXPECTED_MISSING_VALIDATION_GROUPS = frozenset(
    {"Ar1_Vhigh_Ac1_Rp10_UNC085", "Ar1_Vhigh_Ac1_Rp22_UNC304"}
)
MIN_TRAIN_OUTPUT_FREE_BYTES = 1024**3


def _group_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {row[key] for row in rows for key in ("audio_group", "video_group")}


def _validate_train_only_coverage(
    *, expected_groups: set[str], train_groups: set[str], present_groups: set[str]
) -> set[str]:
    if not train_groups or not train_groups <= expected_groups:
        raise ValueError("training group IDs are empty or outside the frozen manifest")
    if not present_groups <= expected_groups:
        raise ValueError("feature cache contains groups outside the frozen manifest")
    missing = expected_groups - present_groups
    if not train_groups <= present_groups:
        raise ValueError("one or more frozen training groups lack verified features")
    if missing != EXPECTED_MISSING_VALIDATION_GROUPS:
        raise ValueError("incomplete cache does not have the two expected validation gaps")
    if missing & train_groups:
        raise ValueError("missing feature group intersects the training split")
    return missing


def run(manifest_path: Path, cache: Path, output: Path, *, steps: int = fit.TRAIN_STEPS):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite train-only output: {output}")
    if steps != fit.TRAIN_STEPS:
        raise ValueError("train-only stage must use the preregistered fixed update count")
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    free_bytes = shutil.disk_usage(cache).free
    if free_bytes < MIN_TRAIN_OUTPUT_FREE_BYTES:
        raise OSError(
            f"train-only output reserve reached: {free_bytes} free bytes "
            f"< {MIN_TRAIN_OUTPUT_FREE_BYTES}"
        )
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_MANIFEST_SHA256 or manifest_sha != fit.MANIFEST_SHA256:
        raise ValueError("frozen RoboMNIST manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = validate_examples(manifest)
    expected_groups, groups_by_split = group_order(manifest)
    expected_group_set = set(expected_groups)
    train_group_ids = _group_ids(rows["train"])
    validation_group_ids = _group_ids(rows["validation"])
    if train_group_ids & validation_group_ids:
        raise ValueError("source groups cross the frozen train/validation split")
    if train_group_ids != set(groups_by_split["train"]):
        raise ValueError("training rows do not cover the frozen training group list")

    cache_meta = json.loads((cache / "run.json").read_text(encoding="utf-8"))
    effective = cache_meta.get("effective", {})
    if (
        cache_meta.get("status") != "INCOMPLETE"
        or cache_meta.get("training") is not False
        or cache_meta.get("sealed_audit_opened") is not False
        or effective.get("manifest_sha256") != manifest_sha
        or effective.get("group_count") != len(expected_groups)
        or effective.get("split_group_counts")
        != {
            "train": len(groups_by_split["train"]),
            "validation": len(groups_by_split["validation"]),
        }
        or effective.get("minimum_free_bytes") != 2 * 1024**3
    ):
        raise ValueError("existing feature run metadata does not match the fixed cache contract")
    group_root = cache / "groups"
    present_group_ids = {path.name for path in group_root.iterdir() if path.is_dir()}
    missing = _validate_train_only_coverage(
        expected_groups=expected_group_set,
        train_groups=train_group_ids,
        present_groups=present_group_ids,
    )
    if missing != EXPECTED_MISSING_VALIDATION_GROUPS or not missing <= validation_group_ids:
        raise ValueError("missing cache entries are not the two fixed validation groups")

    # This call receives TRAIN IDs only: no validation feature file is opened.
    train_features, feature_hashes = fit._load_group_features(cache, train_group_ids)
    if set(train_features) != train_group_ids:
        raise ValueError("verified training feature coverage is incomplete")
    summaries, normalizer_hashes = fit._pool_features(train_features, rows["train"])
    train_audio, train_video, train_target, _ = fit._arrays(rows["train"], summaries)
    train_hash = hashlib.sha256(
        "\n".join(f"{key}:{feature_hashes[key]}" for key in sorted(train_group_ids)).encode()
    ).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < MIN_TRAIN_OUTPUT_FREE_BYTES:
        raise OSError("less than 1 GiB remains before train-only output creation")
    output.mkdir()
    started = time.perf_counter()
    meta: dict[str, Any] = {
        "status": "RUNNING",
        "stage": "train_only_validation_pending",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": fit.subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=fit.REPO_ROOT, text=True
        ).strip(),
        "implementation_sha256": {
            "train_only_script": sha256_file(Path(__file__).resolve()),
            "readout_fit_script": sha256_file(Path(fit.__file__).resolve()),
            "readout_module": sha256_file(
                REPO_ROOT / "src" / "tiny_omni_decision" / "robomnist_av_match_readout.py"
            ),
            "feature_extractor": sha256_file(
                REPO_ROOT / "scripts" / "extract_robomnist_av_match_feature_cache.py"
            ),
        },
        "manifest_sha256": manifest_sha,
        "input_feature_run": str(cache),
        "input_cache_status": cache_meta["status"],
        "input_cache_group_count": len(present_group_ids),
        "expected_group_count": len(expected_groups),
        "missing_validation_group_ids": sorted(missing),
        "train_group_count": len(train_group_ids),
        "train_row_count": len(rows["train"]),
        "train_feature_hashes_sha256": train_hash,
        "normalizer_hashes": normalizer_hashes,
        "training": {
            "seed": fit.SEED,
            "steps": steps,
            "rank": fit.RANK,
            "learning_rate": fit.LEARNING_RATE,
            "weight_decay": fit.WEIGHT_DECAY,
            "optimizer": "AdamW",
            "batch": "full 1,560-row train set each update; deterministic order",
            "gradient_clip_norm": 5.0,
            "validation_used": False,
        },
        "validation_features_read": False,
        "validation_predictions_written": False,
        "sealed_audit_opened": False,
        "gpu_used": False,
    }
    fit._write_json(output / "run.json", meta)
    config = {
        **meta["training"],
        "stage_order_deviation": (
            "train-only stage ran before two validation features were available; "
            "the frozen cache remains unchanged and no validation features were read"
        ),
        "training_feature_hashes_sha256": train_hash,
        "normalizer_hashes": normalizer_hashes,
    }
    fit._write_json(output / "config.json", config)
    meta["config_sha256"] = sha256_file(output / "config.json")
    fit._write_json(output / "run.json", meta)

    try:
        torch.set_num_threads(4)
        torch.manual_seed(fit.SEED)
        models = {
            "joint": BilinearAVMatchScorer(rank=fit.RANK),
            "audio_only": SingleModalityMatchScorer(train_audio.shape[1]),
            "video_only": SingleModalityMatchScorer(train_video.shape[1]),
        }
        histories: dict[str, Any] = {}
        training_seconds: dict[str, float] = {}
        for name, model in models.items():
            torch.manual_seed(fit.SEED)
            inputs = (
                (train_audio, train_video)
                if name == "joint"
                else (train_audio if name == "audio_only" else train_video,)
            )
            started_model = time.perf_counter()
            histories[name] = fit._fit(model, inputs, train_target, steps=steps)
            training_seconds[name] = time.perf_counter() - started_model

        checkpoint_path = output / "readouts.pt"
        torch.save({name: model.state_dict() for name, model in models.items()}, checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded = {
            "joint": BilinearAVMatchScorer(rank=fit.RANK),
            "audio_only": SingleModalityMatchScorer(train_audio.shape[1]),
            "video_only": SingleModalityMatchScorer(train_video.shape[1]),
        }
        for name, model in reloaded.items():
            model.load_state_dict(state[name], strict=True)
        for name, model in models.items():
            inputs = (
                (train_audio, train_video)
                if name == "joint"
                else (train_audio if name == "audio_only" else train_video,)
            )
            reloaded_inputs = inputs
            original_logits = model(*inputs) if name == "joint" else model(inputs[0])
            reloaded_logits = (
                reloaded[name](*reloaded_inputs)
                if name == "joint"
                else reloaded[name](reloaded_inputs[0])
            )
            if not torch.equal(original_logits, reloaded_logits):
                raise RuntimeError(f"save/reload changed train logits for {name}")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only train-only fit unexpectedly initialized CUDA")
        fit._write_json(output / "training-history.json", histories)
        result = {
            **meta,
            "status": "TRAIN_COMPLETE_VALIDATION_PENDING",
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "training_history": histories,
            "training_seconds_by_readout": training_seconds,
            "parameters": {name: model.trainable_parameters for name, model in models.items()},
            "readout_checkpoint_sha256": sha256_file(checkpoint_path),
            "readout_checkpoint_bytes": checkpoint_path.stat().st_size,
            "reload_train_logits_exact": True,
            "runtime": {
                "python": platform.python_version(),
                "numpy": fit.np.__version__,
                "torch": torch.__version__,
                "device": "CPU",
                "cuda_initialized": torch.cuda.is_initialized(),
            },
            "quality_metrics": None,
            "quality_claim_scope": None,
            "limitations": [
                "no validation feature or prediction was read during fit",
                "validation must use all 1,000 fixed groups and this exact checkpoint",
                "this staged execution differs from the preflight's one-shot operation order",
            ],
        }
        fit._write_json(output / "result.json", result)
        fit._write_json(output / "run.json", result)
        return result
    except BaseException as exc:
        meta.update(
            {
                "status": "INCOMPLETE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        fit._write_json(output / "run.json", meta)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=fit.TRAIN_STEPS)
    args = parser.parse_args()
    result = run(
        args.manifest.resolve(strict=True),
        args.cache.resolve(strict=True),
        args.output.resolve(),
        steps=args.steps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
