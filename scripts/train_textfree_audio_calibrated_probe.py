"""CPU-only speaker-held-out calibration check for the frozen audio probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from train_textfree_audio_probe import (
    AUDIO_REVISION,
    LABELS,
    SOURCE,
    _check_groups,
    _load_rows,
    _metrics,
)

from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.textfree_audio import (
    TextFreeAudioReadout,
    fit_scalar_temperature,
    speaker_group_split,
)


def run(feature_cache: Path, sample_dir: Path, output_dir: Path) -> dict[str, Any]:
    feature_cache, sample_dir, output_dir = (
        feature_cache.resolve(),
        sample_dir.resolve(),
        output_dir.resolve(),
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output path: {output_dir}")
    train_rows = _load_rows(sample_dir / "train.jsonl", "train", sample_dir)
    validation_rows = _load_rows(sample_dir / "validation.jsonl", "validation", sample_dir)
    _check_groups(train_rows, validation_rows)
    cache = torch.load(feature_cache, map_location="cpu", weights_only=True)
    if cache.get("source_revision") != SOURCE or cache.get("encoder_revision") != AUDIO_REVISION:
        raise ValueError("frozen feature cache source or encoder revision mismatch")
    if cache["train_ids"] != [str(row["id"]) for row in train_rows] or cache["validation_ids"] != [
        str(row["id"]) for row in validation_rows
    ]:
        raise ValueError("feature cache IDs/order do not match frozen manifests")
    train_targets = torch.tensor([LABELS.index(row["target"]) for row in train_rows])
    validation_targets = torch.tensor([LABELS.index(row["target"]) for row in validation_rows])
    if (
        cache["train_targets"] != train_targets.tolist()
        or cache["validation_targets"] != validation_targets.tolist()
    ):
        raise ValueError("cached targets differ from frozen manifests")
    x_train = cache["train_features"].float()
    x_validation = cache["validation_features"].float()
    if x_train.shape != (len(train_rows), 384) or x_validation.shape != (len(validation_rows), 384):
        raise ValueError("unexpected cached feature shape")
    if not torch.isfinite(x_train).all() or not torch.isfinite(x_validation).all():
        raise ValueError("nonfinite cached features")

    fit_indices, calibration_indices = speaker_group_split(
        [row["speaker_group_sha256"] for row in train_rows], seed=17, calibration_fraction=0.2
    )
    fit_index = torch.tensor(fit_indices)
    calibration_index = torch.tensor(calibration_indices)
    fit_x, fit_y = x_train[fit_index], train_targets[fit_index]
    calibration_x, calibration_y = x_train[calibration_index], train_targets[calibration_index]
    if len({train_rows[i]["speaker_group_sha256"] for i in calibration_indices}) != math.ceil(
        len({row["speaker_group_sha256"] for row in train_rows}) * 0.2
    ):
        raise ValueError("unexpected calibration speaker count")

    seed = 17
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    model = TextFreeAudioReadout(384, LABELS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, 9):
        order = list(range(len(fit_indices)))
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), 64):
            indexes = torch.tensor(order[start : start + 64])
            logits = model(fit_x[indexes])
            loss = F.cross_entropy(logits, fit_y[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "fit_train_ce": sum(losses) / len(losses)})
    training_seconds = time.perf_counter() - started

    model.eval()
    with torch.inference_mode():
        calibration_logits = model(calibration_x).detach()
        validation_logits = model(x_validation).detach()
    temperature = fit_scalar_temperature(calibration_logits, calibration_y)
    uncalibrated = _metrics(validation_logits, validation_targets)
    calibrated = _metrics(validation_logits / temperature, validation_targets)
    output_dir.mkdir(parents=True)
    weights_path = output_dir / "last-readout.pt"
    torch.save(model.state_dict(), weights_path)
    reloaded = TextFreeAudioReadout(384, LABELS).eval()
    reloaded.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        reloaded_logits = reloaded(x_validation)
    if not torch.equal(reloaded_logits, validation_logits):
        raise ValueError("saved/reloaded model logits differ")

    predictions_path = output_dir / "validation-predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row, target, logits in zip(
            validation_rows, validation_targets.tolist(), validation_logits, strict=True
        ):
            raw_probabilities = torch.softmax(logits.float(), dim=-1).tolist()
            scaled_probabilities = torch.softmax(logits.float() / temperature, dim=-1).tolist()
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "speaker_group_sha256": row["speaker_group_sha256"],
                        "target_class_id": LABELS[target],
                        "prediction_class_id": LABELS[int(logits.argmax())],
                        "probabilities_uncalibrated": raw_probabilities,
                        "probabilities_temperature_scaled": scaled_probabilities,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    config = {
        "seed": seed,
        "epochs": 8,
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint": "last epoch; no validation checkpoint selection",
        "calibration": (
            "single scalar temperature fitted by CPU float64 LBFGS on "
            "speaker-held-out train subset"
        ),
        "calibration_seed_key": "sha256('17:calibration:' + speaker_group_sha256)",
        "calibration_speaker_fraction": 0.2,
        "feature_width": 384,
        "classes": LABELS,
    }
    report = {
        "schema_version": 1,
        "status": "complete_text_free_audio_calibration_comparison",
        "claim_scope": (
            "same small fixed-taxonomy Speech Commands task; development-only comparison"
        ),
        "seed": seed,
        "config": config,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_revision": SOURCE,
        "audio_encoder_revision": AUDIO_REVISION,
        "feature_cache_sha256": sha256_file(feature_cache),
        "train_jsonl_sha256": sha256_file(sample_dir / "train.jsonl"),
        "validation_jsonl_sha256": sha256_file(sample_dir / "validation.jsonl"),
        "train_ids_sha256": hashlib.sha256("\n".join(cache["train_ids"]).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(cache["validation_ids"]).encode()
        ).hexdigest(),
        "fit_examples": len(fit_indices),
        "fit_unique_speakers": len({train_rows[i]["speaker_group_sha256"] for i in fit_indices}),
        "calibration_examples": len(calibration_indices),
        "calibration_unique_speakers": len(
            {train_rows[i]["speaker_group_sha256"] for i in calibration_indices}
        ),
        "fit_calibration_speaker_overlap": 0,
        "fit_class_counts": dict(Counter(train_rows[i]["target"] for i in fit_indices)),
        "calibration_class_counts": dict(
            Counter(train_rows[i]["target"] for i in calibration_indices)
        ),
        "calibration_temperature": temperature,
        "calibration_fit_nll_before": _metrics(calibration_logits, calibration_y)["nll"],
        "calibration_fit_nll_after": _metrics(calibration_logits / temperature, calibration_y)[
            "nll"
        ],
        "validation_examples": len(validation_rows),
        "validation_speaker_overlap_with_fit": 0,
        "validation_speaker_overlap_with_calibration": 0,
        "validation_metrics_uncalibrated": uncalibrated,
        "validation_metrics_temperature_scaled": calibrated,
        "validation_metrics_scope": (
            "one post-fit evaluation on previously inspected validation IDs; "
            "not blind or sealed audit"
        ),
        "training_history": history,
        "optimizer_updates": 8 * math.ceil(len(fit_indices) / 64),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds_cpu": training_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_used": False,
        "sealed_audit_loaded": False,
        "save_reload_exact_logits": True,
        "weights_sha256": sha256_file(weights_path),
        "predictions_sha256": sha256_file(predictions_path),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "readout_implementation_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src" / "tiny_omni_decision" / "textfree_audio.py"
        ),
        "limitation": (
            "Training excludes 264 train examples relative to the prior 1,280-example fit, "
            "so this is a calibration-method experiment, not a matched replication of the "
            "prior checkpoint."
        ),
    }
    (output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.feature_cache, args.sample_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
