"""Run a raw-audio, TRAIN-only Path A integration smoke without fitting."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher, TypedDecisionSpec
from tiny_omni_decision.textfree_audio_runtime import (
    AUDIO_ENCODER_PARAMETERS,
    AUDIO_ENCODER_REVISION,
    AUDIO_ENCODER_SHA256,
    AUDIO_FEATURE_SIZE,
    AUDIO_READOUT_SHA256,
    SPEECH_COMMANDS_CLASSES,
    sha256_file,
)

SOURCE = "google/speech_commands@a751309c0fd613e8a5d30d77900f30e8b42bc2da"


def _training_examples(sample_dir: Path, limit: int) -> list[dict[str, str]]:
    path = sample_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("split") != "train" or row.get("source") != SOURCE for row in rows):
        raise ValueError("training manifest contains an unexpected split or source")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("training manifest contains duplicate IDs")
    selected: list[dict[str, str]] = []
    seen_speakers: set[str] = set()
    for row in rows:
        speaker = row.get("speaker_group_sha256")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError("training row is missing its speaker group identity")
        if speaker in seen_speakers:
            continue
        selected.append(row)
        seen_speakers.add(speaker)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError("training manifest lacks the requested distinct speaker groups")
    return selected


def _option_payload(classes: tuple[str, ...]) -> tuple[list[dict[str, str]], TypedDecisionSpec]:
    ids = tuple(f"speech-command:{class_id}" for class_id in classes)
    options = [
        {"option_id": option_id, "class_id": class_id}
        for option_id, class_id in zip(ids, classes, strict=True)
    ]
    spec = TypedDecisionSpec(
        request_id=f"native-audio-{len(classes)}-{classes[0]}",
        decision_type="choice",
        option_ids=ids,
        option_texts=classes,
    )
    return options, spec


def run(args: argparse.Namespace) -> dict[str, object]:
    sample_dir = args.sample_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    selected = _training_examples(sample_dir, args.examples)
    media_rows = []
    for row in selected:
        media = (sample_dir / row["media_path"]).resolve(strict=True)
        if not media.is_relative_to(sample_dir):
            raise ValueError("training media path escaped the frozen sample directory")
        actual = sha256_file(media)
        if actual != row["media_sha256"]:
            raise ValueError(f"training media hash mismatch: {row['id']}")
        media_rows.append((row, media, actual))

    output_dir.mkdir(parents=True)
    encoder_dir = args.encoder_dir.resolve(strict=True)
    readout = args.readout.resolve(strict=True)
    bundle_dir = args.assembly_dir.resolve(strict=True)
    runtime_dir = args.videoprism_source.resolve(strict=True)
    torch.set_num_threads(4)
    records: list[dict[str, object]] = []
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    with PathARuntimeDispatcher(
        bundle_dir,
        videoprism_python=args.videoprism_python.resolve(strict=True),
        videoprism_source=runtime_dir,
        worker_log_dir=output_dir / "worker-logs",
        device="cpu",
        textfree_audio_encoder_dir=encoder_dir,
        textfree_audio_readout_path=readout,
    ) as dispatcher:
        orders = [SPEECH_COMMANDS_CLASSES] * len(media_rows)
        orders.append(tuple(reversed(SPEECH_COMMANDS_CLASSES)))
        for index, classes in enumerate(orders):
            row, media, media_sha = media_rows[0 if index == len(media_rows) else index]
            options, spec = _option_payload(classes)
            response = dispatcher.dispatch_typed(
                "audio_native",
                {
                    "audio_path": str(media),
                    "expected_media_sha256": media_sha,
                    "options": options,
                },
                decision_specs=[spec],
            )
            records.append(
                {
                    "sample_id": row["id"],
                    "media_sha256": media_sha,
                    "class_order": list(classes),
                    "decision_json": response["decision_json"],
                    "encoder_metrics": response["encoder_metrics"],
                    "dispatch_wall_seconds": response["dispatch_wall_seconds"],
                    "worker_startup_wall_seconds": response["worker_startup_wall_seconds"],
                }
            )
        worker = dispatcher.worker_metadata["textfree_audio"]
        if not worker or worker.get("decoder_loaded") or worker.get("text_encoder_loaded"):
            raise ValueError("text-free worker loaded an unexpected model component")
        if worker.get("cuda_initialized"):
            raise ValueError("CPU smoke unexpectedly initialized CUDA")
        if worker.get("resident_parameters") != AUDIO_ENCODER_PARAMETERS + 3_850:
            raise ValueError("text-free worker parameter count differs from the pinned contract")

    first_probabilities = {
        option["id"].removeprefix("speech-command:"): option["probability"]
        for option in records[0]["decision_json"]["options"]
    }
    repeated_probabilities = {
        option["id"].removeprefix("speech-command:"): option["probability"]
        for option in records[-1]["decision_json"]["options"]
    }
    order_max_abs_difference = max(
        abs(first_probabilities[key] - repeated_probabilities[key])
        for key in first_probabilities
    )
    order_parity = math.isclose(order_max_abs_difference, 0.0, abs_tol=1e-6)
    if not order_parity:
        raise ValueError("option permutation changed class-bound probabilities")

    run_report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_RAW_AUDIO_PATH_A_TRAIN_ONLY_SMOKE",
        "scope": (
            "Raw WAV to typed probability JSON through the Path A dispatcher; no fitting, "
            "quality metrics, or label access."
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "device": "cpu",
        "gpu_used": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "sealed_audit_loaded": False,
        "validation_manifest_opened": False,
        "target_labels_used_for_inference_or_metrics": False,
        "source": SOURCE,
        "source_split_read": "train.jsonl only",
        "source_manifest_sha256": sha256_file(sample_dir / "train.jsonl"),
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "encoder_revision": AUDIO_ENCODER_REVISION,
        "encoder_sha256": AUDIO_ENCODER_SHA256,
        "readout_sha256": AUDIO_READOUT_SHA256,
        "resident_parameters": AUDIO_ENCODER_PARAMETERS + 3_850,
        "encoder_feature_width": AUDIO_FEATURE_SIZE,
        "artifacts": {
            "encoder_config_sha256": sha256_file(encoder_dir / "config.json"),
            "encoder_weights_sha256": sha256_file(encoder_dir / "encoder.safetensors"),
            "preprocessor_sha256": sha256_file(encoder_dir / "preprocessor_config.json"),
            "readout_sha256": sha256_file(readout),
        },
        "unique_train_assets": len(media_rows),
        "unique_speakers": len({row["speaker_group_sha256"] for row, _, _ in media_rows}),
        "request_count": len(records),
        "encoder_calls_expected": len(media_rows),
        "encoder_calls_observed": records[-1]["encoder_metrics"]["single_event_encoder_calls"],
        "repeated_request_cache_hit": records[-1]["encoder_metrics"]["cache_hit"],
        "option_permutation_probability_parity": order_parity,
        "option_permutation_max_abs_probability_delta": order_max_abs_difference,
        "option_permutation_float32_tolerance": 1e-6,
        "worker": worker,
        "records": records,
    }
    output_path = output_dir / "run-report.json"
    output_path.write_text(
        json.dumps(run_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_report["run_report_sha256"] = sha256_file(output_path)
    return run_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--assembly-dir", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()
    if args.examples < 1 or args.examples > 8:
        parser.error("--examples must be between 1 and 8")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
