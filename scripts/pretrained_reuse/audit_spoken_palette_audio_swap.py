"""TRAIN-only fixed-checkpoint audio-to-screen target-swap intervention."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import zipfile
import zlib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pretrained_reuse.audit_spoken_palette_absent_target import (
    DEFAULT_AUDIO_ARCHIVES,
    DEFAULT_CORPUS,
    DEFAULT_RUNTIME_ASSEMBLY,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_VIDEOPRISM_PYTHON,
    DEFAULT_VIDEOPRISM_SOURCE,
    EXPECTED_AUDIO_ARCHIVE,
    EXPECTED_CORPUS_MANIFEST,
    EXPECTED_RUNTIME_ASSEMBLY,
    EXPECTED_SOURCE_TRAIN_LAYOUT,
    EXPECTED_TRAIN_JSONL,
    select_train_pairs,
)
from scripts.pretrained_reuse.package_spoken_palette_runtime import sha256_file
from scripts.spoken_palette_decision_data import COLORS, POSITIONS
from tiny_omni_decision.modular_bundle import verify_modular_bundle
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher

DEFAULT_BASELINE = (
    Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
    / "spoken-palette-absent-target-audit-20261122-attempt-01"
)
DEFAULT_OUTPUT = (
    Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
    / "spoken-palette-audio-swap-audit-20261122-attempt-01"
)
EXPECTED_BASELINE_PREDICTIONS = "8bdc08e35e147acd182047d203e2740a27ba86479cc56585238c4f3e52040b6f"
EXPECTED_BASELINE_REPORT = "641126eea9c80824db7d111b7506f163ba82a6e2d23318e352ee36bddb1bcf0a"


def make_audio_swap_plan(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each screen fixed and rotate same-speaker audio to the next color."""
    by_group_color = {(row["speaker_group"], row["target_color"]): row for row in pairs}
    if len(by_group_color) != len(pairs):
        raise ValueError("speaker/color TRAIN pairs must be unique")
    if any(
        {color for speaker, color in by_group_color if speaker == group} != set(COLORS)
        for group in {str(row["speaker_group"]) for row in pairs}
    ):
        raise ValueError("speaker/color pair missing for deterministic audio rotation")
    plan = []
    for screen_pair in pairs:
        group = str(screen_pair["speaker_group"])
        screen_color = str(screen_pair["target_color"])
        audio_color = COLORS[(COLORS.index(screen_color) + 1) % len(COLORS)]
        audio_pair = by_group_color.get((group, audio_color))
        if audio_pair is None:
            raise ValueError("speaker/color pair missing for deterministic audio rotation")
        layout = list(screen_pair["layout_colors_by_position"])
        if sorted(layout) != sorted(COLORS):
            raise ValueError("screen layout does not contain each palette color exactly once")
        target_position = POSITIONS[layout.index(audio_color)]
        matches = [
            option
            for option in screen_pair["options"]
            if (float(option["x_norm"]), float(option["y_norm"])) == target_position
        ]
        if len(matches) != 1:
            raise ValueError("rotated audio target does not map to one supplied option")
        plan.append(
            {
                "screen_pair": screen_pair,
                "audio_pair": audio_pair,
                "screen_color": screen_color,
                "spoken_color": audio_color,
                "target_position": list(target_position),
                "target_option_id": str(matches[0]["id"]),
                "target_position_changed": target_position
                != tuple(float(value) for value in screen_pair["target_position"]),
            }
        )
    if len({row["audio_pair"]["source_id"] for row in plan}) != len(plan):
        raise ValueError("audio rotation must use each selected TRAIN source exactly once")
    return plan


def summarize(
    records: list[dict[str, Any]], *, replicates: int = 10_000, seed: int = 17
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["speaker_group"])].append(row)
    if len(groups) != 24 or len(records) != 96 or any(len(rows) != 4 for rows in groups.values()):
        raise ValueError("audio-swap result must contain four pairs for each of 24 TRAIN groups")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")

    def metrics(selected: list[str]) -> dict[str, float]:
        rows = [row for group in selected for row in groups[group]]
        changed = [row for row in rows if row["target_position_changed"]]
        return {
            "rotated_audio_target_accuracy": float(
                np.mean([row["prediction_matches_spoken_color"] for row in rows])
            ),
            "changed_position_accuracy": float(
                np.mean([row["prediction_matches_spoken_color"] for row in changed])
            ),
            "changed_position_support_fraction": len(changed) / len(rows),
            "mean_probability_on_spoken_color_option": float(
                np.mean([row["target_option_probability"] for row in rows])
            ),
            "mean_prediction_change_from_matched": float(
                np.mean([row["prediction_changed_from_matched"] for row in rows])
            ),
        }

    group_ids = sorted(groups)
    point = metrics(group_ids)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in point}
    for _ in range(replicates):
        draw = metrics(rng.choice(group_ids, size=len(group_ids), replace=True).tolist())
        for name, value in draw.items():
            draws[name].append(value)
    return {
        "method": "24-speaker cluster bootstrap; sample whole groups with replacement",
        "seed": seed,
        "replicates": replicates,
        "independent_groups": len(group_ids),
        "source_pairs": len(records),
        "ci_percentiles": [0.025, 0.975],
        "metrics": {
            name: {
                "estimate": value,
                "ci95": [
                    float(np.quantile(draws[name], 0.025)),
                    float(np.quantile(draws[name], 0.975)),
                ],
            }
            for name, value in point.items()
        },
        "limit": (
            "TRAIN-only intervention on a fixed checkpoint; "
            "not a blind or generalization estimate."
        ),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run(
    *,
    baseline_dir: Path,
    corpus_root: Path,
    source_train_layout: Path,
    audio_archive_dir: Path,
    assembly: Path,
    videoprism_python: Path,
    videoprism_source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite audio-swap audit: {output_dir}")
    baseline_report_path = baseline_dir / "result.json"
    baseline_predictions_path = baseline_dir / "predictions.jsonl"
    if sha256_file(baseline_report_path) != EXPECTED_BASELINE_REPORT:
        raise ValueError("fixed absent-target baseline report hash mismatch")
    if sha256_file(baseline_predictions_path) != EXPECTED_BASELINE_PREDICTIONS:
        raise ValueError("fixed absent-target baseline prediction hash mismatch")
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    if (
        baseline_report.get("status")
        != "completed_train_only_absent_target_counterfactual_diagnostic"
        or baseline_report.get("sealed_audit_loaded") is not False
        or baseline_report.get("training_performed") is not False
    ):
        raise ValueError("baseline is not the fixed TRAIN-only inference artifact")
    baseline_matched = {
        str(row["sample_id"]): row
        for row in _load_jsonl(baseline_predictions_path)
        if row.get("condition") == "matched"
    }
    if len(baseline_matched) != 96:
        raise ValueError("baseline must contain exactly 96 matched TRAIN predictions")

    audio_receipt = json.loads((audio_archive_dir / "result.json").read_text(encoding="utf-8"))
    audio_archive = audio_archive_dir / "audio.zip"
    if (
        audio_receipt.get("status") != "PASS"
        or audio_receipt.get("archives", {}).get("audio", {}).get("sha256")
        != EXPECTED_AUDIO_ARCHIVE
        or sha256_file(audio_archive) != EXPECTED_AUDIO_ARCHIVE
    ):
        raise ValueError("approved Lombard TRAIN audio archive failed its pinned hash gate")
    assembly_manifest = assembly / "package-manifest.json"
    if sha256_file(assembly_manifest) != EXPECTED_RUNTIME_ASSEMBLY:
        raise ValueError("encoder-only Path A assembly hash mismatch")
    package = verify_modular_bundle(assembly.resolve(strict=True))
    if package.get("storage_variant") != "whisper_encoder_only":
        raise ValueError("expected the pinned Whisper encoder-only runtime assembly")
    pairs, preflight = select_train_pairs(corpus_root, source_train_layout)
    if (
        preflight["corpus_manifest_sha256"] != EXPECTED_CORPUS_MANIFEST
        or preflight["train_jsonl_sha256"] != EXPECTED_TRAIN_JSONL
        or preflight["source_train_layout_sha256"] != EXPECTED_SOURCE_TRAIN_LAYOUT
    ):
        raise ValueError("frozen TRAIN input hashes differ from the approved source set")
    plan = make_audio_swap_plan(pairs)
    if len(plan) != 96 or len({row["screen_pair"]["speaker_group"] for row in plan}) != 24:
        raise ValueError("fixed audio-swap plan must cover 96 samples and 24 TRAIN speakers")
    if {row["screen_pair"]["sample_id"] for row in plan} != set(baseline_matched):
        raise ValueError("audio-swap plan does not match the fixed baseline sample identities")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "preflight.json").write_text(
        json.dumps(
            {
                "status": "PASS_TRAIN_ONLY_AUDIO_SWAP_PREFLIGHT",
                "seed": 17,
                "baseline_result_sha256": EXPECTED_BASELINE_REPORT,
                "baseline_predictions_sha256": EXPECTED_BASELINE_PREDICTIONS,
                "source_manifest_sha256": EXPECTED_CORPUS_MANIFEST,
                "train_jsonl_sha256": EXPECTED_TRAIN_JSONL,
                "source_train_layout_sha256": EXPECTED_SOURCE_TRAIN_LAYOUT,
                "audio_archive_sha256": EXPECTED_AUDIO_ARCHIVE,
                "runtime_assembly_sha256": EXPECTED_RUNTIME_ASSEMBLY,
                "pairs": 96,
                "speaker_groups": 24,
                "paired_input_rule": (
                    "fixed screen/options; audio rotates to next color from the same "
                    "frozen TRAIN speaker group"
                ),
                "sealed_audit_opened": False,
                "validation_or_calibration_opened": False,
                "training_performed": False,
                "plan": [
                    {
                        "sample_id": row["screen_pair"]["sample_id"],
                        "speaker_group": row["screen_pair"]["speaker_group"],
                        "screen_source_id": row["screen_pair"]["source_id"],
                        "audio_source_id": row["audio_pair"]["source_id"],
                        "screen_color": row["screen_color"],
                        "spoken_color": row["spoken_color"],
                        "target_position": row["target_position"],
                        "target_option_id": row["target_option_id"],
                        "target_position_changed": row["target_position_changed"],
                    }
                    for row in plan
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    audio_root = output_dir / "audio"
    audio_root.mkdir()
    records: list[dict[str, Any]] = []
    with (
        zipfile.ZipFile(audio_archive) as archive,
        PathARuntimeDispatcher(
            assembly,
            videoprism_python=videoprism_python,
            videoprism_source=videoprism_source,
            worker_log_dir=output_dir / "worker-logs",
            device="cpu",
            audio_encoder_precision="fp32",
        ) as dispatcher,
    ):
        worker = dispatcher._ensure_torch()
        if not worker.ready.get("spoken_palette_av_loaded"):
            raise RuntimeError("integrated spoken-palette AV scorer did not load")
        if not worker.ready.get("spoken_palette_shares_audio_encoder"):
            raise RuntimeError("spoken-palette AV scorer did not reuse the pinned Whisper encoder")
        for index, item in enumerate(plan):
            screen = item["screen_pair"]
            audio = item["audio_pair"]
            audio_bytes = archive.read(audio["audio_member"])
            if (
                len(audio_bytes) != audio["audio_bytes"]
                or zlib.crc32(audio_bytes) & 0xFFFFFFFF != audio["audio_crc32"]
            ):
                raise ValueError("rotated audio bytes differ from official TRAIN metadata")
            audio_path = audio_root / f"{audio['source_id']}.wav"
            audio_path.write_bytes(audio_bytes)
            audio_sha = sha256_file(audio_path)
            response = dispatcher.dispatch(
                "spoken_palette_av",
                {
                    "audio_path": str(audio_path),
                    "image_path": screen["base_image_path"],
                    "options": screen["options"],
                },
            )
            options = response.get("options")
            if not isinstance(options, list) or [str(row.get("id")) for row in options] != [
                str(row["id"]) for row in screen["options"]
            ]:
                raise RuntimeError("runtime changed the fixed option IDs or order")
            probabilities = [float(row["probability"]) for row in options]
            if (
                not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
            ):
                raise RuntimeError("runtime returned an invalid option probability distribution")
            predicted_id = str(response["selected_option_id"])
            target_option = next(
                row for row in options if str(row["id"]) == item["target_option_id"]
            )
            original_target_options = [
                option
                for option in screen["options"]
                if (float(option["x_norm"]), float(option["y_norm"]))
                == tuple(screen["target_position"])
            ]
            if len(original_target_options) != 1:
                raise ValueError("original screen target does not map to one option")
            matched = baseline_matched[str(screen["sample_id"])]
            row = {
                "sample_id": screen["sample_id"],
                "speaker_group": screen["speaker_group"],
                "screen_source_id": screen["source_id"],
                "audio_source_id": audio["source_id"],
                "screen_source_color": item["screen_color"],
                "spoken_color": item["spoken_color"],
                "target_position": item["target_position"],
                "target_option_id": item["target_option_id"],
                "target_position_changed": item["target_position_changed"],
                "predicted_option_id": predicted_id,
                "prediction_matches_spoken_color": predicted_id == item["target_option_id"],
                "prediction_matches_original_screen_color": predicted_id
                == str(original_target_options[0]["id"]),
                "prediction_changed_from_matched": predicted_id
                != str(matched["predicted_option_id"]),
                "target_option_probability": float(target_option["probability"]),
                "max_probability": max(probabilities),
                "uncertainty_score": 1.0 - max(probabilities),
                "probabilities_by_option_id": {
                    str(option["id"]): float(option["probability"]) for option in options
                },
                "audio_sha256": audio_sha,
                "image_sha256": screen["base_image_sha256"],
                "baseline_matched_prediction_id": str(matched["predicted_option_id"]),
                "worker_inference_seconds": response.get("worker_inference_seconds"),
            }
            records.append(row)
            with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_pairs": index + 1,
                        "total_pairs": len(plan),
                        "updated_utc": datetime.now(UTC).isoformat(),
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    result = {
        "schema_version": 1,
        "status": "completed_train_only_fixed_checkpoint_audio_swap_diagnostic",
        "scope": (
            "fixed-checkpoint same-speaker audio rotation across fixed TRAIN screens; "
            "no training, selection, or product quality claim"
        ),
        "source_commit": __import__("subprocess")
        .run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        .stdout.strip(),
        "created_start_utc": started_utc,
        "created_end_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "baseline_result_sha256": EXPECTED_BASELINE_REPORT,
        "baseline_predictions_sha256": EXPECTED_BASELINE_PREDICTIONS,
        "implementation_sha256": {
            "audio_swap_script": sha256_file(Path(__file__).resolve()),
            "runtime_dispatch": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src/tiny_omni_decision/runtime_dispatch.py"
            ),
        },
        "source_and_runtime": {
            **{key: value for key, value in preflight.items() if key not in {"selected_pairs"}},
            "runtime_assembly_sha256": EXPECTED_RUNTIME_ASSEMBLY,
            "package_id": package.get("package_id"),
            "logical_runtime_parameters": package.get("unique_runtime_parameters"),
            "package_bytes": package.get("package_file_bytes_excluding_manifest"),
            "device": "cpu",
            "python": platform.python_version(),
            "training_performed": False,
            "validation_or_calibration_opened": False,
            "sealed_audit_opened": False,
        },
        "intervention": {
            "pairs": len(records),
            "speaker_groups": len({row["speaker_group"] for row in records}),
            "rotation": (
                "within each speaker group, rotate the audio to the next palette color; "
                "retain the original screen, geometry, option order and IDs"
            ),
            "screen_color_and_spoken_color_differ": all(
                row["screen_source_color"] != row["spoken_color"] for row in records
            ),
            "audio_source_ids_unique": len({row["audio_source_id"] for row in records})
            == len(records),
            "predictions": len(records),
        },
        "baseline_matched_train_accuracy": float(
            np.mean(
                [
                    bool(row["prediction_matches_target_position"])
                    for row in baseline_matched.values()
                ]
            )
        ),
        "summary": summarize(records),
        "sealed_audit_loaded": False,
        "training_performed": False,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result_path.with_suffix(".sha256").write_text(sha256_file(result_path) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--source-train-layout", type=Path, default=DEFAULT_SOURCE_ROOT / "train.json"
    )
    parser.add_argument("--audio-archives", type=Path, default=DEFAULT_AUDIO_ARCHIVES)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_RUNTIME_ASSEMBLY)
    parser.add_argument("--videoprism-python", type=Path, default=DEFAULT_VIDEOPRISM_PYTHON)
    parser.add_argument("--videoprism-source", type=Path, default=DEFAULT_VIDEOPRISM_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        baseline_dir=args.baseline_dir,
        corpus_root=args.corpus,
        source_train_layout=args.source_train_layout,
        audio_archive_dir=args.audio_archives,
        assembly=args.assembly,
        videoprism_python=args.videoprism_python,
        videoprism_source=args.videoprism_source,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
