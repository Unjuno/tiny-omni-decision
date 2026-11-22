"""Train-only paired counterfactual audit for target-absent screen decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pretrained_reuse.package_spoken_palette_runtime import sha256_file  # noqa: E402
from scripts.spoken_palette_decision_data import COLORS, POSITIONS, RGB  # noqa: E402
from tiny_omni_decision.modular_bundle import verify_modular_bundle  # noqa: E402
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher  # noqa: E402

DEFAULT_ARTIFACT_ROOT = Path(r"D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2")
DEFAULT_CORPUS = DEFAULT_ARTIFACT_ROOT / "spoken-palette-av-decision-v1-attempt-02"
DEFAULT_SOURCE_ROOT = DEFAULT_ARTIFACT_ROOT / "lombard-grid-layout-02-media-gated"
DEFAULT_AUDIO_ARCHIVES = DEFAULT_ARTIFACT_ROOT / "lombard-grid-archives-01"
DEFAULT_RUNTIME_ASSEMBLY = (
    DEFAULT_ARTIFACT_ROOT
    / "path-a-modular-runtime-assembly-v4-whisper-encoder-only-20261122-attempt-01"
)
DEFAULT_VIDEOPRISM_PYTHON = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\venv\Scripts\python.exe"
)
DEFAULT_VIDEOPRISM_SOURCE = Path(r"C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-r1\source")
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_ROOT / "spoken-palette-absent-target-audit-20261122-attempt-01"

EXPECTED_CORPUS_MANIFEST = "b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a"
EXPECTED_TRAIN_JSONL = "7c2988c5d688d89e605c8e9683cb5db1155faf127f6a2b3e61e1d72ad71aa170"
EXPECTED_SOURCE_TRAIN_LAYOUT = "259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0"
EXPECTED_AUDIO_ARCHIVE = "e36707602fdbeaa2e2284acb29bba883261d5e86857f694ac40cbb56cdcc5e10"
EXPECTED_RUNTIME_ASSEMBLY = "616b9b61e983d875a6a29e71d568f1ba0a6aeba6705da61ee66a88b4be43b3fd"
SEED = 17
CONDITIONS = ("matched", "target_absent", "distractor_absent")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _group_train_sources(
    source_by_id: dict[str, dict[str, Any]], corpus_source_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Restrict the official TRAIN layout to source IDs in the frozen corpus TRAIN split."""
    missing = corpus_source_ids - source_by_id.keys()
    if missing:
        raise ValueError("corpus TRAIN references source IDs absent from the official TRAIN layout")
    selected_rows = [source_by_id[source_id] for source_id in sorted(corpus_source_ids)]
    group_colors: dict[str, set[str]] = defaultdict(set)
    for item in selected_rows:
        group = str(item.get("speaker_group", ""))
        color = (item.get("targets") or {}).get("color")
        if not group or color not in COLORS:
            raise ValueError("corpus TRAIN contains invalid speaker or color metadata")
        group_colors[group].add(str(color))
    return selected_rows, group_colors


def select_train_pairs(
    corpus_root: Path,
    source_train_layout: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one deterministic source per TRAIN speaker/color, balanced by target position."""
    corpus_manifest_path = corpus_root / "manifest.json"
    corpus_manifest_sha = sha256_file(corpus_manifest_path)
    if corpus_manifest_sha != EXPECTED_CORPUS_MANIFEST:
        raise ValueError("frozen corpus manifest hash mismatch")
    manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    policy = manifest.get("evaluation_policy", {})
    source = manifest.get("source", {})
    if (
        policy.get("sealed_audit_opened") is not False
        or source.get("original_calibration_validation_or_audit_labels_loaded") is not False
        or source.get("source_split_used") != "original TRAIN only"
        or manifest.get("split_row_sha256", {}).get("train") != EXPECTED_TRAIN_JSONL
    ):
        raise ValueError("frozen corpus is not proven TRAIN-only and sealed-safe")
    train_path = corpus_root / "train.jsonl"
    if sha256_file(train_path) != EXPECTED_TRAIN_JSONL:
        raise ValueError("frozen TRAIN JSONL hash mismatch")
    if sha256_file(source_train_layout) != EXPECTED_SOURCE_TRAIN_LAYOUT:
        raise ValueError("official TRAIN source-layout hash mismatch")

    source_rows = json.loads(source_train_layout.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("official source TRAIN layout must be a nonempty list")
    source_by_id: dict[str, dict[str, Any]] = {}
    for item in source_rows:
        if item.get("split") != "train":
            raise ValueError("source layout contains a non-TRAIN row")
        source_id = str(item.get("source_id", ""))
        group = str(item.get("speaker_group", ""))
        color = (item.get("targets") or {}).get("color")
        if not source_id or not group or color not in COLORS or source_id in source_by_id:
            raise ValueError("invalid or duplicate source TRAIN metadata")
        source_by_id[source_id] = item

    corpus_rows: list[dict[str, Any]] = []
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("TRAIN JSONL contains a non-TRAIN row")
            if row.get("source_id") not in source_by_id:
                raise ValueError("corpus row references an unknown source TRAIN ID")
            corpus_rows.append(row)
    if len(corpus_rows) != int(
        manifest["speaker_group_split"]["generated_decision_counts"]["train"]
    ):
        raise ValueError("TRAIN row count differs from the frozen corpus manifest")
    train_source_rows, group_colors = _group_train_sources(
        source_by_id, {str(row["source_id"]) for row in corpus_rows}
    )
    if len(group_colors) != 24 or any(colors != set(COLORS) for colors in group_colors.values()):
        raise ValueError("frozen corpus TRAIN must contain 24 speaker groups with all four colors")

    layouts = tuple(
        tuple(str(color) for color in layout) for layout in manifest["task"]["screen_layouts"]["0"]
    )
    layout_id_to_colors = {_digest("|".join(layout))[:16]: layout for layout in layouts}
    if len(layout_id_to_colors) != 12:
        raise ValueError("expected the fixed twelve even-parity TRAIN layouts")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in corpus_rows:
        by_source[str(row["source_id"])].append(row)

    chosen_sources: dict[tuple[str, str], dict[str, Any]] = {}
    for group in sorted(group_colors):
        for color in COLORS:
            candidates = [
                row
                for row in train_source_rows
                if row["speaker_group"] == group and row["targets"]["color"] == color
            ]
            selected = min(
                candidates,
                key=lambda row: (
                    _digest(f"absent-target-v1|{SEED}|{group}|{color}|{row['source_id']}"),
                    row["source_id"],
                ),
            )
            chosen_sources[(group, color)] = selected

    pairs: list[dict[str, Any]] = []
    position_counts: dict[str, list[int]] = {color: [0, 0, 0, 0] for color in COLORS}
    for color in COLORS:
        groups = sorted(group for group in group_colors if (group, color) in chosen_sources)
        for group_index, group in enumerate(groups):
            source_row = chosen_sources[(group, color)]
            target_position = group_index % len(POSITIONS)
            source_id = str(source_row["source_id"])
            candidate_rows = [
                row
                for row in by_source[source_id]
                if tuple(
                    row["options"][row["target_index"]].get(key) for key in ("x_norm", "y_norm")
                )
                == POSITIONS[target_position]
            ]
            if not candidate_rows:
                raise ValueError("no frozen TRAIN screen row matches the balanced target position")
            row = min(
                candidate_rows,
                key=lambda item: (
                    _digest(f"absent-target-layout-v1|{SEED}|{item['sample_id']}"),
                    item["sample_id"],
                ),
            )
            layout = layout_id_to_colors.get(str(row["layout_id"]))
            if layout is None or layout.index(color) != target_position:
                raise ValueError("TRAIN sample target position disagrees with its fixed layout")
            if row["screenshot_sha256"] == "" or len(row["options"]) != 4:
                raise ValueError("invalid frozen TRAIN screen or supplied-option count")
            if (
                tuple(
                    (float(option["x_norm"]), float(option["y_norm"])) for option in row["options"]
                ).count(POSITIONS[target_position])
                != 1
            ):
                raise ValueError("target option position is not unique")
            image_path = (corpus_root / str(row["screenshot_path"])).resolve(strict=True)
            if corpus_root.resolve(strict=True) not in image_path.parents:
                raise ValueError("TRAIN screenshot path escaped its frozen corpus")
            if sha256_file(image_path) != row["screenshot_sha256"]:
                raise ValueError("frozen TRAIN screenshot hash mismatch")
            pairs.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "source_id": source_id,
                    "speaker_group": group,
                    "target_color": color,
                    "target_position_index": target_position,
                    "target_position": list(POSITIONS[target_position]),
                    "layout_id": str(row["layout_id"]),
                    "layout_colors_by_position": list(layout),
                    "base_image_path": str(image_path),
                    "base_image_sha256": str(row["screenshot_sha256"]),
                    "options": [
                        {
                            "id": str(option["option_id"]),
                            "text": f"Screen action candidate {index + 1}",
                            "x_norm": float(option["x_norm"]),
                            "y_norm": float(option["y_norm"]),
                        }
                        for index, option in enumerate(row["options"])
                    ],
                    "audio_member": str(source_row["assets"]["audio"]["name"]),
                    "audio_bytes": int(source_row["assets"]["audio"]["bytes"]),
                    "audio_crc32": int(source_row["assets"]["audio"]["crc32"]),
                    "source_target_index_integrity_check": int(row["target_index"]),
                }
            )
            position_counts[color][target_position] += 1
    if len(pairs) != 96 or len({pair["source_id"] for pair in pairs}) != 96:
        raise ValueError("expected 96 unique source-audio pairs")
    if any(count != 6 for counts in position_counts.values() for count in counts):
        raise ValueError("target color/position support is not exactly balanced")
    return pairs, {
        "corpus_manifest_sha256": corpus_manifest_sha,
        "train_jsonl_sha256": sha256_file(train_path),
        "source_train_layout_sha256": sha256_file(source_train_layout),
        "source_train_rows": len(source_rows),
        "corpus_train_rows": len(corpus_rows),
        "speaker_groups": len(group_colors),
        "selected_sources": len(pairs),
        "unique_speaker_groups": len({pair["speaker_group"] for pair in pairs}),
        "target_color_position_counts": position_counts,
        "selection_rule": (
            "one SHA-256-selected source per TRAIN speaker group and color; "
            "target position balanced 6 per color-position cell; "
            "one SHA-256-selected matching TRAIN screen row"
        ),
    }


def replace_palette_color(
    image: Image.Image,
    *,
    removed_color: str,
    replacement_color: str,
) -> tuple[Image.Image, int]:
    """Replace only exact palette fill pixels, retaining the original screen layout."""
    if (
        removed_color not in COLORS
        or replacement_color not in COLORS
        or removed_color == replacement_color
    ):
        raise ValueError("two distinct colors from the fixed palette are required")
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    old = np.asarray(RGB[removed_color], dtype=np.uint8)
    new = np.asarray(RGB[replacement_color], dtype=np.uint8)
    mask = np.all(pixels == old, axis=-1)
    count = int(mask.sum())
    if count < 100:
        raise ValueError("source screen does not contain the expected palette fill area")
    pixels[mask] = new
    return Image.fromarray(pixels, mode="RGB"), count


def binary_auc(labels: list[int], scores: list[float]) -> float:
    """Compute rank-based binary AUROC with average ranks for ties."""
    if len(labels) != len(scores) or not labels or any(label not in (0, 1) for label in labels):
        raise ValueError("labels and scores must be matching nonempty binary sequences")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if (
        not positive_count
        or not negative_count
        or not all(math.isfinite(score) for score in scores)
    ):
        raise ValueError("AUROC needs both classes and finite scores")
    ordered = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and scores[ordered[stop]] == scores[ordered[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        rank_sum += average_rank * sum(labels[index] for index in ordered[start:stop])
        start = stop
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def score_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_sample[str(row["sample_id"])][str(row["condition"])] = row
        by_speaker[str(row["speaker_group"])].append(row)
    if not by_sample or any(set(rows) != set(CONDITIONS) for rows in by_sample.values()):
        raise ValueError("every paired source must have exactly the three fixed conditions")

    condition_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [sample[condition] for sample in by_sample.values()]
        condition_summary[condition] = {
            "count": len(rows),
            "target_present": condition != "target_absent",
            "target_present_accuracy": (
                sum(bool(row["prediction_matches_target_position"]) for row in rows) / len(rows)
                if condition != "target_absent"
                else None
            ),
            "mean_max_probability": float(np.mean([row["max_probability"] for row in rows])),
            "median_max_probability": float(np.median([row["max_probability"] for row in rows])),
            "mean_top_two_margin": float(np.mean([row["top_two_margin"] for row in rows])),
            "mean_normalized_entropy": float(np.mean([row["normalized_entropy"] for row in rows])),
        }
    invalid_rows = [sample["target_absent"] for sample in by_sample.values()]
    valid_rows = [
        sample[condition]
        for sample in by_sample.values()
        for condition in ("matched", "distractor_absent")
    ]

    def auc_for(
        rows_pos: list[dict[str, Any]], rows_neg: list[dict[str, Any]], score_name: str
    ) -> float:
        return binary_auc(
            [1] * len(rows_pos) + [0] * len(rows_neg),
            [float(row[score_name]) for row in rows_pos + rows_neg],
        )

    per_speaker_auc = {}
    for speaker, rows in sorted(by_speaker.items()):
        positives = [row for row in rows if row["condition"] == "target_absent"]
        negatives = [row for row in rows if row["condition"] != "target_absent"]
        per_speaker_auc[speaker] = auc_for(positives, negatives, "uncertainty_score")

    deltas = {
        condition: {
            "mean_max_probability_delta_vs_matched": float(
                np.mean(
                    [
                        sample[condition]["max_probability"] - sample["matched"]["max_probability"]
                        for sample in by_sample.values()
                    ]
                )
            ),
            "median_max_probability_delta_vs_matched": float(
                np.median(
                    [
                        sample[condition]["max_probability"] - sample["matched"]["max_probability"]
                        for sample in by_sample.values()
                    ]
                )
            ),
            "mean_normalized_entropy_delta_vs_matched": float(
                np.mean(
                    [
                        sample[condition]["normalized_entropy"]
                        - sample["matched"]["normalized_entropy"]
                        for sample in by_sample.values()
                    ]
                )
            ),
        }
        for condition in ("target_absent", "distractor_absent")
    }
    return {
        "source_pairs": len(by_sample),
        "speaker_groups": len(by_speaker),
        "conditions": condition_summary,
        "target_absence_detection": {
            "positive": "target absent",
            "negative": "matched or non-target distractor absent",
            "pooled_auc_uncertainty_1_minus_max_probability": auc_for(
                invalid_rows, valid_rows, "uncertainty_score"
            ),
            "pooled_auc_uncertainty_1_minus_margin": auc_for(
                invalid_rows, valid_rows, "uncertainty_margin_score"
            ),
            "pooled_auc_normalized_entropy": auc_for(
                invalid_rows, valid_rows, "normalized_entropy"
            ),
            "macro_speaker_auc_uncertainty_1_minus_max_probability": float(
                np.mean(list(per_speaker_auc.values()))
            ),
            "speaker_auc_range_uncertainty_1_minus_max_probability": [
                float(min(per_speaker_auc.values())),
                float(max(per_speaker_auc.values())),
            ],
            "per_speaker_auc_uncertainty_1_minus_max_probability": per_speaker_auc,
            "threshold_selected": False,
            "threshold_applied": False,
        },
        "paired_deltas": deltas,
        "runtime_abstention_supported": False,
        "runtime_response_policy": (
            "always returns a four-option choice distribution; no abstention option "
            "or invalid-input threshold is defined"
        ),
    }


def run(
    *,
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
        raise FileExistsError(f"refusing to overwrite absent-target diagnostic: {output_dir}")
    audio_receipt_path = audio_archive_dir / "result.json"
    audio_receipt = json.loads(audio_receipt_path.read_text(encoding="utf-8"))
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
        raise ValueError("encoder-only Path A assembly hash differs from the pinned artifact")
    package = verify_modular_bundle(assembly.resolve(strict=True))
    if package.get("storage_variant") != "whisper_encoder_only":
        raise ValueError("expected the immutable Whisper encoder-only assembly")
    pairs, preflight = select_train_pairs(corpus_root, source_train_layout)
    output_dir.mkdir(parents=True, exist_ok=False)
    media_root = output_dir / "media"
    for condition in CONDITIONS:
        if condition != "matched":
            (media_root / condition).mkdir(parents=True)
    preflight.update(
        {
            "status": "PASS_TRAIN_ONLY_COUNTERFACTUAL_PREFLIGHT",
            "seed": SEED,
            "assembly_manifest_sha256": sha256_file(assembly_manifest),
            "audio_archive_sha256": EXPECTED_AUDIO_ARCHIVE,
            "sealed_audit_opened": False,
            "selection_validation_calibration_labels_opened": False,
            "training_performed": False,
            "selected_pairs": pairs,
        }
    )
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    progress_path = output_dir / "predictions.jsonl"
    worker_log_dir = output_dir / "worker-logs"
    with (
        zipfile.ZipFile(audio_archive) as archive,
        progress_path.open("x", encoding="utf-8", newline="\n") as predictions_stream,
    ):
        with PathARuntimeDispatcher(
            assembly,
            videoprism_python=videoprism_python,
            videoprism_source=videoprism_source,
            worker_log_dir=worker_log_dir,
            device="cpu",
            audio_encoder_precision="fp32",
        ) as dispatcher:
            worker = dispatcher._ensure_torch()
            worker_metadata = worker.ready
            if not worker_metadata.get("spoken_palette_av_loaded"):
                raise RuntimeError("integrated Path A worker did not load the spoken-palette task")
            if not worker_metadata.get("spoken_palette_shares_audio_encoder"):
                raise RuntimeError("Path A worker failed to share the pinned Whisper encoder")
            for pair_index, pair in enumerate(pairs):
                audio_bytes = archive.read(pair["audio_member"])
                if (
                    len(audio_bytes) != pair["audio_bytes"]
                    or zlib.crc32(audio_bytes) & 0xFFFFFFFF != pair["audio_crc32"]
                ):
                    raise ValueError(
                        "Lombard source audio bytes or CRC differs from TRAIN metadata"
                    )
                audio_path = media_root / "audio" / f"{pair['source_id']}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(audio_bytes)
                audio_sha = sha256_file(audio_path)
                base_path = Path(pair["base_image_path"])
                with Image.open(base_path) as source_image:
                    base_image = source_image.convert("RGB")
                    target = str(pair["target_color"])
                    nontargets = [color for color in COLORS if color != target]
                    donor_order = sorted(
                        nontargets,
                        key=lambda color: _digest(
                            f"absent-target-donor-v1|{SEED}|{pair['source_id']}|{pair['layout_id']}|{color}"
                        ),
                    )
                    target_donor = donor_order[0]
                    distractor_removed = donor_order[1]
                    distractor_donor = donor_order[2]
                    target_absent_image, target_pixels = replace_palette_color(
                        base_image,
                        removed_color=target,
                        replacement_color=target_donor,
                    )
                    distractor_absent_image, distractor_pixels = replace_palette_color(
                        base_image,
                        removed_color=distractor_removed,
                        replacement_color=distractor_donor,
                    )
                target_absent_path = media_root / "target_absent" / f"{pair['sample_id']}.png"
                distractor_absent_path = (
                    media_root / "distractor_absent" / f"{pair['sample_id']}.png"
                )
                target_absent_image.save(target_absent_path, format="PNG", optimize=False)
                distractor_absent_image.save(distractor_absent_path, format="PNG", optimize=False)
                image_inputs = {
                    "matched": (base_path, pair["base_image_sha256"]),
                    "target_absent": (target_absent_path, sha256_file(target_absent_path)),
                    "distractor_absent": (
                        distractor_absent_path,
                        sha256_file(distractor_absent_path),
                    ),
                }
                for condition in CONDITIONS:
                    image_path, image_sha = image_inputs[condition]
                    response = dispatcher.dispatch(
                        "spoken_palette_av",
                        {
                            "audio_path": str(audio_path),
                            "image_path": str(image_path),
                            "options": pair["options"],
                        },
                    )
                    option_rows = response.get("options")
                    if not isinstance(option_rows, list) or len(option_rows) != 4:
                        raise RuntimeError("runtime did not return exactly four supplied options")
                    ids = [str(option["id"]) for option in option_rows]
                    if ids != [str(option["id"]) for option in pair["options"]]:
                        raise RuntimeError("runtime changed supplied option IDs or order")
                    probabilities = [float(option["probability"]) for option in option_rows]
                    if not all(
                        math.isfinite(probability) and 0 <= probability <= 1
                        for probability in probabilities
                    ) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
                        raise RuntimeError("runtime returned an invalid option distribution")
                    sorted_probabilities = sorted(probabilities, reverse=True)
                    max_probability = sorted_probabilities[0]
                    margin = sorted_probabilities[0] - sorted_probabilities[1]
                    entropy = -sum(
                        probability * math.log(max(probability, 1e-300))
                        for probability in probabilities
                    ) / math.log(len(probabilities))
                    predicted_id = str(response["selected_option_id"])
                    predicted_option = next(
                        option for option in pair["options"] if option["id"] == predicted_id
                    )
                    predicted_position = [
                        float(predicted_option["x_norm"]),
                        float(predicted_option["y_norm"]),
                    ]
                    is_target_present = condition != "target_absent"
                    row = {
                        "sample_id": pair["sample_id"],
                        "source_id": pair["source_id"],
                        "speaker_group": pair["speaker_group"],
                        "target_color": target,
                        "target_position": pair["target_position"],
                        "condition": condition,
                        "target_present": is_target_present,
                        "predicted_option_id": predicted_id,
                        "predicted_position": predicted_position,
                        "prediction_matches_target_position": (
                            predicted_position == pair["target_position"]
                            if is_target_present
                            else None
                        ),
                        "probabilities_by_option_id": dict(zip(ids, probabilities, strict=True)),
                        "max_probability": max_probability,
                        "top_two_margin": margin,
                        "normalized_entropy": entropy,
                        "uncertainty_score": 1.0 - max_probability,
                        "uncertainty_margin_score": 1.0 - margin,
                        "audio_member": pair["audio_member"],
                        "audio_sha256": audio_sha,
                        "image_sha256": image_sha,
                        "target_fill_pixels_edited": target_pixels
                        if condition == "target_absent"
                        else 0,
                        "nontarget_fill_pixels_edited": distractor_pixels
                        if condition == "distractor_absent"
                        else 0,
                        "runtime_request_id": response.get("dispatch_request_id"),
                        "worker_inference_seconds": response.get("worker_inference_seconds"),
                    }
                    records.append(row)
                    predictions_stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                predictions_stream.flush()
                (output_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            "completed_pairs": pair_index + 1,
                            "total_pairs": len(pairs),
                            "prediction_rows": len(records),
                            "updated_utc": datetime.now(UTC).isoformat(),
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    finished = time.perf_counter()
    result = {
        "schema_version": 1,
        "status": "completed_train_only_absent_target_counterfactual_diagnostic",
        "scope": (
            "fixed checkpoint, 96 TRAIN-only paired audio/screen sources, "
            "counterfactual input diagnostic; no model selection or quality claim"
        ),
        "created_start_utc": started_utc,
        "created_end_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": finished - started,
        "source_commit": __import__("subprocess")
        .run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip(),
        "source_worktree_status": __import__("subprocess")
        .run(
            ["git", "-C", str(ROOT), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip(),
        "implementation_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/tiny_omni_decision/runtime_dispatch.py",
                ROOT / "scripts/path_a_torch_worker.py",
                ROOT / "src/tiny_omni_decision/audio_runtime.py",
                ROOT / "src/tiny_omni_decision/spoken_palette_runtime.py",
            )
        },
        "runtime_assembly": {
            "path": str(assembly.resolve()),
            "manifest_sha256": sha256_file(assembly_manifest),
            "package_id": package.get("package_id"),
            "unique_runtime_parameters": package.get("unique_runtime_parameters"),
            "package_file_bytes_excluding_manifest": package.get(
                "package_file_bytes_excluding_manifest"
            ),
            "whisper_storage_variant": package.get("storage_variant"),
        },
        "source_and_data": {
            **{key: value for key, value in preflight.items() if key not in {"selected_pairs"}},
            "audio_archive_sha256": EXPECTED_AUDIO_ARCHIVE,
            "selected_audio_wav_sha256": {
                row["source_id"]: row["audio_sha256"]
                for row in records
                if row["condition"] == "matched"
            },
            "source_license": "Lombard GRID TRAIN, CC BY 4.0, already admitted ALLOW source",
            "used_splits": ["original TRAIN only"],
            "selection_calibration_validation_opened": False,
            "sealed_audit_opened": False,
        },
        "runtime": {
            "device": "cpu",
            "python": platform.python_version(),
            "torch_worker": worker_metadata,
            "training_performed": False,
            "checkpoint_updated": False,
            "threshold_selected_or_applied": False,
        },
        "counterfactual_construction": {
            "matched": "original frozen TRAIN screen with all four colors present",
            "target_absent": (
                "replace all exact target-color fill pixels with a deterministic "
                "non-target palette color; geometry, source audio, option IDs/order "
                "and target coordinates remain fixed"
            ),
            "distractor_absent": (
                "replace one exact non-target fill color with another non-target "
                "palette color; target color remains visible; geometry and options "
                "remain fixed"
            ),
            "pairs": len(pairs),
            "predictions": len(records),
            "all_conditions_use_same_audio_and_options": True,
            "abstain_option_added": False,
            "confidence_threshold_added": False,
        },
        "summary": score_summary(records),
        "sealed_audit_loaded": False,
        "training_performed": False,
        "threshold_selection_performed": False,
    }
    report_path = output_dir / "result.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.sha256").write_text(sha256_file(report_path) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
