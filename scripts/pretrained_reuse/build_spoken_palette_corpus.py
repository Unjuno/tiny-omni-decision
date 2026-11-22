"""Freeze a new train-only factorial speech-to-screen Decision corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.spoken_palette_decision_data import (
    COLORS,
    SPLIT_ORDER,
    build_rows_for_split,
    factorial_support,
    layouts_by_parity,
    split_speaker_groups,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "manifests/pretrained_reuse/lombard-grid-source-20261014.json"
PROTOCOL = ROOT / "configs/pretrained_reuse/lombard-grid-pooled-slot-v1.json"
SOURCE_MANIFEST_SHA256 = "5876e57a4c71108fe3483fde6e18f3977121942c9208567488f042247626577d"
TRAIN_SHA256 = "259f3ee36aa956618a89819338c2ecb67b194c82dbf636df70b551e00a5d78b0"
AUDIO_POOLED_SHA256 = "e09aab88f31957c53c2e48f38a7e79971739efc916547cd5fd3dd6a1bca0a509"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    source_root = args.source_root.resolve(strict=True)
    if artifact_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {artifact_root}")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("pinned Lombard Grid source manifest hash mismatch")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["source_manifest_sha256"] != SOURCE_MANIFEST_SHA256:
        raise ValueError("pooled-slot protocol and source manifest pins disagree")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest.get("policy") != "ALLOW" or source_manifest.get("license") != "CC-BY-4.0":
        raise ValueError("explicit approved Lombard Grid rights are required")

    train_path = source_root / "lombard-grid-layout-02-media-gated" / "train.json"
    if sha256_file(train_path) != TRAIN_SHA256:
        raise ValueError("frozen Lombard Grid TRAIN rows hash mismatch")
    train_rows = json.loads(train_path.read_text(encoding="utf-8"))
    if len(train_rows) != 4080 or any(row.get("split") != "train" for row in train_rows):
        raise ValueError("the exact 4,080-row TRAIN corpus is required")
    if any(row.get("label_status") != "eligible" for row in train_rows):
        raise ValueError("unresolved labels must not enter the compositional task")

    audio_root = source_root / "lombard-grid-whisper-features-01"
    audio_info_path = audio_root / "result.json"
    audio_info = json.loads(audio_info_path.read_text(encoding="utf-8"))
    pooled_path = audio_root / "pooled.npy"
    if (
        audio_info.get("status") != "PASS"
        or audio_info.get("training") is not False
        or audio_info.get("model_parameters") != 8_208_384
        or audio_info.get("feature_hashes", {}).get("pooled.npy") != AUDIO_POOLED_SHA256
        or sha256_file(pooled_path) != AUDIO_POOLED_SHA256
    ):
        raise ValueError("frozen Whisper pooled-feature bank failed its pinned integrity gate")
    audio_features = np.load(pooled_path, mmap_mode="r", allow_pickle=False)
    feature_rows = audio_info.get("rows", [])
    if audio_features.shape != (len(feature_rows), 384) or audio_features.dtype != np.float32:
        raise ValueError("frozen Whisper pooled-feature shape/dtype mismatch")
    feature_indices: dict[str, int] = {}
    audio_hash_by_source: dict[str, str] = {}
    for index, feature_row in enumerate(feature_rows):
        if feature_row.get("split") != "train":
            continue
        source_id = feature_row.get("source_id")
        if source_id in feature_indices:
            raise ValueError("duplicate source identity in frozen feature index")
        feature_indices[source_id] = index
        audio_hash_by_source[source_id] = feature_row.get("input_sha256")
    train_ids = {row["source_id"] for row in train_rows}
    if set(feature_indices) != train_ids or any(
        not isinstance(audio_hash_by_source[source_id], str)
        or len(audio_hash_by_source[source_id]) != 64
        for source_id in train_ids
    ):
        raise ValueError("feature bank does not exactly cover the pinned TRAIN source IDs")

    speaker_splits = split_speaker_groups(train_rows, seed=17, counts=(24, 4, 4, 4))
    assignment_rows = {
        split: [row for row in train_rows if speaker_splits[row["speaker_group"]] == split]
        for split in SPLIT_ORDER
    }
    if any(not rows for rows in assignment_rows.values()):
        raise ValueError("every predeclared speaker split must have TRAIN examples")
    # No phrase/content group or exact source asset may cross the new splits.
    seen_codes: dict[str, str] = {}
    seen_audio_hashes: dict[str, str] = {}
    for split, rows in assignment_rows.items():
        for row in rows:
            for code_key in ("actual_code", "intended_code"):
                code = row.get(code_key)
                if not isinstance(code, str) or len(code) != 6:
                    continue
                owner = seen_codes.setdefault(code, split)
                if owner != split:
                    raise ValueError("normalized utterance content overlaps speaker splits")
            audio_hash = audio_hash_by_source[row["source_id"]]
            owner = seen_audio_hashes.setdefault(audio_hash, split)
            if owner != split:
                raise ValueError("audio media identity overlaps speaker splits")

    artifact_root.mkdir(parents=True, exist_ok=False)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_support: dict[str, dict[str, Any]] = {}
    row_hashes: dict[str, str] = {}
    screen_hashes: dict[str, set[str]] = {}
    for split in SPLIT_ORDER:
        generated = build_rows_for_split(
            train_rows,
            split=split,
            speaker_splits=speaker_splits,
            audio_feature_index=feature_indices,
            output_root=artifact_root,
            variants_per_layout=2,
            seed=17,
        )
        if len({row["source_id"] for row in generated}) != len(assignment_rows[split]):
            raise ValueError("generated rows do not preserve the assigned TRAIN source IDs")
        row_path = artifact_root / f"{split}.jsonl"
        row_hashes[split] = write_jsonl(row_path, generated)
        split_rows[split] = generated
        split_support[split] = factorial_support(generated, assignment_rows[split])
        screen_hashes[split] = {row["screenshot_sha256"] for row in generated}

    for index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[index + 1 :]:
            if screen_hashes[left] & screen_hashes[right]:
                raise ValueError("rendered media identity overlaps task splits")
            left_sources = {row["source_id"] for row in assignment_rows[left]}
            right_sources = {row["source_id"] for row in assignment_rows[right]}
            if left_sources & right_sources:
                raise ValueError("audio source identity overlaps task splits")
            left_groups = {row["speaker_group"] for row in assignment_rows[left]}
            right_groups = {row["speaker_group"] for row in assignment_rows[right]}
            if left_groups & right_groups:
                raise ValueError("speaker group overlaps task splits")

    screen_files = sorted(path for path in (artifact_root / "screens").rglob("*.png"))
    if len(screen_files) != sum(len(rows) for rows in screen_hashes.values()):
        raise ValueError("expected exactly one generated screen per split/layout/variant")
    manifest = {
        "schema_version": 1,
        "experiment_id": "spoken-color-screen-action-v1-seed17",
        "status": "TRAIN_ONLY_FACTORIAL_PREFLIGHT_PASS_NO_MODEL_FIT",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": 17,
        "purpose": (
            "controlled speech-plus-image supplied-option action mechanism; "
            "not product-quality evidence"
        ),
        "source": {
            "dataset_id": "Sheffield/Lombard-Grid",
            "license": "CC-BY-4.0",
            "source_policy": "ALLOW",
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "train_rows_sha256": TRAIN_SHA256,
            "audio_encoder": "openai/whisper-tiny encoder-only",
            "audio_encoder_revision": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
            "audio_encoder_parameters": 8_208_384,
            "audio_feature_bank_sha256": AUDIO_POOLED_SHA256,
            "audio_feature_shape": list(audio_features.shape),
            "audio_features_frozen": True,
            "source_split_used": "original TRAIN only",
            "original_calibration_validation_or_audit_labels_loaded": False,
        },
        "task": {
            "definition": (
                "choose the supplied screen action showing the color spoken in the audio clip"
            ),
            "colors": list(COLORS),
            "candidate_count": 4,
            "screen_contains_query_or_target_text": False,
            "option_ids_are_opaque_and_order_randomized": True,
            "audio_only_bayes_accuracy_under_factorial_layouts": 0.25,
            "image_only_macro_accuracy_ceiling_under_factorial_color_queries": 0.25,
            "both_inputs_uniquely_determine_action": True,
            "screen_layouts": {
                str(parity): [list(layout) for layout in layouts]
                for parity, layouts in layouts_by_parity().items()
            },
            "train_layout_parity": 0,
            "heldout_layout_parity": 1,
            "render_variants_per_layout": 2,
        },
        "speaker_group_split": {
            "method": "SHA-256 order of spoken-palette-v1|17|speaker|<speaker_group>",
            "group_counts": {
                split: len({r["speaker_group"] for r in assignment_rows[split]})
                for split in SPLIT_ORDER
            },
            "source_row_counts": {split: len(assignment_rows[split]) for split in SPLIT_ORDER},
            "generated_decision_counts": {split: len(split_rows[split]) for split in SPLIT_ORDER},
            "speaker_group_ids": {
                split: sorted({r["speaker_group"] for r in assignment_rows[split]})
                for split in SPLIT_ORDER
            },
            "source_id_overlap": False,
            "speaker_group_overlap": False,
            "normalized_content_overlap": False,
            "audio_media_hash_overlap": False,
            "rendered_screen_hash_overlap": False,
        },
        "split_row_sha256": row_hashes,
        "split_support": split_support,
        "generated_screen_count": len(screen_files),
        "screen_sha256": {
            split: sorted(screen_hashes[split]) for split in SPLIT_ORDER
        },
        "evaluation_policy": {
            "train": "fit only on these rows",
            "selection": "checkpoint selection only",
            "calibration": "one scalar temperature only after checkpoint freeze",
            "evaluation": (
                "one-time grouped development evaluation after model freeze; "
                "not blind or product-level"
            ),
            "sealed_audit_opened": False,
            "additional_seeds_or_hyperparameter_search": False,
        },
        "model_fit_started": False,
    }
    if not all(
        split_support[split]["audio_only_exactly_uninformative"]
        and split_support[split]["image_only_no_instance_signal_beyond_color_prior"]
        and split_support[split]["both_inputs_determine_unique_action"]
        and split_support[split]["sample_count_consistency"]
        for split in SPLIT_ORDER
    ):
        raise ValueError("the generated task failed its both-input factorial support gate")
    manifest_sha256 = write_json(artifact_root / "manifest.json", manifest)
    return {"artifact_root": str(artifact_root), "manifest_sha256": manifest_sha256, **manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\CodexArtifacts\pretrained-reuse-r2"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-attempt-01"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = build_corpus(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
