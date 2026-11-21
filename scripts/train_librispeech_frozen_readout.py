"""Fit a small frozen-feature transcript-choice readout on LibriSpeech train-clean-100."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

from scripts.train_frozen_text_probe import _metrics
from tiny_omni_decision.audio_hard_negatives import (
    TranscriptRecord,
    make_transcript_record,
    normalize_transcript,
    parse_cmudict,
)
from tiny_omni_decision.bundle_validation import verify_bundle
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.librispeech_taskadapt import (
    build_choice_examples,
    equal_condition_validation_nll,
    split_by_speaker,
    validate_audio_feature_cache_metadata,
)

QUESTION = "Which exact transcript matches the spoken audio?"
CONDITIONS = ("random_distractors", "hard_distractors")
TRAIN_CLEAN_100_MD5 = "2a93770f6d5c6c964bc36631d331a522"
CMUDICT_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_LICENSE_SHA256 = "bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a"
QUESTION_TEXT = f"Question: {QUESTION}"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="extracted LibriSpeech root"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--cmudict-license", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="immutable Path A runtime bundle"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-data-dir", type=Path)
    parser.add_argument("--audio-feature-cache", type=Path,
                        help="reuse a hash-checked frozen audio feature cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--experiment-id",
        default="path-a-audio-librispeech-train-clean-100-readout-v1",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_audio_feature_cache(
    path: Path, expected_ids: list[str]
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Load an immutable cache only when IDs, order, shape, and values match."""
    cache_path = path.resolve(strict=True)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"ids", "features"}:
        raise ValueError("audio feature cache must contain only ids and features")
    ids, values = payload["ids"], payload["features"]
    if not isinstance(values, Tensor):
        raise ValueError("audio feature cache must be a tensor")
    validate_audio_feature_cache_metadata(
        ids,
        expected_ids,
        values.shape,
        floating=values.is_floating_point(),
        finite=bool(torch.isfinite(values).all()),
    )
    features = {sample_id: row.float().clone() for sample_id, row in zip(ids, values, strict=True)}
    return features, {
        "audio_feature_count": len(features),
        "audio_feature_extraction_seconds": 0.0,
        "audio_encoder_compute_dtype": "reused_frozen_cache",
        "audio_feature_cache_path": str(cache_path),
        "audio_feature_cache_sha256": _sha256(cache_path),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_rows(
    dataset_root: Path, pronunciations: dict[str, tuple[str, ...]]
) -> tuple[list[dict[str, Any]], list[TranscriptRecord], dict[str, int]]:
    corpus = dataset_root / "LibriSpeech" / "train-clean-100"
    if not corpus.is_dir():
        raise FileNotFoundError(f"expected {corpus}")
    rows: list[dict[str, Any]] = []
    records: list[TranscriptRecord] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    duplicate_content = 0
    malformed = 0
    oov = 0
    for transcript_path in sorted(corpus.rglob("*.trans.txt"), key=lambda p: p.as_posix()):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            sample_id, separator, text = line.partition(" ")
            if not separator or not text.strip():
                malformed += 1
                continue
            if sample_id in seen_ids:
                raise ValueError(f"duplicate LibriSpeech sample ID: {sample_id}")
            seen_ids.add(sample_id)
            fields = sample_id.split("-")
            if len(fields) != 3:
                malformed += 1
                continue
            speaker_id, chapter_id, _ = fields
            audio_path = transcript_path.parent / f"{sample_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if audio_path.stat().st_size == 0:
                raise ValueError(f"empty audio asset: {audio_path}")
            duration = None
            raw_norm = normalize_transcript(text)
            if raw_norm in seen_content:
                duplicate_content += 1
            seen_content.add(raw_norm)
            row = {
                "id": sample_id,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "text": text,
                "audio_path": str(audio_path.resolve()),
                "audio_asset_id": f"openslr-slr12-train-clean-100:{sample_id}",
                "duration_seconds": duration,
            }
            rows.append(row)
            record = make_transcript_record(sample_id, speaker_id, chapter_id, text, pronunciations)
            if record is None:
                oov += 1
            else:
                records.append(record)
    if not rows or not records:
        raise ValueError("LibriSpeech corpus or phonemized records are empty")
    record_ids = {record.sample_id for record in records}
    rows = [row for row in rows if row["id"] in record_ids]
    return (
        rows,
        records,
        {
            "transcript_files": len(list(corpus.rglob("*.trans.txt"))),
            "raw_rows": len(seen_ids),
            "malformed_rows": malformed,
            "duplicate_normalized_transcripts": duplicate_content,
            "excluded_cmudict_oov_rows": oov,
            "phonemized_rows": len(records),
            "unique_speakers": len({row["speaker_id"] for row in rows}),
            "unique_chapters": len({row["chapter_id"] for row in rows}),
            "unique_media_assets": len({row["audio_asset_id"] for row in rows}),
        },
    )


def _read_flac(path: Path) -> np.ndarray:
    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            array = frame.to_ndarray().reshape(-1)
            if np.issubdtype(array.dtype, np.integer):
                array = array.astype(np.float32) / float(np.iinfo(array.dtype).max + 1)
            else:
                array = array.astype(np.float32)
            chunks.append(array)
    if not chunks:
        raise ValueError(f"empty audio: {path}")
    return np.concatenate(chunks)


def _extract_audio_features(
    rows: list[dict[str, Any]],
    model_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    model = WhisperModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    encoder = model.encoder
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    ids = list(dict.fromkeys(str(row["id"]) for row in rows))
    by_id = {str(row["id"]): row for row in rows}
    features: dict[str, Tensor] = {}
    started = time.perf_counter()
    encoder_dtype = (
        "bf16_autocast" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    )
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        waves = [_read_flac(Path(by_id[sample_id]["audio_path"])) for sample_id in batch_ids]
        encoded = processor(
            waves,
            sampling_rate=16_000,
            padding="max_length",
            max_length=processor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = encoded.input_features.to(device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if encoder_dtype == "bf16_autocast"
            else torch.autocast("cpu", enabled=False)
        )
        with torch.inference_mode(), autocast:
            hidden = encoder(model_inputs).last_hidden_state
            pooled = []
            for index, wave in enumerate(waves):
                valid = min(hidden.shape[1], max(1, int(np.ceil(len(wave) / 320))))
                pooled.append(hidden[index, :valid].float().mean(dim=0).cpu())
        for sample_id, vector in zip(batch_ids, pooled, strict=True):
            if vector.shape != (384,) or not torch.isfinite(vector).all():
                raise ValueError(f"invalid Whisper feature for {sample_id}")
            features[sample_id] = vector
    del model, encoder, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, {
        "audio_feature_count": len(features),
        "audio_feature_extraction_seconds": time.perf_counter() - started,
        "audio_encoder_compute_dtype": encoder_dtype,
    }


def _encode_texts(
    texts: list[str], model_path: Path, device: torch.device
) -> tuple[dict[str, Tensor], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vectors: dict[str, Tensor] = {}
    unique = list(dict.fromkeys(texts))
    with torch.inference_mode():
        for start in range(0, len(unique), 128):
            batch = unique[start : start + 128]
            encoded = tokenizer(
                batch, max_length=64, truncation=True, padding=True, return_tensors="pt"
            ).to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().cpu()
            if not torch.isfinite(pooled).all():
                raise ValueError("MiniLM produced non-finite text vectors")
            vectors.update((text, vector) for text, vector in zip(batch, pooled, strict=True))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vectors, parameter_count


def _pair_features(audio: Tensor, question: Tensor, options: Tensor) -> Tensor:
    audio_rows = audio.unsqueeze(0).expand(len(options), -1)
    question_rows = question.unsqueeze(0).expand(len(options), -1)
    return torch.cat(
        [
            audio_rows,
            question_rows,
            options,
            (question_rows - options).abs(),
            (audio_rows - options).abs(),
            question_rows * options,
            audio_rows * options,
        ],
        dim=-1,
    )


def _features_for_batch(
    batch: list[dict[str, Any]], audio_features: dict[str, Tensor], text_features: dict[str, Tensor]
) -> tuple[Tensor, Tensor]:
    x = torch.stack(
        [
            _pair_features(
                audio_features[row["sample_id"]],
                text_features[QUESTION_TEXT],
                torch.stack(
                    [text_features[f"Candidate answer: {text}."] for text in row["options"]]
                ),
            )
            for row in batch
        ]
    )
    y = torch.tensor([int(row["target_index"]) for row in batch], dtype=torch.long)
    return x, y


def _evaluate(
    scorer: nn.Module,
    rows: list[dict[str, Any]],
    audio_features: dict[str, Tensor],
    text_features: dict[str, Tensor],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scorer.eval()
    group_logits: dict[str, list[Tensor]] = defaultdict(list)
    group_targets: dict[str, list[int]] = defaultdict(list)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            x, y = _features_for_batch(batch, audio_features, text_features)
            logits = scorer(x.to(device).reshape(-1, x.shape[-1])).reshape(len(batch), 4).cpu()
            for row, row_logits, target in zip(batch, logits, y.tolist(), strict=True):
                condition = row["condition"]
                group_logits[condition].append(row_logits)
                group_targets[condition].append(target)
                predicted = int(row_logits.argmax())
                probabilities = torch.softmax(row_logits, dim=-1)
                predictions.append(
                    {
                        "sample_id": row["sample_id"],
                        "speaker_id": row["speaker_id"],
                        "chapter_id": row["chapter_id"],
                        "condition": condition,
                        "target_index": target,
                        "prediction_index": predicted,
                        "options": row["options"],
                        "probabilities": probabilities.tolist(),
                    }
                )
    by_condition = {
        condition: (
            _metrics(group_logits[condition], group_targets[condition])
            if group_targets[condition]
            else {"count": 0, "accuracy": None, "nll": None, "brier": None, "ece_15_bins": None}
        )
        for condition in CONDITIONS
    }
    if any(group_targets[condition] for condition in CONDITIONS):
        by_condition["all"] = _metrics(
            [logit for condition in CONDITIONS for logit in group_logits[condition]],
            [target for condition in CONDITIONS for target in group_targets[condition]],
        )
    active_conditions = [condition for condition in CONDITIONS if group_targets[condition]]
    if not active_conditions:
        raise ValueError("cannot evaluate an empty set of decision examples")
    macro = {
        key: sum(float(by_condition[condition][key]) for condition in active_conditions)
        / len(active_conditions)
        for key in ("accuracy", "nll", "brier", "ece_15_bins")
    }
    return {
        "by_condition": by_condition,
        "macro_condition": macro,
        "all": by_condition.get("all"),
    }, predictions


def _flatten_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    split = {condition: [] for condition in CONDITIONS}
    for example in examples:
        for condition in CONDITIONS:
            split[condition].append(
                {
                    "sample_id": example["sample_id"],
                    "speaker_id": example["speaker_id"],
                    "chapter_id": example["chapter_id"],
                    "audio_path": example["audio_path"],
                    "audio_asset_id": example["audio_asset_id"],
                    "target_index": example["target_index"],
                    "options": example["options_by_condition"][condition],
                    "condition": condition,
                }
            )
    return split


def main() -> None:
    args = _args()
    started = datetime.now(UTC)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory {output}")
    if args.seed != 17:
        raise ValueError("primary LibriSpeech readout experiment is frozen to seed 17")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cmudict_sha256 = _sha256(args.cmudict.resolve())
    if cmudict_sha256 != CMUDICT_SHA256:
        raise ValueError("CMUdict SHA-256 mismatch")
    cmudict_license_sha256 = _sha256(args.cmudict_license.resolve())
    if cmudict_license_sha256 != CMUDICT_LICENSE_SHA256:
        raise ValueError("CMUdict license SHA-256 mismatch")
    md5 = hashlib.md5(usedforsecurity=False)
    archive_sha_digest = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            archive_sha_digest.update(block)
    if md5.hexdigest() != TRAIN_CLEAN_100_MD5:
        raise ValueError(f"OpenSLR train-clean-100 MD5 mismatch: {md5.hexdigest()}")
    archive_sha256 = archive_sha_digest.hexdigest()
    bundle = args.bundle.resolve()
    bundle_manifest = json.loads((bundle / "bundle-manifest.json").read_text(encoding="utf-8"))
    verify_bundle(bundle, bundle_manifest)
    text_model = bundle / "models" / "text-minilm-en"
    audio_model = bundle / "models" / "audio-whisper-tiny"
    bundle_sha256 = _sha256(bundle / "bundle-manifest.json")
    if args.preflight_only:
        if args.prepared_data_dir is None:
            raise ValueError("--prepared-data-dir is required for preflight materialization")
        prepared = args.prepared_data_dir.resolve()
        if prepared.exists():
            raise FileExistsError(f"refusing to reuse prepared-data directory: {prepared}")
        pronunciations = parse_cmudict(args.cmudict.read_text(encoding="latin-1"))
        rows, records, input_counts = _load_rows(args.dataset_root.resolve(), pronunciations)
        train_rows, validation_rows, split_report = split_by_speaker(
            rows, seed=args.seed, validation_fraction=args.validation_fraction
        )
        record_map = {record.sample_id: record for record in records}
        train_rows = [row for row in train_rows if row["id"] in record_map]
        validation_rows = [row for row in validation_rows if row["id"] in record_map]
        train_records = [record_map[row["id"]] for row in train_rows]
        validation_records = [record_map[row["id"]] for row in validation_rows]
        tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=True)
        keep = set()
        for row in train_rows + validation_rows:
            tokenized = tokenizer(
                f"Candidate answer: {row['text']}.", truncation=False, padding=False
            )["input_ids"]
            if len(tokenized) <= 64:
                keep.add(row["id"])
        train_rows = [row for row in train_rows if row["id"] in keep]
        validation_rows = [row for row in validation_rows if row["id"] in keep]
        train_records = [record_map[row["id"]] for row in train_rows]
        validation_records = [record_map[row["id"]] for row in validation_rows]
        train_examples, train_construction = build_choice_examples(
            train_rows, train_records, seed=args.seed
        )
        validation_examples, validation_construction = build_choice_examples(
            validation_rows, validation_records, seed=args.seed
        )
        if len({row["speaker_id"] for row in validation_examples}) < 20:
            raise ValueError("fewer than 20 held-out validation speakers")
        if not train_examples or not validation_examples:
            raise ValueError("choice construction produced an empty split")
        del tokenizer
        prepared.mkdir(parents=True)
        split_manifest_path = prepared / "corpus-split-manifest.jsonl"
        split_manifest_path.write_text(
            "".join(
                json.dumps(
                    {
                        "split": split,
                        "sample_id": row["id"],
                        "speaker_id": row["speaker_id"],
                        "chapter_id": row["chapter_id"],
                        "normalized_text": normalize_transcript(row["text"]),
                        "audio_asset_id": row["audio_asset_id"],
                        "audio_path": row["audio_path"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for split, split_rows in (("train", train_rows), ("validation", validation_rows))
                for row in split_rows
            ),
            encoding="utf-8",
        )
        for name, examples in (("train", train_examples), ("validation", validation_examples)):
            (prepared / f"{name}-examples.jsonl").write_text(
                "".join(
                    json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
                    for example in examples
                ),
                encoding="utf-8",
            )
        split_data = {
            "input_counts": input_counts,
            "split": split_report,
            "train_construction": train_construction,
            "validation_construction": validation_construction,
            "train_ids_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in train_rows).encode()
            ).hexdigest(),
            "validation_ids_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in validation_rows).encode()
            ).hexdigest(),
        }
        _write_json(prepared / "split-report.json", split_data)
        prepared_report = {
            "schema_version": 1,
            "status": "complete_librispeech_train_clean_100_preflight",
            "source": {
                "archive_md5": md5.hexdigest(),
                "archive_sha256": archive_sha256,
                "cmudict_sha256": cmudict_sha256,
                "cmudict_license_sha256": cmudict_license_sha256,
                "bundle_manifest_sha256": bundle_sha256,
            },
            "seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "files": {
                name: _sha256(prepared / name)
                for name in (
                    "corpus-split-manifest.jsonl",
                    "split-report.json",
                    "train-examples.jsonl",
                    "validation-examples.jsonl",
                )
            },
            "summary": split_data,
        }
        _write_json(prepared / "preflight-report.json", prepared_report)
        print(
            json.dumps(
                {
                    "preflight": "pass",
                    "prepared_data_dir": str(prepared),
                    "source_md5": prepared_report["source"]["archive_md5"],
                    "source_sha256": prepared_report["source"]["archive_sha256"],
                    "input_counts": split_data["input_counts"],
                    "split": split_data["split"],
                    "train_construction": split_data["train_construction"],
                    "validation_construction": split_data["validation_construction"],
                    "train_unique_audio": len({row["audio_asset_id"] for row in train_examples}),
                    "validation_unique_audio": len(
                        {row["audio_asset_id"] for row in validation_examples}
                    ),
                    "validation_speakers": len({row["speaker_id"] for row in validation_examples}),
                    "validation_chapters": len({row["chapter_id"] for row in validation_examples}),
                    "prepared_files_sha256": prepared_report["files"],
                },
                indent=2,
            )
        )
        return

    if args.prepared_data_dir is None:
        raise ValueError("--prepared-data-dir from a completed preflight is required")
    prepared = args.prepared_data_dir.resolve()
    prepared_report = json.loads((prepared / "preflight-report.json").read_text(encoding="utf-8"))
    expected_source = {
        "archive_md5": md5.hexdigest(),
        "archive_sha256": archive_sha256,
        "cmudict_sha256": cmudict_sha256,
        "cmudict_license_sha256": cmudict_license_sha256,
        "bundle_manifest_sha256": bundle_sha256,
    }
    if prepared_report.get("status") != "complete_librispeech_train_clean_100_preflight":
        raise ValueError("prepared data does not contain a completed preflight")
    if prepared_report.get("source") != expected_source:
        raise ValueError("prepared data source hashes differ from current immutable inputs")
    if (
        prepared_report.get("seed") != args.seed
        or prepared_report.get("validation_fraction") != args.validation_fraction
    ):
        raise ValueError("prepared data split policy differs from requested config")
    for name, expected_hash in prepared_report["files"].items():
        if _sha256(prepared / name) != expected_hash:
            raise ValueError(f"prepared data hash mismatch: {name}")
    split_data = json.loads((prepared / "split-report.json").read_text(encoding="utf-8"))
    train_examples = [
        json.loads(line)
        for line in (prepared / "train-examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    validation_examples = [
        json.loads(line)
        for line in (prepared / "validation-examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if not train_examples or not validation_examples:
        raise ValueError("prepared train/validation examples are empty")
    split_manifest = prepared / "corpus-split-manifest.jsonl"
    manifest_rows = [
        json.loads(line) for line in split_manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    train_manifest = [row for row in manifest_rows if row["split"] == "train"]
    validation_manifest = [row for row in manifest_rows if row["split"] == "validation"]
    if not train_manifest or not validation_manifest:
        raise ValueError("prepared split manifest is empty")
    train_speakers = {row["speaker_id"] for row in train_manifest}
    validation_speakers = {row["speaker_id"] for row in validation_manifest}
    train_assets = {row["audio_asset_id"] for row in train_manifest}
    validation_assets = {row["audio_asset_id"] for row in validation_manifest}
    train_texts = {row["normalized_text"] for row in train_manifest}
    validation_texts = {row["normalized_text"] for row in validation_manifest}
    if (
        train_speakers & validation_speakers
        or train_assets & validation_assets
        or train_texts & validation_texts
    ):
        raise ValueError("prepared split has speaker, asset, or normalized-content overlap")
    split_to_manifest = {
        "train": {row["sample_id"]: row for row in train_manifest},
        "validation": {row["sample_id"]: row for row in validation_manifest},
    }
    for split_name, examples in (("train", train_examples), ("validation", validation_examples)):
        allowed = split_to_manifest[split_name]
        for example in examples:
            source = allowed.get(example["sample_id"])
            if source is None or source["speaker_id"] != example["speaker_id"]:
                raise ValueError(f"prepared example has invalid {split_name} source identity")
            path = Path(example["audio_path"]).resolve()
            if not path.is_relative_to(args.dataset_root.resolve()) or not path.is_file():
                raise ValueError(f"prepared audio path is missing or outside source: {path}")
            for ids in example["distractor_ids_by_condition"].values():
                if any(candidate_id not in allowed for candidate_id in ids):
                    raise ValueError(f"{split_name} distractor escaped its split")
    input_counts = split_data["input_counts"]
    split_report = split_data["split"]
    train_construction = split_data["train_construction"]
    validation_construction = split_data["validation_construction"]
    if len({row["speaker_id"] for row in validation_examples}) < 20:
        raise ValueError("fewer than 20 held-out validation speakers")

    device = torch.device(args.device)
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "epochs": args.epochs,
        "feature_batch_size": args.feature_batch_size,
        "query_batch_size": args.query_batch_size,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "selection_rule": "minimum mean of random/hard held-speaker validation NLL",
        "hard_negative_policy": (
            "filter by split, speaker, chapter, word count, character length, "
            "and shared vocabulary; "
            "rank a deterministic per-target shortlist of at most 256 by phoneme similarity"
        ),
        "trainable_component": "344321-parameter FrozenFeatureCandidateScorer only",
        "frozen_components": ["Whisper Tiny encoder", "MiniLM L3 text encoder"],
        "question": QUESTION,
        "candidate_count": 4,
        "conditions": list(CONDITIONS),
        "audio_feature_cache_sha256": (
            _sha256(args.audio_feature_cache.resolve(strict=True))
            if args.audio_feature_cache is not None else None
        ),
        "sealed_audit_loaded": False,
        "test_split_loaded": False,
    }
    (output / "effective-config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    for name in (
        "corpus-split-manifest.jsonl",
        "split-report.json",
        "train-examples.jsonl",
        "validation-examples.jsonl",
    ):
        shutil.copy2(prepared / name, output / name)
    split_manifest = output / "corpus-split-manifest.jsonl"

    all_examples = _flatten_examples(train_examples + validation_examples)
    audio_rows_by_id = {}
    for split_rows in all_examples.values():
        for row in split_rows:
            audio_rows_by_id.setdefault(
                row["sample_id"],
                {
                    "id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "chapter_id": row["chapter_id"],
                    "audio_path": row["audio_path"],
                    "audio_asset_id": row["audio_asset_id"],
                },
            )
    needed_rows = list(audio_rows_by_id.values())
    expected_audio_ids = [row["id"] for row in needed_rows]
    if args.audio_feature_cache is None:
        audio_features, extraction_report = _extract_audio_features(
            needed_rows, audio_model, device, args.feature_batch_size
        )
    else:
        audio_features, extraction_report = _load_audio_feature_cache(
            args.audio_feature_cache, expected_audio_ids
        )
    cache_path = output / "audio-features.pt"
    torch.save(
        {"ids": list(audio_features), "features": torch.stack(list(audio_features.values()))},
        cache_path,
    )
    question_strings = [QUESTION_TEXT]
    for split_rows in all_examples.values():
        for row in split_rows:
            question_strings.extend(f"Candidate answer: {text}." for text in row["options"])
    text_features, text_parameters = _encode_texts(question_strings, text_model, device)
    if text_parameters != 17_389_824:
        raise ValueError(f"unexpected MiniLM parameters: {text_parameters}")
    del all_examples

    train_queries = [
        row for condition in CONDITIONS for row in _flatten_examples(train_examples)[condition]
    ]
    validation_queries_by_condition = _flatten_examples(validation_examples)
    validation_queries = [
        row for condition in CONDITIONS for row in validation_queries_by_condition[condition]
    ]
    feature_size = 7 * 384
    scorer = FrozenFeatureCandidateScorer(feature_size, hidden_size=128).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in scorer.parameters())
    if trainable_parameters != 344_321:
        raise ValueError(f"unexpected readout size: {trainable_parameters}")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=0.001, weight_decay=0.0001)

    old_state_path = bundle / bundle_manifest["models"]["readout"]["path"]
    old_state = torch.load(old_state_path, map_location="cpu", weights_only=True)
    baseline_scorer = FrozenFeatureCandidateScorer(feature_size, hidden_size=128).to(device)
    baseline_scorer.load_state_dict(old_state, strict=True)
    baseline_eval_started = time.perf_counter()
    baseline_metrics, baseline_predictions = _evaluate(
        baseline_scorer,
        validation_queries,
        audio_features,
        text_features,
        device,
        args.query_batch_size,
    )
    baseline_validation_seconds = time.perf_counter() - baseline_eval_started
    del baseline_scorer, old_state

    best_nll = float("inf")
    best_epoch = 0
    best_state_path = output / "best-readout.pt"
    if best_state_path.is_file():
        best_state_path.unlink()
    history = []
    epoch_validation_seconds = []
    started_training = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_queries)))
        random.Random(args.seed + epoch).shuffle(order)
        scorer.train()
        batch_losses = []
        for start in range(0, len(order), args.query_batch_size):
            batch = [train_queries[index] for index in order[start : start + args.query_batch_size]]
            x, y = _features_for_batch(batch, audio_features, text_features)
            logits = scorer(x.to(device).reshape(-1, feature_size)).reshape(len(batch), 4)
            loss = F.cross_entropy(logits, y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in scorer.parameters()
            ):
                raise FloatingPointError(f"non-finite readout gradient at epoch {epoch}")
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_metrics = {}
        validation_started = time.perf_counter()
        for condition in CONDITIONS:
            epoch_metrics[condition], _ = _evaluate(
                scorer,
                validation_queries_by_condition[condition],
                audio_features,
                text_features,
                device,
                args.query_batch_size,
            )
        epoch_validation_seconds.append(time.perf_counter() - validation_started)
        epoch_metrics["all"], _ = _evaluate(
            scorer, validation_queries, audio_features, text_features, device, args.query_batch_size
        )
        val_nll = equal_condition_validation_nll(epoch_metrics, CONDITIONS)
        history.append(
            {
                "epoch": epoch,
                "train_cross_entropy": sum(batch_losses) / len(batch_losses),
                "validation_by_condition": epoch_metrics,
                "selection_nll": val_nll,
            }
        )
        _write_json(output / "training-history.json", history)
        if val_nll < best_nll:
            best_nll = val_nll
            best_epoch = epoch
            torch.save(scorer.state_dict(), best_state_path)
    train_seconds = time.perf_counter() - started_training
    torch.save(scorer.state_dict(), output / "final-readout.pt")
    best_state = torch.load(output / "best-readout.pt", map_location="cpu", weights_only=True)
    scorer.load_state_dict(best_state, strict=True)
    selected_validation_started = time.perf_counter()
    selected_metrics, selected_predictions = _evaluate(
        scorer, validation_queries, audio_features, text_features, device, args.query_batch_size
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    selected_validation_seconds = time.perf_counter() - selected_validation_started
    predictions_path = output / "selected-validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected_predictions
        ),
        encoding="utf-8",
    )
    baseline_path = output / "baseline-validation-predictions.jsonl"
    baseline_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in baseline_predictions
        ),
        encoding="utf-8",
    )
    report = {
        "status": "complete_librispeech_train_clean_100_frozen_readout_development_experiment",
        "experiment_id": config["experiment_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": _sha256(output / "effective-config.yaml"),
        "split_manifest_sha256": _sha256(split_manifest),
        "source": {
            "official_dataset": "OpenSLR LibriSpeech SLR12 train-clean-100; CC BY 4.0",
            "archive_md5": md5.hexdigest(),
            "archive_sha256": archive_sha256,
            "cmudict_sha256": _sha256(args.cmudict),
            "cmudict_license_sha256": _sha256(args.cmudict_license),
            "bundle_manifest_sha256": _sha256(bundle / "bundle-manifest.json"),
        },
        "split_and_construction": {
            "input_counts": input_counts,
            "split": split_report,
            "train_construction": train_construction,
            "validation_construction": validation_construction,
            "train_unique_audio_assets_used": len(
                {row["audio_asset_id"] for row in train_examples}
            ),
            "validation_unique_audio_assets_used": len(
                {row["audio_asset_id"] for row in validation_examples}
            ),
            "train_speaker_groups_used": len({row["speaker_id"] for row in train_examples}),
            "validation_speaker_groups_used": len(
                {row["speaker_id"] for row in validation_examples}
            ),
        },
        "models": {
            "audio_base_revision": "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
            "audio_weights_sha256": _sha256(audio_model / "model.safetensors"),
            "audio_parameters_frozen": 37_760_640,
            "text_base_revision": (
                "sentence-transformers/paraphrase-MiniLM-L3-v2@"
                "4ca70771034acceecb2e72475f72050fcdde4ddc"
            ),
            "text_weights_sha256": _sha256(text_model / "model.safetensors"),
            "text_parameters_frozen": text_parameters,
            "readout_parameters_trainable": trainable_parameters,
            "full_component_parameters": 37_760_640 + text_parameters + trainable_parameters,
            "backbone_weights_changed": False,
        },
        "feature_extraction": extraction_report,
        "training": {
            "epochs_run": args.epochs,
            "selected_epoch": best_epoch,
            "selection_macro_nll": best_nll,
            "optimizer_updates": sum(
                (len(train_queries) + args.query_batch_size - 1) // args.query_batch_size
                for _ in range(args.epochs)
            ),
            "training_and_validation_loop_seconds": train_seconds,
            "per_epoch_validation_seconds": epoch_validation_seconds,
            "baseline_validation_seconds": baseline_validation_seconds,
            "selected_validation_seconds": selected_validation_seconds,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
        "validation_metrics": {
            "existing_speech_commands_readout": baseline_metrics,
            "selected_librispeech_readout": selected_metrics,
        },
        "artifacts": {
            name: {
                "path": name,
                "sha256": _sha256(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in (
                "best-readout.pt",
                "final-readout.pt",
                "training-history.json",
                "selected-validation-predictions.jsonl",
                "baseline-validation-predictions.jsonl",
                "audio-features.pt",
                "corpus-split-manifest.jsonl",
                "train-examples.jsonl",
                "validation-examples.jsonl",
            )
        },
        "sealed_audit_loaded": False,
        "test_split_loaded": False,
        "product_promotion": False,
        "environment": {
            "command_arguments": sys.argv,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "pyav": importlib.metadata.version("av"),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "limitations": [
            "Public LibriSpeech audiobook speech is not conversational speech or broad audio QA.",
            "Validation is development data used for checkpoint selection, not a blind estimate.",
            "Transcript training tests a narrow audio-to-supplied-transcript matching task.",
            "Frozen backbones dominate parameter count; quantized size remains unmeasured.",
        ],
    }
    _write_json(output / "run-report.json", report)
    print(
        json.dumps(
            {"status": report["status"], "output_dir": str(output), "report": report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
