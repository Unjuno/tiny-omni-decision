"""Verify Whisper feature-cache and fixed-option output parity on one validation clip."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.train_frozen_audio_probe import (
    LABELS,
    CandidateScorer,
    _extract_features,
    _load_encoder,
    _option_embeddings,
    _predict,
    _question_features,
    _rows,
    _validate_splits,
)
from tiny_omni_decision.dataset import sha256_file

EXPECTED_AUDIO_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
EXPECTED_TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--audio-model", type=Path, required=True)
    parser.add_argument("--audio-revision", required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    run_started_at = datetime.now(UTC).isoformat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.audio_revision != EXPECTED_AUDIO_REVISION:
        raise ValueError("unexpected Whisper revision")
    if args.text_revision != EXPECTED_TEXT_REVISION:
        raise ValueError("unexpected text candidate revision")
    if not torch.cuda.is_available():
        raise RuntimeError("this audio cache diagnostic requires the existing local CUDA GPU")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(device)

    sample_dir = args.sample_dir.resolve()
    train_rows = _rows(sample_dir / "train.jsonl")
    validation_rows = _rows(sample_dir / "validation.jsonl")
    _validate_splits(sample_dir, train_rows, validation_rows)
    cached_payload = torch.load(args.feature_cache, map_location="cpu", weights_only=True)
    cached_ids = list(cached_payload["validation_ids"])
    cached_features = cached_payload["validation_features"]
    if len(cached_ids) != len(cached_features) or len(set(cached_ids)) != len(cached_ids):
        raise ValueError("saved validation feature IDs are missing or non-unique")
    cached_index = {sample_id: index for index, sample_id in enumerate(cached_ids)}
    row_by_id = {str(row["id"]): row for row in validation_rows}
    if set(cached_ids) != set(row_by_id):
        raise ValueError("saved validation feature IDs differ from the frozen sample")
    selected_id = cached_ids[0]
    selected_row = row_by_id[selected_id]
    cached_feature = cached_features[cached_index[selected_id]].reshape(1, -1)

    audio_args = SimpleNamespace(audio_model=args.audio_model, audio_revision=args.audio_revision)
    encoder, _, processor = _load_encoder(audio_args, device)
    model_allocated_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize(device)
    extraction_started = time.perf_counter()
    extracted, extracted_ids, _, extraction_seconds = _extract_features(
        [selected_row], sample_dir, encoder, processor, device, batch_size=1
    )
    torch.cuda.synchronize(device)
    total_extraction_seconds = time.perf_counter() - extraction_started
    if extracted_ids != [selected_id]:
        raise ValueError("re-extraction returned a different validation sample ID")
    max_abs_difference = float((cached_feature - extracted).abs().max().item())
    feature_parity = torch.allclose(cached_feature, extracted, rtol=1e-5, atol=1e-5)
    if not feature_parity:
        raise ValueError(f"cached and re-extracted audio features differ: {max_abs_difference}")

    option_embeddings, _ = _option_embeddings(
        args.text_model, args.text_revision, device
    )
    scorer = CandidateScorer(int(extracted.shape[-1]) * 4).to(device)
    scorer.load_state_dict(torch.load(args.readout, map_location=device, weights_only=True))
    scorer.eval()
    cached_logits = _predict(
        scorer, _question_features(cached_feature, option_embeddings.cpu()), device
    )[0]
    extracted_logits = _predict(
        scorer, _question_features(extracted, option_embeddings.cpu()), device
    )[0]
    output_parity = torch.allclose(cached_logits, extracted_logits, rtol=1e-5, atol=1e-5)
    max_abs_logit_difference = float((cached_logits - extracted_logits).abs().max().item())
    if not output_parity:
        raise ValueError(
            f"cached and re-extracted decision logits differ: {max_abs_logit_difference}"
        )

    report = {
        "schema_version": 1,
        "status": "complete_audio_feature_cache_parity_diagnostic",
        "scope": "one existing Speech Commands validation clip and its fixed 10-option readout",
        "started_utc": run_started_at,
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sample_dir": str(sample_dir),
        "train_manifest_sha256": sha256_file(sample_dir / "train.jsonl"),
        "validation_manifest_sha256": sha256_file(sample_dir / "validation.jsonl"),
        "feature_cache_sha256": sha256_file(args.feature_cache),
        "selected_sample_id": selected_id,
        "audio_sha256": selected_row["media_sha256"],
        "target": selected_row["target"],
        "split": "validation",
        "sealed_audit_loaded": False,
        "whisper_revision": args.audio_revision,
        "whisper_weights_sha256": sha256_file(args.audio_model / "model.safetensors"),
        "text_revision": args.text_revision,
        "readout_sha256": sha256_file(args.readout),
        "candidate_count": len(LABELS),
        "cached_feature_shape": list(cached_feature.shape),
        "feature_dtype": str(cached_feature.dtype),
        "cached_vs_reextracted_feature_allclose": feature_parity,
        "max_abs_feature_difference": max_abs_difference,
        "cached_vs_reextracted_logits_allclose": output_parity,
        "max_abs_logit_difference": max_abs_logit_difference,
        "options": LABELS,
        "cached_option_probabilities": torch.softmax(cached_logits.float(), dim=-1).tolist(),
        "reextracted_option_probabilities": torch.softmax(
            extracted_logits.float(), dim=-1
        ).tolist(),
        "feature_cache_entries_reused": 1,
        "encoder_executions_with_feature_cache": 0,
        "encoder_executions_without_feature_cache": 1,
        "encoder_executions_avoided": 1,
        "encoder_extraction_seconds": extraction_seconds,
        "wall_seconds_including_cuda_sync": total_extraction_seconds,
        "cached_feature_bytes": cached_feature.nelement() * cached_feature.element_size(),
        "cuda_model_allocated_bytes_before_extraction": model_allocated_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device": torch.cuda.get_device_name(device),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
        },
        "note": (
            "This checks feature and fixed-option output parity for one clip. It does not measure "
            "a multi-question audio task or establish audio quality."
        ),
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
