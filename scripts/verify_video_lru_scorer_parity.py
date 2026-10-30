"""Verify the observation LRU inside the frozen CLEVRER decision-scoring path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from scripts.diagnose_clevrer_video_multiquery import (
    TEXT_MODEL_WEIGHTS_SHA256,
    TEXT_REVISION,
)
from scripts.fetch_clevrer_probe import VALIDATION_QUESTIONS_SHA256
from scripts.train_frozen_text_probe import _metrics
from scripts.train_frozen_video_probe import (
    _combine,
    _load_rows,
    _predict,
    _text_features,
)
from tiny_omni_decision.cache import (
    InMemoryObservationFeatureLRU,
    ObservationFeatureCache,
    ObservationFeatureKey,
)
from tiny_omni_decision.clevrer import expand_descriptive_questions
from tiny_omni_decision.dataset import sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer


def _key_from_dict(value: dict[str, Any]) -> ObservationFeatureKey:
    fields = dict(value)
    fields.pop("schema_version", None)
    fields["feature_shape"] = tuple(fields["feature_shape"])
    return ObservationFeatureKey(**fields)

EXPECTED_CACHE_MANIFEST_SHA256 = (
    "289b3b435fc318c40f9e03ada0902645debd76f61cb1d6ac36c70e8fc35ed9d4"
)
EXPECTED_READOUT_SHA256 = "77ff3dcc1067da138235fc3082fffc1746e8c5a14d644da5827c82813496df1f"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--raw-validation-questions", type=Path, required=True)
    parser.add_argument("--typed-cache-dir", type=Path, required=True)
    parser.add_argument("--typed-cache-manifest", type=Path, required=True)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--text-revision", required=True)
    parser.add_argument("--capacity-bytes", type=int, default=12_288)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _fetch_feature(
    key: ObservationFeatureKey,
    expected_payload_sha256: str,
    memory: InMemoryObservationFeatureLRU,
    persistent: ObservationFeatureCache,
) -> tuple[torch.Tensor, str]:
    payload = memory.get(key)
    source = "memory_lru"
    if payload is None:
        item = persistent.get(key)
        if item.payload_sha256 != expected_payload_sha256:
            raise ValueError(f"persistent feature hash mismatch for {key.observation_sha256}")
        payload = item.payload
        if not memory.put(key, payload):
            raise ValueError("feature payload exceeds configured LRU byte budget")
        source = "persistent_cache"

    return _decode_feature(key, payload), source


def _decode_feature(key: ObservationFeatureKey, payload: bytes) -> torch.Tensor:
    if key.feature_dtype != "float32":
        raise ValueError(f"unsupported cached feature dtype: {key.feature_dtype}")
    expected_values = 1
    for dimension in key.feature_shape:
        expected_values *= dimension
    if len(payload) != expected_values * torch.tensor([], dtype=torch.float32).element_size():
        raise ValueError("cached feature payload size does not match its declared shape")
    feature = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(key.feature_shape)
    if not torch.isfinite(feature).all():
        raise ValueError("cached feature payload contains non-finite values")
    return feature.contiguous()


def _load_manifest(path: Path) -> dict[str, tuple[ObservationFeatureKey, str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported typed observation cache manifest")
    by_observation: dict[str, tuple[ObservationFeatureKey, str]] = {}
    for entry in manifest.get("entries", []):
        key = _key_from_dict(entry["key"])
        if key.cache_id != entry.get("cache_id"):
            raise ValueError("cache manifest key does not match its cache ID")
        if key.observation_sha256 != entry.get("observation_sha256"):
            raise ValueError("cache manifest key does not match its observation hash")
        if key.modality != "video" or key.feature_shape != (768,):
            raise ValueError("expected pooled 768-dimensional video observation features")
        if key.observation_sha256 in by_observation:
            raise ValueError("duplicate observation identity in typed cache manifest")
        by_observation[key.observation_sha256] = (
            key,
            str(entry["feature_payload_sha256"]),
        )
    if not by_observation:
        raise ValueError("typed observation cache manifest has no entries")
    return by_observation


def main() -> None:
    args = _args()
    paths = (
        args.sample_dir,
        args.raw_validation_questions,
        args.typed_cache_dir,
        args.typed_cache_manifest,
        args.readout_dir,
        args.reference_predictions,
        args.text_model,
    )
    for name, path in zip(
        (
            "sample_dir",
            "raw_validation_questions",
            "typed_cache_dir",
            "typed_cache_manifest",
            "readout_dir",
            "reference_predictions",
            "text_model",
        ),
        paths,
        strict=True,
    ):
        setattr(args, name, path.resolve())
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    if args.text_revision != TEXT_REVISION:
        raise ValueError(f"unexpected MiniLM revision: {args.text_revision}")
    if sha256_file(args.raw_validation_questions) != VALIDATION_QUESTIONS_SHA256:
        raise ValueError("raw CLEVRER validation question hash differs from pinned source")
    if sha256_file(args.typed_cache_manifest) != EXPECTED_CACHE_MANIFEST_SHA256:
        raise ValueError("typed cache manifest differs from the frozen CLEVRER artifact")
    if sha256_file(args.readout_dir / "best-readout.pt") != EXPECTED_READOUT_SHA256:
        raise ValueError("readout checkpoint differs from the frozen CLEVRER artifact")
    text_weights = args.text_model / "model.safetensors"
    if sha256_file(text_weights) != TEXT_MODEL_WEIGHTS_SHA256:
        raise ValueError("MiniLM weights differ from the pinned local checkpoint")

    train = _load_rows(args.sample_dir / "train.jsonl")
    validation = _load_rows(args.sample_dir / "validation.jsonl")
    sample_hashes = {
        "train": sha256_file(args.sample_dir / "train.jsonl"),
        "validation": sha256_file(args.sample_dir / "validation.jsonl"),
        "fetch_report": sha256_file(args.sample_dir / "fetch-report.json"),
    }
    expected_sample_hashes = {
        "train": "25d70a899f934cd84f479ba19e52e109dbe64492446a47baf898e64ccb1824fc",
        "validation": "8d227fb2c49449730e920e3efcb10aa377d64e3926cca10fddc6f37e561dae0e",
        "fetch_report": "ea3918dc66ec8b355405faf36722a0774db6e921d885cf69fdb135578c27134f",
    }
    if sample_hashes != expected_sample_hashes:
        raise ValueError("frozen CLEVRER train/validation manifest hashes changed")
    if {int(row["scene_index"]) for row in train} & {
        int(row["scene_index"]) for row in validation
    }:
        raise ValueError("CLEVRER train and validation scenes overlap")

    raw_questions = json.loads(args.raw_validation_questions.read_text(encoding="utf-8"))
    rows = expand_descriptive_questions(raw_questions, validation)
    reference = [
        json.loads(line)
        for line in args.reference_predictions.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [row["id"] for row in rows] != [row["id"] for row in reference]:
        raise ValueError("expanded validation IDs/order differ from the frozen reference")

    by_observation = _load_manifest(args.typed_cache_manifest)
    persistent = ObservationFeatureCache(args.typed_cache_dir)
    direct_by_observation: dict[str, torch.Tensor] = {}
    for observation_sha, (key, expected_sha) in by_observation.items():
        item = persistent.get(key)
        if item.payload_sha256 != expected_sha:
            raise ValueError(f"direct-load feature hash mismatch for {observation_sha}")
        direct_by_observation[observation_sha] = _decode_feature(key, item.payload)

    device = torch.device("cpu")
    torch.set_num_threads(4)
    started = datetime.now(UTC).isoformat()
    text_load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=True)
    text_encoder = (
        AutoModel.from_pretrained(args.text_model, local_files_only=True).eval().to(device)
    )
    text_load_seconds = time.perf_counter() - text_load_started
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)
    text_started = time.perf_counter()
    text_features, targets = _text_features(rows, tokenizer, text_encoder, device)
    text_encoding_seconds = time.perf_counter() - text_started
    del text_encoder, tokenizer

    readout_started = time.perf_counter()
    state = torch.load(args.readout_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    first_weight = state["network.0.weight"]
    scorer = FrozenFeatureCandidateScorer(
        feature_size=int(first_weight.shape[1]), hidden_size=int(first_weight.shape[0])
    )
    scorer.load_state_dict(state, strict=True)
    scorer.eval()
    readout_load_seconds = time.perf_counter() - readout_started

    direct_video = torch.stack(
        [direct_by_observation[str(row["media_sha256"])] for row in rows]
    )
    direct_candidate_features = _combine(direct_video, text_features)
    direct_started = time.perf_counter()
    direct_logits = _predict(scorer, direct_candidate_features, device)
    direct_scoring_seconds = time.perf_counter() - direct_started

    memory = InMemoryObservationFeatureLRU(args.capacity_bytes)
    lru_video_rows = []
    source_counts: Counter[str] = Counter()
    persistent_reads = 0
    lru_started = time.perf_counter()
    for row in rows:
        observation_sha = str(row["media_sha256"])
        manifest_entry = by_observation.get(observation_sha)
        if manifest_entry is None:
            raise ValueError(f"no typed feature key for validation observation {observation_sha}")
        key, expected_sha = manifest_entry
        feature, source = _fetch_feature(key, expected_sha, memory, persistent)
        source_counts[source] += 1
        persistent_reads += int(source == "persistent_cache")
        lru_video_rows.append(feature)
    lru_retrieval_seconds = time.perf_counter() - lru_started
    lru_video = torch.stack(lru_video_rows)
    lru_candidate_features = _combine(lru_video, text_features)
    lru_scoring_started = time.perf_counter()
    lru_logits = _predict(scorer, lru_candidate_features, device)
    lru_scoring_seconds = time.perf_counter() - lru_scoring_started

    direct_probabilities = [torch.softmax(item.float(), dim=-1) for item in direct_logits]
    lru_probabilities = [torch.softmax(item.float(), dim=-1) for item in lru_logits]
    exact_logits = all(
        torch.equal(left, right)
        for left, right in zip(direct_logits, lru_logits, strict=True)
    )
    exact_probabilities = all(
        torch.equal(left, right)
        for left, right in zip(direct_probabilities, lru_probabilities, strict=True)
    )
    if not exact_logits or not exact_probabilities:
        raise ValueError("inference output changed between direct and LRU feature retrieval")

    reference_by_id = {row["id"]: row for row in reference}
    reference_mismatches = 0
    reference_max_probability_delta = 0.0
    predictions = []
    for row, logits, probabilities, target_index in zip(
        rows, lru_logits, lru_probabilities, targets, strict=True
    ):
        prior = reference_by_id[row["id"]]
        reference_mismatches += int(
            row["options"][int(probabilities.argmax())] != prior["prediction"]
        )
        reference_max_probability_delta = max(
            reference_max_probability_delta,
            max(
                abs(float(current) - float(previous))
                for current, previous in zip(
                    probabilities.tolist(), prior["probabilities"], strict=True
                )
            ),
        )
        predictions.append(
            {
                "id": row["id"],
                "scene_index": int(row["scene_index"]),
                "media_sha256": row["media_sha256"],
                "question_type": row["question_type"],
                "target": row["options"][target_index],
                "prediction": row["options"][int(probabilities.argmax())],
                "logits": logits.tolist(),
                "probabilities": probabilities.tolist(),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "validation-predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": "complete_cpu_video_lru_scorer_parity",
        "claim_scope": (
            "read-only cache/inference integration check on the same eight already-used "
            "CLEVRER development-validation scenes; not an independent evaluation"
        ),
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "sample_manifests_sha256": sample_hashes,
            "raw_validation_questions_sha256": sha256_file(args.raw_validation_questions),
            "typed_cache_manifest_sha256": sha256_file(args.typed_cache_manifest),
            "readout_sha256": sha256_file(args.readout_dir / "best-readout.pt"),
            "text_model_weights_sha256": sha256_file(text_weights),
            "reference_predictions_sha256": sha256_file(args.reference_predictions),
            "validation_id_order_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in rows).encode("utf-8")
            ).hexdigest(),
        },
        "evaluation": {
            "question_count": len(rows),
            "unique_video_observations": len(by_observation),
            "unique_scenes": len({int(row["scene_index"]) for row in rows}),
            "training_performed": False,
            "sealed_audit_loaded": False,
            "device": "cpu",
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
        },
        "lru": {
            "capacity_bytes": memory.max_bytes,
            "unique_payload_bytes": sum(
                len(item.numpy().tobytes()) for item in direct_by_observation.values()
            ),
            "query_count": len(rows),
            "hits": memory.hit_count,
            "misses": memory.miss_count,
            "persistent_cache_reads": persistent_reads,
            "evictions": memory.eviction_count,
            "source_counts_by_query": dict(source_counts),
            "resident_entries_at_end": memory.entry_count,
            "resident_payload_bytes_at_end": memory.bytes_used,
        },
        "exact_parity": {
            "direct_vs_lru_logits_equal_bitwise": exact_logits,
            "direct_vs_lru_probabilities_equal_bitwise": exact_probabilities,
            "direct_vs_lru_prediction_mismatches": sum(
                int(left.argmax().item() != right.argmax().item())
                for left, right in zip(direct_probabilities, lru_probabilities, strict=True)
            ),
        },
        "reference_comparison": {
            "source": "previous CUDA multi-query artifact; backend differences are expected",
            "prediction_mismatches": reference_mismatches,
            "max_probability_absolute_delta": reference_max_probability_delta,
            "lru_validation_metrics": _metrics(lru_logits, targets),
        },
        "timing_seconds": {
            "text_encoder_load": text_load_seconds,
            "text_feature_extraction_cpu": text_encoding_seconds,
            "readout_load": readout_load_seconds,
            "direct_scorer_cpu": direct_scoring_seconds,
            "lru_feature_retrieval_cpu": lru_retrieval_seconds,
            "lru_scorer_cpu": lru_scoring_seconds,
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": sha256_file(predictions_path),
            "bytes": predictions_path.stat().st_size,
        },
        "limitations": [
            "The real frozen scorer and observation LRU are exercised; this is not a packaged "
            "end-user runtime.",
            "The 88 decisions come from eight development scenes already used for checkpoint "
            "selection.",
            "Payload limits exclude Python object/container overhead and "
            "model/text feature memory.",
            "CPU timings are local diagnostics and are not comparable to prior CUDA timings.",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "lru": report["lru"],
                "exact_parity": report["exact_parity"],
                "reference_comparison": report["reference_comparison"],
                "timing_seconds": report["timing_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
