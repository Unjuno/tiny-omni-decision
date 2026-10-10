"""Verify text cache and Decision readout reload against saved validation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.train_frozen_text_observation_probe import (
    _batch_logits,
    _load_examples,
    _metrics,
    _tokenizer_hash,
)
from tiny_omni_decision.cache import ObservationFeatureCache
from tiny_omni_decision.dataset import audit_manifest, sha256_file
from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.observation_identity import text_observation_feature_key
from tiny_omni_decision.schema import DatasetManifest


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _args()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "run-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "complete_text_observation_cache_decision_probe":
        raise ValueError("run report does not describe a completed text observation probe")
    encoder_report = report["encoder"]
    dataset_report = report["dataset"]
    validation_path = args.validation.resolve()
    model_path = args.model.resolve()
    if sha256_file(validation_path) != dataset_report["validation_file_sha256"]:
        raise ValueError("validation file does not match the completed probe")
    if sha256_file(model_path / "model.safetensors") != encoder_report["weights_sha256"]:
        raise ValueError("MiniLM weight hash differs from the completed probe")
    if _tokenizer_hash(model_path) != encoder_report["tokenizer_files_sha256"]:
        raise ValueError("tokenizer files differ from the completed probe")

    manifest = DatasetManifest.model_validate(
        __import__("yaml").safe_load(args.validation_manifest.read_text(encoding="utf-8"))
    )
    if manifest.usage != "evaluation" or audit_manifest(manifest)["project_policy"] != "ALLOW":
        raise ValueError("expected a rights-approved evaluation manifest")
    if (manifest.dataset_id, manifest.revision) != (
        dataset_report["id"],
        dataset_report["revision"],
    ):
        raise ValueError("validation manifest differs from the completed probe")
    examples = _load_examples(validation_path, manifest)
    if len(examples) != dataset_report["validation_examples"]:
        raise ValueError("validation example count differs from the completed probe")

    prediction_path = run_dir / "validation-predictions.jsonl"
    saved_predictions = [
        json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    ids = [example.id for example in examples]
    if [row["id"] for row in saved_predictions] != ids:
        raise ValueError("saved validation predictions use a different ID order")
    if sha256_file(run_dir / "best-readout.pt") != report["artifacts"]["best_readout_sha256"]:
        raise ValueError("best readout hash differs from the completed run report")
    if sha256_file(prediction_path) != report["artifacts"]["validation_predictions_sha256"]:
        raise ValueError("validation prediction hash differs from the completed run report")

    key_ids: dict[tuple[str, str], str] = {}
    keys_by_id = OrderedDict()
    encoder_revision = encoder_report["revision"]
    tokenizer_sha256 = encoder_report["tokenizer_files_sha256"]
    preprocessing_sha256 = encoder_report["preprocessing_sha256"]
    for example in examples:
        parts = (
            ("state", f"State: {example.state}"),
            ("question", f"Question: {example.question}"),
            *(("candidate", f"Candidate answer: {option}") for option in example.options),
        )
        for role, text in parts:
            key = text_observation_feature_key(
                source_id=dataset_report["id"],
                source_revision=dataset_report["revision"],
                observation_text=text,
                encoder_id=encoder_report["id"],
                encoder_revision=encoder_revision,
                encoder_weights_sha256=encoder_report["weights_sha256"],
                tokenizer_sha256=tokenizer_sha256,
                preprocessing_sha256=preprocessing_sha256,
                feature_role=role,
                hidden_size=encoder_report["hidden_size"],
            )
            key_ids[(role, text)] = key.cache_id
            keys_by_id.setdefault(key.cache_id, key)

    cache = ObservationFeatureCache(run_dir / "feature-cache")
    embeddings: dict[str, torch.Tensor] = {}
    cache_payload_hashes: dict[str, str] = {}
    for cache_id, key in keys_by_id.items():
        entry = cache.get(key)
        feature = torch.frombuffer(bytearray(entry.payload), dtype=torch.float32).clone()
        if feature.shape != (encoder_report["hidden_size"],) or not torch.isfinite(feature).all():
            raise ValueError(f"invalid cached validation feature: {cache_id}")
        embeddings[cache_id] = feature
        cache_payload_hashes[cache_id] = entry.payload_sha256

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    scorer = FrozenFeatureCandidateScorer(
        encoder_report["hidden_size"] * 4,
        hidden_size=report["experiment_config"]["hidden_size"],
    ).to(device)
    state = torch.load(run_dir / "best-readout.pt", map_location="cpu", weights_only=True)
    scorer.load_state_dict(state)
    started = time.perf_counter()
    logits, targets = _batch_logits(
        scorer,
        examples,
        embeddings,
        key_ids,
        device,
        report["experiment_config"]["batch_questions"],
    )
    evaluation_seconds = time.perf_counter() - started
    max_probability_delta = 0.0
    exact_classes = True
    exact_options = True
    for example, logits_row, saved in zip(examples, logits, saved_predictions, strict=True):
        probabilities = torch.softmax(logits_row.float(), dim=-1).tolist()
        saved_probabilities = saved["probabilities"]
        max_probability_delta = max(
            max_probability_delta,
            max(
                abs(left - right)
                for left, right in zip(probabilities, saved_probabilities, strict=True)
            ),
        )
        prediction = example.options[int(torch.tensor(probabilities).argmax())]
        exact_classes &= prediction == saved["prediction"]
        exact_options &= saved["options"] == example.options and saved["target"] == example.target
    if not exact_classes or not exact_options:
        raise ValueError("reloaded readout changed saved validation classes, options, or targets")

    verification = {
        "schema_version": 1,
        "status": "passed_save_reload_validation",
        "run_report_sha256": sha256_file(report_path),
        "readout_sha256": sha256_file(run_dir / "best-readout.pt"),
        "validation_predictions_sha256": sha256_file(prediction_path),
        "validation_examples": len(examples),
        "validation_ids_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "unique_validation_cache_entries_reloaded": len(embeddings),
        "validation_cache_payloads_verified": len(cache_payload_hashes),
        "cache_payload_hashes_sha256": hashlib.sha256(
            "\n".join(
                f"{key}:{value}" for key, value in sorted(cache_payload_hashes.items())
            ).encode()
        ).hexdigest(),
        "device": str(device),
        "evaluation_seconds": evaluation_seconds,
        "saved_prediction_classes_exact": exact_classes,
        "saved_options_and_targets_exact": exact_options,
        "max_probability_absolute_delta": max_probability_delta,
        "metrics_after_readout_reload": _metrics(logits, targets),
        "sealed_audit_loaded": False,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite verification output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**verification, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
