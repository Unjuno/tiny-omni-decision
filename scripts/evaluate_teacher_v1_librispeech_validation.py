from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tiny_omni_decision.corpus import macro_metrics
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import BaseModelManifest
from tiny_omni_decision.trainer import (
    _eligible,
    _evaluate,
    _load_pretrained_base,
    _read_examples,
    resolve_media_root,
)
from tiny_omni_decision.training import (
    decision_training_config,
    processor_inputs_for_example,
)

ROOT = Path.cwd()
VALIDATION = ROOT / "data/processed/teacher-quality-next/librispeech/validation.jsonl"
CONFIG = ROOT / "artifacts/tiny-omni-decision-teacher-v1/selected/training-config.yaml"
ADAPTER = ROOT / "artifacts/tiny-omni-decision-teacher-v1/selected"
MODEL_MANIFEST = ROOT / "manifests/base-model.example.yaml"
OUTPUT = ROOT / "artifacts/teacher-quality-next/v1-librispeech-validation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Teacher v1 baseline evaluation requires the available local CUDA GPU")
    config = decision_training_config(load_structured_file(CONFIG))
    configured_max_sequence_length = config.max_sequence_length
    examples, _ = _eligible(_read_examples(VALIDATION), {"audio"})
    if len(examples) != 2703 or len({item.id for item in examples}) != len(examples):
        raise ValueError("LibriSpeech validation corpus is not the expected unique 2,703 examples")

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    model_manifest = BaseModelManifest.model_validate(
        load_structured_file(MODEL_MANIFEST)
    )
    processor = AutoProcessor.from_pretrained(
        model_manifest.processor_repo_id or model_manifest.repo_id,
        revision=model_manifest.processor_revision,
        local_files_only=True,
    )
    sequence_lengths = []
    for index, example in enumerate(examples, start=1):
        inputs, _, _, _ = processor_inputs_for_example(
            processor,
            example,
            data_root=resolve_media_root(VALIDATION, ROOT / "data"),
            video_num_frames=config.video_num_frames,
        )
        sequence_lengths.append(int(inputs["input_ids"].shape[-1]))
        if index % 500 == 0 or index == len(examples):
            print(f"Measured processor length for {index}/{len(examples)} examples", flush=True)
    effective_max_sequence_length = max(sequence_lengths)
    config = config.model_copy(
        update={"max_sequence_length": effective_max_sequence_length}
    )
    base = _load_pretrained_base(
        AutoModelForMultimodalLM.from_pretrained,
        model_manifest.repo_id,
        revision=model_manifest.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False).eval()

    started = time.monotonic()
    metrics, predictions = _evaluate(
        model,
        processor,
        examples,
        data_root=resolve_media_root(VALIDATION, ROOT / "data"),
        config=config,
    )
    evaluation_seconds = time.monotonic() - started
    metrics["macro"] = macro_metrics(metrics)
    metrics["evaluation_seconds"] = evaluation_seconds
    metrics["examples"] = len(examples)
    metrics["configured_v1_max_sequence_length"] = configured_max_sequence_length
    metrics["effective_evaluation_max_sequence_length"] = effective_max_sequence_length
    metrics["examples_over_configured_v1_context"] = sum(
        length > configured_max_sequence_length for length in sequence_lengths
    )
    metrics["validation_corpus_sha256"] = sha256_file(VALIDATION)
    metrics["sample_id_order_sha256"] = hashlib.sha256(
        "".join(f"{example.id}\n" for example in examples).encode("utf-8")
    ).hexdigest()
    metrics["teacher_v1_adapter_sha256"] = sha256_file(ADAPTER / "adapter_model.safetensors")
    metrics["base_revision"] = model_manifest.revision
    metrics["checkpoint_role"] = "read-only Teacher v1 baseline on development validation"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
