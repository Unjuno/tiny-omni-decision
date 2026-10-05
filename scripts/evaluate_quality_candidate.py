from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

from tiny_omni_decision.corpus import macro_metrics
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.schema import BaseModelManifest
from tiny_omni_decision.trainer import (
    _evaluate,
    _load_pretrained_base,
    _read_examples,
    resolve_media_root,
)
from tiny_omni_decision.training import (
    decision_training_config,
    processor_inputs_for_example,
)

ROOT = Path.cwd().resolve()
FRESH_VALIDATION_ROOT = (ROOT / "data/processed/teacher-quality-next").resolve()
OUTPUT_ROOT = (ROOT / "artifacts/teacher-quality-next/evaluations").resolve()
MODEL_MANIFEST = ROOT / "manifests/base-model.example.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_video_decode_cache(
    processor: object, *, video_num_frames: int, capacity: int = 4
) -> None:
    """Reuse exact sampled frames for adjacent questions in one fixed-config evaluation."""
    if capacity < 1:
        raise ValueError("video decode cache capacity must be positive")
    if video_num_frames < 1:
        raise ValueError("video_num_frames must be positive")
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    original_fetch = video_processor.fetch_videos
    cache: OrderedDict[tuple[str, int], object] = OrderedDict()

    def cached_fetch(video: object, sample_indices_fn: object = None) -> object:
        if not isinstance(video, str):
            return original_fetch(video, sample_indices_fn=sample_indices_fn)
        key = (video, video_num_frames)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        result = original_fetch(video, sample_indices_fn=sample_indices_fn)
        cache[key] = result
        if len(cache) > capacity:
            cache.popitem(last=False)
        return result

    video_processor.fetch_videos = cached_fetch


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    adapter = args.adapter.resolve()
    config_path = args.config.resolve()
    if not corpus.is_relative_to(FRESH_VALIDATION_ROOT):
        raise ValueError("only new teacher-quality-next development corpora may be evaluated")
    if corpus.name != "validation.jsonl":
        raise ValueError("evaluation input must be a validation.jsonl file")
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError("evaluation output must remain under teacher-quality-next/evaluations")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter config not found under {adapter}")

    examples = [
        example
        for example in _read_examples(corpus)
        if example.modality in set(args.modalities)
    ]
    if not examples:
        raise ValueError("the requested modalities have no examples in this validation corpus")
    if any(example.split != "validation" for example in examples):
        raise ValueError("evaluation corpus contains a non-validation example")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("validation example IDs are not unique")

    config = decision_training_config(load_structured_file(config_path))
    configured_max_sequence_length = config.max_sequence_length
    model_manifest = BaseModelManifest.model_validate(load_structured_file(MODEL_MANIFEST))
    processor = AutoProcessor.from_pretrained(
        model_manifest.processor_repo_id or model_manifest.repo_id,
        revision=model_manifest.processor_revision,
        local_files_only=True,
    )
    install_video_decode_cache(processor, video_num_frames=config.video_num_frames)

    data_root = resolve_media_root(corpus, ROOT / "data")
    sequence_lengths: list[int] = []
    for index, example in enumerate(examples, start=1):
        inputs, _, _, _ = processor_inputs_for_example(
            processor,
            example,
            data_root=data_root,
            video_num_frames=config.video_num_frames,
        )
        sequence_lengths.append(int(inputs["input_ids"].shape[-1]))
        if index % 500 == 0 or index == len(examples):
            print(f"Measured processor length for {index}/{len(examples)} examples", flush=True)
    config = config.model_copy(
        update={"max_sequence_length": max(configured_max_sequence_length, max(sequence_lengths))}
    )

    if not torch.cuda.is_available():
        raise RuntimeError("candidate evaluation requires the local CUDA GPU")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    base = _load_pretrained_base(
        AutoModelForMultimodalLM.from_pretrained,
        model_manifest.repo_id,
        revision=model_manifest.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    print(f"Starting evaluation of {len(examples)} examples.", flush=True)

    def report_progress(completed: int, total: int) -> None:
        if completed % 100 == 0 or completed == total:
            print(f"Evaluated {completed}/{total} examples.", flush=True)

    metrics, predictions = _evaluate(
        model,
        processor,
        examples,
        data_root=data_root,
        config=config,
        progress_callback=report_progress,
    )
    evaluation_seconds = time.monotonic() - started
    metrics["macro"] = macro_metrics(metrics)
    metrics.update(
        {
            "candidate_id": args.candidate_id,
            "checkpoint_role": "read-only candidate evaluation on fresh development validation",
            "examples": len(examples),
            "modalities": sorted(set(args.modalities)),
            "validation_sha256": sha256_file(corpus),
            "sample_id_order_sha256": hashlib.sha256(
                "".join(f"{item.id}\n" for item in examples).encode("utf-8")
            ).hexdigest(),
            "config_sha256": sha256_file(config_path),
            "configured_max_sequence_length": configured_max_sequence_length,
            "effective_max_sequence_length": config.max_sequence_length,
            "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "base_revision": model_manifest.revision,
            "processor_revision": model_manifest.processor_revision,
            "evaluation_seconds": evaluation_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen candidate only on a new teacher-quality development corpus."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--modalities", nargs="+", choices=("text", "image", "audio", "video"), required=True
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
