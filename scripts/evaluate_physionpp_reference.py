from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.corpus import macro_metrics  # noqa: E402
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import BaseModelManifest  # noqa: E402
from tiny_omni_decision.trainer import (  # noqa: E402
    _evaluate,
    _load_pretrained_base,
    _read_examples,
    resolve_media_root,
)  # noqa: E402
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    processor_inputs_for_example,
)
from tiny_omni_decision.video_cache import install_video_decode_cache  # noqa: E402

CORPUS_ROOT = Path("C:/CodexArtifacts/tqpp/fullclip-v1").resolve()
OUTPUT_ROOT = Path("C:/CodexArtifacts/tqpp/evaluations").resolve()
MODEL_MANIFEST = ROOT / "manifests/base-model.example.yaml"
E_LONG_CONFIG = ROOT / "configs/decision/teacher_v2_candidate_e_long_budget.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    corpus = args.corpus.resolve()
    adapter = args.adapter.resolve()
    output = args.output.resolve()
    if corpus != (CORPUS_ROOT / "validation.jsonl").resolve():
        raise ValueError("only the frozen Physion++ fullclip-v1 validation may be evaluated")
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"evaluation output must remain under {OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter config not found under {adapter}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter weights not found under {adapter}")

    examples = _read_examples(corpus)
    if not examples or any(example.modality != "video" for example in examples):
        raise ValueError("Physion++ validation must contain video examples only")
    if any(example.split != "validation" for example in examples):
        raise ValueError("evaluation corpus contains a non-validation example")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("validation example IDs are not unique")

    config = decision_training_config(load_structured_file(E_LONG_CONFIG))
    if config.video_num_frames != 8:
        raise ValueError("shared reference evaluation requires the fixed 8-frame policy")
    model_manifest = BaseModelManifest.model_validate(load_structured_file(MODEL_MANIFEST))
    processor = AutoProcessor.from_pretrained(
        model_manifest.processor_repo_id or model_manifest.repo_id,
        revision=model_manifest.processor_revision,
        local_files_only=True,
    )
    install_video_decode_cache(processor, video_num_frames=config.video_num_frames)
    data_root = resolve_media_root(corpus, CORPUS_ROOT)
    for example in (examples[0], examples[-1]):
        inputs, _, _, _ = processor_inputs_for_example(
            processor,
            example,
            data_root=data_root,
            video_num_frames=config.video_num_frames,
        )
        observed_length = int(inputs["input_ids"].shape[-1])
        if observed_length > config.max_sequence_length:
            config = config.model_copy(update={"max_sequence_length": observed_length})

    if not torch.cuda.is_available():
        raise RuntimeError("reference evaluation requires the existing local CUDA GPU")
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
    print(
        f"Evaluating {args.candidate_id} on {len(examples)} Physion++ validation clips.",
        flush=True,
    )

    def progress(completed: int, total: int) -> None:
        if completed % 10 == 0 or completed == total:
            print(f"{args.candidate_id}: {completed}/{total}", flush=True)

    metrics, predictions = _evaluate(
        model,
        processor,
        examples,
        data_root=data_root,
        config=config,
        progress_callback=progress,
    )
    evaluation_seconds = time.monotonic() - started
    metrics["macro"] = macro_metrics(metrics)
    metrics.update(
        {
            "candidate_id": args.candidate_id,
            "checkpoint_role": (
                "frozen reference evaluated once on a development-only Physion++ split"
            ),
            "examples": len(examples),
            "validation_sha256": sha256_file(corpus),
            "sample_id_order_sha256": hashlib.sha256(
                "".join(f"{item.id}\n" for item in examples).encode("utf-8")
            ).hexdigest(),
            "evaluation_config_sha256": sha256_file(E_LONG_CONFIG),
            "video_num_frames": config.video_num_frames,
            "max_sequence_length": config.max_sequence_length,
            "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "base_revision": model_manifest.revision,
            "processor_revision": model_manifest.processor_revision,
            "evaluation_seconds": evaluation_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "audit_status": "development validation; not a sealed or final audit",
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
        description=(
            "Compare frozen Teacher adapters on the one-time Physion++ development validation."
        )
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
