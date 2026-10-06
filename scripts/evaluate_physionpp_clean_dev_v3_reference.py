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
from tiny_omni_decision.schema import BaseModelManifest, DecisionExample  # noqa: E402
from tiny_omni_decision.trainer import (  # noqa: E402
    _evaluate,
    _load_pretrained_base,
    resolve_media_root,
)
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    deterministic_validation_subset,
)
from tiny_omni_decision.video_cache import install_video_decode_cache  # noqa: E402

CORPUS_ROOT = Path("C:/CodexArtifacts/tqpp/physionpp-clean-dev-v3-generation-02/corpus").resolve()
OUTPUT_ROOT = Path(
    "C:/CodexArtifacts/tqpp/physionpp-clean-dev-v3-generation-02/references"
).resolve()
MEDIA_ROOT = Path(
    "C:/CodexArtifacts/tqpp/physionpp-clean-dev-v3-generation-02/media-root"
).resolve()
CONFIG = ROOT / "configs/decision/teacher_v2_physionpp_clean_dev_v3.yaml"
BASE_MANIFEST = ROOT / "manifests/base-model.example.yaml"
E_LONG_ADAPTER = Path(
    "C:/Users/junny/AppData/Local/CodexArtifacts/tiny-omni-decision-teacher-v2/"
    "clean-dev-v2-seed17-2048-resume-step1408-exact/best"
).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_ids(examples: list[DecisionExample]) -> str:
    return hashlib.sha256("".join(f"{item.id}\n" for item in examples).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the frozen E-long checkpoint on the clean-dev-v3 selector."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "e-long-selected-step1536",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"reference output must remain under {OUTPUT_ROOT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference result: {output}")
    if not E_LONG_ADAPTER.joinpath("adapter_config.json").is_file():
        raise FileNotFoundError(f"E-long frozen adapter is missing: {E_LONG_ADAPTER}")

    corpus_manifest = json.loads((CORPUS_ROOT / "corpus-manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads(
        (CORPUS_ROOT / "run-preflight-hardlink-v1.json").read_text(encoding="utf-8")
    )
    if corpus_manifest.get("sealed_audit_loaded") or preflight.get("sealed_audit_loaded"):
        raise ValueError("sealed audit data must not enter this development evaluation")
    train_path = CORPUS_ROOT / "train.jsonl"
    validation_path = CORPUS_ROOT / "validation.jsonl"
    if sha256_file(train_path) != corpus_manifest.get("train_sha256"):
        raise ValueError("clean-dev-v3 training corpus changed after preflight")
    if sha256_file(validation_path) != corpus_manifest.get("validation_sha256"):
        raise ValueError("clean-dev-v3 validation corpus changed after preflight")

    with validation_path.open(encoding="utf-8") as handle:
        validation = [DecisionExample.model_validate_json(line) for line in handle if line.strip()]
    config = decision_training_config(load_structured_file(CONFIG))
    examples = deterministic_validation_subset(
        validation,
        seed=config.seed,
        limit=config.selection_eval_examples,
        video_task_weights=config.video_task_weights or None,
    )
    order_hash = hash_ids(examples)
    if order_hash != preflight.get("validation_selector_id_order_sha256"):
        raise ValueError("validation selector differs from the immutable sampler preflight")
    if any(item.split != "validation" for item in examples):
        raise ValueError("selector contains non-validation examples")

    model_manifest = BaseModelManifest.model_validate(load_structured_file(BASE_MANIFEST))
    processor = AutoProcessor.from_pretrained(
        model_manifest.processor_repo_id or model_manifest.repo_id,
        revision=model_manifest.processor_revision,
        local_files_only=True,
    )
    install_video_decode_cache(processor, video_num_frames=config.video_num_frames)
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
    model = PeftModel.from_pretrained(base, E_LONG_ADAPTER, is_trainable=False).eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def progress(completed: int, total: int) -> None:
        if completed % 100 == 0 or completed == total:
            print(f"E-long matched selector: {completed}/{total}", flush=True)

    metrics, predictions = _evaluate(
        model,
        processor,
        examples,
        data_root=resolve_media_root(validation_path, MEDIA_ROOT),
        config=config,
        progress_callback=progress,
    )
    metrics["macro"] = macro_metrics(metrics)
    metrics.update(
        {
            "candidate_id": "teacher-v2-e-long-selected-step1536-on-clean-dev-v3",
            "checkpoint_role": "read-only matched-selector reference",
            "checkpoint_step": 1536,
            "adapter_sha256": sha256_file(E_LONG_ADAPTER / "adapter_model.safetensors"),
            "validation_corpus_sha256": sha256_file(validation_path),
            "validation_selector_id_order_sha256": order_hash,
            "examples": len(examples),
            "video_num_frames": config.video_num_frames,
            "base_revision": model_manifest.revision,
            "processor_revision": model_manifest.processor_revision,
            "evaluation_seconds": time.monotonic() - started,
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


if __name__ == "__main__":
    main()
