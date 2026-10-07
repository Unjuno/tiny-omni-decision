from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option_hash(options: list[str]) -> str:
    payload = json.dumps(options, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen Teacher option logits for train-only student QAT."
    )
    parser.add_argument(
        "--repo-root", type=Path, required=True, help="Teacher repository root; read-only"
    )
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-run-metadata", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tiny_omni_decision.decision import option_logits_from_vocab
    from tiny_omni_decision.io import load_structured_file
    from tiny_omni_decision.schema import BaseModelManifest, DecisionExample
    from tiny_omni_decision.student_training import load_teacher_option_cache
    from tiny_omni_decision.training import processor_inputs_for_example

    if not torch.cuda.is_available():
        raise RuntimeError(
            "frozen Teacher option-cache generation requires the existing local CUDA device"
        )
    if not 0 < args.temperature < float("inf") or args.max_sequence_length < 1:
        raise ValueError("temperature and max_sequence_length must be positive")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < args.minimum_free_vram_gib * 1024**3:
        raise RuntimeError(
            f"only {free_bytes / 1024**3:.2f} GiB free VRAM; "
            f"require {args.minimum_free_vram_gib:.2f} GiB"
        )

    repo = args.repo_root.resolve()
    teacher_manifest_path = args.teacher_manifest.resolve()
    run_metadata_path = args.teacher_run_metadata.resolve()
    base_manifest_path = args.base_model_manifest.resolve()
    train_path = args.train_corpus.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    for path in (teacher_manifest_path, run_metadata_path, base_manifest_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    teacher = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    run = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    base_manifest = BaseModelManifest.model_validate(load_structured_file(base_manifest_path))
    if teacher.get("artifact_role") != "high_precision_reference_before_ternary_compression":
        raise ValueError("Teacher manifest does not identify the frozen high-precision reference")
    if run.get("teacher_id") != teacher.get("teacher_id") or not run.get(
        "checkpoint_reload_verified"
    ):
        raise ValueError("Teacher run metadata is not a verified reload of this Teacher artifact")
    if run.get("best_checkpoint_step") != teacher["artifact"].get("best_checkpoint_step"):
        raise ValueError("Teacher manifest and run metadata select different checkpoints")
    if run.get("base_revision") != base_manifest.revision:
        raise ValueError("Teacher run and base-model manifest revisions differ")
    if teacher.get("base_model", {}).get("revision") != base_manifest.revision:
        raise ValueError("Teacher manifest and base-model manifest revisions differ")
    if teacher.get("base_model", {}).get("repo_id") != base_manifest.repo_id:
        raise ValueError("Teacher manifest and base-model manifest IDs differ")
    if sha256(base_manifest_path) != teacher["base_model"]["model_manifest_sha256"]:
        raise ValueError("base-model manifest hash does not match frozen Teacher manifest")
    if sha256(train_path) != teacher["corpus"]["train_jsonl_sha256"]:
        raise ValueError("training corpus hash does not match frozen Teacher manifest")
    adapter_info = teacher["artifact"]
    adapter_dir = (repo / adapter_info["path"]).resolve()
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if sha256(adapter_weights) != adapter_info["weights_sha256"]:
        raise ValueError("frozen Teacher adapter weights hash mismatch")
    if sha256(adapter_config) != adapter_info["config_sha256"]:
        raise ValueError("frozen Teacher adapter config hash mismatch")
    if sha256(adapter_weights) != run["best_adapter_sha256"]:
        raise ValueError("Teacher run metadata and manifest adapter hashes differ")

    with train_path.open(encoding="utf-8") as stream:
        examples = [DecisionExample.model_validate_json(line) for line in stream if line.strip()]
    if not examples or any(example.split != "train" for example in examples):
        raise ValueError("Teacher cache corpus must contain only train examples")
    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("training corpus contains duplicate sample IDs")
    expected_count = int(teacher["corpus"]["splits"]["train"]["records"])
    if len(examples) != expected_count:
        raise ValueError(f"training corpus has {len(examples)} examples, expected {expected_count}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "cache-run.json"
    partial_path = output_dir / "teacher-options.partial.jsonl"
    final_path = output_dir / "teacher-options.jsonl"
    identity = {
        "teacher_id": teacher["teacher_id"],
        "teacher_revision": base_manifest.revision,
        "teacher_manifest_sha256": sha256(teacher_manifest_path),
        "teacher_run_metadata_sha256": sha256(run_metadata_path),
        "teacher_adapter_sha256": sha256(adapter_weights),
        "base_model_id": base_manifest.repo_id,
        "base_revision": base_manifest.revision,
        "base_manifest_sha256": sha256(base_manifest_path),
        "train_corpus_sha256": sha256(train_path),
        "train_count": len(examples),
        "temperature": args.temperature,
        "max_sequence_length": args.max_sequence_length,
        "device": "cuda",
        "processor_revision": base_manifest.processor_revision,
    }
    if final_path.exists():
        raise FileExistsError(f"completed Teacher cache already exists: {final_path}")
    if run_path.exists():
        metadata = json.loads(run_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise ValueError("existing Teacher cache run metadata does not match this request")
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
    elif partial_path.exists():
        raise ValueError("partial Teacher cache exists without metadata; refusing unsafe resume")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "identity": identity,
            "state": "in_progress",
            "resume_count": 0,
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
    previous = partial_path.read_bytes() if partial_path.exists() else b""
    if previous and not previous.endswith(b"\n"):
        raise ValueError("Teacher cache partial has a truncated final record")
    saved = [json.loads(line) for line in previous.splitlines() if line]
    if len(saved) > len(examples):
        raise ValueError("Teacher cache partial exceeds training corpus length")
    for index, record in enumerate(saved):
        example = examples[index]
        if (
            record.get("sample_id") != example.id
            or record.get("options") != example.options
            or record.get("target") != example.target
            or record.get("target_index") != example.options.index(example.target)
            or record.get("option_order_sha256") != option_hash(example.options)
            or record.get("teacher_id") != teacher["teacher_id"]
            or record.get("teacher_revision") != base_manifest.revision
            or record.get("teacher_temperature") != args.temperature
        ):
            raise ValueError(f"partial cache is not the exact train prefix at row {index + 1}")
    metadata.update(
        {
            "state": "in_progress",
            "partial_records_before_run": len(saved),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "transformers_version": importlib.metadata.version("transformers"),
            "peft_version": importlib.metadata.version("peft"),
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_total_bytes": total_bytes,
            "gpu_free_bytes_at_start": free_bytes,
            "command_line": [sys.executable, *sys.argv],
        }
    )
    atomic_json(run_path, metadata)

    processor = AutoProcessor.from_pretrained(
        base_manifest.processor_repo_id or base_manifest.repo_id,
        revision=base_manifest.processor_revision,
        local_files_only=True,
    )
    base = AutoModelForMultimodalLM.from_pretrained(
        base_manifest.repo_id,
        revision=base_manifest.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen Teacher unexpectedly has trainable parameters")
    model.config.use_cache = False
    started = time.perf_counter()
    mode = "ab" if previous else "wb"
    with partial_path.open(mode) as output:
        for index, example in enumerate(examples[len(saved) :], start=len(saved)):
            inputs, option_ids, position, target_index = processor_inputs_for_example(
                processor, example, data_root=data_root
            )
            if inputs["input_ids"].shape[-1] > args.max_sequence_length:
                raise ValueError(
                    f"{example.id}: sequence length {inputs['input_ids'].shape[-1]} "
                    f"exceeds {args.max_sequence_length}"
                )
            device = model.get_input_embeddings().weight.device
            inputs = {
                key: (value.to(device) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                model_output = model(**inputs, use_cache=False)
                logits = option_logits_from_vocab(
                    model_output.logits[0, position], option_ids
                ).float()
                probabilities = torch.softmax(logits / args.temperature, dim=-1)
            if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
                raise ValueError(f"non-finite Teacher output for {example.id}")
            record = {
                "sample_id": example.id,
                "source": example.source,
                "modality": example.modality,
                "options": list(example.options),
                "target": example.target,
                "target_index": target_index,
                "option_order_sha256": option_hash(example.options),
                "teacher_id": teacher["teacher_id"],
                "teacher_revision": base_manifest.revision,
                "teacher_checkpoint_step": teacher["artifact"]["best_checkpoint_step"],
                "teacher_temperature": args.temperature,
                "teacher_option_token_ids": option_ids,
                "teacher_option_logits": logits.cpu().tolist(),
                "teacher_option_probabilities": probabilities.cpu().tolist(),
            }
            output.write(
                (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
            )
            output.flush()
            os.fsync(output.fileno())
            if (index + 1) % 32 == 0 or index + 1 == len(examples):
                print(
                    f"cached {index + 1}/{len(examples)} train examples: {example.id}", flush=True
                )
    # Re-run the strict cache contract before publishing a complete file.
    load_teacher_option_cache(
        partial_path,
        examples,
        expected_teacher_id=teacher["teacher_id"],
        expected_teacher_revision=base_manifest.revision,
        expected_temperature=args.temperature,
    )
    os.replace(partial_path, final_path)
    metadata.update(
        {
            "state": "complete",
            "ended_at_utc": datetime.now(UTC).isoformat(),
            "last_attempt_duration_seconds": time.perf_counter() - started,
            "record_count": len(examples),
            "cache_sha256": sha256(final_path),
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    atomic_json(run_path, metadata)
    print(f"Teacher option cache complete: {final_path}", flush=True)


if __name__ == "__main__":
    main()
