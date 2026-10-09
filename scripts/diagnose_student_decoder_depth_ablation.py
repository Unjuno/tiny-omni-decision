from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tiny_omni_decision.ternary_diagnostic_utils import group_decoder_layer_targets

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DECODER_LAYER_PATTERN = re.compile(r"^language_model\.layers\.(\d+)\.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256(path)


def _restore_targets(
    model: Any, snapshots: dict[str, Any], target_names: tuple[str, ...]
) -> dict[str, int]:
    import torch

    parameters = dict(model.named_parameters())
    restored_elements = 0
    with torch.no_grad():
        for name in target_names:
            parameter = parameters[name]
            original = snapshots[name]
            if parameter.shape != original.shape or parameter.dtype != original.dtype:
                raise ValueError(f"BF16 restore shape/dtype mismatch: {name}")
            parameter.copy_(original.to(device=parameter.device))
            if not torch.equal(parameter.detach().cpu(), original):
                raise ValueError(f"BF16 restore was not bitwise exact: {name}")
            restored_elements += original.numel()
    return {"restored_tensors": len(target_names), "restored_elements": restored_elements}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-training early/middle/late shared-decoder ternary sensitivity ablation."
    )
    for name in (
        "repo-root",
        "model-path",
        "model-manifest",
        "overlay-dir",
        "baseline-results",
        "qat-run-dir",
        "shadow-checkpoint",
        "validation-snapshot",
        "selection-manifest",
        "source-validation",
        "teacher-validation-predictions",
        "data-root",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-overlay-manifest-sha256", required=True)
    parser.add_argument("--expected-baseline-results-sha256", required=True)
    parser.add_argument("--expected-qat-metadata-sha256", required=True)
    parser.add_argument("--expected-shadow-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse diagnostic output: {output_dir}")
    if args.temperature <= 0 or args.threads < 1:
        raise ValueError("temperature and thread count must be positive")

    import torch
    from safetensors import safe_open
    from transformers import AutoModel, AutoProcessor

    from scripts.diagnose_student_ternary_components import _verify_overlay_matches_shadow
    from tiny_omni_decision.io import load_structured_file
    from tiny_omni_decision.schema import BaseModelManifest
    from tiny_omni_decision.student_eval import (
        evaluate_student_examples,
        load_fixed_validation_snapshot,
        validate_local_media_paths,
    )
    from tiny_omni_decision.student_evaluate import _verify_local_model_files
    from tiny_omni_decision.ternary import (
        load_packed_ternary_overlay,
        select_ternary_parameter_names,
    )

    torch.set_num_threads(args.threads)
    repo_root = args.repo_root.resolve()
    model_path = args.model_path.resolve()
    overlay_dir = args.overlay_dir.resolve()
    qat_run_dir = args.qat_run_dir.resolve()
    shadow_path = args.shadow_checkpoint.resolve()
    baseline_path = args.baseline_results.resolve()
    baseline_predictions_dir = baseline_path.parent
    baseline_hash = sha256(baseline_path)
    if baseline_hash != args.expected_baseline_results_sha256:
        raise ValueError("component-ablation baseline results hash mismatch")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "complete_validation_diagnostic_not_model_selection"
        or baseline.get("qat_best_step") != 512
        or baseline.get("sealed_audit_loaded") is not False
    ):
        raise ValueError("baseline is not the expected completed, non-audit step-512 diagnosis")

    metadata_path = qat_run_dir / "run-metadata.json"
    if sha256(metadata_path) != args.expected_qat_metadata_sha256:
        raise ValueError("QAT run metadata hash mismatch")
    if sha256(shadow_path) != args.expected_shadow_sha256:
        raise ValueError("QAT step-512 BF16 shadow hash mismatch")
    overlay_manifest_path = overlay_dir / "manifest.json"
    if sha256(overlay_manifest_path) != args.expected_overlay_manifest_sha256:
        raise ValueError("packed overlay manifest hash mismatch")
    qat_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    overlay_tensor_path = overlay_dir / overlay_manifest["tensor_file"]
    if sha256(overlay_tensor_path) != overlay_manifest["tensor_file_sha256"]:
        raise ValueError("packed overlay tensor hash mismatch")
    model_manifest = BaseModelManifest.model_validate(load_structured_file(args.model_manifest))
    if (
        qat_metadata.get("state") != "complete"
        or qat_metadata.get("best_validation_step") != 512
        or qat_metadata.get("best_shadow_sha256") != args.expected_shadow_sha256
        or overlay_manifest.get("base_model_id") != model_manifest.repo_id
        or overlay_manifest.get("base_revision") != model_manifest.revision
    ):
        raise ValueError("selected shadow, overlay, QAT run, and base manifest do not align")
    if sha256(args.validation_snapshot.resolve()) != args.expected_validation_sha256:
        raise ValueError("fixed validation snapshot hash mismatch")
    if sha256(args.selection_manifest.resolve()) != args.expected_selection_manifest_sha256:
        raise ValueError("validation selection manifest hash mismatch")

    examples, _ = load_fixed_validation_snapshot(
        args.validation_snapshot,
        args.selection_manifest,
        source_validation_path=args.source_validation,
        teacher_predictions_path=args.teacher_validation_predictions,
    )
    if len(examples) != 256 or len(examples) != int(qat_metadata["validation_ids"]):
        raise ValueError("expected the exact 256-example validation snapshot")
    ordered_ids_sha = hashlib.sha256(
        "\n".join(example.id for example in examples).encode("utf-8")
    ).hexdigest()
    if (
        sha256(args.validation_snapshot.resolve()) != baseline["validation_snapshot_sha256"]
        or baseline["validation_ordered_ids_sha256"] != ordered_ids_sha
        or baseline["validation_selection_manifest_sha256"]
        != args.expected_selection_manifest_sha256
        or qat_metadata["validation_snapshot_sha256"] != args.expected_validation_sha256
    ):
        raise ValueError("validation snapshot or ID order differs from QAT/baseline")
    media_counts = validate_local_media_paths(examples, args.data_root)

    verified_model_files = _verify_local_model_files(model_path, model_manifest)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    targets = select_ternary_parameter_names(model)
    target_records = tuple(sorted(record["name"] for record in overlay_manifest["targets"]))
    if targets != target_records:
        raise ValueError("loaded target inventory differs from the selected overlay")
    if tuple(example.id for example in examples) != tuple(
        row["sample_id"]
        for row in (
            json.loads(line)
            for line in (baseline_predictions_dir / "all_ternary_targets-predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ):
        raise ValueError("validation order differs from the frozen component-ablation baseline")

    decoder_groups = group_decoder_layer_targets(targets)
    if set(decoder_groups) != {"early", "middle", "late"}:
        raise ValueError("decoder grouping failed to produce three depth groups")
    quantization_audit = _verify_overlay_matches_shadow(
        shadow_path=shadow_path,
        overlay_dir=overlay_dir,
        target_names=targets,
        group_size=int(overlay_manifest["group_size"]),
        threshold_factor=float(overlay_manifest["threshold_factor"]),
    )

    parameters = dict(model.named_parameters())
    shadow_snapshots = {}
    with safe_open(shadow_path, framework="pt", device="cpu") as shadow_file:
        if set(shadow_file.keys()) != set(targets):
            raise ValueError("BF16 shadow keys differ from the loaded target inventory")
        for name in targets:
            value = shadow_file.get_tensor(name)
            if value.dtype != torch.bfloat16 or value.shape != parameters[name].shape:
                raise ValueError(f"BF16 shadow shape/dtype mismatch: {name}")
            parameters[name].data.copy_(value.to(device))
            shadow_snapshots[name] = value

    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(UTC).isoformat()
    variant_results = {}
    for group_name, group_targets in decoder_groups.items():
        print(f"validation arm start: decoder_{group_name}_bf16", flush=True)
        load_info = load_packed_ternary_overlay(
            model,
            overlay_dir,
            expected_base_model_id=model_manifest.repo_id,
            expected_base_revision=model_manifest.revision,
        )
        restore_info = _restore_targets(model, shadow_snapshots, group_targets)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        metrics, predictions = evaluate_student_examples(
            model,
            processor,
            examples,
            data_root=args.data_root.resolve(),
            temperature=args.temperature,
            ece_bins=15,
            cache_video_frames=True,
        )
        if [row["sample_id"] for row in predictions] != [example.id for example in examples]:
            raise ValueError(f"prediction order differs from frozen validation IDs: {group_name}")
        prediction_sha = _write_predictions(
            output_dir / f"decoder_{group_name}_bf16-predictions.jsonl", predictions
        )
        variant_results[group_name] = {
            "restored_layer_ids": sorted(
                {int(DECODER_LAYER_PATTERN.match(name).group(1)) for name in group_targets}
            ),
            "restored_target_tensors": len(group_targets),
            "restored_elements": restore_info["restored_elements"],
            "overlay_tensor_sha256": load_info["tensor_file_sha256"],
            "metrics": metrics,
            "prediction_sha256": prediction_sha,
            "prediction_count": len(predictions),
            "evaluation_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if args.device == "cuda"
            else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if args.device == "cuda"
            else None,
        }
        print(f"validation arm complete: decoder_{group_name}_bf16", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    layer_ids = sorted(
        {
            int(DECODER_LAYER_PATTERN.match(name).group(1))
            for names in decoder_groups.values()
            for name in names
        }
    )
    result = {
        "schema_version": 1,
        "experiment_id": "embeddinggemma2-ternary-decoder-depth-ablation-v0",
        "status": "complete_validation_diagnostic_not_model_selection",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "base_model_id": model_manifest.repo_id,
        "base_revision": model_manifest.revision,
        "selected_qat_step": 512,
        "qat_metadata_sha256": sha256(metadata_path),
        "qat_shadow_sha256": sha256(shadow_path),
        "overlay_manifest_sha256": sha256(overlay_manifest_path),
        "overlay_tensor_sha256": sha256(overlay_tensor_path),
        "baseline_results_sha256": baseline_hash,
        "baseline_diagnostic_results": str(baseline_path),
        "validation_snapshot_sha256": sha256(args.validation_snapshot.resolve()),
        "validation_selection_manifest_sha256": sha256(args.selection_manifest.resolve()),
        "validation_ordered_ids_sha256": hashlib.sha256(
            "\n".join(example.id for example in examples).encode("utf-8")
        ).hexdigest(),
        "validation_examples": len(examples),
        "temperature": args.temperature,
        "ece_bins": 15,
        "device": torch.cuda.get_device_name(device) if args.device == "cuda" else "cpu",
        "verified_model_files": verified_model_files,
        "source_validation_media_counts": media_counts,
        "target_inventory": quantization_audit,
        "decoder_layer_ids": layer_ids,
        "decoder_group_policy": "equal contiguous thirds: early, middle, late",
        "decoder_groups": {
            key: {
                "layer_ids": sorted(
                    {int(DECODER_LAYER_PATTERN.match(name).group(1)) for name in names}
                ),
                "target_tensors": len(names),
                "elements": sum(shadow_snapshots[name].numel() for name in names),
            }
            for key, names in decoder_groups.items()
        },
        "selection_rule": {
            "primary": "macro modality NLL and macro modality Brier",
            "clear_improvement": (
                "both improve by at least 1% relative to all-ternary baseline, "
                "and no modality accuracy drops by more than 1/64"
            ),
        },
        "sealed_audit_loaded": False,
        "recovery_lora_loaded": False,
        "training_or_optimizer_updates": 0,
        "variants": variant_results,
        "note": "Small fixed-validation sensitivity scan; no precision policy or model selected.",
    }
    _write_json(output_dir / "diagnostic-results.json", result)
    print(json.dumps({"status": result["status"], "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
