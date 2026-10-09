from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_for_target(name: str) -> str:
    root = name.split(".", maxsplit=1)[0]
    if root in {"audio_tower", "embed_audio"}:
        return "audio_path"
    if root in {"vision_tower", "embed_vision"}:
        return "vision_path"
    if root == "language_model":
        return "shared_decoder"
    raise ValueError(f"unclassified ternary target root: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only validation ablations for a selected packed ternary checkpoint."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--expected-overlay-manifest-sha256", required=True)
    parser.add_argument("--qat-run-dir", type=Path, required=True)
    parser.add_argument("--expected-qat-metadata-sha256", required=True)
    parser.add_argument("--shadow-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-shadow-sha256", required=True)
    parser.add_argument("--validation-snapshot", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--teacher-validation-predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


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


def _verify_overlay_matches_shadow(
    *,
    shadow_path: Path,
    overlay_dir: Path,
    target_names: tuple[str, ...],
    group_size: int,
    threshold_factor: float,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    from tiny_omni_decision.ternary import pack_ternary_codes, quantize_groupwise_ternary

    manifest = json.loads((overlay_dir / "manifest.json").read_text(encoding="utf-8"))
    tensor_path = overlay_dir / manifest["tensor_file"]
    component_counts: dict[str, dict[str, int]] = {}
    trit_counts = Counter()
    checked = 0
    with safe_open(shadow_path, framework="pt", device="cpu") as shadow_file:
        with safe_open(tensor_path, framework="pt", device="cpu") as overlay_file:
            if set(shadow_file.keys()) != set(target_names):
                raise ValueError("selected QAT shadow keys do not match the overlay targets")
            if set(overlay_file.keys()) != {
                key for name in target_names for key in (f"code__{name}", f"scale__{name}")
            }:
                raise ValueError("overlay tensor keys do not match the loaded target inventory")
            for name in target_names:
                original = shadow_file.get_tensor(name)
                if original.dtype != torch.bfloat16:
                    raise ValueError(f"selected shadow tensor is not BF16: {name}")
                codes, scales = quantize_groupwise_ternary(
                    original, group_size=group_size, threshold_factor=threshold_factor
                )
                packed = pack_ternary_codes(codes)
                saved_packed = overlay_file.get_tensor(f"code__{name}")
                saved_scales = overlay_file.get_tensor(f"scale__{name}")
                if not torch.equal(packed, saved_packed):
                    raise ValueError(f"packed ternary codes differ from selected shadow: {name}")
                if not torch.equal(scales, saved_scales):
                    raise ValueError(f"ternary group scales differ from selected shadow: {name}")
                component = component_for_target(name)
                entry = component_counts.setdefault(component, {"tensors": 0, "elements": 0})
                entry["tensors"] += 1
                entry["elements"] += original.numel()
                component_codes = torch.bincount(
                    (codes.to(torch.int64) + 1).reshape(-1), minlength=3
                )
                trit_counts["negative"] += int(component_codes[0])
                trit_counts["zero"] += int(component_codes[1])
                trit_counts["positive"] += int(component_codes[2])
                checked += 1
    return {
        "status": "exact_requantization_match",
        "target_tensors_checked": checked,
        "target_elements_checked": sum(row["elements"] for row in component_counts.values()),
        "components": component_counts,
        "trit_counts": dict(trit_counts),
    }


def _restore_bf16_groups(
    model: Any, snapshots: dict[str, Any], components: set[str]
) -> dict[str, Any]:
    import torch

    from tiny_omni_decision.ternary import select_ternary_parameter_names

    parameters = dict(model.named_parameters())
    names = [
        name
        for name in select_ternary_parameter_names(model)
        if component_for_target(name) in components
    ]
    restored = 0
    restored_elements = 0
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            original = snapshots[name]
            if parameter.shape != original.shape or parameter.dtype != original.dtype:
                raise ValueError(f"BF16 restore shape/dtype mismatch: {name}")
            parameter.copy_(original.to(device=parameter.device))
            if not torch.equal(parameter.detach().cpu(), original):
                raise ValueError(f"BF16 restore was not bitwise exact: {name}")
            restored += 1
            restored_elements += original.numel()
    return {"restored_tensors": restored, "restored_elements": restored_elements}


def _gradient_path_audit(
    model: Any,
    processor: Any,
    examples: list[Any],
    *,
    data_root: Path,
    temperature: float,
    shadow_snapshots: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import parametrize

    from tiny_omni_decision.student import (
        decision_option_text,
        model_sentence_embeddings,
        processor_inputs_for_decision_example,
        supplied_option_logits,
    )
    from tiny_omni_decision.ternary import apply_ternary_qat, select_ternary_parameter_names

    targets = select_ternary_parameter_names(model)
    target_set = set(targets)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in target_set)
    apply_ternary_qat(model)
    model.eval()
    examples_by_modality = {}
    for example in examples:
        examples_by_modality.setdefault(example.modality, example)
    if set(examples_by_modality) != {"text", "image", "audio", "video"}:
        raise ValueError("fixed validation snapshot must contain all four modalities")

    module_map = dict(model.named_modules())
    gradients: dict[str, Any] = {}
    expected_component = {
        "text": "shared_decoder",
        "image": "vision_path",
        "video": "vision_path",
        "audio": "audio_path",
    }
    for modality in ("text", "image", "audio", "video"):
        print(f"gradient audit: {modality}", flush=True)
        example = examples_by_modality[modality]
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            query_inputs = processor_inputs_for_decision_example(
                processor, example, data_root=data_root
            )
            query = model_sentence_embeddings(model, query_inputs)[0]
            option_inputs = processor(
                text=[decision_option_text(option) for option in example.options],
                return_tensors="pt",
            )
            options = model_sentence_embeddings(model, option_inputs)
            logits = supplied_option_logits(query, options, temperature=temperature)
            loss = F.cross_entropy(
                logits.float().unsqueeze(0),
                torch.tensor([example.options.index(example.target)], device=logits.device),
            )
            loss.backward()

        by_component: dict[str, dict[str, float | int]] = {}
        for target in targets:
            component = component_for_target(target)
            module_name, _, leaf = target.rpartition(".")
            original = module_map[module_name].parametrizations[leaf].original
            grad = original.grad
            row = by_component.setdefault(
                component,
                {
                    "target_tensors": 0,
                    "with_gradient": 0,
                    "nonzero_gradient": 0,
                    "gradient_l2_sum": 0.0,
                },
            )
            row["target_tensors"] += 1
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise ValueError(f"non-finite {component} gradient for {modality}")
                row["with_gradient"] += 1
                nonzero = int(torch.count_nonzero(grad))
                row["nonzero_gradient"] += int(nonzero > 0)
                row["gradient_l2_sum"] += float(grad.detach().float().norm().cpu())

        expected = expected_component[modality]
        if by_component.get(expected, {}).get("nonzero_gradient", 0) == 0:
            raise ValueError(f"ternary STE did not deliver a {expected} gradient for {modality}")
        if by_component.get("shared_decoder", {}).get("nonzero_gradient", 0) == 0:
            raise ValueError(
                f"ternary STE did not deliver a shared decoder gradient for {modality}"
            )
        gradients[modality] = {"loss": float(loss.detach().cpu()), "components": by_component}
        model.zero_grad(set_to_none=True)
        del query_inputs, query, option_inputs, options, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name in targets:
        module_name, _, leaf = name.rpartition(".")
        original = module_map[module_name].parametrizations[leaf].original
        if not torch.equal(original.detach().cpu(), shadow_snapshots[name]):
            raise ValueError(f"gradient-only audit modified selected shadow weights: {name}")
    for name in targets:
        module_name, _, leaf = name.rpartition(".")
        parametrize.remove_parametrizations(module_map[module_name], leaf, leave_parametrized=False)
    return {
        "status": "finite_component_gradients_without_optimizer_updates",
        "modalities": gradients,
        "parameter_values_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    model_path = args.model_path.resolve()
    overlay_dir = args.overlay_dir.resolve()
    qat_run = args.qat_run_dir.resolve()
    shadow_path = args.shadow_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse diagnostic output: {output_dir}")
    if not args.temperature > 0 or args.threads < 1:
        raise ValueError("temperature and thread count must be positive")

    import torch
    from safetensors import safe_open
    from transformers import AutoModel, AutoProcessor

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
    model_manifest = BaseModelManifest.model_validate(load_structured_file(args.model_manifest))
    verified_model_files = _verify_local_model_files(model_path, model_manifest)
    qat_metadata_path = qat_run / "run-metadata.json"
    qat_metadata = json.loads(qat_metadata_path.read_text(encoding="utf-8"))
    overlay_manifest_path = overlay_dir / "manifest.json"
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))
    overlay_tensor_path = overlay_dir / overlay_manifest["tensor_file"]
    if sha256(qat_metadata_path) != args.expected_qat_metadata_sha256:
        raise ValueError("QAT run metadata SHA-256 mismatch")
    if sha256(shadow_path) != args.expected_shadow_sha256:
        raise ValueError("selected BF16 shadow SHA-256 mismatch")
    if sha256(overlay_manifest_path) != args.expected_overlay_manifest_sha256:
        raise ValueError("selected overlay manifest SHA-256 mismatch")
    if sha256(overlay_tensor_path) != overlay_manifest["tensor_file_sha256"]:
        raise ValueError("selected overlay tensor SHA-256 mismatch")
    if (
        qat_metadata.get("state") != "complete"
        or int(qat_metadata.get("best_validation_step", -1)) != 512
        or qat_metadata.get("best_shadow_sha256") != args.expected_shadow_sha256
        or overlay_manifest.get("base_model_id") != model_manifest.repo_id
        or overlay_manifest.get("base_revision") != model_manifest.revision
    ):
        raise ValueError("selected QAT run, BF16 shadow, overlay, and pinned base do not align")

    examples, _teacher_rows = load_fixed_validation_snapshot(
        args.validation_snapshot,
        args.selection_manifest,
        source_validation_path=args.source_validation,
        teacher_predictions_path=args.teacher_validation_predictions,
    )
    media_counts = validate_local_media_paths(examples, args.data_root)
    validation_sha256 = sha256(args.validation_snapshot)
    if validation_sha256 != qat_metadata["validation_snapshot_sha256"] or len(examples) != int(
        qat_metadata["validation_ids"]
    ):
        raise ValueError("diagnostic validation snapshot differs from selected QAT run")

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
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model.eval()
    targets = select_ternary_parameter_names(model)
    manifest_target_names = tuple(sorted(record["name"] for record in overlay_manifest["targets"]))
    if targets != manifest_target_names:
        raise ValueError("loaded architecture target inventory differs from selected overlay")
    components = {component_for_target(name) for name in targets}
    if components != {"audio_path", "vision_path", "shared_decoder"}:
        raise ValueError(f"unexpected target component inventory: {sorted(components)}")

    parameter_map = dict(model.named_parameters())
    shadow_snapshots: dict[str, Any] = {}
    with safe_open(shadow_path, framework="pt", device="cpu") as shadow_file:
        if set(shadow_file.keys()) != set(targets):
            raise ValueError("selected BF16 shadow keys differ from target inventory")
        for name in targets:
            value = shadow_file.get_tensor(name)
            if tuple(value.shape) != tuple(parameter_map[name].shape):
                raise ValueError(f"shadow tensor shape mismatch: {name}")
            if value.dtype != torch.bfloat16 or value.dtype != parameter_map[name].dtype:
                raise ValueError(f"shadow tensor dtype mismatch: {name}")
            parameter_map[name].data.copy_(value.to(device))
            shadow_snapshots[name] = value

    quantization_audit = _verify_overlay_matches_shadow(
        shadow_path=shadow_path,
        overlay_dir=overlay_dir,
        target_names=targets,
        group_size=int(overlay_manifest["group_size"]),
        threshold_factor=float(overlay_manifest["threshold_factor"]),
    )

    started_at = datetime.now(UTC).isoformat()
    output_dir.mkdir(parents=True, exist_ok=False)
    gradient_audit = _gradient_path_audit(
        model,
        processor,
        examples,
        data_root=args.data_root.resolve(),
        temperature=args.temperature,
        shadow_snapshots=shadow_snapshots,
    )

    arms = (
        ("all_ternary_targets", set()),
        ("audio_path_bf16", {"audio_path"}),
        ("vision_path_bf16", {"vision_path"}),
        ("shared_decoder_bf16", {"shared_decoder"}),
    )
    model.eval()
    variant_results: dict[str, Any] = {}
    for arm_name, restored_components in arms:
        print(f"validation arm start: {arm_name}", flush=True)
        load_info = load_packed_ternary_overlay(
            model,
            overlay_dir,
            expected_base_model_id=model_manifest.repo_id,
            expected_base_revision=model_manifest.revision,
        )
        restore_info = _restore_bf16_groups(model, shadow_snapshots, restored_components)
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
            raise ValueError(f"prediction order differs from frozen validation IDs: {arm_name}")
        prediction_sha = _write_predictions(
            output_dir / f"{arm_name}-predictions.jsonl", predictions
        )
        variant_results[arm_name] = {
            "restored_components": sorted(restored_components),
            "restored_high_precision": restore_info,
            "overlay_tensor_sha256": load_info["tensor_file_sha256"],
            "metrics": metrics,
            "prediction_sha256": prediction_sha,
            "evaluation_seconds": time.perf_counter() - started,
            "prediction_count": len(predictions),
        }
        print(
            f"validation arm complete: {arm_name} ({len(predictions)} examples)",
            flush=True,
        )
        torch.cuda.empty_cache() if args.device == "cuda" else None

    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        source_commit = "UNKNOWN"
    result = {
        "schema_version": 1,
        "experiment_id": "embeddinggemma2-ternary-component-ablation-v0",
        "status": "complete_validation_diagnostic_not_model_selection",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": source_commit,
        "base_model_id": model_manifest.repo_id,
        "base_revision": model_manifest.revision,
        "model_manifest_sha256": sha256(args.model_manifest),
        "verified_model_files": verified_model_files,
        "qat_run_metadata_sha256": sha256(qat_metadata_path),
        "qat_best_step": 512,
        "qat_best_shadow_sha256": sha256(shadow_path),
        "overlay_manifest_sha256": sha256(overlay_manifest_path),
        "overlay_tensor_sha256": sha256(overlay_tensor_path),
        "validation_snapshot_sha256": validation_sha256,
        "validation_selection_manifest_sha256": sha256(args.selection_manifest),
        "validation_ordered_ids_sha256": hashlib.sha256(
            "\n".join(example.id for example in examples).encode("utf-8")
        ).hexdigest(),
        "validation_examples": len(examples),
        "score_temperature": args.temperature,
        "ece_bins": 15,
        "threads": args.threads,
        "sealed_audit_loaded": False,
        "recovery_lora_loaded": False,
        "training_or_optimizer_updates": 0,
        "source_validation_media_counts": media_counts,
        "target_inventory": quantization_audit,
        "quantization_code_forward_path": "selected_qat_overlay_against_same_qat_best_bf16_shadow",
        "quantization_target_restore_groups": {
            "audio_path": ["audio_tower.*", "embed_audio.*"],
            "vision_path": ["vision_tower.*", "embed_vision.*"],
            "shared_decoder": ["language_model.*"],
        },
        "gradient_path_audit": gradient_audit,
        "variants": variant_results,
        "device": torch.cuda.get_device_name(device) if args.device == "cuda" else "cpu",
        "cuda_total_bytes": torch.cuda.get_device_properties(device).total_memory
        if args.device == "cuda"
        else None,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if args.device == "cuda"
        else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if args.device == "cuda"
        else None,
        "note": (
            "Validation-only component ablation; not a product precision decision "
            "or sealed evaluation."
        ),
    }
    _write_json(output_dir / "diagnostic-results.json", result)
    print(
        json.dumps(
            {"status": result["status"], "output_dir": str(output_dir)},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
