from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
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


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"JSONL file has an incomplete final record: {path}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen EmbeddingGemma candidates against exact shared choices."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--teacher-repo-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "final"), required=True)
    parser.add_argument(
        "--variant", choices=("unquantized", "ternary", "recovery", "all"), default="all"
    )
    parser.add_argument(
        "--cache-video-frames",
        action="store_true",
        help="Decode each consecutive video scene once and keep only one clip in host memory.",
    )
    return parser.parse_args()


def _verify_bundle(package: Path) -> dict[str, Any]:
    manifest_path = package / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "tiny-omni-embeddinggemma2-ternary-recovery-bundle-v1":
        raise ValueError("unsupported student bundle format")
    if manifest.get("status") != "exported_validation_candidate_not_product_promoted":
        raise ValueError("bundle is not marked as a validation candidate")
    for item in manifest["files"]:
        path = (package / item["path"]).resolve()
        if package not in path.parents:
            raise ValueError("bundle manifest contains an unsafe path")
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"bundle file missing or size mismatch: {item['path']}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"bundle file hash mismatch: {item['path']}")
    return manifest


def _load_examples_and_teacher(
    split: str,
    *,
    repo: Path,
    teacher_repo: Path,
    config: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    from tiny_omni_decision.schema import DecisionExample
    from tiny_omni_decision.student_eval import load_fixed_validation_snapshot
    from tiny_omni_decision.student_export import align_teacher_predictions

    data = config["data"]
    data_root = (teacher_repo / data["data_root"]).resolve()
    if split == "validation":
        snapshot = Path(data["validation_snapshot"]).resolve()
        selection_manifest = (repo / data["validation_selection_manifest"]).resolve()
        source_validation = (teacher_repo / data["source_validation_corpus"]).resolve()
        teacher_validation = (teacher_repo / data["teacher_validation_predictions"]).resolve()
        examples, teacher = load_fixed_validation_snapshot(
            snapshot,
            selection_manifest,
            source_validation_path=source_validation,
            teacher_predictions_path=teacher_validation,
        )
        return (
            examples,
            align_teacher_predictions(examples, teacher),
            {
                "split": split,
                "examples_sha256": sha256(snapshot),
                "teacher_predictions_sha256": sha256(teacher_validation),
                "data_root": str(data_root),
            },
        )

    corpus_dir = teacher_repo / "data" / "processed" / "durable-teacher-v0"
    eval_path = corpus_dir / "eval.jsonl"
    train_path = corpus_dir / "train.jsonl"
    validation_path = corpus_dir / "validation.jsonl"
    corpus_manifest_path = corpus_dir / "corpus-manifest.json"
    teacher_artifacts = teacher_repo / "artifacts" / "tiny-omni-decision-teacher-v0"
    teacher_run_path = teacher_artifacts / "run-metadata.json"
    teacher_predictions_path = teacher_artifacts / "decision-lora-predictions.jsonl"
    required = (
        eval_path,
        train_path,
        validation_path,
        corpus_manifest_path,
        teacher_run_path,
        teacher_predictions_path,
    )
    missing = next((path for path in required if not path.is_file()), None)
    if missing:
        raise FileNotFoundError(missing)
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    teacher_run = json.loads(teacher_run_path.read_text(encoding="utf-8"))
    eval_hash = sha256(eval_path)
    if (
        eval_hash != corpus_manifest["evaluation"]["sha256"]
        or eval_hash != teacher_run["eval_rows_sha256"]
        or sha256(train_path) != teacher_run["train_rows_sha256"]
        or sha256(validation_path) != teacher_run["validation_rows_sha256"]
        or corpus_manifest["overlap"]["status"] != "disjoint"
        or corpus_manifest["overlap"]["shared_source_ids"] != 0
        or corpus_manifest["overlap"]["shared_content_fingerprints"] != 0
    ):
        raise ValueError("frozen corpus hashes or split-overlap gate failed")
    raw_examples = [
        DecisionExample.model_validate(json.loads(line))
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    teacher_rows = read_jsonl(teacher_predictions_path)
    teacher = align_teacher_predictions(raw_examples, teacher_rows)
    return (
        raw_examples,
        teacher,
        {
            "split": split,
            "examples_sha256": eval_hash,
            "examples_count": len(raw_examples),
            "corpus_manifest_sha256": sha256(corpus_manifest_path),
            "teacher_run_metadata_sha256": sha256(teacher_run_path),
            "teacher_predictions_sha256": sha256(teacher_predictions_path),
            "train_rows_sha256": sha256(train_path),
            "validation_rows_sha256": sha256(validation_path),
            "historical_teacher_evaluation_already_observed": True,
            "data_root": str(data_root),
        },
    )


def _load_candidate(
    variant: str,
    *,
    package: Path,
    bundle: dict[str, Any],
    device: Any,
) -> tuple[Any, Any, float]:
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    from tiny_omni_decision.ternary import load_packed_ternary_overlay

    base_path = package / "base"
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(base_path), local_files_only=True, trust_remote_code=False
    )
    base = AutoModel.from_pretrained(
        str(base_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)
    if variant in {"ternary", "recovery"}:
        load_packed_ternary_overlay(
            base,
            package / "ternary-overlay",
            expected_base_model_id=bundle["base_model_id"],
            expected_base_revision=bundle["base_revision"],
        )
    model = base
    if variant == "recovery":
        model = PeftModel.from_pretrained(
            base, str(package / "recovery-adapter"), is_trainable=False
        )
    model.eval()
    return model, processor, time.perf_counter() - started


def _run_variant(
    variant: str,
    *,
    output: Path,
    examples: list[Any],
    teacher: list[dict[str, Any]],
    data_root: Path,
    package: Path,
    bundle: dict[str, Any],
    evaluation_identity: dict[str, Any],
    cache_video_frames: bool,
) -> dict[str, Any]:
    import torch

    from tiny_omni_decision.student_eval import (
        compare_student_predictions,
        evaluate_student_examples,
        validate_local_media_paths,
    )

    result_path = output / variant / "metrics.json"
    prediction_path = output / variant / "predictions.jsonl"
    candidate_dir = result_path.parent
    candidate_dir.mkdir(parents=True, exist_ok=True)
    expected_run = {
        "variant": variant,
        "split": evaluation_identity["split"],
        "examples_sha256": evaluation_identity["examples_sha256"],
        "bundle_manifest_sha256": sha256(package / "bundle-manifest.json"),
    }
    identity_path = candidate_dir / "run-identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != expected_run:
            raise ValueError(f"existing evaluation identity differs for {variant}")
    else:
        write_json(identity_path, expected_run)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("run_identity") != expected_run:
            raise ValueError(f"existing metrics identity differs for {variant}")
        return result

    existing = read_jsonl(prediction_path) if prediction_path.exists() else []
    validate_local_media_paths(examples, data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("this pinned evaluation environment requires the CUDA RTX device")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model, processor, load_seconds = _load_candidate(
        variant, package=package, bundle=bundle, device=device
    )
    timings: dict[str, list[float]] = defaultdict(list)
    previous_callback = time.perf_counter()
    callback_count = 0
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("a", encoding="utf-8", newline="\n") as stream:

        def on_prediction(record: dict[str, Any]) -> None:
            nonlocal previous_callback, callback_count
            now = time.perf_counter()
            timings[record["modality"]].append(now - previous_callback)
            previous_callback = now
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            callback_count += 1
            if callback_count % 32 == 0:
                stream.flush()

        started = time.perf_counter()
        metrics, predictions = evaluate_student_examples(
            model,
            processor,
            examples,
            data_root=data_root,
            temperature=float(bundle["score_temperature"]),
            ece_bins=15,
            existing_predictions=existing,
            on_prediction=on_prediction,
            cache_video_frames=cache_video_frames,
        )
        elapsed = time.perf_counter() - started
        stream.flush()

    paired = compare_student_predictions(predictions, teacher, ece_bins=15)
    timing_summary = {
        modality: {
            "count": len(values),
            "total_seconds": sum(values),
            "mean_seconds": statistics.mean(values),
            "p50_seconds": statistics.median(values),
            "p95_seconds": sorted(values)[min(len(values) - 1, int(0.95 * len(values)))],
            "measurement": (
                "wall intervals between newly emitted predictions, including decode, "
                "tokenize, and cache work; resumed rows have no per-example interval"
            ),
        }
        for modality, values in sorted(timings.items())
    }
    result = {
        "run_identity": expected_run,
        "variant": variant,
        "validation_or_final_metrics": metrics,
        "paired_teacher_metrics": paired,
        "predictions_sha256": sha256(prediction_path),
        "prediction_count": len(predictions),
        "model_load_seconds": load_seconds,
        "evaluation_wall_seconds": elapsed,
        "video_frame_cache_enabled": cache_video_frames,
        "wall_time_by_modality": timing_summary,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "mean_confidence": metrics["all"]["mean_confidence"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(result_path, result)
    del model, processor
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    import yaml

    from tiny_omni_decision.student_eval import validate_local_media_paths

    repo = args.repo_root.resolve()
    teacher_repo = args.teacher_repo_root.resolve()
    package = args.package_dir.resolve()
    output = args.output_dir.resolve()
    bundle = _verify_bundle(package)
    config = yaml.safe_load((package / "recovery-config.yaml").read_text(encoding="utf-8"))
    examples, teacher, identity = _load_examples_and_teacher(
        args.split, repo=repo, teacher_repo=teacher_repo, config=config
    )
    data_root = Path(identity.pop("data_root"))
    validate_local_media_paths(examples, data_root)
    output.mkdir(parents=True, exist_ok=True)
    root_identity = {
        **identity,
        "bundle_manifest_sha256": sha256(package / "bundle-manifest.json"),
        "candidate_variants": ["unquantized", "ternary", "recovery"],
        "video_frame_cache_enabled": args.cache_video_frames,
        "final_eval_is_not_used_for_model_selection": args.split == "final",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    root_identity_path = output / "evaluation-identity.json"
    if root_identity_path.exists():
        previous = json.loads(root_identity_path.read_text(encoding="utf-8"))
        for key, value in root_identity.items():
            if key == "started_at_utc":
                continue
            if previous.get(key) != value:
                raise ValueError("existing evaluation directory is for a different frozen run")
        root_identity = previous
    else:
        write_json(root_identity_path, root_identity)
    variants = ["unquantized", "ternary", "recovery"] if args.variant == "all" else [args.variant]
    for variant in variants:
        result = _run_variant(
            variant,
            output=output,
            examples=examples,
            teacher=teacher,
            data_root=data_root,
            package=package,
            bundle=bundle,
            evaluation_identity=identity,
            cache_video_frames=args.cache_video_frames,
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "completed": True,
                    "examples": result["prediction_count"],
                    "accuracy": result["validation_or_final_metrics"]["all"]["accuracy"],
                    "nll": result["validation_or_final_metrics"]["all"]["nll"],
                    "wall_seconds": result["evaluation_wall_seconds"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    variants_complete = all(
        (output / variant / "metrics.json").is_file()
        for variant in ("unquantized", "ternary", "recovery")
    )
    if variants_complete:
        summary = {
            "evaluation_identity_sha256": sha256(root_identity_path),
            "variants": {
                variant: json.loads((output / variant / "metrics.json").read_text(encoding="utf-8"))
                for variant in ("unquantized", "ternary", "recovery")
            },
        }
        write_json(output / "paired-summary.json", summary)


if __name__ == "__main__":
    main()
