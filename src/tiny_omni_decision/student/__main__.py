"""Separate student CLI: no changes to the active Teacher training commands."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path


def validate_config(config: dict) -> None:
    required = {
        "group_size", "exclude", "max_length", "head_hidden_dim", "seed", "qat_steps",
        "lora_steps", "evaluation_interval", "learning_rate", "lora_learning_rate",
        "lora_rank", "lora_alpha", "temperature", "ce_weight", "brier_weight",
        "reload_atol", "limits", "max_artifact_bytes",
    }
    if set(config) != required:
        raise ValueError(f"config keys missing/extra: {sorted(set(config) ^ required)}")
    for key in ("group_size", "max_length", "head_hidden_dim", "qat_steps", "lora_steps",
                "evaluation_interval", "lora_rank", "max_artifact_bytes"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_length"] > 8192 or type(config["seed"]) is not int:
        raise ValueError("invalid max_length/seed")
    if not isinstance(config["exclude"], list) or not all(
        isinstance(n, str) and n for n in config["exclude"]
    ):
        raise ValueError("exclude must contain exact parameter names")
    for key in ("learning_rate", "lora_learning_rate", "lora_alpha",
                "temperature", "ce_weight", "brier_weight", "reload_atol"):
        if type(config[key]) not in (float, int) or not math.isfinite(config[key]):
            raise ValueError(f"invalid {key}")
        zero_ok = key in {"ce_weight", "brier_weight", "reload_atol"}
        if config[key] < 0 or (config[key] == 0 and not zero_ok):
            raise ValueError(f"invalid {key}")
    from .recovery import quality_gate
    metrics = {"all": dict(accuracy=0.0, nll=0.0, brier=0.0, ece=0.0)}
    quality_gate(metrics, metrics, config["limits"])


def _status(event: str, **fields) -> None:
    payload = {"event": event, **fields}
    print(
        "student-progress "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def _environment(device: str = "cpu") -> dict:
    import platform

    import torch
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("sentence-transformers", "transformers", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["device"] = device
    versions["hardware"] = (
        torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU"
    )
    return versions


def _smoke(output: Path) -> dict:
    import torch

    from .data import Cache, Record
    from .recovery import run_recovery
    from .ternary import TernaryController

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
        def forward(self, record):
            return self.linear(torch.tensor([1.0, 0.5, -0.5, 2.0]))

    def item(identifier):
        return Record(identifier, "g" + identifier, "c" + identifier, "text", {"text": "fixture"},
                      ["a", "b", "c"], 1, [-100.0, 100.0, -100.0], {})

    torch.manual_seed(17)
    model = Toy()
    controller = TernaryController(model, group_size=4)
    config = dict(
        qat_steps=3, lora_steps=3, evaluation_interval=1, learning_rate=0.01,
        lora_learning_rate=0.01, seed=17, lora_rank=2, lora_alpha=4.0,
        limits=dict(max_accuracy_drop=0.0, max_nll_increase=0.0,
                    max_brier_increase=0.0, max_ece_increase=0.0),
    )
    result = run_recovery(model, controller,
                          Cache("train", "synthetic", [item("1")], "a"*64),
                          Cache("validation", "synthetic", [item("2")], "b"*64),
                          output, config, metadata={"kind": "synthetic"})
    if not result["lora_used"] or result["reload_max_abs_error"] != 0.0:
        raise RuntimeError("synthetic QAT/LoRA/export/reload smoke failed")
    return {"kind": "synthetic_pipeline_check_not_model_quality", "lora_used": True,
            "reload_max_abs_error": result["reload_max_abs_error"],
            "quality_gate_status": result["status"], "output": str(output)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    pin = subs.add_parser("pin", help="resolve and save a student revision (no weights downloaded)")
    pin.add_argument("--output", type=Path, required=True)
    pin.add_argument("--revision", default="main")
    cache = subs.add_parser("cache-teacher", help="run ONLY after freezing the Teacher")
    cache.add_argument("--examples", type=Path, required=True)
    cache.add_argument(
        "--base-manifest", type=Path, default=Path("manifests/base-model.example.yaml")
    )
    cache.add_argument("--adapter", type=Path, required=True)
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--role", choices=["train", "validation", "evaluation"], required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-length", type=int, default=1024)
    cache.add_argument("--allow-download", action="store_true")
    for name in ("inspect", "train"):
        cmd = subs.add_parser(name)
        cmd.add_argument("--pin", type=Path, required=True)
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--data-root", type=Path, required=True)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--allow-download", action="store_true")
        cmd.add_argument("--output", type=Path, required=True)
        if name == "train":
            cmd.add_argument("--train-cache", type=Path, required=True)
            cmd.add_argument("--validation-cache", type=Path, required=True)
    evaluation = subs.add_parser(
        "evaluate", help="evaluate the selected frozen export, never train"
    )
    evaluation.add_argument("--bundle", type=Path, required=True)
    evaluation.add_argument("--cache", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--allow-download", action="store_true")
    evaluation.add_argument("--output", type=Path, required=True)
    smoke = subs.add_parser("smoke", help="CPU synthetic check; downloads no model")
    smoke.add_argument("--output", type=Path, required=True)
    return p


def dispatch(args: argparse.Namespace) -> dict:
    from .data import write_json
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "smoke":
        return _smoke(args.output)
    if args.command == "pin":
        from huggingface_hub import HfApi
        from transformers import AutoConfig

        from .model import MODEL_ID, validate_pin
        info = HfApi().model_info(MODEL_ID, revision=args.revision)
        config = AutoConfig.from_pretrained(MODEL_ID, revision=info.sha, trust_remote_code=False)
        card = info.card_data.to_dict() if info.card_data else {}
        result = {"model_id": MODEL_ID, "revision": info.sha,
                  "model_type": config.model_type, "license": card.get("license")}
        validate_pin(result)
        write_json(args.output, result)
        return result
    if args.command == "cache-teacher":
        from .teacher_cache import create_teacher_cache
        return create_teacher_cache(
            examples_path=args.examples, base_manifest=args.base_manifest,
            adapter_dir=args.adapter, data_root=args.data_root, output=args.output,
            role=args.role, device=args.device, max_length=args.max_length,
            allow_download=args.allow_download,
        )
    import torch

    from .artifacts import bundle_info, load_bundle
    from .data import assert_disjoint, assert_heldout, load_cache, validate_media
    from .model import load_student
    from .recovery import evaluate, run_recovery
    from .ternary import TernaryController, inspect_quantization_inventory

    if args.command == "evaluate":
        bundle = bundle_info(args.bundle)
        metadata = bundle["metadata"]
        config = metadata["config"]
        validate_config(config)
        cache = load_cache(args.cache, "evaluation")
        if cache.teacher_id != metadata["teacher_id"]:
            raise ValueError("evaluation Teacher differs from training Teacher")
        assert_heldout(cache, metadata["split_guard"])
        validate_media(cache, args.data_root)
        student = load_student(
            metadata["student_pin"],
            data_root=args.data_root,
            device=args.device,
            max_length=config["max_length"],
            head_hidden_dim=config["head_hidden_dim"],
            allow_download=args.allow_download,
        )
        exclusions = list(dict.fromkeys(config["exclude"] + student.ternary_exclusions()))
        controller = TernaryController(student, config["group_size"], exclusions)
        load_bundle(controller, args.bundle)
        metrics, logits = evaluate(student, cache.records)
        teacher_metrics, _ = evaluate(None, cache.records)
        result = {"metrics": metrics, "teacher_metrics": teacher_metrics,
                  "sample_ids": [r.sample_id for r in cache.records], "option_logits": logits,
                  "environment": _environment(args.device), "role": "evaluation",
                  "execution": "reference_dequantized_float", "bundle": str(args.bundle)}
        write_json(args.output, result)
        return result
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    pin = json.loads(args.pin.read_text(encoding="utf-8"))
    caches = None
    if args.command == "train":
        train = load_cache(args.train_cache, "train")
        validation = load_cache(args.validation_cache, "validation")
        assert_disjoint(train, validation)
        for cache in (train, validation):
            validate_media(cache, args.data_root)
        caches = (train, validation)
    torch.manual_seed(config["seed"])
    _status("student_load_start", device=args.device)
    load_started = time.perf_counter()
    student = load_student(
        pin,
        data_root=args.data_root,
        device=args.device,
        max_length=config["max_length"],
        head_hidden_dim=config["head_hidden_dim"],
        allow_download=args.allow_download,
    )
    _status(
        "student_load_done",
        device=args.device,
        elapsed_seconds=time.perf_counter() - load_started,
    )
    baseline = None
    if caches:
        _status(
            "unquantized_validation_start",
            records=len(caches[1].records),
        )
        baseline_started = time.perf_counter()
        baseline = evaluate(student, caches[1].records)[0]
        _status(
            "unquantized_validation_done",
            records=len(caches[1].records),
            elapsed_seconds=time.perf_counter() - baseline_started,
        )
    exclusions = list(dict.fromkeys(config["exclude"] + student.ternary_exclusions()))
    metadata = {
        "student_pin": pin,
        "environment": _environment(args.device),
        "readout": "custom_mean_pool_tiny_variable_option_mlp_v1",
        "head_hidden_dim": config["head_hidden_dim"],
        "unquantized_validation": baseline,
    }
    if args.command == "inspect":
        result = metadata | {
            "inventory": inspect_quantization_inventory(
                student, config["group_size"], exclusions
            )
        }
        write_json(args.output, result)
        return result
    _status("ternary_controller_start")
    controller_started = time.perf_counter()
    controller = TernaryController(student, config["group_size"], exclusions)
    _status(
        "ternary_controller_done",
        targets=len(controller.targets),
        elapsed_seconds=time.perf_counter() - controller_started,
    )
    student.enable_decision_head_training()
    return run_recovery(student, controller, *caches, args.output, config, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"student: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
