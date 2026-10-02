from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

import typer
from rich.console import Console

from .decision_math import label_token_ids_from_prompt, prompt_for_decision
from .io import load_structured_file
from .schema import BaseModelManifest, DatasetManifest, TextDecision

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def validate_model_manifest(path: Path) -> None:
    """Validate a base-model provenance manifest."""
    manifest = BaseModelManifest.model_validate(load_structured_file(path))
    console.print(f"[green]valid[/green] model manifest: {manifest.repo_id}@{manifest.revision}")
    if manifest.revision is None:
        console.print("[yellow]warning:[/yellow] upstream revision is not pinned yet")


@app.command()
def validate_dataset_manifest(path: Path) -> None:
    """Validate a dataset provenance manifest."""
    manifest = DatasetManifest.model_validate(load_structured_file(path))
    console.print(f"[green]valid[/green] dataset manifest: {manifest.dataset_id}")
    if manifest.revision is None:
        console.print("[yellow]warning:[/yellow] dataset revision is not pinned yet")
    if manifest.license.upper() == "UNKNOWN":
        console.print("[yellow]warning:[/yellow] dataset license is unresolved")


@app.command()
def status() -> None:
    """Print the current implementation boundary."""
    console.print("Phase 1: complete; reproducible text decision LoRA smoke validated.")
    console.print("Current focus: Phase 2 dataset provenance, licensing, and split gates.")
    console.print("Later hard gate: architecture-compatible ternary runtime verification.")


@app.command()
def preflight() -> None:
    """Report Python, optional ML versions, and CUDA device memory."""
    report: dict[str, object] = {"python": platform.python_version(), "cuda_available": False}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "huggingface-hub",
    ):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            devices = []
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    free, total = torch.cuda.mem_get_info()
                devices.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_bytes": total,
                        "free_memory_bytes": free,
                    }
                )
            report["cuda_devices"] = devices
    except ImportError:
        report["ml_status"] = "Install the optional ML dependencies to inspect CUDA."
    console.print(json.dumps(report, indent=2))


def _load_model(model_id: str, revision: str, dtype: str = "auto"):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return processor, model


@app.command("inspect-model")
def inspect_model(
    manifest_path: Path = Path("manifests/base-model.example.yaml"),
    config_only: bool = typer.Option(
        False, "--config-only", help="Inspect instantiated architecture without downloading weights"
    ),
) -> None:
    """Load and inspect the pinned model; no architecture/module names are assumed."""
    manifest = BaseModelManifest.model_validate(load_structured_file(manifest_path))
    if not manifest.revision:
        raise typer.BadParameter("manifest must pin an immutable revision")
    try:
        if config_only:
            import torch
            from accelerate import init_empty_weights
            from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                manifest.repo_id, revision=manifest.processor_revision or manifest.revision
            )
            config = AutoConfig.from_pretrained(manifest.repo_id, revision=manifest.revision)
            with init_empty_weights():
                model = AutoModelForMultimodalLM.from_config(config, dtype=torch.bfloat16)
            model.tie_weights()
        else:
            processor, model = _load_model(manifest.repo_id, manifest.revision)
    except Exception as exc:
        raise typer.ClickException(f"Model inspection failed: {exc}") from exc

    modules = list(model.named_modules())
    names = [name for name, _ in modules]
    params = list(model.named_parameters())
    total = sum(param.numel() for _, param in params)
    trainable = sum(param.numel() for _, param in params if param.requires_grad)
    import torch

    linear_modules = [name for name, module in modules if isinstance(module, torch.nn.Linear)]
    candidates = [
        name
        for name in linear_modules
        if name.startswith("model.language_model.")
        and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    ]
    modality = sorted(
        {
            ".".join(name.split(".")[:2])
            for name in names
            if any(term in name.lower() for term in ("vision", "image", "audio", "video"))
        }
    )
    tokenizer = processor.tokenizer
    components = {
        name: type(getattr(processor, name)).__name__
        for name in ("image_processor", "video_processor", "feature_extractor")
        if hasattr(processor, name)
    }
    dtype_counts: dict[str, int] = {}
    for _, param in params:
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + param.numel()
    memory = sum(param.numel() * param.element_size() for _, param in params)
    console.print(
        json.dumps(
            {
                "repo_id": manifest.repo_id,
                "revision": manifest.revision,
                "inspection_mode": "config_only_meta" if config_only else "loaded_weights",
                "architecture": model.config.architectures,
                "total_parameters": total,
                "trainable_parameters": trainable,
                "main_modules": [name for name, _ in model.named_children()]
                + [f"model.{name}" for name, _ in model.model.named_children()],
                "candidate_lora_target_modules": candidates,
                "modality_related_modules": modality,
                "dtype_parameter_counts": dtype_counts,
                "parameter_memory_bytes": memory,
                "tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer_model_max_length": tokenizer.model_max_length,
                "tokenizer_special_tokens": {
                    name: str(value) for name, value in tokenizer.special_tokens_map.items()
                },
                "chat_template_available": bool(tokenizer.chat_template),
                "processor_class": processor.__class__.__name__,
                "processor_components": components,
                "processor_revision": manifest.processor_revision,
                "option_label_candidates": ["A", "B", "C", "D"],
            },
            indent=2,
        )
    )


def _decision_logits(model, processor, sample: TextDecision) -> tuple[object, list[int]]:
    from .decision import option_logits_from_vocab

    labels = [chr(ord("A") + i) for i in range(len(sample.options))]
    text = processor.apply_chat_template(
        [{"role": "user", "content": prompt_for_decision(sample, labels)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids, prompt_length = label_token_ids_from_prompt(processor.tokenizer, text, labels)
    inputs = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    result = model(**inputs, use_cache=False)
    vocab = result.logits[0, prompt_length - 1]
    return option_logits_from_vocab(vocab, ids), ids


@app.command("smoke-text-decision")
def smoke_text_decision(
    output_dir: Path = Path("artifacts/smoke-text-decision"),
    seed: int = 17,
    manifest_path: Path = Path("manifests/base-model.example.yaml"),
) -> None:
    """Run a one-step LoRA training, checkpoint roundtrip, and inference smoke test."""
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model

        from .decision import decision_loss, normalize_probabilities
    except Exception as exc:
        raise typer.ClickException(
            'Install the ML extra plus a supported CUDA build: pip install ".[ml]"'
        ) from exc
    manifest = BaseModelManifest.model_validate(load_structured_file(manifest_path))
    if not manifest.revision:
        raise typer.BadParameter("manifest must pin an immutable revision")
    if not torch.cuda.is_available():
        raise typer.ClickException("Smoke training requires CUDA; CPU-only mode is for CI tests")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    processor, base = _load_model(manifest.repo_id, manifest.revision)
    for param in base.parameters():
        param.requires_grad_(False)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    modules = [name for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)]
    if not modules:
        raise typer.ClickException(
            "No Linear modules found; inspect the loaded model before choosing targets"
        )
    # Use decoder module paths to avoid similarly named vision/audio layers.
    targets = sorted(
        name
        for name in modules
        if name.startswith("model.language_model.")
        and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    )
    if not targets:
        raise typer.ClickException(
            "Could not identify attention projection modules; refusing guessed LoRA targets"
        )
    model = get_peft_model(
        base,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"
        ),
    )
    adapted_targets = sorted(
        name.removeprefix("base_model.model.")
        for name, module in model.named_modules()
        if hasattr(module, "lora_A")
    )
    if adapted_targets != targets:
        raise typer.ClickException(
            f"LoRA target mismatch: expected {targets}, adapted {adapted_targets}"
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_base = [p for name, p in model.named_parameters() if "lora_" not in name]
    if not trainable or any(p.requires_grad for p in frozen_base):
        raise typer.ClickException("Base frozen / LoRA trainable invariant failed")
    base_versions = [param._version for param in frozen_base]
    sample = TextDecision(
        state="A device battery is low.",
        question="Which action best preserves battery?",
        options=["Lower screen brightness", "Increase speaker volume", "Enable the flashlight"],
        target="Lower screen brightness",
    )
    target = sample.options.index(sample.target)
    model.train()
    logits, _ = _decision_logits(model, processor, sample)
    loss, ce, brier = decision_loss(
        logits.unsqueeze(0), torch.tensor([target], device=logits.device)
    )
    if not torch.isfinite(loss):
        raise typer.ClickException("Loss is non-finite")
    loss.backward()
    if not trainable or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
        raise typer.ClickException("No finite LoRA gradient")
    if any(p.grad is not None for p in frozen_base):
        raise typer.ClickException("Frozen base received gradients")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.step()
    if [param._version for param in frozen_base] != base_versions:
        raise typer.ClickException("A frozen base parameter was modified")
    model.eval()
    with torch.no_grad():
        before_reload_logits, _ = _decision_logits(model, processor, sample)
        before_reload_probabilities = normalize_probabilities(before_reload_logits).cpu()
    metrics = {
        "loss": float(loss.detach()),
        "ce": float(ce.detach()),
        "brier": float(brier.detach()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    weights_sha256 = next(
        (item["sha256"] for item in manifest.files if item.get("path") == "model.safetensors"),
        None,
    )
    (output_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "base_repo_id": manifest.repo_id,
                "base_revision": manifest.revision,
                "base_weights_sha256": weights_sha256,
                "base_config": model.config.to_dict(),
                "seed": seed,
                "target_modules": targets,
                "lora_rank": 16,
                "lora_alpha": 32,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del before_reload_logits, logits, loss, ce, brier
    del model, base, processor, optimizer, trainable, frozen_base, base_versions
    gc.collect()
    torch.cuda.empty_cache()
    processor, base = _load_model(manifest.repo_id, manifest.revision)
    reloaded = PeftModel.from_pretrained(base, output_dir).eval()
    verify_sample = sample
    with torch.no_grad():
        verify_logits, _ = _decision_logits(reloaded, processor, verify_sample)
        probabilities = normalize_probabilities(verify_logits)
    if not torch.isfinite(probabilities).all() or not torch.allclose(
        probabilities.sum(), torch.tensor(1.0, device=probabilities.device), atol=1e-5
    ):
        raise typer.ClickException("Reloaded inference probability check failed")
    if not torch.allclose(before_reload_probabilities, probabilities.cpu(), atol=1e-4, rtol=1e-4):
        raise typer.ClickException("Adapter probabilities changed after save/reload")
    console.print(
        {
            **metrics,
            "options": sample.options,
            "option_logits": verify_logits.tolist(),
            "probabilities": probabilities.tolist(),
            "prediction": sample.options[int(probabilities.argmax())],
            "confidence": float(probabilities.max()),
            "checkpoint": str(output_dir),
        }
    )


if __name__ == "__main__":
    app()
