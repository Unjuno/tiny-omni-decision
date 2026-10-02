from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import random
import tempfile
from pathlib import Path

import click
import typer
from rich.console import Console

from .decision_math import label_token_ids_from_prompt, prompt_for_decision
from .io import load_structured_file
from .schema import (
    BaseModelManifest,
    DatasetCatalog,
    DatasetManifest,
    DecisionExample,
    TextDecision,
)

app = typer.Typer(no_args_is_help=True)
console = Console()
MANIFEST_OPTION = typer.Option(..., "--manifest")
OUTPUT_OPTION = typer.Option(..., "--output")
ADAPTER_OPTION = typer.Option("generic", "--adapter")
LIMIT_OPTION = typer.Option(None, "--limit", min=1)
SEED_OPTION = typer.Option(None, "--seed")
COMPONENTS_OPTION = typer.Option(None, "--components")


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
    from .dataset import audit_manifest

    result = audit_manifest(manifest)
    console.print(json.dumps(result, indent=2))


@app.command("validate-dataset-catalog")
def validate_dataset_catalog(path: Path) -> None:
    """Validate a separate training or evaluation candidate catalog."""
    catalog = DatasetCatalog.model_validate(load_structured_file(path))
    for item in catalog.sources:
        if not item.include:
            continue
        manifest_path = (path.parent / item.manifest).resolve()
        manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
        if manifest.usage not in {catalog.purpose, "both"}:
            raise click.ClickException(
                f"{item.manifest} usage={manifest.usage} cannot enter {catalog.purpose} catalog"
            )
        if catalog.purpose == "evaluation" and item.split == "train":
            raise click.ClickException(
                f"training split cannot enter evaluation catalog: {item.manifest}"
            )
    console.print(f"valid {catalog.purpose} dataset catalog: {len(catalog.sources)} sources")


@app.command("audit-dataset-manifest")
def audit_dataset_manifest(path: Path) -> None:
    """Report source facts and the fail-closed project license policy decision."""
    manifest = DatasetManifest.model_validate(load_structured_file(path))
    from .dataset import audit_manifest

    result = audit_manifest(manifest)
    console.print(json.dumps(result, indent=2))
    if result["project_policy"] == "DENY":
        raise typer.Exit(code=2)


@app.command()
def status() -> None:
    """Print the current implementation boundary."""
    console.print(
        "Phase 1: complete; reproducible text decision LoRA smoke and CPU CI are implemented."
    )
    console.print(
        "Phase 2: pinned candidates, license audits, modality adapters, normalization, "
        "and split gates are implemented."
    )
    console.print(
        "Open hard gates: full-corpus split verification, mixed-source rights review, "
        "ternary runtime."
    )


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


@app.command("dataset-inspect")
def dataset_inspect(manifest_path: Path) -> None:
    """Inspect a pinned source manifest without loading media."""
    manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    from .dataset import audit_manifest

    console.print(
        json.dumps(
            {**manifest.model_dump(mode="json"), "audit": audit_manifest(manifest)}, indent=2
        )
    )


@app.command("dataset-normalize")
def dataset_normalize(
    source: str,
    manifest_path: Path = MANIFEST_OPTION,
    output: Path = OUTPUT_OPTION,
    adapter: str = ADAPTER_OPTION,
    limit: int | None = LIMIT_OPTION,
    seed: int | None = SEED_OPTION,
    components_path: Path | None = COMPONENTS_OPTION,
) -> None:
    """Normalize local JSON/JSONL or a Hub dataset in streaming mode."""
    from .dataset import (
        audit_license,
        audit_manifest,
        iter_hub_rows,
        iter_local_rows,
        iter_mixed_license_rows,
        load_component_csv,
        normalize_jsonl,
    )

    manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
    audit = audit_manifest(manifest)
    components = {
        item.component_id: item.model_dump(mode="json") for item in manifest.source_components
    }
    if components_path:
        components.update(load_component_csv(components_path))
    mixed_policy = (
        adapter in {"onejev", "onejev-data"}
        and manifest.license.casefold() == "per-source"
        and manifest.usage in {"training", "both"}
    )
    if (
        manifest.usage in {"training", "both"}
        and audit["project_policy"] != "ALLOW"
        and not mixed_policy
    ):
        raise click.ClickException(
            "training-safe normalization requires ALLOW; "
            f"manifest policy is {audit['project_policy']}"
        )
    source_path = Path(source)
    if source_path.is_file():
        rows = iter_local_rows(source_path)
    else:
        if source != manifest.dataset_id:
            raise typer.BadParameter("Hub source must match dataset_id in the pinned manifest")
        rows = iter_hub_rows(manifest)
    filter_counts = {"ALLOW": 0, "REVIEW": 0, "DENY": 0, "unresolved_source": 0}
    if mixed_policy:
        rows = iter_mixed_license_rows(rows, components, counts=filter_counts)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for item in normalize_jsonl(
                rows, manifest, adapter, limit=limit, seed=seed, components=components
            ):
                if manifest.usage in {"training", "both"}:
                    provenance = item.provenance
                    record_policy, unresolved = audit_license(
                        provenance.license,
                        commercial_use=provenance.commercial_use,
                        derivative_model_training_allowed=provenance.derivative_model_training_allowed,
                        redistribution_allowed=provenance.redistribution_allowed,
                        media_redistribution_allowed=provenance.media_redistribution_allowed,
                        has_media=bool(item.media),
                        trust_status=provenance.trust_status,
                    )
                    if record_policy != "ALLOW":
                        raise ValueError(
                            f"training-safe row {item.id} has policy {record_policy}; "
                            f"unresolved={unresolved}"
                        )
                handle.write(item.model_dump_json() + "\n")
                count += 1
            if count == 0:
                raise ValueError("normalization produced zero eligible examples")
        Path(temp_name).replace(output)
    except Exception as exc:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
        raise click.ClickException(f"Dataset normalization failed: {exc}") from exc
    console.print(
        {
            "normalized": count,
            "output": str(output),
            "dataset_id": manifest.dataset_id,
            "revision": manifest.revision,
            "mixed_license_filter": filter_counts if mixed_policy else None,
        }
    )


@app.command("dataset-check-splits")
def dataset_check_splits(training_path: Path, evaluation_path: Path) -> None:
    """Fail on duplicate source identities or normalized content across splits."""
    from .dataset import check_train_eval_splits, iter_local_rows

    training = (DecisionExample.model_validate(row) for row in iter_local_rows(training_path))
    evaluation = (DecisionExample.model_validate(row) for row in iter_local_rows(evaluation_path))
    try:
        console.print(check_train_eval_splits(training, evaluation))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


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
        raise click.ClickException(f"Model inspection failed: {exc}") from exc

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
        raise click.ClickException(
            'Install the ML extra plus a supported CUDA build: pip install ".[ml]"'
        ) from exc
    manifest = BaseModelManifest.model_validate(load_structured_file(manifest_path))
    if not manifest.revision:
        raise typer.BadParameter("manifest must pin an immutable revision")
    if not torch.cuda.is_available():
        raise click.ClickException("Smoke training requires CUDA; CPU-only mode is for CI tests")
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
        raise click.ClickException(
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
        raise click.ClickException(
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
        raise click.ClickException(
            f"LoRA target mismatch: expected {targets}, adapted {adapted_targets}"
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen_base = [p for name, p in model.named_parameters() if "lora_" not in name]
    if not trainable or any(p.requires_grad for p in frozen_base):
        raise click.ClickException("Base frozen / LoRA trainable invariant failed")
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
        raise click.ClickException("Loss is non-finite")
    loss.backward()
    if not trainable or any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
        raise click.ClickException("No finite LoRA gradient")
    if any(p.grad is not None for p in frozen_base):
        raise click.ClickException("Frozen base received gradients")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    optimizer.step()
    if [param._version for param in frozen_base] != base_versions:
        raise click.ClickException("A frozen base parameter was modified")
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
        raise click.ClickException("Reloaded inference probability check failed")
    if not torch.allclose(before_reload_probabilities, probabilities.cpu(), atol=1e-4, rtol=1e-4):
        raise click.ClickException("Adapter probabilities changed after save/reload")
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
