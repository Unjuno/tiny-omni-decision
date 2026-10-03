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
    from .dataset import audit_manifest

    included: list[tuple[DatasetManifest, str]] = []
    for item in catalog.sources:
        if not item.include:
            continue
        manifest_path = (path.parent / item.manifest).resolve()
        manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
        if manifest.usage not in {catalog.purpose, "both"}:
            raise click.ClickException(
                f"{item.manifest} usage={manifest.usage} cannot enter {catalog.purpose} catalog"
            )
        if catalog.purpose == "evaluation" and manifest.split == "train":
            raise click.ClickException(
                f"training split cannot enter evaluation catalog: {item.manifest}"
            )
        if item.split and manifest.split != item.split:
            raise click.ClickException(
                f"catalog split {item.split} does not match {item.manifest} split {manifest.split}"
            )
        result = audit_manifest(manifest)
        if catalog.purpose == "training" and result["project_policy"] != "ALLOW":
            raise click.ClickException(
                f"included source {item.manifest} has project policy {result['project_policy']}"
            )
        if catalog.purpose == "evaluation" and manifest.usage != "evaluation":
            raise click.ClickException(
                f"evaluation catalog source must be evaluation-only: {item.manifest}"
            )
        included.append((manifest, item.manifest))
    if catalog.purpose == "training":
        evaluation_path = path.with_name("evaluation-candidates.yaml")
        if evaluation_path.is_file():
            evaluation_catalog = DatasetCatalog.model_validate(
                load_structured_file(evaluation_path)
            )
            evaluation_sources: dict[str, list[tuple[DatasetManifest, str]]] = {}
            for eval_item in evaluation_catalog.sources:
                if not eval_item.include:
                    continue
                eval_manifest_path = (evaluation_path.parent / eval_item.manifest).resolve()
                eval_manifest = DatasetManifest.model_validate(
                    load_structured_file(eval_manifest_path)
                )
                evaluation_sources.setdefault(eval_manifest.dataset_id, []).append(
                    (eval_manifest, eval_item.manifest)
                )
            for train_manifest, train_name in included:
                for eval_manifest, eval_name in evaluation_sources.get(
                    train_manifest.dataset_id, []
                ):
                    if train_manifest.split == eval_manifest.split:
                        raise click.ClickException(
                            f"same dataset split included for training and evaluation: "
                            f"{train_name}, {eval_name}"
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
        "Phase 2: approved text/image/audio/video candidates and split gates are implemented."
    )
    console.print(
        "Phase 3: bounded local high-precision LoRA pipeline is implemented; run the "
        "frozen corpus and GPU result in docs/PHASE3.md. This is a bounded experiment, "
        "not a durable teacher or model-quality claim."
    )
    console.print(
        "Still gated: larger durable training, benchmark expansion, component-level "
        "rights review for OneJev, and ternary runtime compatibility."
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


@app.command("freeze-corpus")
def freeze_corpus(
    train_catalog_path: Path = Path("manifests/phase3-training-corpus.yaml"),
    eval_catalog_path: Path = Path("manifests/phase3-evaluation-corpus.yaml"),
    output_dir: Path = Path("data/processed/phase3-frozen"),
    seed: int = 17,
    max_records_per_source: int = typer.Option(256, min=1),
) -> None:
    """Materialize capped, reproducible corpora and hash their provenance."""
    from .corpus import partition_heldout_records
    from .dataset import (
        audit_license,
        audit_manifest,
        check_train_eval_splits,
        corpus_statistics,
        deterministic_reservoir_sample,
        iter_hub_rows,
        iter_local_rows,
        normalize_jsonl,
        sha256_file,
    )

    def normalize_catalog(
        catalog_path: Path, split_output: Path
    ) -> tuple[
        list[DecisionExample], list[dict[str, object]], dict[tuple[str, str], str]
    ]:
        catalog = DatasetCatalog.model_validate(load_structured_file(catalog_path))
        if catalog_path == train_catalog_path and catalog.purpose != "training":
            raise click.ClickException("train catalog must have purpose=training")
        if catalog_path == eval_catalog_path and catalog.purpose != "evaluation":
            raise click.ClickException("evaluation catalog must have purpose=evaluation")
        examples: list[DecisionExample] = []
        source_rows: list[dict[str, object]] = []
        heldout_partitions: dict[tuple[str, str], str] = {}
        for entry in catalog.sources:
            if not entry.include:
                continue
            manifest_path = (catalog_path.parent / entry.manifest).resolve()
            manifest = DatasetManifest.model_validate(load_structured_file(manifest_path))
            if catalog.purpose == "evaluation":
                if entry.heldout_partition is None:
                    raise click.ClickException(
                        f"evaluation source needs heldout_partition: {entry.manifest}"
                    )
                source_split = (manifest.dataset_id, manifest.split)
                previous_partition = heldout_partitions.setdefault(
                    source_split, entry.heldout_partition
                )
                if previous_partition != entry.heldout_partition:
                    raise click.ClickException(
                        f"conflicting heldout partition roles for {source_split}"
                    )
            policy = audit_manifest(manifest)["project_policy"]
            if catalog.purpose == "training" and policy != "ALLOW":
                raise click.ClickException(
                    f"training source not ALLOW: {entry.manifest} ({policy})"
                )
            limit = min(entry.row_limit or max_records_per_source, max_records_per_source)
            local_source = manifest.notes.get("local_source")
            if local_source:
                source = Path(str(local_source))
                if not source.is_absolute():
                    source = Path.cwd() / source
                rows = iter_local_rows(source)
            else:
                source = None
                rows = iter_hub_rows(manifest)
            adapter = entry.adapter or "generic"
            if adapter in {"clevr4", "clevr-4"}:
                rows = (
                    row for row in rows if row.get("split", manifest.split) == manifest.split
                )
            elif adapter in {"speech-commands", "speech_commands"}:
                from .dataset import _speech_command_label

                rows = (
                    row
                    for row in rows
                    if row.get("split", manifest.split) == manifest.split
                    and _speech_command_label(row) is not None
                )
            elif adapter in {"clevrer", "clevrer-video"}:
                from .dataset import _clevrer_row_matches_split

                rows = (row for row in rows if _clevrer_row_matches_split(row, manifest.split))
            rows = deterministic_reservoir_sample(
                rows,
                limit=limit,
                seed=seed,
                source_key=f"{manifest.dataset_id}@{manifest.revision}:{manifest.split}",
            )
            normalized = list(
                normalize_jsonl(rows, manifest, adapter, seed=seed)
            )
            if not normalized:
                raise click.ClickException(f"source normalized to zero rows: {entry.manifest}")
            for item in normalized:
                if catalog.purpose == "training":
                    record_policy, unresolved = audit_license(
                        item.provenance.license,
                        commercial_use=item.provenance.commercial_use,
                        derivative_model_training_allowed=item.provenance.derivative_model_training_allowed,
                        redistribution_allowed=item.provenance.redistribution_allowed,
                        media_redistribution_allowed=item.provenance.media_redistribution_allowed,
                        has_media=bool(item.media),
                        trust_status=item.provenance.trust_status,
                    )
                    if record_policy != "ALLOW":
                        raise click.ClickException(
                            f"record is {record_policy}: {item.id}; {unresolved}"
                        )
            examples.extend(normalized)
            source_rows.append(
                {
                    "catalog_manifest": entry.manifest,
                    "dataset_id": manifest.dataset_id,
                    "source_revision": manifest.revision,
                    "split": manifest.split,
                    "adapter": entry.adapter,
                    "seed": seed,
                    "source_row_limit": limit,
                    "source_row_selection": "seeded reservoir sample without replacement",
                    "heldout_partition": entry.heldout_partition,
                    "records": len(normalized),
                    "manifest_sha256": sha256_file(manifest_path),
                    "source_file": str(source) if source else None,
                    "source_file_sha256": sha256_file(source)
                    if source and source.is_file()
                    else None,
                }
            )
        split_output.parent.mkdir(parents=True, exist_ok=True)
        with split_output.open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(example.model_dump_json() + "\n")
        return examples, source_rows, heldout_partitions

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    heldout_path = output_dir / "heldout-candidates.jsonl"
    validation_path = output_dir / "validation.jsonl"
    eval_path = output_dir / "eval.jsonl"
    try:
        train, train_sources, _ = normalize_catalog(train_catalog_path, train_path)
        heldout_candidates, heldout_sources, heldout_partitions = normalize_catalog(
            eval_catalog_path, heldout_path
        )
        clevr4_manifest = DatasetManifest.model_validate(
            load_structured_file(Path("manifests/candidates/clevr4.yaml"))
        )
        from .media import (
            materialize_clevr4_images,
            materialize_clevrer_videos,
            materialize_speech_commands_audio,
        )

        train, train_media = materialize_clevr4_images(
            train,
            data_root=Path("data"),
            archive_url=str(clevr4_manifest.notes["archive"]),
        )
        heldout_candidates, heldout_media = materialize_clevr4_images(
            heldout_candidates,
            data_root=Path("data"),
            archive_url=str(clevr4_manifest.notes["archive"]),
        )
        clevrer_train_manifest = DatasetManifest.model_validate(
            load_structured_file(Path("manifests/candidates/clevrer.yaml"))
        )
        clevrer_eval_manifest = DatasetManifest.model_validate(
            load_structured_file(Path("manifests/candidates/clevrer-validation.yaml"))
        )
        train, train_video_media = materialize_clevrer_videos(
            train,
            data_root=Path("data"),
            archive_urls={"train": str(clevrer_train_manifest.notes["video_archive_url"])},
        )
        heldout_candidates, heldout_video_media = materialize_clevrer_videos(
            heldout_candidates,
            data_root=Path("data"),
            archive_urls={"validation": str(clevrer_eval_manifest.notes["video_archive_url"])},
        )
        speech_train_manifest = DatasetManifest.model_validate(
            load_structured_file(Path("manifests/candidates/speech-commands.yaml"))
        )
        speech_eval_manifests = {
            split: DatasetManifest.model_validate(
                load_structured_file(
                    Path(f"manifests/candidates/speech-commands-{split}.yaml")
                )
            )
            for split in ("validation", "test")
        }
        train, train_audio_media = materialize_speech_commands_audio(
            train, manifest=speech_train_manifest, data_root=Path("data")
        )
        heldout_candidates, heldout_audio_media = materialize_speech_commands_audio(
            heldout_candidates, manifest=speech_eval_manifests, data_root=Path("data")
        )
        validation, evaluation = partition_heldout_records(
            heldout_candidates, seed=seed, heldout_partitions=heldout_partitions
        )
        for path, examples in ((train_path, train), (eval_path, evaluation)):
            path.write_text(
                "".join(example.model_dump_json() + "\n" for example in examples),
                encoding="utf-8",
            )
        validation_path.write_text(
            "".join(example.model_dump_json() + "\n" for example in validation),
            encoding="utf-8",
        )
        overlaps = {
            "train_validation": check_train_eval_splits(train, validation),
            "train_evaluation": check_train_eval_splits(train, evaluation),
            "validation_evaluation": check_train_eval_splits(validation, evaluation),
        }
        overlap = {
            "status": "disjoint"
            if all(item["status"] == "disjoint" for item in overlaps.values())
            else "overlap",
            "pairwise": overlaps,
            "training_records": len(train),
            "validation_records": len(validation),
            "evaluation_records": len(evaluation),
            "shared_source_ids": sum(item["shared_source_ids"] for item in overlaps.values()),
            "shared_content_fingerprints": sum(
                item["shared_content_fingerprints"] for item in overlaps.values()
            ),
        }
    except Exception as exc:
        train_path.unlink(missing_ok=True)
        heldout_path.unlink(missing_ok=True)
        validation_path.unlink(missing_ok=True)
        eval_path.unlink(missing_ok=True)
        raise click.ClickException(f"Corpus freeze failed: {exc}") from exc
    heldout_path.unlink(missing_ok=True)
    import hashlib

    corpus_manifest = {
        "schema_version": 1,
        "status": "frozen_durable_corpus",
        "seed": seed,
        "max_source_rows": max_records_per_source,
        "source_row_selection": "deterministic reservoir sampling without replacement",
        "heldout_partition": (
            "official validation/evaluation roles preserve source splits; entries marked "
            "split are partitioned 50/50 by source record with a deterministic seed"
        ),
        "option_order_policy": "stable per-example shuffle seeded by global seed and sample id",
        "media_policy": (
            "selected Clevr-4 PNG, Speech Commands WAV, and CLEVRER MP4 members "
            "are materialized and sha256 pinned; no benchmark or full archive media is copied"
        ),
        "media_materialization": {
            "train": {
                "images": train_media,
                "audio": train_audio_media,
                "videos": train_video_media,
            },
            "heldout": {
                "images": heldout_media,
                "audio": heldout_audio_media,
                "videos": heldout_video_media,
            },
        },
        "train": {
            "path": train_path.name,
            **corpus_statistics(train),
            "sha256": sha256_file(train_path),
            "sources": train_sources,
        },
        "evaluation": {
            "path": eval_path.name,
            **corpus_statistics(evaluation),
            "sha256": sha256_file(eval_path),
            "sources": heldout_sources,
        },
        "validation": {
            "path": validation_path.name,
            **corpus_statistics(validation),
            "sha256": sha256_file(validation_path),
            "sources": heldout_sources,
        },
        "overlap": overlap,
        "manifest_sha256": sha256_file(Path("manifests/base-model.example.yaml")),
        "training_catalog_sha256": sha256_file(train_catalog_path),
        "evaluation_catalog_sha256": sha256_file(eval_catalog_path),
        "content_fingerprint_algorithm": (
            "SHA-256 over NFKC/casefold/whitespace-normalized state, question, "
            "sorted options, and media identity"
        ),
        "corpus_pair_sha256": hashlib.sha256(
            (
                sha256_file(train_path)
                + sha256_file(validation_path)
                + sha256_file(eval_path)
            ).encode()
        ).hexdigest(),
    }
    if overlap["status"] != "disjoint":
        raise click.ClickException("Corpus overlap gate failed")
    (output_dir / "corpus-manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2), encoding="utf-8"
    )
    console.print(json.dumps(corpus_manifest, indent=2))


@app.command("train-decision")
def train_decision(
    train_manifest: Path = Path("data/processed/phase3-frozen/train.jsonl"),
    eval_manifest: Path = Path("data/processed/phase3-frozen/eval.jsonl"),
    validation_manifest: Path | None = None,
    config: Path = Path("configs/decision/e2b_qat_lora.yaml"),
    output: Path = Path("artifacts/decision-teacher-v0"),
    seed: int | None = None,
    max_train_examples: int | None = typer.Option(None, min=1),
    max_eval_examples: int | None = typer.Option(None, min=1),
    max_steps: int | None = typer.Option(None, min=1),
    gradient_accumulation_steps: int | None = typer.Option(None, min=1),
    checkpoint_interval: int | None = typer.Option(None, min=1),
    evaluation_interval: int | None = typer.Option(None, min=1),
    resume_from: Path | None = None,
    modalities: str = typer.Option("text,image,audio,video"),
    tiny_overfit: bool = False,
) -> None:
    """Train, select, and evaluate the frozen-base vocabulary-readout Decision LoRA."""
    from .trainer import run_training

    selected_modalities = {item.strip() for item in modalities.split(",") if item.strip()}
    try:
        result = run_training(
            train_path=train_manifest,
            eval_path=eval_manifest,
            validation_path=validation_manifest,
            config_path=config,
            output_dir=output,
            seed_override=seed,
            max_train_examples=max_train_examples,
            max_eval_examples=max_eval_examples,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            checkpoint_interval=checkpoint_interval,
            evaluation_interval=evaluation_interval,
            resume_from=resume_from,
            modalities=selected_modalities,
            tiny_overfit=tiny_overfit,
        )
    except Exception as exc:
        raise click.ClickException(f"Decision LoRA training failed: {exc}") from exc
    console.print(
        json.dumps(
            {
                "artifact": str(output),
                "steps": result["global_steps"],
                "consumed": result["actual_train_consumption_by_modality_source"],
                "baseline": result["baseline_metrics"],
                "decision_teacher": result["decision_teacher_metrics"],
                "metric_deltas_teacher_minus_base": result[
                    "metric_deltas_teacher_minus_base"
                ],
                "best_checkpoint_step": result["best_checkpoint_step"],
                "skipped_train": result["skipped_train_examples"],
                "skipped_eval": result["skipped_eval_examples"],
                "max_allocated_vram_bytes": result["max_allocated_vram_bytes"],
                "checkpoint_reload_verified": result["checkpoint_reload_verified"],
            },
            indent=2,
        )
    )


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
