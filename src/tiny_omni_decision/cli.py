from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

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
        "Teacher v0 is a legacy observed reference. Teacher v1 adds fail-closed "
        "validation, asset-level split checks, an isolated one-time sealed audit, "
        "and validation-only learning-curve selection."
    )
    console.print(
        "Teacher v1 data freeze, local learning curves, policy/seed comparisons, "
        "final audit, and quality claims remain gated. No ternary conversion or "
        "Recovery LoRA is performed."
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


@app.command("freeze-teacher-v1-corpus")
def freeze_teacher_v1_corpus(
    train_catalog_path: Path = Path("manifests/teacher-v1-training-corpus.yaml"),
    heldout_catalog_path: Path = Path("manifests/teacher-v1-heldout-corpus.yaml"),
    legacy_corpus_dir: Path = Path("data/processed/durable-teacher-v0"),
    output_dir: Path = Path("data/processed/durable-teacher-v1"),
    sealed_audit_dir: Path = Path("data/sealed/durable-teacher-v1"),
    seed: int = 17,
    max_records_per_source: int = typer.Option(8192, min=1),
) -> None:
    """Freeze disjoint Teacher v1 train/validation and an isolated sealed audit split."""
    import uuid

    from .corpus import filter_previously_seen_records, source_asset_identity
    from .dataset import check_train_eval_splits, corpus_statistics, sha256_file

    if seed != 17:
        raise click.ClickException("Teacher v1 corpus seed is immutable at 17")
    output_dir = output_dir.resolve()
    sealed_audit_dir = sealed_audit_dir.resolve()
    legacy_corpus_dir = legacy_corpus_dir.resolve()
    if output_dir.exists() or sealed_audit_dir.exists():
        raise click.ClickException(
            "Teacher v1 corpus destinations already exist; refusing overwrite"
        )
    if output_dir == sealed_audit_dir or output_dir in sealed_audit_dir.parents:
        raise click.ClickException("processed and sealed-audit destinations must be separate")
    legacy_paths = {
        "train": legacy_corpus_dir / "train.jsonl",
        "validation": legacy_corpus_dir / "validation.jsonl",
        "evaluation": legacy_corpus_dir / "eval.jsonl",
    }
    absent = [str(path) for path in legacy_paths.values() if not path.is_file()]
    if absent:
        raise click.ClickException(f"legacy Teacher v0 corpora are required: {absent}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    sealed_audit_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=".teacher-v1-source-pool-", dir=output_dir.parent)
    )
    processed_build = Path(
        tempfile.mkdtemp(prefix=".teacher-v1-build-", dir=output_dir.parent)
    )
    audit_build = Path(
        tempfile.mkdtemp(prefix=".teacher-v1-audit-build-", dir=sealed_audit_dir.parent)
    )

    def read_examples(path: Path) -> list[DecisionExample]:
        with path.open(encoding="utf-8") as handle:
            return [
                DecisionExample.model_validate_json(line)
                for line in handle
                if line.strip()
            ]

    def write_examples(path: Path, examples: list[DecisionExample]) -> None:
        path.write_text(
            "".join(item.model_dump_json() + "\n" for item in examples), encoding="utf-8"
        )

    try:
        freeze_corpus(
            train_catalog_path=train_catalog_path,
            eval_catalog_path=heldout_catalog_path,
            output_dir=stage_dir,
            seed=seed,
            max_records_per_source=max_records_per_source,
        )
        legacy = {name: read_examples(path) for name, path in legacy_paths.items()}
        previously_observed = [item for rows in legacy.values() for item in rows]
        stage_train = read_examples(stage_dir / "train.jsonl")
        stage_validation = read_examples(stage_dir / "validation.jsonl")
        stage_audit = read_examples(stage_dir / "eval.jsonl")

        new_train, train_exclusion = filter_previously_seen_records(
            stage_train, previously_observed
        )
        train = [*legacy["train"], *new_train]
        validation, validation_exclusion = filter_previously_seen_records(
            stage_validation, [*previously_observed, *train]
        )
        audit, audit_exclusion = filter_previously_seen_records(
            stage_audit, [*previously_observed, *train, *validation]
        )
        modality_names = {"text", "image", "audio", "video"}
        for name, rows in (("train", train), ("validation", validation), ("sealed_audit", audit)):
            present = {item.modality for item in rows}
            missing = sorted(modality_names - present)
            if missing:
                raise click.ClickException(
                    f"{name} split has no examples for modalities: {missing}"
                )

        pairwise = {
            "train_validation": check_train_eval_splits(train, validation),
            "train_sealed_audit": check_train_eval_splits(train, audit),
            "validation_sealed_audit": check_train_eval_splits(validation, audit),
        }
        stage_manifest = json.loads(
            (stage_dir / "corpus-manifest.json").read_text(encoding="utf-8")
        )

        def source_split_statistics(examples: list[DecisionExample]) -> dict[str, dict[str, int]]:
            buckets: dict[str, dict[str, object]] = {}
            for item in examples:
                key = f"{item.source}:{item.split}"
                bucket = buckets.setdefault(key, {"records": 0, "assets": set()})
                bucket["records"] = int(bucket["records"]) + 1
                assets = bucket["assets"]
                assert isinstance(assets, set)
                assets.add(source_asset_identity(item))
            return {
                key: {"records": int(value["records"]), "unique_asset_groups": len(value["assets"])}
                for key, value in sorted(buckets.items())
            }

        validation_source_stats = source_split_statistics(validation)
        audit_source_stats = source_split_statistics(audit)
        new_train_source_stats = source_split_statistics(new_train)
        train_source_coverage = {}
        train_source_shortages = []
        for source in stage_manifest["train"]["sources"]:
            key = f"{source['dataset_id']}:{source['split']}"
            selected_source = new_train_source_stats.get(
                key, {"records": 0, "unique_asset_groups": 0}
            )
            train_source_coverage[key] = {
                "candidate_records": source["records"],
                "previously_unseen_records": selected_source["records"],
                "previously_unseen_asset_groups": selected_source["unique_asset_groups"],
                "records_removed_by_legacy_gates": max(
                    0, source["records"] - selected_source["records"]
                ),
            }
            if selected_source["records"] == 0:
                train_source_shortages.append(
                    {
                        "source_split": key,
                        "candidate_records": source["records"],
                        "reason": "no previously unseen training asset groups remained",
                    }
                )
        heldout_source_coverage = {}
        source_shortages = []
        for source in stage_manifest["evaluation"]["sources"]:
            key = f"{source['dataset_id']}:{source['split']}"
            validation_source = validation_source_stats.get(
                key, {"records": 0, "unique_asset_groups": 0}
            )
            audit_source = audit_source_stats.get(
                key, {"records": 0, "unique_asset_groups": 0}
            )
            role = source["heldout_partition"]
            heldout_source_coverage[key] = {
                "heldout_role": role,
                "candidate_records": source["records"],
                "validation_records": validation_source["records"],
                "validation_unique_asset_groups": validation_source["unique_asset_groups"],
                "sealed_audit_records": audit_source["records"],
                "sealed_audit_unique_asset_groups": audit_source["unique_asset_groups"],
                "records_removed_by_legacy_or_cross_split_gates": max(
                    0,
                    source["records"]
                    - validation_source["records"]
                    - audit_source["records"],
                ),
            }
            expected = {
                "validation": (validation_source, "validation"),
                "evaluation": (audit_source, "sealed_audit"),
                "split": None,
            }
            expected_splits = (
                [("validation", validation_source), ("sealed_audit", audit_source)]
                if role == "split"
                else [(expected[role][1], expected[role][0])]
            )
            for split_name, split_stats in expected_splits:
                if split_stats["records"] == 0:
                    source_shortages.append(
                        {
                            "source_split": key,
                            "missing_partition": split_name,
                            "candidate_records": source["records"],
                            "reason": "no previously unused asset groups survived the gates",
                        }
                    )
        train_path = processed_build / "train.jsonl"
        validation_path = processed_build / "validation.jsonl"
        audit_path = audit_build / "sealed_audit.jsonl"
        write_examples(train_path, train)
        write_examples(validation_path, validation)
        write_examples(audit_path, audit)
        audit_hash = sha256_file(audit_path)
        source_revisions = {}
        for split_name in ("train", "evaluation"):
            for source in stage_manifest[split_name]["sources"]:
                source_revisions[
                    f"{source['dataset_id']}:{source['split']}"
                ] = {
                    "revision": source["source_revision"],
                    "manifest_sha256": source["manifest_sha256"],
                }

        audit_manifest = {
            "schema_version": 1,
            "teacher_id": "tiny-omni-decision-teacher-v1",
            "split": "sealed_audit",
            "seed": seed,
            "immutable_seed": True,
            "source_revisions": source_revisions,
            "train_catalog_sha256": sha256_file(train_catalog_path),
            "heldout_catalog_sha256": sha256_file(heldout_catalog_path),
            "legacy_corpus_sha256": {
                name: sha256_file(path) for name, path in legacy_paths.items()
            },
            "sealed_audit_path": audit_path.name,
            "sealed_audit_sha256": audit_hash,
            "sealed_audit_statistics": corpus_statistics(audit),
            "source_coverage": heldout_source_coverage,
            "source_shortages": source_shortages,
            "excluded_previously_seen": audit_exclusion,
            "pairwise_overlap": pairwise,
            "content_fingerprint_algorithm": stage_manifest["content_fingerprint_algorithm"],
            "media_identity_policy": (
                "sha256 for materialized assets; normalized source URI otherwise"
            ),
            "sealed_from_iterative_training": True,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        audit_manifest_path = audit_build / "sealed-audit-manifest.json"
        audit_manifest_path.write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")

        corpus_manifest = {
            "schema_version": 1,
            "status": "frozen_teacher_v1_train_validation_sealed_audit",
            "seed": seed,
            "immutable_seed": True,
            "source_revisions": source_revisions,
            "base_model_manifest_sha256": stage_manifest["manifest_sha256"],
            "train_catalog_sha256": sha256_file(train_catalog_path),
            "heldout_catalog_sha256": sha256_file(heldout_catalog_path),
            "legacy_corpus_sha256": {
                name: sha256_file(path) for name, path in legacy_paths.items()
            },
            "train": {
                "path": "train.jsonl",
                **corpus_statistics(train),
                "sha256": sha256_file(train_path),
                "previously_unseen_additions": corpus_statistics(new_train),
                "source_coverage": train_source_coverage,
                "source_shortages": train_source_shortages,
                "excluded_previously_seen": train_exclusion,
            },
            "validation": {
                "path": "validation.jsonl",
                **corpus_statistics(validation),
                "sha256": sha256_file(validation_path),
                "source_coverage": validation_source_stats,
                "excluded_previously_seen": validation_exclusion,
            },
            "sealed_audit": {
                "manifest_path": "data/sealed/durable-teacher-v1/sealed-audit-manifest.json",
                "audit_sha256": audit_hash,
                "audit_manifest_sha256": sha256_file(audit_manifest_path),
            },
            "sealed_audit_sha256": audit_hash,
            "pairwise_overlap": pairwise,
            "source_coverage": heldout_source_coverage,
            "source_shortages": source_shortages,
            "train_source_shortages": train_source_shortages,
            "corpus_pair_sha256": hashlib.sha256(
                (sha256_file(train_path) + sha256_file(validation_path) + audit_hash).encode()
            ).hexdigest(),
            "experiment_iteration_may_load": ["train.jsonl", "validation.jsonl"],
            "sealed_audit_may_load_only_after_selection_freeze": True,
            "stage_candidate_pool_manifest_sha256": sha256_file(
                stage_dir / "corpus-manifest.json"
            ),
            "build_id": uuid.uuid4().hex,
        }
        (processed_build / "corpus-manifest.json").write_text(
            json.dumps(corpus_manifest, indent=2), encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        sealed_audit_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists() or sealed_audit_dir.exists():
            raise click.ClickException("Teacher v1 corpus destination appeared during freeze")
        processed_build.rename(output_dir)
        try:
            audit_build.rename(sealed_audit_dir)
        except Exception:
            shutil.rmtree(output_dir)
            raise
    except Exception as exc:
        raise click.ClickException(f"Teacher v1 corpus freeze failed: {exc}") from exc
    finally:
        for temporary in (stage_dir, processed_build, audit_build):
            if temporary.exists():
                shutil.rmtree(temporary)

    console.print(json.dumps(corpus_manifest, indent=2))


@app.command("train-decision")
def train_decision(
    validation_manifest: Annotated[
        Path, typer.Option(help="Independent corpus used for checkpoint selection.")
    ],
    train_manifest: Path = Path("data/processed/durable-teacher-v1/train.jsonl"),
    config: Path = Path("configs/decision/teacher_v1.yaml"),
    output: Path = Path(
        "artifacts/tiny-omni-decision-teacher-v1/candidates/seed17-rank16-512-balanced"
    ),
    reference_adapter: Path | None = Path("artifacts/tiny-omni-decision-teacher-v0/best"),
    seed: int | None = None,
    max_train_examples: int | None = typer.Option(None, min=1),
    max_steps: int | None = typer.Option(None, min=1),
    gradient_accumulation_steps: int | None = typer.Option(None, min=1),
    checkpoint_interval: int | None = typer.Option(None, min=1),
    evaluation_interval: int | None = typer.Option(None, min=1),
    resume_from: Path | None = None,
    modalities: str = typer.Option("text,image,audio,video"),
    tiny_overfit: bool = False,
) -> None:
    """Train and select a candidate using train plus independent validation only."""
    from .trainer import run_training

    selected_modalities = {item.strip() for item in modalities.split(",") if item.strip()}
    try:
        result = run_training(
            train_path=train_manifest,
            validation_path=validation_manifest,
            config_path=config,
            output_dir=output,
            reference_adapter_path=reference_adapter,
            seed_override=seed,
            max_train_examples=max_train_examples,
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
                "validation_macro": result["validation_macro_metrics"],
                "validation_teacher_v0": result["validation_teacher_v0_metrics"],
                "validation_teacher_v1_candidate": result[
                    "validation_teacher_v1_candidate_metrics"
                ],
                "best_checkpoint_step": result["best_checkpoint_step"],
                "skipped_train": result["skipped_train_examples"],
                "sample_accounting": result["sample_accounting"],
                "max_allocated_vram_bytes": result["max_allocated_vram_bytes"],
                "checkpoint_reload_verified": result["checkpoint_reload_verified"],
            },
            indent=2,
        )
    )


@app.command("freeze-teacher-selection")
def freeze_teacher_selection_command(
    candidate: Annotated[Path, typer.Option(..., "--candidate")],
    config_path: Annotated[Path, typer.Option(..., "--config-path")],
    train_path: Annotated[Path, typer.Option(..., "--train-path")],
    validation_path: Annotated[Path, typer.Option(..., "--validation-path")],
    audit_manifest_path: Path = Path(
        "data/sealed/durable-teacher-v1/sealed-audit-manifest.json"
    ),
    lock_path: Path = Path("artifacts/tiny-omni-decision-teacher-v1/selection-lock.json"),
) -> None:
    """Freeze the validation-selected candidate before the single audit evaluation."""
    from .corpus import file_sha256
    from .sealed_audit import artifact_sha256, freeze_teacher_selection

    candidate = candidate.resolve()
    metadata_path = candidate.parent / "run-metadata.json"
    if not metadata_path.is_file():
        raise click.ClickException("candidate run-metadata.json is missing beside its adapter")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_role") != "validation_selected_experiment_candidate":
        raise click.ClickException("only a validation-selected experiment candidate can be frozen")
    if metadata.get("seed") != 17:
        raise click.ClickException(
            "the sealed-audit candidate must use the predefined primary seed 17"
        )
    selected_adapter = (
        metadata_path.parent / str(metadata.get("best_adapter_path"))
    ).resolve()
    if candidate != selected_adapter:
        raise click.ClickException("candidate path must be the validation-selected best adapter")
    weights_path = candidate / "adapter_model.safetensors"
    if not weights_path.is_file() or metadata.get("best_adapter_sha256") != file_sha256(
        str(weights_path)
    ):
        raise click.ClickException("selected adapter weights do not match run metadata")
    expected_hashes = {
        "training_config_sha256": file_sha256(str(config_path)),
        "train_rows_sha256": file_sha256(str(train_path)),
        "validation_rows_sha256": file_sha256(str(validation_path)),
    }
    for key, actual in expected_hashes.items():
        if metadata.get(key) != actual:
            raise click.ClickException(f"selected candidate {key} does not match frozen input")
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    corpus_manifest_path = train_path.resolve().parent / "corpus-manifest.json"
    if not corpus_manifest_path.is_file():
        raise click.ClickException("frozen train corpus manifest is missing")
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    audit_hash = audit_manifest.get("sealed_audit_sha256")
    audit_manifest_hash = file_sha256(str(audit_manifest_path))
    if corpus_manifest.get("sealed_audit_sha256") != audit_hash:
        raise click.ClickException(
            "sealed audit manifest does not match the frozen corpus manifest"
        )
    if (
        corpus_manifest.get("sealed_audit", {}).get("audit_manifest_sha256")
        != audit_manifest_hash
    ):
        raise click.ClickException(
            "sealed audit manifest hash does not match the frozen corpus manifest"
        )
    sampling = metadata.get("sampling_mixture") or {}
    selection = {
        "teacher_id": "tiny-omni-decision-teacher-v1",
        "checkpoint_sha256": artifact_sha256(candidate),
        "config_sha256": expected_hashes["training_config_sha256"],
        "train_corpus_sha256": expected_hashes["train_rows_sha256"],
        "validation_corpus_sha256": expected_hashes["validation_rows_sha256"],
        "sealed_audit_sha256": audit_hash,
        "sealed_audit_manifest_sha256": audit_manifest_hash,
        "seed": int(metadata["seed"]),
        "best_step": int(metadata["best_checkpoint_step"]),
        "selection_rule": (
            "teacher-v1-validation-only: macro_nll + 0.2*macro_brier + "
            "0.1*macro_ece - 0.25*macro_accuracy - 0.25*minimum_modality_accuracy; "
            "checkpoint chosen by this score and configured early stopping"
        ),
        "sampling_policy": {
            "modality_weights": sampling.get("modality_weights", {}),
            "source_weights": sampling.get("source_weights", {}),
            "max_sample_repeats": sampling.get("max_sample_repeats"),
            "consumed_by_modality_source": sampling.get("consumed_by_modality_source", {}),
        },
        "base_repo_id": metadata.get("base_repo_id"),
        "base_revision": metadata.get("base_revision"),
        "best_validation_metrics": metadata.get("validation_best_metrics"),
        "candidate_metadata_sha256": file_sha256(str(metadata_path)),
        "candidate_path": str(candidate),
    }
    try:
        frozen = freeze_teacher_selection(lock_path, selection)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Could not freeze teacher selection: {exc}") from exc
    console.print(json.dumps(frozen, indent=2))


@app.command("evaluate-sealed-audit")
def evaluate_sealed_audit(
    candidate: Annotated[Path, typer.Option(..., "--candidate")],
    config_path: Annotated[Path, typer.Option(..., "--config-path")],
    train_path: Annotated[Path, typer.Option(..., "--train-path")],
    validation_path: Annotated[Path, typer.Option(..., "--validation-path")],
    audit_path: Path = Path("data/sealed/durable-teacher-v1/sealed_audit.jsonl"),
    audit_manifest_path: Path = Path(
        "data/sealed/durable-teacher-v1/sealed-audit-manifest.json"
    ),
    lock_path: Path = Path("artifacts/tiny-omni-decision-teacher-v1/selection-lock.json"),
    claim_path: Path = Path("data/sealed/durable-teacher-v1/sealed-audit-claim.json"),
    result_path: Path = Path("data/sealed/durable-teacher-v1/sealed-audit-result.json"),
    model_manifest_path: Path = Path("manifests/base-model.example.yaml"),
) -> None:
    """Evaluate exactly once after hashes and the candidate selection are frozen."""
    from .corpus import file_sha256, macro_metrics
    from .sealed_audit import artifact_sha256, claim_sealed_audit_evaluation_once

    candidate = candidate.resolve()
    lock_path = lock_path.resolve()
    audit_path = audit_path.resolve()
    audit_manifest_path = audit_manifest_path.resolve()
    if result_path.exists():
        raise click.ClickException(
            "sealed audit result already exists; a second attempt is forbidden"
        )
    if not lock_path.is_file():
        raise click.ClickException("freeze-teacher-selection must run before audit evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    metadata_path = candidate.parent / "run-metadata.json"
    if not metadata_path.is_file():
        raise click.ClickException("candidate run-metadata.json is missing beside its adapter")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if lock.get("candidate_metadata_sha256") != file_sha256(str(metadata_path)):
        raise click.ClickException("candidate metadata differs from the frozen selection lock")
    if metadata.get("artifact_role") != "validation_selected_experiment_candidate":
        raise click.ClickException("locked candidate is not a validation-selected experiment")
    selected_adapter = (metadata_path.parent / str(metadata.get("best_adapter_path"))).resolve()
    if candidate != selected_adapter:
        raise click.ClickException("candidate path is not the locked best adapter")
    weights_path = candidate / "adapter_model.safetensors"
    if not weights_path.is_file() or metadata.get("best_adapter_sha256") != file_sha256(
        str(weights_path)
    ):
        raise click.ClickException("candidate adapter weights differ from run metadata")
    actual_hashes = {
        "checkpoint_sha256": artifact_sha256(candidate),
        "config_sha256": file_sha256(str(config_path)),
        "train_corpus_sha256": file_sha256(str(train_path)),
        "validation_corpus_sha256": file_sha256(str(validation_path)),
    }
    for key, actual in actual_hashes.items():
        if lock.get(key) != actual:
            raise click.ClickException(f"{key} differs from the frozen selection lock")
    if lock.get("sealed_audit_manifest_sha256") != file_sha256(
        str(audit_manifest_path)
    ):
        raise click.ClickException(
            "sealed audit manifest differs from the frozen selection lock"
        )
    if (
        metadata.get("base_repo_id") != lock.get("base_repo_id")
        or metadata.get("base_revision") != lock.get("base_revision")
    ):
        raise click.ClickException("candidate base model differs from the frozen selection lock")
    if metadata.get("best_checkpoint_step") != lock.get("best_step"):
        raise click.ClickException(
            "candidate checkpoint step differs from the frozen selection lock"
        )
    try:
        claim = claim_sealed_audit_evaluation_once(
            lock_path, audit_manifest_path, audit_path, claim_path.resolve()
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Sealed audit access denied: {exc}") from exc

    try:
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        from .corpus import macro_metrics
        from .schema import BaseModelManifest
        from .trainer import _evaluate, _read_examples
        from .training import decision_training_config

        base_manifest = BaseModelManifest.model_validate(load_structured_file(model_manifest_path))
        if (
            base_manifest.repo_id != lock.get("base_repo_id")
            or base_manifest.revision != lock.get("base_revision")
        ):
            raise ValueError("base model manifest differs from the frozen selection")
        processor = AutoProcessor.from_pretrained(
            base_manifest.processor_repo_id or base_manifest.repo_id,
            revision=base_manifest.processor_revision,
        )
        base_model = AutoModelForMultimodalLM.from_pretrained(
            base_manifest.repo_id,
            revision=base_manifest.revision,
            dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, candidate, is_trainable=False).eval()
        examples = _read_examples(audit_path)
        required_modalities = {"text", "image", "audio", "video"}
        present_modalities = {item.modality for item in examples}
        if not examples or present_modalities != required_modalities:
            raise ValueError(
                f"sealed audit modality coverage mismatch: {sorted(present_modalities)}"
            )
        config = decision_training_config(load_structured_file(config_path))
        data_root = audit_path.parent.parent.parent
        metrics, _ = _evaluate(
            model, processor, examples, data_root=data_root, config=config
        )
        metrics["macro"] = macro_metrics(metrics)
        result = {
            "schema_version": 1,
            "status": "completed",
            "teacher_id": lock["teacher_id"],
            "checkpoint_sha256": claim["checkpoint_sha256"],
            "selection_lock_sha256": claim["selection_lock_sha256"],
            "sealed_audit_sha256": claim["sealed_audit_sha256"],
            "record_count": len(examples),
            "metrics": metrics,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "teacher_id": lock.get("teacher_id"),
            "sealed_audit_sha256": claim["sealed_audit_sha256"],
            "failure": f"{type(exc).__name__}: {exc}",
            "attempted_at_utc": claim["attempted_at_utc"],
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with result_path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(failure, handle, indent=2)
                handle.write("\n")
        except FileExistsError:
            pass
        raise click.ClickException(
            f"Sealed audit evaluation failed after one-time claim: {exc}"
        ) from exc
    console.print(json.dumps(result, indent=2))


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
