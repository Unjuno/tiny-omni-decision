"""Read-only bridge to the existing pinned Gemma 4 Teacher pipeline."""
from __future__ import annotations

from pathlib import Path

from .data import Record, digest, file_hash, media_path, write_cache


def create_teacher_cache(
    *, examples_path: Path, base_manifest: Path, adapter_dir: Path, data_root: Path,
    output: Path, role: str, device: str = "cuda", max_length: int = 1024,
    allow_download: bool = False,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from ..dataset import audit_license
    from ..io import load_structured_file
    from ..schema import BaseModelManifest, DecisionExample
    from ..trainer import _forward_decision

    manifest = BaseModelManifest.model_validate(load_structured_file(base_manifest))
    adapter_files = [adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors"]
    hashes = {p.name: file_hash(p) for p in adapter_files}
    teacher_id = digest({"base": manifest.model_dump(), "adapter": hashes,
                         "input_pipeline": "existing_gemma4_processor_v1",
                         "max_length": max_length, "video_num_frames": 4})
    source_hash = file_hash(examples_path)
    examples = [DecisionExample.model_validate_json(line) for line in
                examples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError("Teacher cache needs nonempty unique examples")
    if role not in {"train", "validation", "evaluation"}:
        raise ValueError("invalid cache role")
    for example in examples:
        if role == "train":
            p = example.provenance
            policy, reason = audit_license(
                p.license, commercial_use=p.commercial_use,
                derivative_model_training_allowed=p.derivative_model_training_allowed,
                redistribution_allowed=p.redistribution_allowed,
                media_redistribution_allowed=p.media_redistribution_allowed,
                has_media=bool(example.media), trust_status=p.trust_status,
            )
            if policy != "ALLOW":
                raise ValueError(f"training sample is not ALLOW: {example.id}: {reason}")
    base = AutoModelForMultimodalLM.from_pretrained(
        manifest.repo_id, revision=manifest.revision, torch_dtype=torch.bfloat16,
        trust_remote_code=False, local_files_only=not allow_download,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        manifest.processor_repo_id or manifest.repo_id, revision=manifest.processor_revision,
        trust_remote_code=False, local_files_only=not allow_download,
    )
    teacher = PeftModel.from_pretrained(
        base, str(adapter_dir), is_trainable=False, local_files_only=True
    )
    teacher.requires_grad_(False)
    teacher.eval()
    records = []
    media_checksums = {}
    with torch.no_grad():
        for example in examples:
            inputs = {"text": f"{example.state}\nQuestion: {example.question}"}
            row_hashes = {}
            for media in example.media:
                if not media.path:
                    raise ValueError("cache requires already materialized local media")
                if media.path not in media_checksums:
                    media_checksums[media.path] = file_hash(media_path(data_root, media.path))
                actual = media_checksums[media.path]
                if media.sha256 and media.sha256 != actual:
                    raise ValueError(f"media checksum mismatch: {media.path}")
                row_hashes[media.path] = actual
                if media.kind in inputs:
                    raise ValueError("existing Teacher bridge supports one media item per modality")
                inputs[media.kind] = media.path
            logits, target = _forward_decision(
                teacher, processor, example, data_root=data_root, max_sequence_length=max_length
            )
            def normalized(value: str) -> str:
                return " ".join(value.casefold().split())
            content_id = digest({
                "text": normalized(inputs["text"]),
                "options": sorted(normalized(o) for o in example.options),
                "media": sorted(row_hashes.values()),
            })
            group_id = digest([example.source, example.source_revision, example.source_record_id])
            records.append(Record(example.id, group_id, content_id, example.modality, inputs,
                                  example.options, target, logits.float().cpu().tolist(),
                                  row_hashes))
    if hashes != {p.name: file_hash(p) for p in adapter_files}:
        raise ValueError("Teacher adapter changed while caching; use a frozen checkpoint")
    if source_hash != file_hash(examples_path):
        raise ValueError("source examples changed while caching")
    write_cache(output, records, role=role, teacher_id=teacher_id, source_sha256=source_hash)
    return {"path": str(output), "role": role, "count": len(records), "teacher_id": teacher_id}
