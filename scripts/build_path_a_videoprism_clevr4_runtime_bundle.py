"""Build an immutable, task-scoped VideoPrism CLEVR-4 inference bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from tiny_omni_decision.videoprism_runtime import (
    CHECKPOINT_SHA256,
    CLEVR4_TAXONOMIES,
    MODEL_REVISION,
    READOUT_SHA256,
)

TEXT_REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
DEFAULT_ROOT = Path(r"C:\CodexArtifacts\pretrained-reuse-r2")
DEFAULT_OUTPUT = DEFAULT_ROOT / "path-a-image-videoprism-clevr4-runtime-bundle-v2"
DEFAULT_MODELS = DEFAULT_ROOT / "models"
DEFAULT_RUN = DEFAULT_ROOT / "image-videoprism-clevr4-full-seed17"
DEFAULT_SAMPLE = DEFAULT_ROOT / "clevr4-full-coverage-seed17"
DEFAULT_CHECKPOINT = Path(r"C:\CodexArtifacts\pretrained-reuse-r1\videoprism-base-f16r288.npz")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> tuple[list[dict[str, object]], int]:
    rows = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        rows.append(
            {"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": sha256_file(path)}
        )
        total += size
    return rows, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    models_dir = args.models_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve(strict=True)
    sample_dir = args.sample_dir.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime bundle: {output}")
    report_path = run_dir / "run-report.json"
    readout_path = run_dir / "best-readout.pt"
    text_dir = models_dir / "text-minilm-en"
    for path in (
        report_path,
        readout_path,
        text_dir / "model.safetensors",
        checkpoint,
        sample_dir / "val-image-manifest.jsonl",
        sample_dir / "val-labels.jsonl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("pinned VideoPrism checkpoint hash mismatch")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("encoder_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("source run used a different VideoPrism checkpoint")
    if source_report.get("candidate_text_revision") != TEXT_REVISION:
        raise ValueError("source run used an unexpected MiniLM revision")
    if sha256_file(readout_path) != READOUT_SHA256:
        raise ValueError("validation-selected readout hash mismatch")
    if source_report.get("sealed_audit_loaded") or source_report.get("test_split_loaded"):
        raise ValueError("source run unexpectedly loaded a sealed/test split")

    tokenizer = AutoTokenizer.from_pretrained(text_dir, local_files_only=True)
    text_model = AutoModel.from_pretrained(
        text_dir, local_files_only=True, dtype=torch.float32
    ).eval()
    prompts = {
        f"{taxonomy}__{option}": f"Image attribute: {taxonomy}. Candidate class: {option}."
        for taxonomy, options in CLEVR4_TAXONOMIES.items()
        for option in options
    }
    keys = list(prompts)
    encoded = tokenizer(
        [prompts[key] for key in keys],
        max_length=64,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = text_model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).float().numpy()
    if embeddings.shape != (40, 384) or not np.isfinite(embeddings).all():
        raise ValueError("MiniLM option embedding export has invalid shape or values")
    state = torch.load(readout_path, map_location="cpu", weights_only=True)
    if set(state) != {"network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"}:
        raise ValueError("saved readout parameter layout differs from the pinned MLP")
    if sum(value.numel() for value in state.values()) != 147_713:
        raise ValueError("saved readout parameter count mismatch")

    output.mkdir(parents=True)
    shutil.copy2(checkpoint, output / "videoprism-base-f16r288.npz")
    np.savez_compressed(
        output / "option-embeddings.npz",
        **{key: value for key, value in zip(keys, embeddings, strict=True)},
    )
    np.savez(
        output / "readout.npz",
        linear1_weight=state["network.0.weight"].numpy(),
        linear1_bias=state["network.0.bias"].numpy(),
        linear2_weight=state["network.2.weight"].numpy(),
        linear2_bias=state["network.2.bias"].numpy(),
    )
    runtime_dir = output / "runtime" / "tiny_omni_decision"
    runtime_dir.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).resolve().parents[1] / "src/tiny_omni_decision/videoprism_runtime.py",
        runtime_dir,
    )
    (runtime_dir / "__init__.py").write_text(
        '"""Task-scoped VideoPrism runtime bundle."""\n', encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# VideoPrism CLEVR-4 runtime bundle\n\n"
        "Task-scoped development runtime for static CLEVR-4 taxonomy choices. "
        "It repeats each image across eight VideoPrism frame slots. It does not "
        "support arbitrary open-vocabulary questions, temporal video decisions, "
        "or general image understanding. The pinned VideoPrism checkpoint is "
        "457 MB, so this bundle exceeds the roadmap's provisional 200 MB target.\n\n"
        "The runtime requires the pinned VideoPrism upstream source plus JAX and "
        "TensorFlow. The readout and 40 fixed MiniLM option embeddings are stored "
        "locally; MiniLM is not loaded at inference. See bundle-manifest.json.\n",
        encoding="utf-8",
    )
    files, byte_count = tree_manifest(output)
    manifest = {
        "schema_version": 1,
        "bundle_id": "path-a-image-videoprism-clevr4-runtime-v0",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip(),
        "scope": (
            "static CLEVR-4 taxonomy choices only; not temporal video or general image decisions"
        ),
        "encoder": {
            "model_id": "google/videoprism-base-f16r288",
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "parameters": 114_365_184,
            "license": "CC-BY-4.0 for released model checkpoints; preserve attribution",
            "frames": 8,
            "preprocessing": (
                "RGB resize-short-side-288 bilinear, center crop, repeat image over frames, [0,1]"
            ),
            "pooling": "mean over all VideoPrism output tokens to 768 dimensions",
        },
        "option_embeddings": {
            "model_id": "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "revision": TEXT_REVISION,
            "license": "Apache-2.0",
            "count": len(keys),
            "dimension": 384,
            "export_sha256": sha256_file(output / "option-embeddings.npz"),
            "parameters_not_resident_at_inference": 0,
        },
        "readout": {
            "source_sha256": READOUT_SHA256,
            "export_sha256": sha256_file(output / "readout.npz"),
            "parameters": 147_713,
            "selection": "source run minimum NLL on prior 512-image validation selector, epoch 8",
        },
        "source_run_report_sha256": sha256_file(report_path),
        "source_validation_image_manifest_sha256": sha256_file(
            sample_dir / "val-image-manifest.jsonl"
        ),
        "source_validation_labels_sha256": sha256_file(sample_dir / "val-labels.jsonl"),
        "environment_at_build": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
        },
        "runtime_environment": (
            "separate JAX/Flax and TensorFlow profile; see R2 probe log for exact versions"
        ),
        "total_parameters": 114_512_897,
        "bundle_bytes_excluding_manifest": byte_count,
        "package_target_bytes": 200_000_000,
        "package_target_met": byte_count <= 200_000_000,
        "files": files,
        "limitations": [
            "synthetic CLEVR-4 only; image repeated across eight temporal slots",
            "fixed taxonomies and vocabulary; arbitrary candidate text is unsupported",
            "checkpoint-selected validation is development data, not a blind audit",
            "JAX/Flax and TensorFlow dependencies are external and not vendored",
            "package exceeds the provisional 200 MB target",
        ],
    }
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "bundle": str(output),
                "payload_bytes": byte_count,
                "manifest": str(output / "bundle-manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
