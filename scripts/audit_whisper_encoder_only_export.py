"""Export and verify only the Whisper encoder used by Path A inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import WhisperConfig, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from tiny_omni_decision.audio_runtime import AUDIO_MODEL_ID, AUDIO_MODEL_REVISION


def extract_encoder_state(
    source_state: Mapping[str, torch.Tensor], *, prefix: str = "model.encoder."
) -> dict[str, torch.Tensor]:
    """Drop the unused Whisper decoder namespace without changing encoder tensors."""
    if not prefix or not prefix.endswith("."):
        raise ValueError("encoder prefix must be a non-empty dotted namespace")
    extracted = {
        name.removeprefix(prefix): tensor.contiguous()
        for name, tensor in source_state.items()
        if name.startswith(prefix)
    }
    if not extracted:
        raise ValueError("source checkpoint has no encoder tensors under the expected prefix")
    return extracted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_encoder(source_model_dir: Path, output_dir: Path) -> dict[str, object]:
    source_model_dir = source_model_dir.resolve(strict=True)
    source_weights = source_model_dir / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()

    source_state: dict[str, torch.Tensor] = {}
    with safe_open(source_weights, framework="pt", device="cpu") as source:
        for key in source.keys():
            if key.startswith("model.encoder."):
                source_state[key] = source.get_tensor(key)
    encoder_state = extract_encoder_state(source_state)

    config = WhisperConfig.from_pretrained(source_model_dir, local_files_only=True)
    encoder = WhisperEncoder(config).eval()
    load_result = encoder.load_state_dict(encoder_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError("encoder-only state did not load strictly")
    encoder_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    if encoder_parameter_count != sum(tensor.numel() for tensor in encoder_state.values()):
        raise ValueError("encoder parameter count differs from exported tensor count")

    model = WhisperModel.from_pretrained(
        source_model_dir, local_files_only=True, dtype=torch.float32
    ).eval()
    source_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    original_encoder_state = model.encoder.state_dict()
    if set(original_encoder_state) != set(encoder_state):
        raise ValueError("source encoder and exported tensor names differ")
    max_weight_delta = max(
        (original_encoder_state[name] - encoder_state[name]).abs().max().item()
        for name in encoder_state
    )
    torch.manual_seed(1701)
    input_features = torch.randn(1, config.num_mel_bins, 3000)
    with torch.inference_mode():
        original_output = model.encoder(input_features).last_hidden_state
        exported_output = encoder(input_features).last_hidden_state
    max_output_delta = (original_output - exported_output).abs().max().item()
    mean_output_delta = (original_output - exported_output).abs().mean().item()
    if max_weight_delta != 0 or max_output_delta != 0:
        raise ValueError("encoder-only model does not exactly reproduce source encoder")

    output_weights = output_dir / "encoder.safetensors"
    save_file(encoder_state, output_weights, metadata={"format": "pt"})
    shutil.copy2(source_model_dir / "config.json", output_dir / "config.json")
    preprocessor = source_model_dir / "preprocessor_config.json"
    if preprocessor.is_file():
        shutil.copy2(preprocessor, output_dir / "preprocessor_config.json")
    total_file_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    source_file_bytes = source_weights.stat().st_size
    report: dict[str, object] = {
        "status": "complete_exact_encoder_only_export",
        "source_model_dir": str(source_model_dir),
        "source_model_id": AUDIO_MODEL_ID,
        "source_model_revision": AUDIO_MODEL_REVISION,
        "source_weights_sha256": sha256_file(source_weights),
        "source_weights_bytes": source_file_bytes,
        "source_checkpoint_logical_parameters": source_parameter_count,
        "inference_encoder_parameters": encoder_parameter_count,
        "unused_decoder_parameter_count": source_parameter_count - encoder_parameter_count,
        "encoder_state_sha256": sha256_file(output_weights),
        "encoder_state_bytes": output_weights.stat().st_size,
        "output_total_bytes": total_file_bytes,
        "bytes_removed_from_model_weight_file": source_file_bytes - output_weights.stat().st_size,
        "max_encoder_weight_delta": max_weight_delta,
        "encoder_smoke_input_shape": list(input_features.shape),
        "encoder_smoke_max_output_delta": max_output_delta,
        "encoder_smoke_mean_output_delta": mean_output_delta,
        "elapsed_seconds": time.perf_counter() - started,
        "device": "cpu",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "smoke_input": "seeded synthetic log-mel tensor; no dataset audio was loaded",
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "audit-result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = export_encoder(args.source_model_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
