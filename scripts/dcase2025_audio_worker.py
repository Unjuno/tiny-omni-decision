"""Persistent frozen EfficientAT feature worker for DCASE stereo audio."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EfficientATWorker:
    def __init__(self, source: Path, checkpoint: Path, dependency_root: Path):
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("EfficientAT checkpoint SHA-256 mismatch")
        import subprocess

        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != SOURCE_COMMIT:
            raise ValueError("EfficientAT source commit mismatch")
        sys.path.insert(0, str(dependency_root.resolve()))
        sys.path.insert(0, str(source.resolve()))
        import torch
        import torchaudio
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        if torch.__version__ != "2.6.0+cu124" or torchaudio.__version__ != "2.6.0+cu124":
            raise ValueError(
                "EfficientAT runtime differs from the measured PyTorch/torchaudio pair"
            )
        self.torch = torch
        self.torchaudio = torchaudio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with contextlib.redirect_stdout(sys.stderr):
            self.model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
            self.model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
            )
            self.model.eval().requires_grad_(False).to(self.device)
            self.mel = AugmentMelSTFT(fmax=15000).eval().to(self.device)

    def process(self, request: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        media_path = Path(str(request["media_npz"]))
        output_path = Path(str(request["output_npy"]))
        with np.load(media_path, allow_pickle=False) as archive:
            audio = archive["audio"]
            valid = archive["valid"]
            sample_ids = archive["sample_ids"].tolist()
        if (
            audio.ndim != 3
            or audio.shape[1:] != (2, 120000)
            or valid.shape != (len(audio),)
            or len(sample_ids) != len(audio)
        ):
            raise ValueError("unexpected DCASE audio batch shape or row count")

        torch = self.torch
        waveforms = torch.from_numpy(audio.reshape((-1, 120000)).copy()).to(self.device)
        waveforms = self.torchaudio.functional.resample(waveforms, 24000, 32000)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            _, features = self.model(self.mel(waveforms).unsqueeze(1))
        values = features.float().cpu().numpy().reshape((len(audio), 2, 384))
        values = values.reshape((len(audio), 768))
        if not np.isfinite(values).all():
            raise ValueError("EfficientAT returned non-finite features")
        np.save(output_path, values, allow_pickle=False)
        return {
            "status": "PASS",
            "batch_id": request.get("batch_id"),
            "sample_ids": sample_ids,
            "feature_shape": list(values.shape),
            "device": str(self.device),
            "valid_count": int(valid.sum()),
            "peak_allocated_vram_bytes": (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "wall_seconds": time.perf_counter() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    args = parser.parse_args()
    worker = EfficientATWorker(args.source, args.checkpoint, args.dependency_root)
    print(
        json.dumps(
            {
                "status": "READY",
                "source_commit": SOURCE_COMMIT,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "device": str(worker.device),
                "torch": worker.torch.__version__,
                "torchaudio": worker.torchaudio.__version__,
                "parameters": sum(parameter.numel() for parameter in worker.model.parameters()),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            response = worker.process(json.loads(line))
        except Exception as error:
            response = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
