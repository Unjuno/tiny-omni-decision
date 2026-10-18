"""Extract frozen EfficientAT and VideoPrism features for the fixed AV corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EFFICIENTAT_REVISION = "a425fdce92572e602a1d5634799bd9f1f2efa806"
EFFICIENTAT_CHECKPOINT_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
VIDEOPRISM_REVISION = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
VIDEOPRISM_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
EXPECTED_GROUPS = {"train": 126, "calibration": 38, "validation": 56}
REPO = Path(__file__).resolve().parents[1]
R1 = Path("C:/CodexArtifacts/pretrained-reuse-r1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_groups(inputs_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_path = inputs_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    groups_path = inputs_dir / "groups.jsonl"
    if result.get("status") != "PASS_FULL_DECODE_AND_NORMALIZED_CONTENT_SPLIT_GATES":
        raise ValueError("full decoded input and normalized split audit is required")
    if result.get("completed_groups") != sum(EXPECTED_GROUPS.values()):
        raise ValueError("all 220 fixed source groups are required")
    if result.get("artifact_hashes", {}).get("groups.jsonl") != sha256_file(groups_path):
        raise ValueError("prepared group manifest hash differs")
    rows = [json.loads(line) for line in groups_path.read_text(encoding="utf-8").splitlines()]
    expected_total = sum(EXPECTED_GROUPS.values())
    if len(rows) != expected_total or len({row["source_group"] for row in rows}) != expected_total:
        raise ValueError("prepared group identity/count differs")
    if {
        split: sum(row["split"] == split for row in rows) for split in EXPECTED_GROUPS
    } != EXPECTED_GROUPS:
        raise ValueError("prepared split group counts differ")
    return result, rows


def _load_encoder(modality: str, device: str, report: dict[str, Any]):
    import numpy as np

    if modality == "audio":
        import torch
        import torchaudio

        source = R1 / "efficientat-audit-v1/source"
        checkpoint = R1 / "efficientat-audit-v1/dymn04_as.pt"
        if sha256_file(checkpoint) != EFFICIENTAT_CHECKPOINT_SHA256:
            raise ValueError("pinned EfficientAT checkpoint changed")
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != EFFICIENTAT_REVISION:
            raise ValueError("pinned EfficientAT source revision changed")
        sys.path.insert(0, str(source))
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        torch.set_num_threads(4)
        model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        model.eval().requires_grad_(False).to(device)
        mel = AugmentMelSTFT().eval().to(device)
        report.update(
            model="EfficientAT DyMN-0.4 AudioSet",
            revision=revision,
            checkpoint_sha256=EFFICIENTAT_CHECKPOINT_SHA256,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            feature_shape=[384],
            torch=torch.__version__,
            torchaudio=torchaudio.__version__,
            preprocessing=(
                "decoded mono PCM 24 kHz; frozen evaluation mel; internal resample to 32 kHz"
            ),
        )

        def encode(inputs: dict[str, Any]):
            waveform = torch.from_numpy(inputs["audio"][:, 0].copy()).unsqueeze(0).to(device)
            waveform = torchaudio.functional.resample(waveform, 24000, 32000)
            with torch.inference_mode():
                _, feature = model(mel(waveform).unsqueeze(1))
            return feature.squeeze(0).float().cpu().numpy()

        def parameter_device():
            return str(next(model.parameters()).device)

        return encode, parameter_device

    if modality != "video":
        raise ValueError("modality must be audio or video")
    if device != "cpu":
        raise ValueError("VideoPrism extraction is pinned to its CPU runtime")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    checkpoint = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/path-a-modular-runtime-assembly-v4/components/image/videoprism-base-f16r288.npz"
    )
    source = R1 / "videoprism-base-r1/source"
    if sha256_file(checkpoint) != VIDEOPRISM_CHECKPOINT_SHA256:
        raise ValueError("pinned VideoPrism checkpoint changed")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != VIDEOPRISM_REVISION:
        raise ValueError("pinned VideoPrism source revision changed")
    sys.path.insert(0, str(source))
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    from videoprism import models as vp

    if jax.default_backend() != "cpu":
        raise RuntimeError("VideoPrism environment did not honor CPU-only extraction")
    model = vp.get_model("videoprism_public_v1_base")
    variables = vp.load_pretrained_weights(
        "videoprism_public_v1_base", checkpoint_path=str(checkpoint)
    )
    forward = jax.jit(lambda frames: model.apply(variables, frames, train=False))
    parameter_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(variables))
    report.update(
        model="Google VideoPrism-B",
        revision=revision,
        checkpoint_sha256=VIDEOPRISM_CHECKPOINT_SHA256,
        parameters=parameter_count,
        feature_shape=[768],
        jax=jax.__version__,
        tensorflow=tf.__version__,
        preprocessing=(
            "8 endpoint-inclusive 288x288 RGB frames, resize to 288 and divide by 255, "
            "uniform token mean"
        ),
        backend=jax.default_backend(),
    )

    def encode(inputs: dict[str, Any]):
        frames = np.asarray(tf.image.resize(inputs["video"], (288, 288)), dtype=np.float32) / 255
        output = forward(jnp.asarray(frames[None]))
        tokens = output[0] if isinstance(output, tuple) else output
        return np.asarray(tokens).mean(axis=1).squeeze(0).astype(np.float32)

    def parameter_device():
        return jax.default_backend()

    return encode, parameter_device


def extract(modality: str, device: str, inputs_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    _, rows = read_groups(inputs_dir)
    output_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "RUNNING",
        "modality": modality,
        "device": device,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "runner_sha256": sha256_file(Path(__file__)),
        "inputs_result_sha256": sha256_file(inputs_dir / "result.json"),
        "groups_manifest_sha256": sha256_file(inputs_dir / "groups.jsonl"),
        "started_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "training": False,
        "evaluation": False,
        "sealed_audit_opened": False,
        "optimizer_updates": 0,
        "groups": [],
    }
    start = time.perf_counter()
    try:
        if modality == "audio" and device == "cuda":
            import torch

            if not torch.cuda.is_available() or not os.getenv("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
                raise RuntimeError("CUDA feature extraction requires the HotUGPU reservation gate")
        encode, device_fn = _load_encoder(modality, device, result)
        result["actual_device"] = device_fn()
        if modality == "audio":
            if device not in {"cpu", "cuda"}:
                raise ValueError("audio device must be cpu or cuda")
        result["load_seconds"] = time.perf_counter() - start
        for index, row in enumerate(rows):
            input_path = inputs_dir / row["input_file"]
            if sha256_file(input_path) != row["input_sha256"]:
                raise ValueError("prepared model input changed")
            with np.load(input_path, allow_pickle=False) as data:
                if modality == "audio":
                    if data["audio"].ndim != 2 or data["audio"].shape[1] != 1:
                        raise ValueError("mono 24 kHz audio input shape differs")
                elif data["video"].shape != (8, 288, 288, 3):
                    raise ValueError("fixed eight-frame video input shape differs")
                feature = encode(data)
                repeated = encode(data)
            expected_shape = (384,) if modality == "audio" else (768,)
            if feature.shape != expected_shape:
                raise ValueError("encoder feature shape differs")
            delta = float(np.max(np.abs(feature - repeated)))
            if not np.isfinite(feature).all() or not np.allclose(
                feature, repeated, rtol=1e-5, atol=1e-6
            ):
                raise ValueError("feature is nonfinite or repeated extraction differs")
            filename = f"feature-{row['source_group']}.npy"
            feature_path = output_dir / filename
            np.save(feature_path, feature.astype(np.float32), allow_pickle=False)
            manifest_row = {
                "source_group": row["source_group"],
                "split": row["split"],
                "input_file": row["input_file"],
                "input_sha256": row["input_sha256"],
                "feature_file": filename,
                "feature_sha256": sha256_file(feature_path),
                "feature_shape": list(feature.shape),
                "repeat_max_abs_delta": delta,
            }
            result["groups"].append(manifest_row)
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "completed": index + 1,
                        "expected": len(rows),
                        "seconds": time.perf_counter() - start,
                        "last_source_group": row["source_group"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        (output_dir / "features.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["groups"]),
            encoding="utf-8",
        )
        result["status"] = "PASS_FROZEN_FEATURE_EXTRACTION_REPLAY"
    except Exception as error:
        result.update(status="FAILED", error=repr(error))
        raise
    finally:
        result.update(
            finished_utc=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.perf_counter() - start,
            completed=len(result["groups"]),
        )
        if (output_dir / "features.jsonl").exists():
            result["features_manifest_sha256"] = sha256_file(output_dir / "features.jsonl")
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("audio", "video"), required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.modality, args.device, args.inputs_dir, args.output_dir)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "modality", "completed", "elapsed_seconds")}
        )
    )


if __name__ == "__main__":
    main()
