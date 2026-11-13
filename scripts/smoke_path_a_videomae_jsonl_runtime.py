"""Verify the pinned VideoMAE Path A readout through the real JSONL worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES
from tiny_omni_decision.videomae_runtime import (
    ENCODER_CHECKPOINT_SHA256,
    ENCODER_SOURCE_SHA256,
    READOUT_SHA256,
    TaxonomyConditionedScorer,
    sha256_file,
)

TRAIN_MANIFEST_SHA256 = "e48687af785a74864d2d4eb6e632a50d381d821a3ade8c2227eda7624647f410"
FEATURE_MANIFEST_SHA256 = "019cb10f6943c1e6b58ff671215e657d310a65cd53162538c4e3c3789d0694f6"
FEATURES_SHA256 = "7b6cef6b6451eb7cafd8d918e34fe527fac20fb658ae79c692cac28c9130e4b1"
INPUTS_RECEIPT_SHA256 = "350e7d4dec9f113d941f1415dc3bda4e556315ea57baffb89fe2efa62717a16b"
BEST_READOUT_SHA256 = READOUT_SHA256
ASSEMBLY_MANIFEST_SHA256 = "79fbca05e6062b66fe6ffb8a831d6c635edffe361c76ea4ce1574f0736ae9519"
ARCHIVE_BYTES = 36_057_238_315


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_expected_probabilities(
    feature_path: Path,
    feature_manifest: Path,
    readout: Path,
    sample_id: str,
    expected_media_sha256: str,
    taxonomy: str,
    options: list[str],
) -> list[float]:
    manifest_rows = _jsonl(feature_manifest)
    if sha256_file(feature_path) != FEATURES_SHA256:
        raise ValueError("frozen VideoMAE feature cache hash mismatch")
    with np.load(feature_path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str).tolist()
        hashes = archive["media_sha256"].astype(str).tolist()
        features = archive["features"]
    if len(ids) != len(set(ids)) or ids != [str(row["id"]) for row in manifest_rows]:
        raise ValueError("cached feature IDs differ from the pinned manifest")
    if hashes != [str(row["media_sha256"]) for row in manifest_rows]:
        raise ValueError("cached feature media hashes differ from the pinned manifest")
    try:
        index = ids.index(sample_id)
    except ValueError as error:
        raise ValueError("selected train image is absent from the frozen feature cache") from error
    if hashes[index] != expected_media_sha256:
        raise ValueError("selected train media hash differs from its cached feature identity")
    model = TaxonomyConditionedScorer().cpu().eval()
    model.load_state_dict(torch.load(readout, map_location="cpu", weights_only=True), strict=True)
    task_index = list(CLEVR4_TAXONOMIES).index(taxonomy)
    option_indices = [list(CLEVR4_TAXONOMIES[taxonomy]).index(value) for value in options]
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(np.asarray(features[index], dtype=np.float32).copy()).reshape(1, -1),
            torch.tensor([task_index], dtype=torch.long),
        )[0, option_indices]
        return torch.softmax(logits, dim=-1).tolist()


def run_smoke(
    *,
    archive: Path,
    train_manifest: Path,
    feature_manifest: Path,
    feature_path: Path,
    readout: Path,
    model_root: Path,
    dependency_root: Path,
    assembly: Path,
    videoprism_python: Path,
    videoprism_source: Path,
    output: Path,
    device: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    checks = {
        "train_media_manifest_sha256": (train_manifest, TRAIN_MANIFEST_SHA256),
        "feature_manifest_sha256": (feature_manifest, FEATURE_MANIFEST_SHA256),
        "frozen_feature_cache_sha256": (feature_path, FEATURES_SHA256),
        "readout_sha256": (readout, BEST_READOUT_SHA256),
    }
    checked_hashes: dict[str, str] = {}
    for name, (path, expected) in checks.items():
        actual = sha256_file(path.resolve(strict=True))
        if actual != expected:
            raise ValueError(f"{name} mismatch: {actual}")
        checked_hashes[name] = actual
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Clevr-4 archive size differs from the pinned source preflight")
    if sha256_file(assembly / "package-manifest.json") != ASSEMBLY_MANIFEST_SHA256:
        raise ValueError("Path A assembly manifest hash mismatch")
    receipt_path = train_manifest.parent / "receipt.json"
    if sha256_file(receipt_path) != INPUTS_RECEIPT_SHA256:
        raise ValueError("frozen VideoMAE inputs receipt hash mismatch")
    if sha256_file(model_root / "modeling_finetune.py") != ENCODER_SOURCE_SHA256:
        raise ValueError("VideoMAE source hash mismatch")
    if sha256_file(model_root / "vit_s_k710_dl_from_giant.pth") != ENCODER_CHECKPOINT_SHA256:
        raise ValueError("VideoMAE checkpoint hash mismatch")

    rows = _jsonl(train_manifest)
    if not rows or any(row.get("split", "train") != "train" for row in rows):
        raise ValueError("smoke input manifest is not a nonempty train-only split")
    sample = rows[0]
    sample_id = str(sample["id"])
    sample_hash = str(sample["media_sha256"])
    taxonomy = "color"
    options = list(CLEVR4_TAXONOMIES[taxonomy])[:2]
    expected_probabilities = _load_expected_probabilities(
        feature_path,
        feature_manifest,
        readout,
        sample_id,
        sample_hash,
        taxonomy,
        options,
    )

    output.mkdir(parents=True, exist_ok=False)
    sample_path = output / "train-image.png"
    import zipfile

    with zipfile.ZipFile(archive) as source:
        image_bytes = source.read(str(sample["archive_member"]))
    actual_media_hash = hashlib.sha256(image_bytes).hexdigest()
    if actual_media_hash != sample_hash:
        raise ValueError("extracted training image hash mismatch")
    sample_path.write_bytes(image_bytes)

    request = {
        "request_id": "videomae-path-a-runtime-parity-train-only-001",
        "task_id": "path-a-image-videomaev2-clevr4-100k-runtime-v0",
        "payload": {
            "image_path": str(sample_path),
            "expected_media_sha256": sample_hash,
            "taxonomy": taxonomy,
            "options": options,
        },
        "decision_specs": [
            {
                "request_id": "videomae-path-a-runtime-parity-train-only-001",
                "decision_type": "choice",
                "options": [{"id": option, "text": option} for option in options],
            }
        ],
    }
    request_path = output / "request.jsonl"
    response_path = output / "response.jsonl"
    request_path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project_root / "src"), str(project_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["PYTHONUTF8"] = "1"
    log_dir = output / "worker-logs"
    command = [
        sys.executable,
        "-B",
        "-m",
        "tiny_omni_decision.cli",
        "path-a-jsonl",
        "--assembly",
        str(assembly),
        "--videoprism-python",
        str(videoprism_python),
        "--videoprism-source",
        str(videoprism_source),
        "--videomae-model-dir",
        str(model_root),
        "--videomae-readout",
        str(readout),
        "--videomae-dependency-dir",
        str(dependency_root),
        "--worker-log-dir",
        str(log_dir),
        "--device",
        device,
    ]
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        input=request_path.read_text(encoding="utf-8"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        cwd=project_root,
        check=False,
    )
    elapsed = time.perf_counter() - started
    response_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(
            f"JSONL runtime exited {completed.returncode}: {completed.stderr[-3000:]}"
        )
    response = json.loads(completed.stdout.strip().splitlines()[-1])
    if response.get("status") != "ok":
        raise RuntimeError(f"JSONL image inference failed: {response}")
    record = response["decision"]
    actual_probabilities = [float(item["probability"]) for item in record["options"]]
    max_probability_delta = max(
        abs(actual - expected)
        for actual, expected in zip(actual_probabilities, expected_probabilities, strict=True)
    )
    if record["request_id"] != request["request_id"] or max_probability_delta > 1e-6:
        raise ValueError("JSONL typed output differs from the frozen cached-feature scorer")

    result = {
        "status": "PASS_JSONL_IMAGE_RUNTIME_PARITY_NOT_QUALITY",
        "experiment_id": output.name,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip(),
        "device": response.get("runtime", {}).get("device", device),
        "torch": torch.__version__,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "runtime_elapsed_seconds": elapsed,
        "runtime_metrics": response.get("runtime", {}),
        "run_type": "single frozen image inference through the Path A JSONL dispatcher",
        "input_split": "official Clevr-4 train only; first frozen train-manifest row",
        "sample_id": sample_id,
        "sample_media_sha256": actual_media_hash,
        "input_manifest_hashes": checked_hashes,
        "input_receipt_sha256": INPUTS_RECEIPT_SHA256,
        "encoder_source_sha256": ENCODER_SOURCE_SHA256,
        "encoder_checkpoint_sha256": ENCODER_CHECKPOINT_SHA256,
        "readout_sha256": READOUT_SHA256,
        "assembly_manifest_sha256": ASSEMBLY_MANIFEST_SHA256,
        "runtime_module_sha256": sha256_file(
            project_root / "src/tiny_omni_decision/videomae_runtime.py"
        ),
        "worker_script_sha256": sha256_file(project_root / "scripts/path_a_videomae_worker.py"),
        "smoke_script_sha256": sha256_file(Path(__file__).resolve()),
        "request_sha256": sha256_file(request_path),
        "response_sha256": sha256_file(response_path),
        "expected_probabilities_from_frozen_feature_cache": expected_probabilities,
        "runtime_probabilities": actual_probabilities,
        "max_probability_delta": max_probability_delta,
        "parameter_count": 22_205_968,
        "training_performed": False,
        "validation_or_reporting_labels_loaded": False,
        "sealed_audit_loaded": False,
        "accuracy_or_calibration_measured": False,
        "videoprism_worker_started": False,
        "claims": [
            "Image JSONL request used pinned VideoMAE and verified local media.",
            "Typed probabilities match the cached-feature scorer path within 1e-6.",
            "This is a train-image runtime/parity smoke, not a quality or temporal-video result.",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "archive",
        "train-manifest",
        "feature-manifest",
        "feature-cache",
        "readout",
        "model-root",
        "dependency-root",
        "assembly",
        "videoprism-python",
        "videoprism-source",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_smoke(
        archive=args.archive,
        train_manifest=args.train_manifest,
        feature_manifest=args.feature_manifest,
        feature_path=args.feature_cache,
        readout=args.readout,
        model_root=args.model_root,
        dependency_root=args.dependency_root,
        assembly=args.assembly,
        videoprism_python=args.videoprism_python,
        videoprism_source=args.videoprism_source,
        output=args.output,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
