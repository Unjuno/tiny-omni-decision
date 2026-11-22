"""Package the fixed spoken-color pilot without copying training media."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-json-runtime-attempt-01"
)
WHISPER_ROOT = Path(r"C:\CodexArtifacts\pretrained-reuse-r1\whisper-encoder-runtime-01")
DINO_ROOT = Path(r"C:\CodexArtifacts\pretrained-reuse-r1\dinov2-small-audit-01\model")
BASELINE_ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-fit-attempt-01"
)
INT8_ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-dynamic-int8-attempt-02"
)
CALIBRATION_ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-dino-int8-calibration-attempt-01"
)
CORPUS_ROOT = Path(
    r"C:\CodexArtifacts\pretrained-reuse-r2\spoken-palette-av-decision-v1-attempt-02"
)
EXPECTED_WHISPER_ENCODER = "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8"
EXPECTED_DINO_CONFIG = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
EXPECTED_DINO_PROCESSOR = "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
EXPECTED_DINO_INT8 = "cd8083402b68ed41f9a3367258698ba7cc70c455b58f31828e29346e281381d8"
EXPECTED_SCORER = "48b4a12401f9da88472b290536ca5e3fe3b708319af9d221caa5ce4b75e9c1b5"
EXPECTED_CORPUS = "b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a"
EXPECTED_SOURCE_MANIFEST = "5876e57a4c71108fe3483fde6e18f3977121942c9208567488f042247626577d"
EXPECTED_WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
EXPECTED_DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(
    source: Path, output_root: Path, relative: str, expected: str | None = None
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    actual = sha256_file(source)
    if expected is not None and actual != expected:
        raise ValueError(f"pinned source hash mismatch: {source}")
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != actual:
        raise ValueError(f"copy hash mismatch: {relative}")
    return {"path": relative.replace("\\", "/"), "sha256": actual, "bytes": target.stat().st_size}


def build(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite runtime attempt: {output_root}")

    baseline = json.loads((BASELINE_ROOT / "result.json").read_text(encoding="utf-8"))
    int8_result = json.loads((INT8_ROOT / "result.json").read_text(encoding="utf-8"))
    calibration = json.loads((CALIBRATION_ROOT / "result.json").read_text(encoding="utf-8"))
    corpus_manifest_path = CORPUS_ROOT / "manifest.json"
    corpus_sha = sha256_file(corpus_manifest_path)
    source_manifest = ROOT / "manifests/pretrained_reuse/lombard-grid-source-20261014.json"
    if (
        baseline.get("status") != "PASS_EXECUTION_WITH_LIMITED_DEVELOPMENT_EVALUATION"
        or int8_result.get("status") != "PASS_FIXED_DEVELOPMENT_QUANTIZATION_COMPATIBILITY_CHECK"
        or calibration.get("status") != "PASS_CALIBRATION_ONLY_TEMPERATURE_FIT"
        or int8_result.get("inputs", {}).get("baseline_checkpoint_sha256")
        != baseline.get("selected_checkpoint", {}).get("sha256")
        or calibration.get("inputs", {}).get("float32_scorer_sha256")
        != baseline.get("selected_checkpoint", {}).get("sha256")
        or calibration.get("inputs", {}).get("int8_state_sha256") != EXPECTED_DINO_INT8
        or int8_result.get("inputs", {}).get("corpus_manifest_sha256") != corpus_sha
        or corpus_sha != EXPECTED_CORPUS
        or sha256_file(source_manifest) != EXPECTED_SOURCE_MANIFEST
        or int8_result.get("sealed_audit_opened") is not False
        or calibration.get("sealed_audit_opened") is not False
        or sha256_file(BASELINE_ROOT / "checkpoint-best.safetensors") != EXPECTED_SCORER
        or sha256_file(INT8_ROOT / "dinov2-small-dynamic-int8-state.pt") != EXPECTED_DINO_INT8
    ):
        raise ValueError("source artifact lineage or read-only audit gate failed")

    output_root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    for source_name, relative, expected in [
        ("config.json", "whisper/config.json", None),
        ("preprocessor_config.json", "whisper/preprocessor_config.json", None),
        ("encoder.safetensors", "whisper/encoder.safetensors", EXPECTED_WHISPER_ENCODER),
        ("LICENSE_APACHE_2.txt", "licenses/whisper-LICENSE_APACHE_2.txt", None),
        ("MODEL_CARD_PINNED.md", "licenses/whisper-MODEL_CARD_PINNED.md", None),
    ]:
        files[f"whisper_{source_name}"] = _copy(
            WHISPER_ROOT / source_name, output_root, relative, expected
        )
    for source_name, relative, expected in [
        ("config.json", "dinov2/config.json", EXPECTED_DINO_CONFIG),
        ("preprocessor_config.json", "dinov2/preprocessor_config.json", EXPECTED_DINO_PROCESSOR),
        ("README.md", "licenses/dinov2-MODEL_CARD.md", None),
    ]:
        files[f"dinov2_{source_name}"] = _copy(
            DINO_ROOT / source_name, output_root, relative, expected
        )
    files["training_data_source_manifest"] = _copy(
        source_manifest,
        output_root,
        "licenses/lombard-grid-source-manifest.json",
        EXPECTED_SOURCE_MANIFEST,
    )
    files["whisper_encoder"] = files["whisper_encoder.safetensors"]
    files["dinov2_dynamic_int8_state"] = _copy(
        INT8_ROOT / "dinov2-small-dynamic-int8-state.pt",
        output_root,
        "dinov2/dynamic-int8-state.pt",
        EXPECTED_DINO_INT8,
    )
    files["scorer"] = _copy(
        BASELINE_ROOT / "checkpoint-best.safetensors",
        output_root,
        "decision/scorer.safetensors",
        EXPECTED_SCORER,
    )
    files["normalizer"] = _copy(
        BASELINE_ROOT / "train-normalizer.npz", output_root, "decision/train-normalizer.npz"
    )
    files["calibration_source_result"] = _copy(
        CALIBRATION_ROOT / "result.json",
        output_root,
        "decision/calibration-source-result.json",
    )

    temperature = calibration["selector"]["selected_temperature"]
    if not isinstance(temperature, (float, int)) or temperature <= 0:
        raise ValueError("selected calibration temperature is invalid")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "device": "cpu",
        "experiment_id": "spoken-color-screen-action-json-runtime-v1",
        "task_scope": (
            "four-choice spoken color to generated screen action; task-specific prototype"
        ),
        "source_revisions": {
            "whisper": f"openai/whisper-tiny@{EXPECTED_WHISPER_REVISION}",
            "dinov2": f"facebook/dinov2-small@{EXPECTED_DINO_REVISION}",
        },
        "source_hashes": {
            "corpus_manifest_sha256": corpus_sha,
            "baseline_scorer_sha256": EXPECTED_SCORER,
            "dynamic_int8_state_sha256": EXPECTED_DINO_INT8,
            "baseline_result_sha256": sha256_file(BASELINE_ROOT / "result.json"),
            "dynamic_int8_result_sha256": sha256_file(INT8_ROOT / "result.json"),
            "calibration_result_sha256": sha256_file(CALIBRATION_ROOT / "result.json"),
        },
        "logical_prequantization_parameters": {
            "whisper_encoder": 8_208_384,
            "dinov2_small": 22_056_576,
            "decision_scorer": 98_561,
            "total": 30_363_521,
            "note": (
                "DINOv2 Linear weights use dynamic qint8; embeddings, norms, Whisper, "
                "and readout remain float32."
            ),
        },
        "decision": {
            "architecture": "FrozenFeatureCandidateScorer(feature_size=1538, hidden_size=64)",
            "selected_temperature": float(temperature),
            "temperature_scope": (
                "selected on the already observed calibration split; not independent "
                "quality evidence"
            ),
            "input_format": (
                "mono 16-bit PCM or float32 PCM WAV, 16 kHz, <=30 seconds; "
                "plus RGB screen image and "
                ">=2 options with normalized coordinates"
            ),
            "output": (
                "typed JSON probabilities over caller-supplied option IDs in caller-supplied order"
            ),
        },
        "limitations": [
            "This is a task-specific prototype, not a general multimodal Decision model.",
            "The calibration metric is in-sample on the rows used to choose temperature.",
            "The source task uses generated screen assets and does not establish "
            "natural audiovisual quality.",
            "No Video input, device latency target, or complete dependency-inclusive "
            "installer is provided.",
            "Whisper encoder was feature-extracted in BF16 on CUDA during training; "
            "the CPU FP32 runtime needs measured input-feature parity and is not "
            "claimed bitwise equivalent.",
        ],
        "files": files,
    }
    bundle_path = output_root / "runtime-bundle.json"
    bundle_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle_hash = sha256_file(bundle_path)
    (output_root / "bundle-receipt.json").write_text(
        json.dumps(
            {
                "runtime_bundle_sha256": bundle_hash,
                "payload_files": len(files),
                "sealed_audit_opened": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    payload_files = [path for path in output_root.rglob("*") if path.is_file()]
    return {
        "status": "PASS_BUNDLE_CREATED",
        "bundle_path": str(output_root),
        "bundle_manifest_sha256": bundle_hash,
        "payload_file_count": len(payload_files),
        "payload_bytes": sum(path.stat().st_size for path in payload_files),
        "selected_temperature": float(temperature),
        "sealed_audit_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
