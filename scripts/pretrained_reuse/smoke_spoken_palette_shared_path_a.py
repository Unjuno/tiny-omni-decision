"""Replay one fixed TRAIN-only AV request through the shared Path A worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from tiny_omni_decision.modular_bundle import verify_modular_bundle
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_fixed_train_row(corpus: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = corpus / "manifest.json"
    expected_manifest = "b75a9e5ea629b8e1d8cb188ef4ee9c9a9b3e784b0f387ce9eed0a27072d72c7a"
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("frozen spoken-palette corpus manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_policy", {}).get("sealed_audit_opened") is not False:
        raise ValueError("corpus does not preserve the sealed-audit isolation marker")
    train_path = corpus / "train.jsonl"
    if sha256_file(train_path) != manifest["split_row_sha256"]["train"]:
        raise ValueError("frozen TRAIN rows hash mismatch")
    with train_path.open(encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    if row.get("split") != "train" or row.get("source_id") != "s11_l_bbab2n":
        raise ValueError("fixed source is not the expected first TRAIN audiovisual item")
    return row, manifest


def run(
    assembly: Path,
    source_bundle: Path,
    corpus: Path,
    audio_archive: Path,
    audio_layout: Path,
    videoprism_python: Path,
    videoprism_source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite shared-runtime smoke: {output}")
    package_manifest = verify_modular_bundle(assembly.resolve(strict=True))
    if package_manifest["unique_runtime_parameters"] != 192_654_853:
        raise ValueError("integrated assembly has unexpected unique parameter accounting")
    if package_manifest["package_target_met"]:
        raise ValueError("reported package target unexpectedly differs from measured manifest")
    task = next(
        (
            item
            for item in package_manifest["components"]
            if item.get("name") == "spoken_palette_av"
        ),
        None,
    )
    if task is None or not task.get("shared_parameter_refs"):
        raise ValueError("integrated assembly does not declare its shared Whisper asset")

    baseline_path = source_bundle / "runtime-smoke-train-first-clamped.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    row, corpus_manifest = _read_fixed_train_row(corpus)
    screenshot = (corpus / row["screenshot_path"]).resolve(strict=True)
    if corpus.resolve() not in screenshot.parents:
        raise ValueError("TRAIN screenshot escapes frozen corpus")
    screenshot_hash = sha256_file(screenshot)
    if screenshot_hash != row["screenshot_sha256"]:
        raise ValueError("TRAIN screenshot hash mismatch")

    layout = json.loads(audio_layout.read_text(encoding="utf-8"))
    if not layout or layout[0].get("source_id") != row["source_id"]:
        raise ValueError("audio layout does not identify the same fixed TRAIN source")
    audio_member = layout[0]["assets"]["audio"]["name"]
    with zipfile.ZipFile(audio_archive) as archive:
        info = archive.getinfo(audio_member)
        if info.file_size != layout[0]["assets"]["audio"]["bytes"]:
            raise ValueError("TRAIN WAV length differs from the pinned media layout")
        if info.CRC != layout[0]["assets"]["audio"]["crc32"]:
            raise ValueError("TRAIN WAV CRC differs from the pinned media layout")
        wav_bytes = archive.read(audio_member)
    audio_hash = hashlib.sha256(wav_bytes).hexdigest()
    if audio_hash != "8408220551860f06b8ea61187802066a0b784578f88e130bfe9fdbec97cedd87":
        raise ValueError("fixed TRAIN audio bytes differ from the existing runtime smoke")

    options = [
        {
            "id": str(option["option_id"]),
            "text": f"Screen action candidate {index + 1}",
            "x_norm": float(option["x_norm"]),
            "y_norm": float(option["y_norm"]),
        }
        for index, option in enumerate(row["options"])
    ]
    output.mkdir(parents=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="shared-path-a-wav-", dir=output.parent) as temp:
        wav_path = Path(temp) / "fixed-train.wav"
        wav_path.write_bytes(wav_bytes)
        with PathARuntimeDispatcher(
            assembly,
            videoprism_python=videoprism_python,
            videoprism_source=videoprism_source,
            worker_log_dir=output / "worker-logs",
            device="cpu",
        ) as dispatcher:
            first_result = dispatcher.dispatch(
                "spoken_palette_av",
                {
                    "audio_path": str(wav_path),
                    "image_path": str(screenshot),
                    "options": options,
                },
            )
            warm_result = dispatcher.dispatch(
                "spoken_palette_av",
                {
                    "audio_path": str(wav_path),
                    "image_path": str(screenshot),
                    "options": options,
                },
            )
            worker = dispatcher.worker_metadata["torch"]
            if not worker.get("spoken_palette_shares_audio_encoder"):
                raise ValueError("Torch worker did not reuse the resident Whisper encoder")
            if not worker.get("spoken_palette_av_loaded"):
                raise ValueError("Torch worker did not load the joint task head")

    reference_decision = baseline["decision_json"]
    reference = {
        item["id"]: float(item["probability"]) for item in reference_decision["options"]
    }

    def compare(result: dict[str, Any]) -> list[float]:
        actual = {item["id"]: float(item["probability"]) for item in result["options"]}
        if list(reference) != list(actual):
            raise ValueError("integrated and standalone output option order differs")
        deltas = [abs(reference[key] - actual[key]) for key in reference]
        if result.get("selected_option_id") != reference_decision.get("selected_option_id"):
            raise ValueError("integrated runtime changed the fixed request argmax")
        if max(deltas, default=1.0) > 1e-5:
            raise ValueError("integrated runtime probability replay exceeds 1e-5 parity gate")
        if not math.isclose(sum(actual.values()), 1.0, abs_tol=1e-5):
            raise ValueError("integrated runtime returned non-normalized probabilities")
        return deltas

    first_deltas = compare(first_result)
    warm_deltas = compare(warm_result)

    report = {
        "status": "PASS_SHARED_ENCODER_TRAIN_ONLY_RUNTIME_PARITY",
        "scope": (
            "one previously used TRAIN raw-input request; mechanics and saved-output "
            "parity only, no label use or quality evaluation"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "implementation_sha256": {
            "package_builder": sha256_file(
                Path(__file__).resolve().with_name("package_spoken_palette_shared_path_a.py")
            ),
            "runtime": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src/tiny_omni_decision/spoken_palette_runtime.py"
            ),
            "dispatcher": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src/tiny_omni_decision/runtime_dispatch.py"
            ),
            "torch_worker": sha256_file(
                Path(__file__).resolve().parents[2] / "scripts/path_a_torch_worker.py"
            ),
            "smoke_script": sha256_file(Path(__file__).resolve()),
        },
        "assembly_manifest_sha256": sha256_file(assembly / "package-manifest.json"),
        "base_runtime_bundle_sha256": sha256_file(source_bundle / "runtime-bundle.json"),
        "source_corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
        "source_train_jsonl_sha256": corpus_manifest["split_row_sha256"]["train"],
        "sample_id": row["sample_id"],
        "source_id": row["source_id"],
        "split": "train",
        "audio_member": audio_member,
        "audio_sha256": audio_hash,
        "screenshot_sha256": screenshot_hash,
        "same_whisper_encoder_object": worker["spoken_palette_shares_audio_encoder"],
        "worker_resident_unique_parameters": worker["resident_unique_parameters"],
        "full_manifest_unique_parameters": package_manifest["unique_runtime_parameters"],
        "package_file_bytes_excluding_manifest": package_manifest[
            "package_file_bytes_excluding_manifest"
        ],
        "package_target_met": package_manifest["package_target_met"],
        "runtime_load_seconds": worker["model_load_seconds"],
        "first_dispatch_seconds_including_worker_startup": first_result[
            "dispatch_wall_seconds"
        ],
        "first_worker_inference_seconds": first_result["worker_inference_seconds"],
        "warm_dispatch_seconds": warm_result["dispatch_wall_seconds"],
        "warm_worker_inference_seconds": warm_result["worker_inference_seconds"],
        "worker_rss_after_request_bytes": warm_result["rss_after_request_bytes"],
        "max_probability_delta_vs_standalone": max(
            [*first_deltas, *warm_deltas], default=0.0
        ),
        "first_per_option_probability_delta": first_deltas,
        "warm_per_option_probability_delta": warm_deltas,
        "decision_json": {
            key: warm_result[key]
            for key in (
                "schema_version",
                "request_id",
                "decision_type",
                "options",
                "selected_option_id",
                "confidence",
            )
        },
        "sealed_audit_opened": False,
        "evaluation_split_opened": False,
        "model_updated": False,
    }
    report["total_wall_seconds"] = time.perf_counter() - started
    report_path = output / "shared-path-a-runtime-smoke.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    report["report_sha256"] = sha256_file(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--audio-layout", type=Path, required=True)
    parser.add_argument("--videoprism-python", type=Path, required=True)
    parser.add_argument("--videoprism-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
