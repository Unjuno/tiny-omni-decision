"""Replay the frozen Path A TRAIN mechanics request through an encoder-only bundle."""

# ruff: noqa: E402 - direct-script entrypoint installs repository source paths first.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

# Direct-script entrypoint support.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.pretrained_reuse.smoke_spoken_palette_runtime import (
    CORPUS_ROOT,
    DEFAULT_AUDIO_ARCHIVE,
    DEFAULT_AUDIO_LAYOUT,
    _read_first_train_row,
    sha256_file,
)
from tiny_omni_decision.audio_runtime import AUDIO_ENCODER_STATE_SHA256, AUDIO_WEIGHTS_SHA256
from tiny_omni_decision.modular_bundle import verify_modular_bundle
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher

DEFAULT_BASELINE = Path(
    r"D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\path-a-shared-whisper-audio-image-smoke-20261122-attempt-03\shared-path-a-runtime-smoke.json"
)


def run_smoke(
    assembly: Path,
    baseline_report: Path,
    output_dir: Path,
    *,
    corpus_root: Path = CORPUS_ROOT,
    audio_archive: Path = DEFAULT_AUDIO_ARCHIVE,
    audio_layout: Path = DEFAULT_AUDIO_LAYOUT,
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite encoder-only smoke directory: {output}")
    output.mkdir(parents=True)
    manifest = verify_modular_bundle(assembly.resolve(strict=True))
    baseline = json.loads(baseline_report.resolve(strict=True).read_text(encoding="utf-8"))
    row, corpus_manifest = _read_first_train_row(corpus_root.resolve(strict=True))
    sample_id = str(row["sample_id"])
    if baseline.get("sample_id") != sample_id or baseline.get("split") != "train":
        raise ValueError("baseline and frozen corpus do not name the same TRAIN request")
    screenshot_path = (corpus_root / str(row["screenshot_path"])).resolve(strict=True)
    if corpus_root.resolve() not in screenshot_path.parents:
        raise ValueError("TRAIN screenshot escaped its frozen corpus")
    screenshot_sha = sha256_file(screenshot_path)
    if (
        screenshot_sha != row["screenshot_sha256"]
        or screenshot_sha != baseline["screenshot_sha256"]
    ):
        raise ValueError("TRAIN screenshot hash differs from the fixed baseline")

    layout = json.loads(audio_layout.resolve(strict=True).read_text(encoding="utf-8"))
    if not layout or layout[0].get("split") != "train":
        raise ValueError("audio layout does not begin with the frozen TRAIN source")
    media = layout[0]
    if media.get("source_id") != row.get("source_id"):
        raise ValueError("audio source identity differs from the frozen corpus row")
    audio_member = str(media["assets"]["audio"]["name"])
    with zipfile.ZipFile(audio_archive.resolve(strict=True)) as archive:
        audio_info = archive.getinfo(audio_member)
        if audio_info.file_size != int(media["assets"]["audio"]["bytes"]) or audio_info.CRC != int(
            media["assets"]["audio"]["crc32"]
        ):
            raise ValueError("TRAIN WAV size or CRC differs from the pinned audio layout")
        wav_bytes = archive.read(audio_member)
    audio_sha = hashlib.sha256(wav_bytes).hexdigest()
    if audio_sha != baseline["audio_sha256"]:
        raise ValueError("TRAIN WAV hash differs from the fixed baseline")

    option_ids = tuple(str(option["option_id"]) for option in row["options"])
    option_texts = tuple(f"Screen action candidate {index + 1}" for index in range(len(option_ids)))
    options = [
        {
            "id": option_ids[index],
            "text": option_texts[index],
            "x_norm": float(option["x_norm"]),
            "y_norm": float(option["y_norm"]),
        }
        for index, option in enumerate(row["options"])
    ]
    baseline_decision = baseline["decision_json"]
    if [item["id"] for item in baseline_decision["options"]] != list(option_ids):
        raise ValueError("baseline option order differs from frozen TRAIN request")
    baseline_probabilities = [float(item["probability"]) for item in baseline_decision["options"]]

    worker_logs = output / "worker-logs"
    worker_logs.mkdir()
    with tempfile.TemporaryDirectory(prefix="path-a-encoder-only-", dir=output) as scratch:
        wav_path = Path(scratch) / "train-input.wav"
        wav_path.write_bytes(wav_bytes)
        started = time.perf_counter()
        with PathARuntimeDispatcher(
            assembly.resolve(strict=True),
            videoprism_python=Path(sys.executable),
            videoprism_source=ROOT,
            worker_log_dir=worker_logs,
            device="cpu",
        ) as dispatcher:
            response = dispatcher.dispatch(
                "spoken_palette_av",
                {
                    "audio_path": str(wav_path),
                    "image_path": str(screenshot_path),
                    "options": options,
                },
            )
            elapsed = time.perf_counter() - started
            worker_metadata = dispatcher.worker_metadata["torch"]
        if worker_metadata.get("audio_encoder_storage_format") != "encoder_only":
            raise ValueError("Path A worker did not load the encoder-only Whisper state")
        if worker_metadata.get("audio_encoder_weights_sha256") != AUDIO_ENCODER_STATE_SHA256:
            raise ValueError("Path A worker did not report the pinned encoder-only state hash")
        if worker_metadata.get("audio_encoder_source_checkpoint_sha256") != AUDIO_WEIGHTS_SHA256:
            raise ValueError("Path A worker source checkpoint provenance differs")

    decision_json = response
    probabilities = [float(item["probability"]) for item in decision_json["options"]]
    if (
        [item["id"] for item in decision_json["options"]] != list(option_ids)
        or not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("encoder-only Path A runtime returned invalid typed JSON")
    deltas = [
        abs(left - right) for left, right in zip(probabilities, baseline_probabilities, strict=True)
    ]
    maximum_probability_delta = max(deltas)
    if maximum_probability_delta > 1e-9:
        raise ValueError("encoder-only Path A probabilities differ from the fixed baseline")

    result = {
        "status": "PASS_ENCODER_ONLY_PATH_A_TRAIN_REPLAY",
        "scope": "same historical TRAIN mechanics request; no quality evaluation or label use",
        "assembly_manifest_sha256": sha256_file(assembly / "package-manifest.json"),
        "baseline_report_sha256": sha256_file(baseline_report),
        "corpus_manifest_sha256": sha256_file(corpus_root / "manifest.json"),
        "train_jsonl_sha256": corpus_manifest["split_row_sha256"]["train"],
        "sample_id": sample_id,
        "source_id": str(row["source_id"]),
        "split": "train",
        "audio_member": audio_member,
        "audio_sha256": audio_sha,
        "screenshot_sha256": screenshot_sha,
        "encoder_state_sha256": AUDIO_ENCODER_STATE_SHA256,
        "source_whisper_checkpoint_sha256": AUDIO_WEIGHTS_SHA256,
        "runtime_parameter_count": worker_metadata["resident_unique_parameters"],
        "runtime_worker_metadata": worker_metadata,
        "package_bytes": manifest["package_file_bytes_excluding_manifest"],
        "unique_runtime_parameters": manifest["unique_runtime_parameters"],
        "baseline_probabilities": baseline_probabilities,
        "encoder_only_probabilities": probabilities,
        "max_probability_delta": maximum_probability_delta,
        "typed_decision_json": decision_json,
        "elapsed_seconds_including_worker_startup": elapsed,
        "evaluation_split_opened": False,
        "sealed_audit_opened": False,
        "model_updated": False,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--audio-archive", type=Path, default=DEFAULT_AUDIO_ARCHIVE)
    parser.add_argument("--audio-layout", type=Path, default=DEFAULT_AUDIO_LAYOUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.assembly,
                args.baseline_report,
                args.output_dir,
                corpus_root=args.corpus,
                audio_archive=args.audio_archive,
                audio_layout=args.audio_layout,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
