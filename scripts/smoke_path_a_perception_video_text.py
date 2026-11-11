"""Run label-blind Perception TRAIN canaries through the real Path A JSONL worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tiny_omni_decision.path_a_runtime_cli import run_jsonl
from tiny_omni_decision.runtime_dispatch import PathARuntimeDispatcher

EXPECTED_EXPERIMENT_ID = "r2-perception-test-mcqa-path-a-20261111"
EXPECTED_TRAIN_SHA256 = "2f2fa8ed4e81baaadcede50cc4343f0f0c55a02dd63703d3313e8d2188cb60a7"
TASK_ID = "path-a-video-text-videomaev2-perception-test-experimental-v0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canary_rows(corpus_dir: Path, media_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("unexpected frozen Perception TRAIN corpus")
    train_meta = manifest.get("train", {})
    records_path = corpus_dir / "train.jsonl"
    if (
        train_meta.get("records_sha256") != EXPECTED_TRAIN_SHA256
        or sha256_file(records_path) != EXPECTED_TRAIN_SHA256
    ):
        raise ValueError("Perception TRAIN corpus hash differs from the frozen source")

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with records_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("target", None)
            row.pop("target_option_index", None)
            if row.get("split") != "train":
                raise ValueError("canary corpus contains a non-TRAIN row")
            source_video_id = str(row.get("source_video_id", ""))
            question = row.get("question")
            options = row.get("options")
            if (
                not source_video_id
                or not isinstance(question, str)
                or not question.strip()
                or not isinstance(options, list)
                or len(options) != 3
            ):
                raise ValueError("malformed Perception TRAIN canary row")
            projected = {
                "id": str(row["id"]),
                "group_id": str(row["group_id"]),
                "source_video_id": source_video_id,
                "question": question,
                "options": [str(value) for value in options],
            }
            grouped.setdefault(source_video_id, []).append(projected)

    for video_id, rows in grouped.items():
        media_path = media_dir / f"{video_id}.mp4"
        if len(rows) >= 2 and media_path.is_file():
            return rows[:2], media_path.resolve(strict=True)
    raise FileNotFoundError("no TRAIN video with two questions and present media was found")


def _request(
    request_id: str,
    video_path: Path,
    media_sha256: str,
    row: dict[str, Any],
    options: list[str],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "task_id": TASK_ID,
        "payload": {
            "video_path": str(video_path),
            "expected_media_sha256": media_sha256,
            "question": row["question"],
            "options": options,
        },
        "decision_specs": [
            {
                "request_id": request_id,
                "decision_type": "choice",
                "options": [
                    {"id": f"option-{index}", "text": option}
                    for index, option in enumerate(options)
                ],
            }
        ],
    }


def run_smoke(
    *,
    corpus_dir: Path,
    media_dir: Path,
    model_root: Path,
    dependency_dir: Path,
    assembly_dir: Path,
    readout_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA canary requested but CUDA is unavailable")
    rows, video_path = _canary_rows(corpus_dir.resolve(strict=True), media_dir.resolve(strict=True))
    media_sha256 = sha256_file(video_path)
    options = rows[0]["options"]
    permuted_options = [options[2], options[1], options[0]]
    requests = [
        _request(
            "perception-train-canary-question-1", video_path, media_sha256, rows[0], options
        ),
        _request(
            "perception-train-canary-question-2",
            video_path,
            media_sha256,
            rows[1],
            rows[1]["options"],
        ),
        _request(
            "perception-train-canary-option-permutation",
            video_path,
            media_sha256,
            rows[0],
            permuted_options,
        ),
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    worker_log_dir = output_dir / "worker-logs"
    dispatcher = PathARuntimeDispatcher(
        assembly_dir,
        worker_log_dir=worker_log_dir,
        videomae_model_dir=model_root,
        videomae_dependency_dir=dependency_dir,
        perception_video_readout=readout_path,
        device=device,
    )
    request_path = output_dir / "requests.jsonl"
    response_path = output_dir / "responses.jsonl"
    request_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in requests),
        encoding="utf-8",
    )
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    stdout = io.StringIO()
    try:
        run_jsonl(io.StringIO(request_path.read_text(encoding="utf-8")), stdout, dispatcher)
    finally:
        worker_metadata = dispatcher.worker_metadata
        dispatcher.close()
    elapsed = time.perf_counter() - started
    response_path.write_text(stdout.getvalue(), encoding="utf-8")
    responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    if len(responses) != 3 or any(item.get("status") != "ok" for item in responses):
        raise RuntimeError(f"one or more video-text canary requests failed: {responses}")
    probabilities = [item["decision"]["options"] for item in responses]
    first = {row["text"]: row["probability"] for row in probabilities[0]}
    permuted = {row["text"]: row["probability"] for row in probabilities[2]}
    max_permutation_delta = max(abs(first[text] - permuted[text]) for text in first)
    if max_permutation_delta > 1e-5:
        raise RuntimeError(
            "option permutation changed same-request probability mapping by "
            f"{max_permutation_delta}"
        )
    if device == "cuda":
        torch.cuda.synchronize()
    runtimes = [item["runtime"] for item in responses]
    runtime_metadata = runtimes[0]["video_text_encoder"]
    text_model_dir = (
        assembly_dir.resolve(strict=True)
        / "components"
        / "audio"
        / "models"
        / "text-minilm-en"
    )
    weight_files = {
        "videomae_checkpoint_bytes": (model_root / "vit_s_k710_dl_from_giant.pth").stat().st_size,
        "minilm_checkpoint_bytes": (text_model_dir / "model.safetensors").stat().st_size,
        "decision_readout_checkpoint_bytes": readout_path.stat().st_size,
    }
    weight_files["sum_checkpoint_bytes"] = sum(weight_files.values())
    project_root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1,
        "status": "TRAIN_RUNTIME_CANARY_PASS",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "command_line": sys.argv,
        "split": "official TRAIN only",
        "canary_sample_ids": [row["id"] for row in rows],
        "video_group_id": rows[0]["group_id"],
        "source_video_id": rows[0]["source_video_id"],
        "media_sha256": media_sha256,
        "corpus_train_sha256": EXPECTED_TRAIN_SHA256,
        "runtime_task_id": TASK_ID,
        "request_count": len(requests),
        "same_video_questions_reused": True,
        "option_permutation_max_probability_delta": max_permutation_delta,
        "elapsed_seconds": elapsed,
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "timm_version": importlib.metadata.version("timm"),
        "pyav_version": importlib.metadata.version("av"),
        "psutil_version": importlib.metadata.version("psutil"),
        "resident_unique_parameters": runtime_metadata["resident_unique_parameters"],
        "runtime_model_provenance": runtime_metadata,
        "parameter_breakdown": {
            "videomae": runtime_metadata["video_encoder_parameters"],
            "minilm": runtime_metadata["text_encoder_parameters"],
            "decision_readout": runtime_metadata["readout_parameters"],
        },
        "weight_file_bytes": weight_files,
        "worker_rss_bytes_after_each_request": [item.get("worker_rss_bytes") for item in runtimes],
        "worker_peak_working_set_bytes": max(
            item.get("worker_peak_working_set_bytes") or 0 for item in runtimes
        ),
        "worker_inference_seconds": [item.get("worker_inference_seconds") for item in runtimes],
        "dispatcher_seconds": [item.get("dispatch_wall_seconds") for item in runtimes],
        "worker_startup_seconds": runtimes[0].get("worker_startup_wall_seconds", {}),
        "video_cache_hits_after_each_request": [
            item.get("video_cache_hits") for item in runtimes
        ],
        "video_cache_misses_after_each_request": [
            item.get("video_cache_misses") for item in runtimes
        ],
        "cuda_peak_allocated_bytes": (
            max(
                item["runtime"].get("cuda_peak_allocated_bytes") or 0 for item in responses
            )
            if device == "cuda"
            else 0
        ),
        "worker_metadata": worker_metadata,
        "request_sha256": sha256_file(request_path),
        "response_sha256": sha256_file(response_path),
        "video_text_readout_sha256": sha256_file(readout_path.resolve(strict=True)),
        "assembly_manifest_sha256": sha256_file(
            assembly_dir.resolve(strict=True) / "package-manifest.json"
        ),
        "readout_module_sha256": sha256_file(
            project_root / "src/tiny_omni_decision/perception_video_readout.py"
        ),
        "runtime_module_sha256": sha256_file(
            project_root / "src/tiny_omni_decision/perception_video_runtime.py"
        ),
        "dispatcher_module_sha256": sha256_file(
            project_root / "src/tiny_omni_decision/runtime_dispatch.py"
        ),
        "worker_script_sha256": sha256_file(project_root / "scripts/path_a_videomae_worker.py"),
        "smoke_script_sha256": sha256_file(Path(__file__).resolve()),
        "training_performed": False,
        "targets_or_labels_used": False,
        "accuracy_or_calibration_computed": False,
        "validation_or_sealed_audit_opened": False,
        "claims": [
            "The real Path A JSONL dispatcher loaded the pinned VideoMAE, MiniLM and readout.",
            "Three official TRAIN-only requests returned finite typed option probabilities.",
            "Two questions for one TRAIN video reused cached frozen video tokens.",
            "A candidate permutation preserved text-to-probability mapping within tolerance.",
            "This canary measures runtime wiring only, not task quality or calibration.",
        ],
    }
    report_path = output_dir / "result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "corpus-dir",
        "media-dir",
        "model-root",
        "dependency-dir",
        "assembly-dir",
        "readout",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = run_smoke(
        corpus_dir=args.corpus_dir,
        media_dir=args.media_dir,
        model_root=args.model_root,
        dependency_dir=args.dependency_dir,
        assembly_dir=args.assembly_dir,
        readout_path=args.readout,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
