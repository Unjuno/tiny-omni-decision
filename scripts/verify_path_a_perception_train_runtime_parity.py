"""Compare the pinned raw-input runtime with saved features on fit-side TRAIN only.

This verifier is deliberately not a quality evaluation: it never reads targets,
never loads validation/sealed-audit files, and writes no Accuracy or calibration
metrics. It checks whether the raw video/text route reproduces the frozen feature
cache and readout logits used by the original Path A fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tiny_omni_decision.perception_video_readout import PerceptionVideoDecisionReadout
from tiny_omni_decision.perception_video_runtime import (
    VIDEO_TOKEN_SHAPE,
    PerceptionVideoTextRuntime,
    _candidate_text,
    _file_sha256,
    pool_candidate_queries,
)

TRAIN_SHA256 = "2f2fa8ed4e81baaadcede50cc4343f0f0c55a02dd63703d3313e8d2188cb60a7"
TRAIN_FEATURE_SHA256 = "9802470f0ca145f051d64f6f46341817e0f14b96a84f54fd2c25c69384f1b3e2"
VIDEO_MANIFEST_SHA256 = "19893d3d37a8833d815c0f0277950f51d6e438cd5a9679cac8b3e5e8ebfc84f0"
VIDEO_MODEL_SHA256 = "24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172"
FIT_GROUPS_SHA256 = "9a053ff7ae6d1c70b56df610f8b310dc81426b0af83c5cd1a325ec92af08d7ec"
SELECTION_GROUPS_SHA256 = "357115c13cce7e837beaec5abe635942945347572d08e2889a39444d2e8cf138"
FIT_GROUP_COUNT = 1661
SELECTION_GROUP_COUNT = 294
READOUT_SHA256 = "e9516be6d58ba82c3c4f111e59d6b724e856b3ac412bbc2c57b683d35deecb8c"
TEXT_MODEL_SHA256 = "cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433"

# Pre-registered absolute tolerances. They are not adjusted after the run.
VIDEO_FEATURE_ATOL = 1e-3
TEXT_FEATURE_ATOL = 2e-4
LOGIT_ATOL = 2e-3
PREVIOUS_RUNTIME_CANARY_VIDEO_IDS = {"video_10001"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_group_partition(
    groups: list[str], *, seed: int = 17, selection_fraction: float = 0.15
) -> tuple[set[str], set[str]]:
    """Reproduce the reader's deterministic source-video split without labels."""
    if not groups or any(not group for group in groups):
        raise ValueError("every row must have a source video identity")
    if not 0 < selection_fraction < 1:
        raise ValueError("selection_fraction must be between zero and one")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two video groups are required")
    permutation = np.random.default_rng(seed).permutation(np.asarray(unique, dtype=str)).tolist()
    selection_count = min(len(unique) - 1, int(np.ceil(selection_fraction * len(unique))))
    selection = set(permutation[:selection_count])
    fit = set(unique) - selection
    if fit & selection:
        raise AssertionError("video groups crossed the internal split")
    return fit, selection


def project_runtime_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop labels before any row enters parity selection or output construction."""
    return {
        key: raw[key] for key in ("id", "source_video_id", "question", "options", "reasoning_type")
    }


def _group_split(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Mirror the frozen fit/selection split and verify both pinned group hashes."""
    groups = [str(row["source_video_id"]) for row in rows]
    fit, selection = deterministic_group_partition(groups)
    if len(set(groups)) != 1955:
        raise ValueError(f"unexpected frozen TRAIN video group count: {len(set(groups))}")
    fit_hash = hashlib.sha256("\n".join(sorted(fit)).encode()).hexdigest()
    selection_hash = hashlib.sha256("\n".join(sorted(selection)).encode()).hexdigest()
    if (
        len(fit) != FIT_GROUP_COUNT
        or len(selection) != SELECTION_GROUP_COUNT
        or fit_hash != FIT_GROUPS_SHA256
        or selection_hash != SELECTION_GROUPS_SHA256
    ):
        raise ValueError("reconstructed fit/selection group split differs from frozen config")
    return fit, selection


def select_fit_rows(
    rows: list[dict[str, Any]],
    *,
    video_count: int = 8,
    questions_per_video: int = 2,
    excluded_video_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Choose a deterministic subset from fit groups without exposing label fields."""
    fit_groups, selection_groups = _group_split(rows)
    excluded_video_ids = excluded_video_ids or set()
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        # Explicit projection: targets and label-derived fields never enter the run.
        row = project_runtime_fields(raw)
        by_group[str(row["source_video_id"])].append(row)
    if fit_groups & selection_groups:
        raise AssertionError("fit and internal-selection video groups overlap")
    chosen: list[dict[str, Any]] = []
    for group in sorted(fit_groups):
        if group in excluded_video_ids:
            continue
        examples = sorted(by_group[group], key=lambda row: row["id"])
        if len(examples) >= questions_per_video:
            chosen.extend(examples[:questions_per_video])
        if len(chosen) >= video_count * questions_per_video:
            break
    chosen = chosen[: video_count * questions_per_video]
    if len(chosen) != video_count * questions_per_video:
        raise ValueError("not enough fit-side groups with the requested question count")
    if any(row["source_video_id"] not in fit_groups for row in chosen):
        raise AssertionError("selected parity rows escaped fit-side TRAIN groups")
    if any(set(row) & {"target", "target_option_index", "label"} for row in chosen):
        raise AssertionError("label fields leaked into parity selection")
    return chosen


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("non-finite tensor in parity comparison")
    return float((left.float() - right.float()).abs().max().item())


def _load_train_inputs(args: argparse.Namespace):
    corpus_dir = args.corpus_dir.resolve(strict=True)
    train_path = corpus_dir / "train.jsonl"
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != "r2-perception-test-mcqa-path-a-20261111"
        or _file_sha256(train_path) != TRAIN_SHA256
        or manifest.get("train", {}).get("records_sha256") != TRAIN_SHA256
    ):
        raise ValueError("frozen official TRAIN corpus identity/hash mismatch")
    raw_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    if len(raw_rows) != 7392 or any(row.get("split") != "train" for row in raw_rows):
        raise ValueError("unexpected TRAIN row count/split")
    rows = select_fit_rows(raw_rows, excluded_video_ids=PREVIOUS_RUNTIME_CANARY_VIDEO_IDS)

    text_path = args.text_features.resolve(strict=True)
    if sha256_file(text_path) != TRAIN_FEATURE_SHA256:
        raise ValueError("frozen TRAIN text feature archive hash mismatch")
    report_path = args.text_report.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "COMPLETE"
        or report.get("split") != "train"
        or report.get("records_sha256") != TRAIN_SHA256
        or report.get("labels_used") is not False
        or report.get("official_test_opened") is not False
        or report.get("feature_sha256") != TRAIN_FEATURE_SHA256
        or report.get("model_files_sha256", {}).get("model.safetensors") != TEXT_MODEL_SHA256
    ):
        raise ValueError("frozen TRAIN text report identity mismatch")
    with np.load(text_path, allow_pickle=False) as archive:
        ids = archive["sample_ids"].astype(str).tolist()
        videos = archive["video_ids"].astype(str).tolist()
        features = archive["features"]
    if features.shape != (7392, 3, 384) or features.dtype != np.float32:
        raise ValueError("unexpected frozen TRAIN text feature contract")
    id_index = {sample_id: index for index, sample_id in enumerate(ids)}
    if len(id_index) != len(ids):
        raise ValueError("duplicate IDs in frozen text feature archive")
    for row in rows:
        index = id_index.get(row["id"])
        if index is None or videos[index] != row["source_video_id"]:
            raise ValueError("selected train sample does not align with saved text features")
        row["text_feature_index"] = index

    video_manifest_path = args.video_feature_manifest.resolve(strict=True)
    if sha256_file(video_manifest_path) != VIDEO_MANIFEST_SHA256:
        raise ValueError("frozen TRAIN VideoMAE manifest hash mismatch")
    video_manifest = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    if (
        video_manifest.get("status") != "COMPLETE"
        or video_manifest.get("split") != "train"
        or video_manifest.get("checkpoint_sha256") != VIDEO_MODEL_SHA256
    ):
        raise ValueError("frozen TRAIN VideoMAE manifest identity mismatch")
    receipts = {row["video_id"]: row for row in video_manifest["feature_receipts"]}
    for row in rows:
        receipt = receipts.get(row["source_video_id"])
        feature_path = args.video_feature_dir / f"{row['source_video_id']}.npy"
        if receipt is None or sha256_file(feature_path) != receipt.get("feature_sha256"):
            raise ValueError(f"saved video feature receipt mismatch: {row['source_video_id']}")
        row["video_feature_path"] = feature_path
        row["video_feature_sha256"] = receipt["feature_sha256"]
        row["expected_media_sha256"] = receipt["media_sha256"]
    return rows, features


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.drive.upper() != "D:" or args.output_dir.exists():
        raise ValueError("choose a new, non-existing output directory on D:")
    if args.device not in {"cpu", "cuda"} or (
        args.device == "cuda" and not torch.cuda.is_available()
    ):
        raise RuntimeError("requested parity device is unavailable")
    rows, saved_text = _load_train_inputs(args)
    media_root = args.media_dir.resolve(strict=True)
    media_by_video = {}
    for row in rows:
        video_id = row["source_video_id"]
        media_path = media_root / f"{video_id}.mp4"
        media_path.resolve(strict=True)
        if _file_sha256(media_path) != row["expected_media_sha256"]:
            raise ValueError(f"media hash differs from its TRAIN feature receipt: {video_id}")
        media_by_video[video_id] = media_path

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    runtime = PerceptionVideoTextRuntime(
        model_root=args.model_root,
        dependency_root=args.dependency_root,
        text_model_dir=args.text_model_dir,
        readout_path=args.readout,
        device=args.device,
    )
    readout_offline = PerceptionVideoDecisionReadout().to(runtime.device).eval()
    payload = torch.load(args.readout, map_location="cpu", weights_only=True)
    readout_offline.load_state_dict(payload["state_dict"], strict=True)
    observations = []
    video_deltas: dict[str, float] = {}
    text_deltas: dict[str, float] = {}
    logit_deltas: dict[str, float] = {}
    probability_deltas: dict[str, float] = {}
    try:
        with torch.inference_mode():
            for row in rows:
                video_id = row["source_video_id"]
                media_path = media_by_video[video_id]
                media_sha = row["expected_media_sha256"]
                if not torch.cuda.is_available() and args.device == "cuda":
                    raise RuntimeError("CUDA became unavailable during parity run")
                raw_video = runtime._video_tokens(media_path, media_sha).reshape(1, -1, 384)
                saved_video_np = np.load(row["video_feature_path"], allow_pickle=False)
                if tuple(saved_video_np.shape) != VIDEO_TOKEN_SHAPE:
                    raise ValueError("saved video tokens have unexpected shape")
                saved_video = torch.from_numpy(saved_video_np.reshape(1, -1, 384).copy()).to(
                    device=runtime.device, dtype=torch.float32
                )
                video_delta = _max_abs_delta(raw_video, saved_video)
                video_deltas[row["source_video_id"]] = video_delta
                if video_delta > VIDEO_FEATURE_ATOL:
                    raise AssertionError(
                        f"video token parity exceeded fixed atol for {video_id}: {video_delta}"
                    )

                candidates = [_candidate_text(option) for option in row["options"]]
                prompts = [
                    f"Question: {row['question'].strip()}\nCandidate answer: {candidate}"
                    for candidate in candidates
                ]
                encoded = runtime.tokenizer(
                    prompts,
                    max_length=256,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(runtime.device)
                hidden = runtime.text_encoder(**encoded).last_hidden_state
                raw_text = pool_candidate_queries(hidden, encoded["attention_mask"])[0]
                saved_text_tensor = torch.from_numpy(
                    saved_text[row["text_feature_index"]].copy()
                ).to(device=runtime.device, dtype=torch.float32)
                text_delta = _max_abs_delta(raw_text, saved_text_tensor)
                text_deltas[row["id"]] = text_delta
                if text_delta > TEXT_FEATURE_ATOL:
                    raise AssertionError(
                        f"text feature parity exceeded fixed atol for {row['id']}: {text_delta}"
                    )

                offline_logits = readout_offline(saved_video, saved_text_tensor.unsqueeze(0))[0]
                runtime_logits = runtime.readout(raw_video, raw_text.unsqueeze(0))[0]
                logit_delta = _max_abs_delta(offline_logits, runtime_logits)
                offline_prob = torch.softmax(offline_logits, dim=-1)
                runtime_prob = torch.softmax(runtime_logits, dim=-1)
                probability_delta = _max_abs_delta(offline_prob, runtime_prob)
                logit_deltas[row["id"]] = logit_delta
                probability_deltas[row["id"]] = probability_delta
                if logit_delta > LOGIT_ATOL:
                    raise AssertionError(
                        "end-to-end logit parity exceeded fixed atol for "
                        f"{row['id']}: {logit_delta}"
                    )
                observations.append(
                    {
                        "sample_id": row["id"],
                        "source_video_id": video_id,
                        "reasoning_type": row["reasoning_type"],
                        "video_feature_max_abs_delta": video_delta,
                        "text_feature_max_abs_delta": text_delta,
                        "logit_max_abs_delta": logit_delta,
                        "probability_max_abs_delta": probability_delta,
                    }
                )
        if args.device == "cuda":
            torch.cuda.synchronize()
    finally:
        runtime.close()

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "status": "TRAIN_RUNTIME_PARITY_PASS",
        "purpose": (
            "raw-input runtime parity with frozen saved features/readout; not quality evaluation"
        ),
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "command_line": sys.argv,
        "split": "official TRAIN fit-side groups only",
        "fit_video_group_count": FIT_GROUP_COUNT,
        "fit_video_group_ids_sha256": FIT_GROUPS_SHA256,
        "internal_selection_groups_excluded": SELECTION_GROUP_COUNT,
        "internal_selection_group_ids_sha256": SELECTION_GROUPS_SHA256,
        "previous_runtime_canary_video_ids_excluded": sorted(
            PREVIOUS_RUNTIME_CANARY_VIDEO_IDS
        ),
        "sample_count": len(observations),
        "video_group_count": len(video_deltas),
        "sample_ids": [item["sample_id"] for item in observations],
        "sample_id_order_sha256": hashlib.sha256(
            "\n".join(item["sample_id"] for item in observations).encode()
        ).hexdigest(),
        "video_ids": sorted(video_deltas),
        "video_feature_sha256": {
            row["source_video_id"]: row["video_feature_sha256"] for row in rows
        },
        "input_hashes": {
            "train_records_sha256": TRAIN_SHA256,
            "train_text_features_sha256": TRAIN_FEATURE_SHA256,
            "train_text_report_sha256": sha256_file(args.text_report),
            "video_feature_manifest_sha256": VIDEO_MANIFEST_SHA256,
            "readout_sha256": _file_sha256(args.readout),
            "video_checkpoint_sha256": VIDEO_MODEL_SHA256,
            "text_checkpoint_sha256": TEXT_MODEL_SHA256,
            "media_sha256": {
                video_id: _file_sha256(path) for video_id, path in media_by_video.items()
            },
        },
        "fixed_absolute_tolerances": {
            "video_feature": VIDEO_FEATURE_ATOL,
            "text_feature": TEXT_FEATURE_ATOL,
            "logit": LOGIT_ATOL,
        },
        "max_absolute_deltas": {
            "video_feature_by_video": video_deltas,
            "text_feature_by_sample": text_deltas,
            "logits_by_sample": logit_deltas,
            "probabilities_by_sample": probability_deltas,
        },
        "observations": observations,
        "elapsed_seconds": elapsed,
        "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
        "targets_or_labels_used": False,
        "validation_or_sealed_audit_opened": False,
        "accuracy_or_calibration_metrics_computed": False,
        "training_performed": False,
    }
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    result["result_sha256"] = sha256_file(path)
    (args.output_dir / "result.sha256").write_text(result["result_sha256"] + "  result.json\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--text-report", type=Path, required=True)
    parser.add_argument("--video-feature-dir", type=Path, required=True)
    parser.add_argument("--video-feature-manifest", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, required=True)
    parser.add_argument("--text-model-dir", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
