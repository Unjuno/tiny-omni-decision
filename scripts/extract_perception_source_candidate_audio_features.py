"""Extract label-free source/offset audio-window embeddings under HotUGPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tiny_omni_decision.perception_source_candidate_audio import (
    materialize_audio_window_at_offset,
)

WEIGHTS_SHA256 = "49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e"
SOURCE_COMMIT = "a425fdce92572e602a1d5634799bd9f1f2efa806"
MODEL_PARAMETERS = 1_961_239
SAMPLE_RATE = 32_000
WINDOW_SAMPLES = 2 * SAMPLE_RATE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_inputs(plan, report, split_name, plan_path, report_path):
    if plan.get("format") != "perception_test_source_candidate_feature_plan_v1":
        raise ValueError("unexpected source-candidate feature plan format")
    if not plan.get("label_free") or plan.get("model_loaded") or plan.get("metrics_computed"):
        raise ValueError("feature plan must be label-free and unscored")
    if plan.get(f"{split_name}_media_report_sha256") != sha256_file(report_path):
        raise ValueError("media report hash differs from frozen feature plan")
    if report.get("status") not in {"PASS", "PASS_WITH_EXPLICIT_TERMINAL_FRAME_REPAIR"}:
        raise ValueError("media report did not pass fixed-split preflight")
    report_rows = {row["record_id"]: row for row in report["records"]}
    split_rows = plan[split_name]["records"]
    if len(report_rows) != len(report["records"]):
        raise ValueError("duplicate record IDs in media report")
    if set(report_rows) != {row["record_id"] for row in split_rows}:
        raise ValueError("media report IDs differ from frozen feature plan")
    for row in split_rows:
        checked = report_rows[row["record_id"]]
        if (
            checked.get("video_id") != row["video_id"]
            or checked.get("terminal_frame_repair") != row.get("terminal_frame_repair")
            or not checked.get("media_sha256")
        ):
            raise ValueError(f"media report mismatch for {row['record_id']}")
    return report_rows


def decode_resampled_mono_audio(path: Path, *, sample_rate_hz: int = SAMPLE_RATE):
    """Decode the media audio to a PTS-aligned mono float32 waveform."""
    import av

    chunks = []
    origin_us = None
    cursor = 0
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            raise ValueError(f"audio stream missing: {path}")
        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate_hz
        )
        decoded_frames = []
        for source_frame in container.decode(streams[0]):
            decoded_frames.extend(resampler.resample(source_frame))
        decoded_frames.extend(resampler.resample(None))
        for frame in decoded_frames:
            if frame.time is not None:
                frame_start_us = round(float(frame.time) * 1_000_000)
            elif frame.pts is not None and frame.time_base is not None:
                frame_start_us = round(float(frame.pts * frame.time_base) * 1_000_000)
            else:
                raise ValueError(f"resampled audio frame has no PTS: {path}")
            if origin_us is None:
                origin_us = max(0, frame_start_us)
            values = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError(f"resampled audio contains nonfinite samples: {path}")
            start = round((frame_start_us - origin_us) * sample_rate_hz / 1_000_000)
            if start < 0:
                trim = min(len(values), -start)
                values = values[trim:]
                start += trim
            delta = start - cursor
            if delta < 0:
                if -delta > sample_rate_hz // 100:
                    raise ValueError(f"resampled audio frames overlap unexpectedly: {path}")
                overlap = min(len(values), -delta)
                values = values[overlap:]
                start += overlap
                delta = start - cursor
            if delta > sample_rate_hz // 10:
                raise ValueError(f"resampled audio has an unexpected timestamp gap: {path}")
            if delta:
                chunks.append(np.zeros(delta, dtype=np.float32))
                cursor += delta
            if len(values):
                chunks.append(values)
                cursor += len(values)
    if origin_us is None or not chunks:
        raise ValueError(f"video has no decodable audio: {path}")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise ValueError(f"decoded audio waveform is invalid: {path}")
    return waveform, origin_us


def collect_unique_windows(records):
    unique = []
    lookup = {}
    references = []
    for record in records:
        for variant, name in (("source", "audio_windows"), ("offset", "shifted_audio_windows")):
            windows = record.get(name)
            if windows is None:
                continue
            for window_index, window in enumerate(windows):
                key = (
                    record["video_id"],
                    window["source_start_us"],
                    window["source_end_us"],
                    window["pad_left_us"],
                    window["pad_right_us"],
                )
                feature_index = lookup.get(key)
                if feature_index is None:
                    feature_index = len(unique)
                    lookup[key] = feature_index
                    unique.append({"video_id": record["video_id"], **window})
                references.append(
                    {
                        "record_id": record["record_id"],
                        "video_id": record["video_id"],
                        "variant": variant,
                        "window_index": window_index,
                        "feature_index": feature_index,
                        **window,
                    }
                )
    return unique, references


def _load_model(weights: Path, source: Path):
    if sha256_file(weights) != WEIGHTS_SHA256:
        raise ValueError("EfficientAT checkpoint SHA-256 mismatch")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("EfficientAT source differs from pinned pristine revision")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing audio feature extraction")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(source))
    with __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

    model = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != MODEL_PARAMETERS:
        raise ValueError(f"unexpected EfficientAT parameter count: {parameters}")
    model.eval().requires_grad_(False).cuda()
    mel = AugmentMelSTFT(freqm=0, timem=0).eval().requires_grad_(False).cuda()
    return torch, model, mel, parameters, commit


def _state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("TINY_OMNI_GPU_GATE_RESERVED_MIB"):
        raise RuntimeError("audio feature extraction must run under run_with_hotgpu_guard.py")
    plan_path = args.plan.resolve(strict=True)
    acquisition_path = args.acquisition_manifest.resolve(strict=True)
    report_path = args.media_report.resolve(strict=True)
    weights = args.weights.resolve(strict=True)
    source = args.source.resolve(strict=True)
    media_dir = args.media_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audio feature output: {output}")
    if not 1 <= args.batch_size <= 64:
        raise ValueError("batch size must be between 1 and 64")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    media_report = json.loads(report_path.read_text(encoding="utf-8"))
    report_rows = _validate_inputs(plan, media_report, args.split, plan_path, report_path)
    acquisition_sha = sha256_file(acquisition_path)
    if (
        plan.get("acquisition_manifest_sha256") != acquisition_sha
        or media_report.get("acquisition_manifest_sha256") != acquisition_sha
    ):
        raise ValueError("acquisition manifest hash mismatch")

    clips = {}
    for clip in acquisition["clips"]:
        video_id = clip["metadata"]["video_id"]
        if video_id in clips:
            raise ValueError(f"duplicate media receipt for {video_id}")
        clips[video_id] = clip
    records = plan[args.split]["records"]
    unique_windows, references = collect_unique_windows(records)
    windows_by_video = {}
    for index, window in enumerate(unique_windows):
        windows_by_video.setdefault(window["video_id"], []).append((index, window))
    if not unique_windows:
        raise ValueError("fixed split contains no audio windows")

    torch, model, mel, parameters, commit = _load_model(weights, source)
    before_state = _state_sha(model)
    device_name = torch.cuda.get_device_name(0)
    gate = {
        key: value for key, value in os.environ.items() if key.startswith("TINY_OMNI_GPU_GATE_")
    }
    output.mkdir(parents=True)
    run_metadata = {
        "status": "RUNNING",
        "plan_sha256": sha256_file(plan_path),
        "acquisition_manifest_sha256": acquisition_sha,
        "media_report_sha256": sha256_file(report_path),
        "weights_sha256": WEIGHTS_SHA256,
        "source_commit": commit,
        "model_parameters": parameters,
        "device": device_name,
        "dtype": str(next(model.parameters()).dtype),
        "split": args.split,
        "records": len(records),
        "unique_audio_windows": len(unique_windows),
        "audio_references": len(references),
        "sample_rate_hz": SAMPLE_RATE,
        "window_seconds": 2,
        "batch_size": args.batch_size,
        "gpu_gate": gate,
        "labels_loaded": False,
        "metrics_computed": False,
    }
    atomic_json(output / "run.json", run_metadata)

    features = np.empty((len(unique_windows), 384), dtype=np.float32)
    started = time.perf_counter()
    peak_allocated = 0
    peak_reserved = 0
    completed_windows = 0
    with torch.inference_mode():
        for video_index, (video_id, video_windows) in enumerate(windows_by_video.items(), start=1):
            clip = clips.get(video_id)
            media_hashes = {
                row["media_sha256"] for row in report_rows.values() if row["video_id"] == video_id
            }
            if clip is None or media_hashes != {clip["sha256"]}:
                raise ValueError(f"preflight media hash differs from acquisition: {video_id}")
            media_path = media_dir / f"{video_id}.mp4"
            if (
                media_path.stat().st_size != clip["bytes"]
                or sha256_file(media_path) != clip["sha256"]
            ):
                raise ValueError(f"audio media file differs from acquisition: {video_id}")
            waveform, waveform_start_us = decode_resampled_mono_audio(media_path)
            materialized = [
                (
                    index,
                    materialize_audio_window_at_offset(
                        waveform, window, waveform_start_us=waveform_start_us
                    ),
                )
                for index, window in video_windows
            ]
            for start in range(0, len(materialized), args.batch_size):
                batch_rows = materialized[start : start + args.batch_size]
                batch_wave = np.stack([row[1] for row in batch_rows])
                if batch_wave.shape[1] != WINDOW_SAMPLES or not np.isfinite(batch_wave).all():
                    raise ValueError(f"invalid fixed audio window batch: {video_id}")
                waveform_tensor = torch.from_numpy(batch_wave).cuda()
                mel_input = mel(waveform_tensor).unsqueeze(1)
                logits, embeddings = model(mel_input)
                torch.cuda.synchronize()
                if (
                    logits.shape != (len(batch_rows), 527)
                    or embeddings.shape != (len(batch_rows), 384)
                    or not torch.isfinite(embeddings).all()
                ):
                    raise ValueError(f"invalid EfficientAT output: {video_id}")
                for (feature_index, _), embedding in zip(
                    batch_rows, embeddings.float().cpu().numpy(), strict=True
                ):
                    features[feature_index] = embedding
                completed_windows += len(batch_rows)
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            print(
                json.dumps(
                    {
                        "completed_videos": video_index,
                        "total_videos": len(windows_by_video),
                        "completed_windows": completed_windows,
                        "total_windows": len(unique_windows),
                        "elapsed_seconds": time.perf_counter() - started,
                        "latest_video_id": video_id,
                        "peak_allocated_bytes": peak_allocated,
                    }
                ),
                flush=True,
            )

    if completed_windows != len(unique_windows) or not np.isfinite(features).all():
        raise ValueError("audio feature extraction is incomplete or nonfinite")
    after_state = _state_sha(model)
    if before_state != after_state:
        raise RuntimeError("frozen EfficientAT weights changed during extraction")

    feature_path = output / "audio_window_features.npy"
    with feature_path.open("xb") as stream:
        np.save(stream, features, allow_pickle=False)
    unique_index_path = output / "unique_window_index.jsonl"
    with unique_index_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, window in enumerate(unique_windows):
            stream.write(json.dumps({"feature_index": index, **window}, sort_keys=True) + "\n")
    reference_path = output / "record_window_references.jsonl"
    with reference_path.open("x", encoding="utf-8", newline="\n") as stream:
        for reference in references:
            stream.write(json.dumps(reference, sort_keys=True) + "\n")

    result = {
        "status": "PASS_SOURCE_CANDIDATE_AUDIO_FEATURES",
        "split": args.split,
        "plan_sha256": sha256_file(plan_path),
        "media_report_sha256": sha256_file(report_path),
        "weights_sha256": WEIGHTS_SHA256,
        "source_commit": commit,
        "device": device_name,
        "model_parameters": parameters,
        "model_state_sha256_before": before_state,
        "model_state_sha256_after": after_state,
        "records": len(records),
        "unique_audio_windows": len(unique_windows),
        "audio_references": len(references),
        "feature_shape": list(features.shape),
        "feature_sha256": sha256_file(feature_path),
        "unique_index_sha256": sha256_file(unique_index_path),
        "reference_index_sha256": sha256_file(reference_path),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_gate": gate,
        "labels_loaded": False,
        "metrics_computed": False,
        "validation_scope": "internal fixed development partition; non-blind",
    }
    atomic_json(output / "result.json", result)
    atomic_json(
        output / "run.json",
        {**run_metadata, **result, "result_sha256": sha256_file(output / "result.json")},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--media-report", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    print(json.dumps(run(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
