"""Evaluate the frozen corrected GRID AV sample with fixed model and controls."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_temporal import digest
from tiny_omni_decision.native_av_correspondence import NativeAVCorrespondence

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SAMPLE = ROOT / "grid-av-direct-official-pilot-20261028-attempt-03"
BUNDLES = ROOT / "grid-official-speaker-audio-20261028-attempt-01"
VIDEOS = ROOT / "grid-validation-video-20261028-attempt-01"
RANGE_DIR = ROOT / "grid-av-direct-video-range-eval-20261028-attempt-01"
SPEECH = Path("C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01")
AUDIO_HEAD = ROOT / "lombard-temporal-full-01/audio"
VIDEO_HEAD = ROOT / "lombard-temporal-full-01/video_ordered"
PARENT = ROOT / "lombard-correspondence-02/result.json"
VIDEO_DEPS = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages")
VIDEO_SOURCE = Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")
WORKER = Path("scripts/native_av_video_worker.py")
PINNED = {
    SAMPLE / "selected.jsonl": "e897ff02a435a463211f0a157495058adcb5c390bf384437e3e56dc8066da281",
    SAMPLE / "summary.json": "cf572ae2e71ccfc1acd939d829b5880d897175422a964e8fe0c5d064d6022412",
    SPEECH
    / "encoder.safetensors": "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8",
    SPEECH / "config.json": "d8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9",
    SPEECH / "preprocessor_config.json": (
        "3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef"
    ),
    AUDIO_HEAD / "best.pt": "ade277dc1adf24bd621c838ebb8be8a7cbf26380889d45a3893400b4723a491a",
    VIDEO_HEAD / "best.pt": "4261a296711b0cfac2a7681bc250ddc4c2d59d8265119f0143e798e2936397d9",
    AUDIO_HEAD
    / "normalizer.npz": "13e3c7367c9c0a079c29f4acdff5ce4263883cfaeffd1714f646210b0698c289",
    VIDEO_HEAD
    / "normalizer.npz": "13e3c7367c9c0a079c29f4acdff5ce4263883cfaeffd1714f646210b0698c289",
    PARENT: "9638d06b7fe4c0e3c53e0a0080d38b350eb135fd8cf819e2e3fcfdbf8e977516",
    Path(
        "scripts/lombard_temporal_model.py"
    ): "ba0e87fdacd9a465cbbac6557b0c66f333752f6fb6929123a53258546b56daee",
    Path("src/tiny_omni_decision/native_av_correspondence.py"): (
        "d18eb335082f8d6a326b07c5ecfbc22e83b059c6e3a4b4b55b45bf9fab2f7311"
    ),
    WORKER: "5857a1d8973c7995c67efed448a87561e43a64ed122e49446d1d1e3f1a074093",
    VIDEOS / "s30.zip": "69e7c2ef14b1bd4f2d0012a8a13fe8e9375c4f11a03c4de9f03f4071f48f12fd",
}
REPO = Path(__file__).resolve().parents[1]


def read_audio_members(path: Path, speaker: str, codes: set[str]) -> dict[str, bytes]:
    wanted = {f"{speaker}/{code}.wav" for code in codes}
    result = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if member.isfile() and member.name in wanted:
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("missing audio payload stream")
                result[member.name.rsplit("/", 1)[-1][:-4]] = source.read()
    if set(result) != codes:
        raise ValueError(f"{speaker}: selected audio members missing")
    return result


def metrics(rows: list[dict], field: str) -> dict:
    if not rows:
        raise ValueError("metrics require predictions")
    correct = np.asarray([r[field]["selected_option_id"] == r["target_id"] for r in rows])
    probs = np.asarray(
        [[r[field]["probabilities"][option] for option in r["option_ids"]] for r in rows],
        dtype=np.float64,
    )
    targets = np.asarray([r["option_ids"].index(r["target_id"]) for r in rows])
    onehot = np.eye(probs.shape[1])[targets]
    confidence = probs.max(1)
    ece = 0.0
    for low in np.linspace(0, 0.9, 10):
        selected = (confidence >= low) & (confidence < low + 0.1)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {
        "n": len(rows),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(np.maximum(probs[np.arange(len(rows)), targets], 1e-15)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece_10bin": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def speaker_bootstrap(rows: list[dict], field_a: str, field_b: str, *, iterations=10000) -> dict:
    speakers = sorted({row["speaker"] for row in rows})
    per_speaker = {
        speaker: np.asarray(
            [
                (row[field_a]["selected_option_id"] == row["target_id"])
                - (row[field_b]["selected_option_id"] == row["target_id"])
                for row in rows
                if row["speaker"] == speaker
            ],
            dtype=np.float64,
        )
        for speaker in speakers
    }
    rng = np.random.default_rng(1701)
    samples = np.empty(iterations, np.float64)
    for i in range(iterations):
        chosen = rng.choice(speakers, len(speakers), replace=True)
        samples[i] = np.concatenate([per_speaker[s] for s in chosen]).mean()
    observed = np.concatenate(list(per_speaker.values())).mean()
    return {
        "group": "speaker",
        "speakers": speakers,
        "iterations": iterations,
        "seed": 1701,
        "mean_accuracy_difference": float(observed),
        "percentile_95_interval": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-range-dir", type=Path, default=RANGE_DIR)
    parser.add_argument("--worker-batch-size", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve() if args.output else None
    if (
        (not args.preflight_only and out is None)
        or (out is not None and (out.exists() or out.is_relative_to(REPO)))
        or not 1 <= args.worker_batch_size <= 40
    ):
        raise ValueError("new external output and safe worker batch size required")
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError(f"pinned input changed: {path}")
    selection_path = SAMPLE / "selected.jsonl"
    rows = [json.loads(line) for line in selection_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 64 or len({row["quartet_id"] for row in rows}) != 64:
        raise ValueError("frozen 64-quartet sample changed")
    codes_by_speaker = {row["speaker"]: set() for row in rows}
    for row in rows:
        codes_by_speaker[row["speaker"]].update(row["codes"])
    if sum(map(len, codes_by_speaker.values())) != 255:
        raise ValueError("frozen distinct media count differs")

    audio_archives = {}
    builder_summary = json.loads((SAMPLE / "summary.json").read_bytes())
    verified_source_hashes = {}
    for speaker, codes in codes_by_speaker.items():
        path = BUNDLES / f"{speaker}.tar.part"
        source = builder_summary["source_audits"][speaker]
        expected_archive = source["input_sha256"]["audio_tar"]
        actual_audio_hash = digest(path)
        if actual_audio_hash != expected_archive:
            raise ValueError(f"{speaker}: audio source archive hash differs")
        actual_alignment_hash = digest(BUNDLES / f"{speaker}-align.tar")
        if actual_alignment_hash != source["input_sha256"]["alignment_tar"]:
            raise ValueError(f"{speaker}: alignment source archive hash differs")
        actual_tail_hash = digest(BUNDLES / f"{speaker}-video-tail.bin")
        if actual_tail_hash != source["input_sha256"]["video_metadata_tail"]:
            raise ValueError(f"{speaker}: video metadata range hash differs")
        verified_source_hashes[speaker] = {
            "audio_tar": actual_audio_hash,
            "alignment_tar": actual_alignment_hash,
            "video_metadata_tail": actual_tail_hash,
        }
        audio_archives[speaker] = read_audio_members(path, speaker, codes)
    audio_bytes = {
        (speaker, code): payload
        for speaker, members in audio_archives.items()
        for code, payload in members.items()
    }
    video_records = json.loads((args.video_range_dir / "range-extraction.json").read_bytes())[
        "files"
    ]
    video_bytes = {}
    for item in video_records:
        key = (item["speaker"], item["code"])
        payload = Path(item["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("range-extracted media hash changed")
        video_bytes[key] = payload
    s30_codes = codes_by_speaker["s30"]
    with zipfile.ZipFile(VIDEOS / "s30.zip") as archive:
        for code in s30_codes:
            payload = archive.read(f"s30/{code}.mpg")
            video_bytes[("s30", code)] = payload
    if set(audio_bytes) != set(video_bytes) or len(video_bytes) != 255:
        raise ValueError("selected audio/video identities do not align")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_PREFLIGHT",
                    "quartets": len(rows),
                    "queries": sum(len(row["queries"]) for row in rows),
                    "unique_audio_assets": len(audio_bytes),
                    "unique_video_assets": len(video_bytes),
                    "range_video_assets": len(video_records),
                    "source_hashes_verified": len(verified_source_hashes),
                    "gpu_used": False,
                    "sealed_audit_read": False,
                }
            )
        )
        return

    assert out is not None
    out.mkdir(parents=True)
    started = time.perf_counter()
    media_dir = out / "media"
    media_dir.mkdir()
    audio_hash = {}
    video_hash = {}
    for (speaker, code), payload in audio_bytes.items():
        sha = hashlib.sha256(payload).hexdigest()
        audio_hash[(speaker, code)] = sha
        (media_dir / f"{sha}.wav").write_bytes(payload)
    for (speaker, code), payload in video_bytes.items():
        sha = hashlib.sha256(payload).hexdigest()
        video_hash[(speaker, code)] = sha
        (media_dir / f"{sha}.mpg").write_bytes(payload)

    import av
    import torch
    from safetensors.torch import load_file
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    from scripts.lombard_temporal_model import TemporalSlotHead

    torch.set_num_threads(4)
    encoder = WhisperEncoder(WhisperConfig.from_pretrained(SPEECH, local_files_only=True))
    encoder.load_state_dict(load_file(SPEECH / "encoder.safetensors"), strict=True)
    encoder.eval().requires_grad_(False)
    processor = WhisperFeatureExtractor.from_pretrained(SPEECH, local_files_only=True)
    audio_head = TemporalSlotHead().eval().requires_grad_(False)
    audio_head.load_state_dict(
        torch.load(AUDIO_HEAD / "best.pt", map_location="cpu", weights_only=True), strict=True
    )
    video_head = TemporalSlotHead().eval().requires_grad_(False)
    video_head.load_state_dict(
        torch.load(VIDEO_HEAD / "best.pt", map_location="cpu", weights_only=True), strict=True
    )
    with np.load(AUDIO_HEAD / "normalizer.npz", allow_pickle=False) as values:
        audio_mean, audio_scale = values["audio_mean"].copy(), values["audio_scale"].copy()
    with np.load(VIDEO_HEAD / "normalizer.npz", allow_pickle=False) as values:
        video_mean, video_scale = values["video_mean"].copy(), values["video_scale"].copy()
    if (
        audio_mean.shape != (384,)
        or audio_scale.shape != (384,)
        or video_mean.shape != (768,)
        or video_scale.shape != (768,)
    ):
        raise ValueError("frozen AV normalizer dimensions differ from the heads")

    audio_logits = {}
    audio_duration = {}
    with torch.inference_mode():
        for index, (key, payload) in enumerate(audio_bytes.items(), 1):
            with av.open(io.BytesIO(payload)) as container:
                stream = container.streams.audio[0]
                if stream.codec_context.sample_rate != 25000:
                    raise ValueError("GRID source audio must be 25 kHz")
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                samples = []
                for frame in container.decode(stream):
                    samples.extend(value.to_ndarray() for value in resampler.resample(frame))
                samples.extend(value.to_ndarray() for value in resampler.resample(None))
            wave = np.concatenate(samples, axis=1)[0]
            features = processor(wave, sampling_rate=16000, return_tensors="pt").input_features
            hidden = encoder(features).last_hidden_state[0]
            duration = len(wave) / 16000
            valid = max(1, min(hidden.shape[0], int((duration * 16000 + 319) // 320)))
            pooled = hidden[:valid].mean(0).numpy()
            normalized = torch.from_numpy(((pooled - audio_mean) / audio_scale)[None])
            logits = audio_head(
                normalized, torch.zeros((1, 16, 768)), torch.tensor([[1.0, 0.0]])
            ).numpy()[0]
            if logits.shape != (51,) or not np.isfinite(logits).all():
                raise ValueError("invalid frozen audio logits")
            audio_logits[key] = logits
            audio_duration[key] = duration
            if index % 32 == 0 or index == len(audio_bytes):
                print(f"audio_features={index}/{len(audio_bytes)}", flush=True)

    video_logits = {}
    video_time_ranges = {}
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = os.pathsep.join(
        map(str, (VIDEO_DEPS, VIDEO_SOURCE, REPO, REPO / "src"))
    )
    worker_env["JAX_PLATFORMS"] = "cpu"
    worker_env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    worker_reports = []
    unique_video_items = list(video_bytes.items())
    for batch_index in range(0, len(unique_video_items), args.worker_batch_size):
        batch = unique_video_items[batch_index : batch_index + args.worker_batch_size]
        worker_out = out / f"video-worker-{batch_index // args.worker_batch_size:03}"
        command = [
            sys.executable,
            "-m",
            "scripts.native_av_video_worker",
            "--output",
            str(worker_out),
        ]
        with (out / f"video-worker-{batch_index // args.worker_batch_size:03}.log").open(
            "w", encoding="utf-8"
        ) as log:
            worker = subprocess.Popen(
                command,
                cwd=REPO,
                env=worker_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                ready = json.loads(worker.stdout.readline())
                if not ready.get("ready"):
                    raise RuntimeError("VideoPrism worker failed initialization")
                for i, (key, _payload) in enumerate(batch, 1):
                    sha = video_hash[key]
                    media_path = media_dir / f"{sha}.mpg"
                    worker.stdin.write(json.dumps({"path": str(media_path), "sha256": sha}) + "\n")
                    worker.stdin.flush()
                    reply = json.loads(worker.stdout.readline())
                    feature = np.asarray(reply["feature"], dtype=np.float32)
                    if feature.shape != (16, 768) or not np.isfinite(feature).all():
                        raise ValueError("invalid VideoPrism feature")
                    normalized = torch.from_numpy(((feature - video_mean) / video_scale)[None])
                    logits = video_head(
                        torch.zeros((1, 384)), normalized, torch.tensor([[0.0, 1.0]])
                    ).numpy()[0]
                    if logits.shape != (51,) or not np.isfinite(logits).all():
                        raise ValueError("invalid frozen video logits")
                    video_logits[key] = logits
                    pts = reply["metadata"]["pts"]
                    if len(pts) != 16 or pts[-1] <= pts[0]:
                        raise ValueError("invalid VideoPrism temporal support")
                    video_time_ranges[key] = (float(pts[0]), float(pts[-1]))
                    if i % 8 == 0 or i == len(batch):
                        print(
                            f"video_features={batch_index + i}/{len(unique_video_items)}",
                            flush=True,
                        )
                worker.stdin.close()
                if worker.wait(timeout=600) != 0:
                    raise RuntimeError("VideoPrism worker failed")
            finally:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait(timeout=15)
        worker_report = json.loads((worker_out / "result.json").read_bytes())
        if worker_report.get("status") != "PASS_NATIVE_VIDEO_WORKER":
            raise ValueError("VideoPrism terminal integrity check failed")
        worker_reports.append(worker_report)

    profile = hashlib.sha256(
        json.dumps({str(p): h for p, h in PINNED.items()}, sort_keys=True).encode()
    ).hexdigest()
    calibration = json.loads(PARENT.read_bytes())
    runtime = NativeAVCorrespondence(
        lambda content: audio_logits[
            next(
                key
                for key, value in audio_hash.items()
                if value == hashlib.sha256(content).hexdigest()
            )
        ],
        lambda content: video_logits[
            next(
                key
                for key, value in video_hash.items()
                if value == hashlib.sha256(content).hexdigest()
            )
        ],
        calibration["models"]["audio"]["temperatures"],
        calibration["models"]["video_ordered"]["temperatures"],
        calibration["query_temperature"],
        profile,
        max_entries=16,
    )
    audio_states = {
        key: runtime.encode(
            "audio", payload, time_range=(0.0, audio_duration[key]), expected_sha256=audio_hash[key]
        )
        for key, payload in audio_bytes.items()
    }
    video_states = {
        key: runtime.encode(
            "video", payload, time_range=video_time_ranges[key], expected_sha256=video_hash[key]
        )
        for key, payload in video_bytes.items()
    }
    decisions = []
    permutation = [2, 0, 3, 1]
    for row in rows:
        videos = [video_states[(row["speaker"], code)] for code in row["codes"]]
        for query in row["queries"]:
            audio = audio_states[(row["speaker"], query["audio_code"])]
            ids = list(row["option_ids"])

            def decision(a, v, option_ids=ids):
                return json.loads(runtime.decide(a, v, option_ids))

            baseline = decision(audio, videos)
            reordered = json.loads(
                runtime.decide(
                    audio, [videos[i] for i in permutation], [ids[i] for i in permutation]
                )
            )
            wrong_audio_code = next(code for code in row["codes"] if code != query["audio_code"])
            wrong_audio = decision(audio_states[(row["speaker"], wrong_audio_code)], videos)
            wrong_video = json.loads(runtime.decide(audio, videos[1:] + videos[:1], ids))
            decisions.append(
                {
                    "speaker": row["speaker"],
                    "quartet_id": row["quartet_id"],
                    "audio_code": query["audio_code"],
                    "target_id": query["target_id"],
                    "option_ids": ids,
                    "joint": baseline,
                    "reordered": reordered,
                    "audio_only_policy": decision(audio, None),
                    "video_only_policy": decision(None, videos),
                    "mismatched_audio": wrong_audio,
                    "mismatched_video": wrong_video,
                }
            )
    if len(decisions) != 256 or any(d["joint"] != d["reordered"] for d in decisions):
        raise ValueError("incomplete query set or order-invariance failure")
    result = {
        "status": "PASS_FIXED_DEVELOPMENT_EVALUATION",
        "scope": "256 queries in frozen corrected development sample; not blind audit",
        "selection_sha256": PINNED[selection_path],
        "query_count": len(decisions),
        "unique_audio": len(audio_bytes),
        "unique_video": len(video_bytes),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": {
            field: metrics(decisions, field)
            for field in (
                "joint",
                "audio_only_policy",
                "video_only_policy",
                "mismatched_audio",
                "mismatched_video",
            )
        },
        "speaker_metrics": {
            speaker: {
                field: metrics([d for d in decisions if d["speaker"] == speaker], field)
                for field in (
                    "joint",
                    "audio_only_policy",
                    "video_only_policy",
                    "mismatched_audio",
                    "mismatched_video",
                )
            }
            for speaker in sorted({d["speaker"] for d in decisions})
        },
        "paired_speaker_bootstrap": {
            field: speaker_bootstrap(decisions, "joint", field)
            for field in (
                "audio_only_policy",
                "video_only_policy",
                "mismatched_audio",
                "mismatched_video",
            )
        },
        "predictions": decisions,
        "worker_batches": len(worker_reports),
        "worker_parameter_sha": {
            "before": worker_reports[0]["parameter_sha_before"],
            "after": worker_reports[-1]["parameter_sha_after"],
            "unchanged_each_batch": all(
                r["parameter_sha_before"] == r["parameter_sha_after"] for r in worker_reports
            ),
        },
        "gpu_used": False,
        "training": False,
        "checkpoint_selection": False,
        "sealed_audit_read": False,
        "interpretation": "One fixed development sample; no tuning or promotion.",
        "input_hashes": {str(path): expected for path, expected in PINNED.items()},
        "verified_source_hashes_by_speaker": verified_source_hashes,
        "video_range_extraction_manifest_sha256": digest(
            args.video_range_dir / "range-extraction.json"
        ),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "pyav": av.__version__,
            "device": "cpu",
            "default_dtype": str(torch.get_default_dtype()),
        },
        "script_sha256": digest(Path(__file__)),
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "queries": len(decisions),
                "accuracy": result["metrics"]["joint"]["accuracy"],
                "seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
