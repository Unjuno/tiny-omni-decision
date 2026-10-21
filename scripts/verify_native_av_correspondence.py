"""Actual raw AV->frozen heads->custom JSON, repeated reuse and fresh replay."""

import argparse
import hashlib
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import psutil
import safetensors
import torch
import transformers
from safetensors.torch import load_file
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from scripts.lombard_correspondence import OPTION_IDS, agreement_scores, choice_json
from scripts.lombard_readout_data import read_json, sha
from scripts.lombard_temporal_model import TemporalSlotHead
from scripts.robomnist_audio_cue import recheck
from scripts.train_lombard_temporal import ROOT, source_scope, state_sha
from tiny_omni_decision.native_av_correspondence import NativeAVCorrespondence

PARENT_SHA = "9638d06b7fe4c0e3c53e0a0080d38b350eb135fd8cf819e2e3fcfdbf8e977516"
CODE = [
    "scripts/verify_native_av_correspondence.py",
    "scripts/native_av_video_worker.py",
    "src/tiny_omni_decision/native_av_correspondence.py",
    "tests/test_native_av_correspondence.py",
    "docs/pretrained_reuse/R5_NATIVE_AV_CORRESPONDENCE_PROTOCOL_20261015.md",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256):
        raise ValueError("paired replay arguments required")
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    worker = None
    log = None
    result = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=sys.argv,
        model_updates=0,
        gpu_used=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        decisions=[],
    )
    observed_total_peak = 0

    def guard():
        nonlocal observed_total_peak
        process = psutil.Process()
        children = process.children(recursive=True)
        resident = process.memory_info().rss
        for child in children:
            try:
                resident += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        observed_total_peak = max(observed_total_peak, resident)
        peak = getattr(process.memory_info(), "peak_wset", resident)
        size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
        if (
            time.perf_counter() - tick > 600
            or peak > 6 * 2**30
            or size + 2**22 > 64 * 2**20
            or shutil.disk_usage(args.output).free < 2**30
        ):
            raise RuntimeError("native AV safety guard")

    try:
        parent = read_json(ROOT / "lombard-correspondence-02/result.json", PARENT_SHA)
        if (
            parent["status"] != "PASS_FROZEN_NATIVE_CORRESPONDENCE"
            or parent["query_temperature"] != 1.0
        ):
            raise ValueError("frozen complete parent required")
        pins = dict(parent["validated_hashes"])
        pins.update({str(Path(p).resolve()): sha(Path(p)) for p in CODE})
        for module in [av, safetensors, transformers]:
            for p in Path(module.__file__).resolve().parent.rglob("*"):
                if p.is_file() and p.suffix in {".py", ".pyd", ".dll"}:
                    pins[str(p)] = sha(p)
        pins[str(ROOT / "lombard-correspondence-02/result.json")] = PARENT_SHA
        for name, h in parent["artifact_hashes"].items():
            pins[str(ROOT / "lombard-correspondence-02" / name)] = h
        speech = Path("C:/CodexArtifacts/pretrained-reuse-r1/whisper-encoder-runtime-01")
        for name, h in [
            (
                "encoder.safetensors",
                "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8",
            ),
            ("config.json", "d8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9"),
            (
                "preprocessor_config.json",
                "3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef",
            ),
        ]:
            pins[str(speech / name)] = h
        recheck(pins)
        guard()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        raw, framed, _, rows, provenance = source_scope(False)
        result["provenance"] = provenance
        quartets = {
            p: np.load(
                ROOT / "lombard-correspondence-02" / f"{p}-train-catalog.npy", allow_pickle=False
            )[0].tolist()
            for p in ["random", "hard"]
        }
        chosen = sorted(set(sum(quartets.values(), [])))
        result.update(
            quartets=quartets,
            selected_source_ids=[rows["train"][i]["source_id"] for i in chosen],
            selection="first TRAIN anchor, frozen random and hard catalogs; no score selection",
        )
        models = {}
        norms = {}
        temps = {}
        before = {}
        for arm in ["audio", "video_ordered"]:
            folder = ROOT / "lombard-temporal-full-01" / arm
            model = TemporalSlotHead().eval().requires_grad_(False)
            model.load_state_dict(
                torch.load(folder / "best.pt", map_location="cpu", weights_only=True), strict=True
            )
            if sum(p.numel() for p in model.parameters()) != 106355:
                raise ValueError("full registered head required")
            models[arm] = model
            before[arm] = state_sha(model)
            with np.load(folder / "normalizer.npz", allow_pickle=False) as value:
                norms[arm] = {k: value[k].copy() for k in value.files}
            temps[arm] = parent["models"][arm]["temperatures"]
        start = time.perf_counter()
        encoder = (
            WhisperEncoder(WhisperConfig.from_pretrained(speech, local_files_only=True))
            .eval()
            .requires_grad_(False)
        )
        encoder.load_state_dict(load_file(speech / "encoder.safetensors"), strict=True)
        if sum(p.numel() for p in encoder.parameters()) != 8208384:
            raise ValueError("encoder count differs")
        encoder_before = state_sha(encoder)
        processor = WhisperFeatureExtractor.from_pretrained(speech, local_files_only=True)
        result["audio_load_seconds"] = time.perf_counter() - start
        index = read_json(ROOT / "lombard-grid-media-integrity-01/rows.json")
        lookup = {r["source_id"]: r for r in index}
        media = {}
        audio_data = {}
        video_data = {}
        by_digest = {}
        for kind in ["audio", "front"]:
            archive_path = ROOT / "lombard-grid-archives-01" / f"{kind}.zip"
            integrity = read_json(ROOT / "lombard-grid-media-integrity-01/result.json")
            pins[str(archive_path)] = integrity["source_archives"][kind]["sha256"]
            recheck({str(archive_path): pins[str(archive_path)]})
            with zipfile.ZipFile(archive_path) as archive:
                for i in chosen:
                    r = rows["train"][i]
                    content = archive.read(r["assets"][kind]["name"])
                    digest = hashlib.sha256(content).hexdigest()
                    if (
                        digest
                        != lookup[r["source_id"]][
                            "audio_sha256" if kind == "audio" else "video_sha256"
                        ]
                    ):
                        raise ValueError("native media SHA differs")
                    media[(kind, i)] = content
                    if kind == "front":
                        path = args.output / (digest + ".avi")
                        path.write_bytes(content)
                        by_digest[digest] = (i, path)
                        pins[str(path.resolve())] = digest
        messages = queue.Queue()
        env = os.environ.copy()
        deps = Path(
            "C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/venv/Lib/site-packages"
        )
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(deps),
                str(Path("C:/CodexArtifacts/pretrained-reuse-r1/videoprism-base-r1/source")),
                str(Path.cwd()),
                str(Path.cwd() / "src"),
            ]
        )
        env["JAX_PLATFORMS"] = "cpu"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        log = (args.output / "video-worker.log").open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "scripts.native_av_video_worker",
            "--output",
            str(args.output / "video-worker"),
        ]
        worker = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
        )
        threading.Thread(
            target=lambda: [messages.put(line) for line in worker.stdout], daemon=True
        ).start()

        def receive():
            while True:
                guard()
                try:
                    return json.loads(messages.get(timeout=0.2))
                except queue.Empty:
                    if worker.poll() is not None:
                        raise RuntimeError(
                            f"native worker exited {worker.returncode}; see video-worker.log"
                        ) from None

        ready = receive()
        if not ready.get("ready"):
            raise ValueError("native worker not ready")
        result["video_load_seconds"] = ready["load_seconds"]
        result["video_command"] = command

        def head(arm, audio=None, video=None):
            n = norms[arm]
            a = (
                np.zeros((1, 384), np.float32)
                if audio is None
                else ((audio - n["audio_mean"]) / n["audio_scale"])[None]
            )
            v = (
                np.zeros((1, 16, 768), np.float32)
                if video is None
                else ((video - n["video_mean"]) / n["video_scale"])[None]
            )
            present = np.array([[audio is not None, video is not None]], np.float32)
            with torch.inference_mode():
                return models[arm](
                    *(torch.from_numpy(np.ascontiguousarray(x)) for x in [a, v, present])
                ).numpy()[0]

        def audio_backend(content):
            start = time.perf_counter()
            with av.open(io.BytesIO(content)) as container:
                stream = container.streams.audio[0]
                if stream.codec_context.sample_rate != 16000:
                    raise ValueError("native16kHz required")
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                chunks = []
                for frame in container.decode(stream):
                    chunks.extend(v.to_ndarray() for v in resampler.resample(frame))
                chunks.extend(v.to_ndarray() for v in resampler.resample(None))
                wave = np.concatenate(chunks, axis=1)[0]
            if not 0 < len(wave) <= 480000 or not np.isfinite(wave).all():
                raise ValueError("complete native audio required")
            data = processor(wave, sampling_rate=16000, return_tensors="pt").input_features
            with torch.inference_mode():
                feature = (
                    encoder(data).last_hidden_state[0, : (len(wave) + 319) // 320].numpy().mean(0)
                )
            digest = hashlib.sha256(content).hexdigest()
            audio_data[digest] = dict(
                feature=feature,
                logits=head("audio", audio=feature),
                seconds=time.perf_counter() - start,
                samples=len(wave),
            )
            return audio_data[digest]["logits"]

        def video_backend(content):
            digest = hashlib.sha256(content).hexdigest()
            _, path = by_digest[digest]
            worker.stdin.write(json.dumps(dict(path=str(path.resolve()), sha256=digest)) + "\n")
            worker.stdin.flush()
            reply = receive()
            feature = np.asarray(reply["feature"], np.float32)
            if feature.shape != (16, 768):
                raise ValueError("native chronological feature shape differs")
            video_data[digest] = dict(
                feature=feature,
                logits=head("video_ordered", video=feature),
                metadata=reply["metadata"],
            )
            return video_data[digest]["logits"]

        profile = hashlib.sha256(
            json.dumps(
                dict(
                    parent_profile={
                        p: h
                        for p, h in pins.items()
                        if not Path(p).is_relative_to(args.output.resolve())
                    },
                    video_profile_sha256=ready["profile_sha256"],
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        runtime = NativeAVCorrespondence(
            audio_backend, video_backend, temps["audio"], temps["video_ordered"], 1.0, profile
        )
        states = {}
        latencies = []
        for i in chosen:
            r = rows["train"][i]
            audit = lookup[r["source_id"]]
            for kind, modality, duration in [
                ("audio", "audio", audit["samples"] / 16000),
                ("front", "video", audit["last_video_pts"]),
            ]:
                start = time.perf_counter()
                states[(modality, i)] = runtime.encode(
                    modality,
                    media[(kind, i)],
                    time_range=(0, duration),
                    expected_sha256=audit[modality + "_sha256"],
                )
                latencies.append(time.perf_counter() - start)
        differences = []
        for i in chosen:
            a = states[("audio", i)]
            v = states[("video", i)]
            af = audio_data[a.media_sha256]["feature"]
            vf = video_data[v.media_sha256]["feature"]
            if not np.array_equal(vf, framed["train"][i]):
                raise ValueError("native CPU video bank is not exact")
            delta = float(np.max(np.abs(af - raw["train"][0][i])))
            cached_audio = head("audio", audio=raw["train"][0][i])
            cached_video = head("video_ordered", video=framed["train"][i])
            ld = float(np.max(np.abs(np.asarray(a.logits) - cached_audio)))
            if delta > 1e-4 or ld > 1e-3 or not np.array_equal(np.asarray(v.logits), cached_video):
                raise ValueError("predeclared backend feature/logit gate failed")
            differences.append(
                dict(
                    source_id=rows["train"][i]["source_id"],
                    audio_max_abs_feature_delta=delta,
                    audio_max_abs_logit_delta=ld,
                    video_features_exact=True,
                    video_logits_exact=True,
                )
            )
        np.savez_compressed(
            args.output / "native-states.npz",
            audio_features=np.stack(
                [audio_data[states[("audio", i)].media_sha256]["feature"] for i in chosen]
            ),
            video_features=np.stack(
                [video_data[states[("video", i)].media_sha256]["feature"] for i in chosen]
            ),
            audio_logits=np.array([states[("audio", i)].logits for i in chosen]),
            video_logits=np.array([states[("video", i)].logits for i in chosen]),
        )
        timings = []
        for policy, q in quartets.items():
            candidates = [states[("video", i)] for i in q]
            cached_v = np.array([head("video_ordered", video=framed["train"][i]) for i in q])
            for i in q:
                start = time.perf_counter()
                value = runtime.decide(states[("audio", i)], candidates, OPTION_IDS)
                timings.append(time.perf_counter() - start)
                cached_a = head("audio", audio=raw["train"][0][i])
                expected = choice_json(
                    agreement_scores(
                        cached_a[None], cached_v[None], temps["audio"], temps["video_ordered"]
                    )[0],
                    OPTION_IDS,
                    1.0,
                )
                given = json.loads(value)
                reference = json.loads(expected)
                delta = max(
                    abs(given["probabilities"][k] - reference["probabilities"][k])
                    for k in OPTION_IDS
                )
                if delta > 1e-4 or given["selected_option_id"] != reference["selected_option_id"]:
                    raise ValueError("predeclared native decision gate failed")
                if value != runtime.decide(
                    states[("audio", i)], candidates[::-1], OPTION_IDS[::-1]
                ):
                    raise ValueError("custom JSON order differs")
                result["decisions"].append(
                    dict(
                        policy=policy,
                        source_id=rows["train"][i]["source_id"],
                        json=given,
                        reference=reference,
                        max_abs_probability_delta=delta,
                    )
                )
        count = dict(runtime.backend_calls)
        for _ in range(4):
            for i in chosen:
                audit = lookup[rows["train"][i]["source_id"]]
                for kind, modality, duration in [
                    ("audio", "audio", audit["samples"] / 16000),
                    ("front", "video", audit["last_video_pts"]),
                ]:
                    runtime.encode(modality, media[(kind, i)], time_range=(0, duration))
        for _ in range(128):
            start = time.perf_counter()
            runtime.decide(
                states[("audio", chosen[0])],
                [states[("video", i)] for i in quartets["hard"]],
                OPTION_IDS,
            )
            timings.append(time.perf_counter() - start)
        if runtime.backend_calls != count:
            raise ValueError("reuse unexpectedly invokes encoder")
        worker.stdin.close()
        while worker.poll() is None:
            guard()
            time.sleep(0.1)
        if worker.returncode != 0:
            raise RuntimeError("native worker terminal verification failed")
        child = read_json(args.output / "video-worker/result.json")
        if child["status"] != "PASS_NATIVE_VIDEO_WORKER" or len(child["calls"]) != len(chosen):
            raise ValueError("actual encoder call accounting differs")
        recheck(pins)
        if (
            encoder_before != state_sha(encoder)
            or any(state_sha(models[k]) != v for k, v in before.items())
            or torch.cuda.is_initialized()
        ):
            raise ValueError("frozen parameters/CUDA gate failed")
        guard()
        result.update(
            status="PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME",
            validated_hashes=pins,
            profile_sha256=profile,
            feature_parity=differences,
            backend_calls=count,
            cache_hits=runtime.cache_hits,
            cache_entries=runtime.cache_entries,
            encode_seconds=latencies,
            decision_seconds=timings,
            decision_p50_seconds=float(np.median(timings)),
            decision_p95_seconds=float(np.quantile(timings, 0.95)),
            observed_sampled_combined_ram_peak=observed_total_peak,
            video_worker_receipt_sha256=sha(args.output / "video-worker/result.json"),
            parameters_and_coefficients=122790899,
            python=sys.version,
            torch=torch.__version__,
            numpy=np.__version__,
            device="cpu",
            neural_dtype="float32",
            scoring_dtype="float64",
            scope="TRAIN functional/cross-backend parity; not new quality or release qualification",
            fresh_process_replay=bool(args.replay_reference),
        )
        if args.replay_reference:
            prior = read_json(args.replay_reference / "result.json", args.replay_sha256)
            if prior["status"] != "PASS_NATIVE_AV_CORRESPONDENCE_RUNTIME":
                raise ValueError("complete primary required")
            recheck(prior["validated_hashes"])
            recheck(
                {str(args.replay_reference / k): h for k, h in prior["artifact_hashes"].items()}
            )
            for k in [
                "selected_source_ids",
                "quartets",
                "decisions",
                "feature_parity",
                "backend_calls",
                "cache_hits",
                "cache_entries",
                "parameters_and_coefficients",
            ]:
                if result[k] != prior[k]:
                    raise ValueError("fresh raw replay differs: " + k)
            with np.load(args.output / "native-states.npz", allow_pickle=False) as current:
                with np.load(
                    args.replay_reference / "native-states.npz", allow_pickle=False
                ) as original:
                    if set(current.files) != set(original.files) or any(
                        not np.array_equal(current[k], original[k]) for k in current.files
                    ):
                        raise ValueError("fresh native state arrays differ")
            recheck(prior["validated_hashes"])
            recheck({str(args.replay_reference / "result.json"): args.replay_sha256})
        result["artifact_hashes"] = {
            str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*") if p.is_file()
        }
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=30)
        if log is not None:
            log.close()
        result.update(
            seconds=time.perf_counter() - tick,
            ended_utc=datetime.now(UTC).isoformat(),
            peak_cpu_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
        )
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
