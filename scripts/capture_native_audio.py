"""Complete pinned development audio features, or independent full feature replay."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import psutil
import torch
from safetensors.torch import load_file
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from scripts.capture_native_language import state_digest, verify, write
from scripts.native_audio_cache import store_features, validate_profile
from scripts.native_audio_sources import audio_plan, iter_audio
from scripts.native_candidate_inputs import require_source_revision
from scripts.native_video_cache import sha
from tiny_omni_decision.native_audio import audio_tokens

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_audio_cache_v1.json"
PROFILE = REPO / "configs/pretrained_reuse/native_audio_profile_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["cache", "replay"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    report = dict(
        status="RUNNING",
        phase=args.phase,
        started_utc=datetime.now(UTC).isoformat(),
        command=sys.argv,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        completed_examples=0,
        optimizer_updates=0,
        quality_evaluation=False,
        calibration_loaded=False,
        sealed_audit_loaded=False,
        cloud_cost=0,
        max_replay_abs=0.0,
        profile_matches=0,
    )
    write(args.output / "result.json", report)
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_profile(config)
        if sha(PROFILE) != config["profile_sha256"]:
            raise ValueError("original audio preprocessing profile changed")
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        source = Path(profile["event_source"])
        require_source_revision(source, profile["event_revision"], "models")
        model_pins = profile["input_pins"].copy()
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0")
        model_pins.update({str(REPO / p): sha(REPO / p) for p in names if p})
        model_pins[str(PROFILE)] = sha(PROFILE)
        model_pins.update({str(p): sha(p) for p in source.rglob("*.py")})
        verify(model_pins)
        audit_path = Path(config["corpus_audit"])
        if sha(audit_path / "result.json") != config["corpus_audit_sha256"]:
            raise ValueError("corpus audit receipt differs")
        audit = json.loads((audit_path / "result.json").read_text(encoding="utf-8"))
        if (
            audit["status"] != "PASS_NATIVE_AUDIO_CORPUS"
            or audit["completed"] != 7410
            or audit["cross_split_decoded_pcm_overlap"] != 0
        ):
            raise ValueError("complete independent corpus audit required")
        verify(audit["validated_hashes"])
        audit_files = {str(audit_path / k): v for k, v in audit["artifact_hashes"].items()}
        audit_files[str(audit_path / "result.json")] = config["corpus_audit_sha256"]
        verify(audit_files)
        plan = json.loads((audit_path / "plan.json").read_text(encoding="utf-8"))
        root = audit_path.parent
        lombard = {
            s: json.loads(
                (root / "lombard-grid-layout-02-media-gated" / f"{s}.json").read_text(
                    encoding="utf-8"
                )
            )
            for s in ("train", "validation")
        }
        media = json.loads(
            (root / "lombard-grid-media-integrity-01/rows.json").read_text(encoding="utf-8")
        )
        starss = {
            s: [
                json.loads(x)
                for x in (root / "starss23-all-windows-v1" / f"{s}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for s in ("train", "validation")
        }
        if audio_plan(lombard, media, starss) != plan:
            raise ValueError("source reconstruction differs from audited plan")
        decoded = [
            json.loads(x)
            for x in (audit_path / "decoded.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if (
            [r["id"] for r in decoded] != [r["id"] for r in plan["rows"]]
            or len(decoded) != config["max_examples"]
            or plan["tensor_bytes"] != config["expected_tensor_bytes"]
        ):
            raise ValueError("complete waveform order/storage differs")
        pins = {**model_pins, **audit["validated_hashes"], **audit_files, str(CONFIG): sha(CONFIG)}
        if sha(args.reference / "result.json") != args.reference_sha256:
            raise ValueError("reference receipt hash differs")
        reference = json.loads((args.reference / "result.json").read_text(encoding="utf-8"))
        expected_status = (
            "PASS_NATIVE_AUDIO_PROFILE" if args.phase == "cache" else "PASS_NATIVE_AUDIO_CACHE"
        )
        expected_pins = model_pins if args.phase == "cache" else pins
        if reference["status"] != expected_status or reference["validated_hashes"] != expected_pins:
            raise ValueError("complete same-source/reference inputs required")
        verify({str(args.reference / k): v for k, v in reference["artifact_hashes"].items()})
        report.update(
            config_sha256=sha(CONFIG),
            validated_hashes=pins,
            reference=dict(path=str(args.reference), sha256=args.reference_sha256),
            counts=plan["counts"],
            shapes=plan["shapes"],
            tensor_bytes=plan["tensor_bytes"],
            expected_examples=len(plan["rows"]),
            example_order_sha256=audit["example_order_sha256"],
        )
        write(args.output / "effective-config.json", config)
        write(args.output / "plan.json", plan)
        needed = plan["tensor_bytes"] if args.phase == "cache" else 0
        if shutil.disk_usage(args.output).free < needed + config["disk_reserve_bytes"]:
            raise RuntimeError("insufficient disk for complete banks plus reserve")
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError("registered CUDA profile unavailable")
        torch.cuda.reset_peak_memory_stats()
        speech_path = Path(profile["speech_model"])
        speech_config = WhisperConfig.from_pretrained(speech_path, local_files_only=True)
        speech_config._attn_implementation = "eager"
        speech = WhisperEncoder(speech_config)
        speech.load_state_dict(load_file(speech_path / "encoder.safetensors"), strict=True)
        speech.eval().requires_grad_(False).to("cuda")
        processor = WhisperFeatureExtractor.from_pretrained(speech_path, local_files_only=True)
        sys.path.insert(0, str(source))
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        with contextlib.redirect_stdout(io.StringIO()):
            event = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
            mel = AugmentMelSTFT()
        event.load_state_dict(
            torch.load(profile["event_weights"], weights_only=True, map_location="cpu"), strict=True
        )
        event.eval().requires_grad_(False).to("cuda")
        mel.eval().requires_grad_(False).to("cuda")
        models = dict(speech=speech, event=event, mel=mel)
        before = {k: state_digest(v) for k, v in models.items()}
        report.update(
            frozen_before=before,
            dtype="float32",
            attention="eager",
            batch_size=1,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            parameters={k: sum(p.numel() for p in v.parameters()) for k, v in models.items()},
            versions={
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "torchaudio",
                    "transformers",
                    "numpy",
                    "soundfile",
                    "safetensors",
                )
            },
        )
        if report["parameters"] != dict(speech=8208384, event=1961239, mel=0):
            raise ValueError("registered audio parameter counts differ")
        if any(
            reference[k] != report[k]
            for k in ("versions", "frozen_before", "parameters", "dtype", "attention")
        ):
            raise ValueError("reference runtime/frozen model differs")
        if args.phase == "cache":
            banks = {
                name: np.lib.format.open_memmap(
                    args.output / name, mode="w+", dtype=np.float32, shape=tuple(shape)
                )
                for name, shape in plan["shapes"].items()
            }
        else:
            if json.loads((args.reference / "plan.json").read_text(encoding="utf-8")) != plan:
                raise ValueError("replay bank plan differs")
            banks = {
                name: np.load(args.reference / name, mmap_mode="r", allow_pickle=False)
                for name in plan["shapes"]
            }
            if any(
                bank.shape != tuple(plan["shapes"][name]) or bank.dtype != np.float32
                for name, bank in banks.items()
            ):
                raise ValueError("reference bank format differs")

        def guard():
            report["os_peak_working_set_bytes"] = psutil.Process().memory_info().peak_wset
            if (
                time.perf_counter() - tick > config["max_wall_seconds"]
                or report["os_peak_working_set_bytes"] > config["max_rss_bytes"]
                or torch.cuda.max_memory_allocated() > config["max_vram_bytes"]
                or shutil.disk_usage(args.output).free < config["disk_reserve_bytes"]
            ):
                raise RuntimeError("registered resource cap exceeded")

        encode_tick = time.perf_counter()
        with (args.output / "features.jsonl").open("x", encoding="utf-8") as log:
            for index, (row, pcm, digest) in enumerate(
                iter_audio(plan["rows"], profile["lombard_archive"])
            ):
                guard()
                expected = decoded[index]
                if (
                    expected["pcm_sha256"] != digest
                    or expected["rate"] != row["rate"]
                    or expected["samples"] != len(pcm)
                    or expected["split"] != row["split"]
                ):
                    raise ValueError("full corpus waveform replay differs")
                acoustic, event_features = audio_tokens(
                    speech, processor, event, mel, torch.from_numpy(pcm), row["rate"]
                )
                values = dict(speech=acoustic.cpu().numpy(), event=event_features.cpu().numpy())
                delta = store_features(banks, row, values, args.phase)
                report["max_replay_abs"] = max(report["max_replay_abs"], delta)
                if args.phase == "cache" and row["id"] in (
                    "lombard:s11_l_bbab2n",
                    "starss:" + profile["starss_id"],
                ):
                    for kind, value in values.items():
                        old = np.load(
                            args.reference / (row["source"] + "-" + kind + ".npy"),
                            allow_pickle=False,
                        )
                        if not np.array_equal(old, value):
                            raise ValueError("fixed TRAIN profile feature differs")
                    report["profile_matches"] += 1
                log.write(
                    json.dumps(
                        dict(
                            id=row["id"],
                            split=row["split"],
                            pcm_sha256=digest,
                            feature_sha256={
                                k: hashlib.sha256(v.tobytes()).hexdigest()
                                for k, v in values.items()
                            },
                        )
                    )
                    + "\n"
                )
                report["completed_examples"] += 1
                if report["completed_examples"] % 128 == 0:
                    write(args.output / "result.json", report)
                    print(
                        "audio",
                        report["completed_examples"],
                        "/",
                        report["expected_examples"],
                        flush=True,
                    )
        report["encoding_seconds"] = time.perf_counter() - encode_tick
        if report["completed_examples"] != config["max_examples"]:
            raise ValueError("incomplete source iteration")
        if args.phase == "cache":
            if report["profile_matches"] != 2:
                raise ValueError("both fixed TRAIN profile examples required")
            for bank in banks.values():
                bank.flush()
            banks.clear()
            for name, shape in plan["shapes"].items():
                bank = np.load(args.output / name, mmap_mode="r", allow_pickle=False)
                if bank.shape != tuple(shape) or bank.dtype != np.float32:
                    raise ValueError("saved bank format differs")
                for offset in range(0, len(bank), 1024):
                    if not np.isfinite(bank[offset : offset + 1024]).all():
                        raise ValueError("nonfinite saved audio bank")
        else:
            if sha(args.output / "features.jsonl") != sha(args.reference / "features.jsonl"):
                raise ValueError("full per-example hash replay differs")
        after = {k: state_digest(v) for k, v in models.items()}
        if before != after:
            raise ValueError("frozen audio state changed")
        verify(pins)
        report.update(
            frozen_after=after,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            artifact_hashes={
                p.name: sha(p)
                for p in args.output.iterdir()
                if p.is_file() and p.name != "result.json"
            },
        )
        guard()
        report.update(
            status="PASS_NATIVE_AUDIO_" + args.phase.upper(),
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - tick,
        )
        write(args.output / "result.json", report)
        print(report["status"], report["seconds"], flush=True)
    except BaseException as error:
        report.update(
            status="INCOMPLETE",
            error=repr(error),
            traceback=traceback.format_exc(),
            finished_utc=datetime.now(UTC).isoformat(),
            seconds=time.perf_counter() - tick,
        )
        write(args.output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
