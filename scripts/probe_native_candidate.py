"""Actual native frozen Path A wiring/count/reload; initialized head, no quality claim."""

import argparse
import contextlib
import hashlib
import io
import json
import math
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

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from safetensors.torch import load_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    WhisperConfig,
    WhisperFeatureExtractor,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from scripts.evaluate_semantic_state_bridge import read, sha, tensor_digest, verify, write
from scripts.native_candidate_inputs import masked_mean, positions, validate_native_sources
from tiny_omni_decision.native_candidate import (
    NativeCandidateReadout,
    NativeTokenBlock,
    candidate_json,
)
from tiny_omni_decision.resident_inventory import parameter_inventory
from tiny_omni_decision.token_transport import read_tokens

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_conditional_candidate_v0.json"
ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
R1 = ROOT.parent / "pretrained-reuse-r1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-reference", type=Path)
    parser.add_argument("--replay-sha256")
    args = parser.parse_args()
    if bool(args.replay_reference) != bool(args.replay_sha256):
        raise ValueError("paired replay arguments required")
    # Refuse dirty source: each native run belongs to a reviewable exact commit.
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise ValueError("commit source before native execution")
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    result = dict(status="RUNNING", started_utc=datetime.now(UTC).isoformat(), command=sys.argv,
                  source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  config_sha256=sha(CONFIG), model_updates=0, head_status="UNTRAINED_SEED17",
                  quality_evaluation=False, validation_loaded=False, calibration_loaded=False,
                  sealed_audit_loaded=False, cloud_cost=0)
    worker = log = None
    arrays = {}
    write(args.output / "result.json", result)
    try:
        config = read(CONFIG)
        pins = config["input_pins"].copy()
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO,
                                        text=True).split("\0")
        pins.update({str(REPO / p): sha(REPO / p) for p in names if p.endswith(".py")})
        pins[str(CONFIG)] = sha(CONFIG)
        for source, revision in [(R1 / "efficientat-audit-v1/source",
                                  config["source_revisions"]["event_audio"]),
                                 (R1 / "videoprism-base-r1/source",
                                  config["source_revisions"]["visual"])]:
            if (subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                        text=True).strip() != revision
                    or subprocess.check_output(["git", "-C", str(source), "diff", "--name-only"],
                                               text=True).strip()):
                raise ValueError("external model source differs")
            pins.update({str(p): sha(p) for p in source.rglob("*.py")})
        verify(pins)
        if shutil.disk_usage(args.output).free < 2 * 2**30:
            raise RuntimeError("less than2GiB free")
        reference = None
        if args.replay_reference:
            if sha(args.replay_reference / "result.json") != args.replay_sha256:
                raise ValueError("reference receipt hash changed")
            reference = read(args.replay_reference / "result.json")
            if (reference["status"] != "PASS_NATIVE_CANDIDATE_INTEGRATION"
                    or reference["config_sha256"] != result["config_sha256"]
                    or reference["validated_hashes"] != pins):
                raise ValueError("complete matching reference required")
            verify({str(args.replay_reference / p): h
                    for p, h in reference["artifact_hashes"].items()})
        native = config["native_inputs"]
        image_path, video_path = Path(native["image"]), Path(native["video"])
        validate_native_sources(
            read(ROOT / "videoprism-spatial-image-cache-01/records.json"),
            read(ROOT / "lombard-grid-media-integrity-01/rows.json"),
            image_id=image_path.stem, image_sha=pins[str(image_path)],
            source_id=native["source_id"], audio_sha=native["audio_sha256"],
            video_sha=pins[str(video_path)],
        )
        write(args.output / "effective-config.json", config)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError("registered local CUDA profile unavailable")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        language_path = config["model_paths"]["language"]
        language, loading = AutoModelForSequenceClassification.from_pretrained(
            language_path, local_files_only=True, dtype=torch.float32,
            attn_implementation="eager", output_loading_info=True,
        )
        if any(loading.get(k) for k in ["missing_keys", "unexpected_keys", "mismatched_keys"]):
            raise ValueError("nonexact language load")
        language.eval().requires_grad_(False).to(device)
        tokenizer = AutoTokenizer.from_pretrained(language_path, local_files_only=True)
        speech_path = Path(config["model_paths"]["speech"])
        speech_config = WhisperConfig.from_pretrained(speech_path, local_files_only=True)
        speech_config._attn_implementation = "eager"
        speech = WhisperEncoder(speech_config).eval().requires_grad_(False)
        speech.load_state_dict(load_file(speech_path / "encoder.safetensors"), strict=True)
        speech.to(device)
        processor = WhisperFeatureExtractor.from_pretrained(speech_path, local_files_only=True)
        sys.path.insert(0, str(R1 / "efficientat-audit-v1/source"))
        from models.dymn.model import get_model
        from models.preprocess import AugmentMelSTFT

        with contextlib.redirect_stdout(sys.stderr):
            event = get_model(width_mult=0.4, pretrained_name=None, T_max=1.0)
            mel = AugmentMelSTFT()
        event.load_state_dict(torch.load(R1 / "efficientat-audit-v1/dymn04_as.pt",
                                        weights_only=True, map_location="cpu"), strict=True)
        event.eval().requires_grad_(False).to(device)
        mel.eval().requires_grad_(False).to(device)
        frozen = dict(language=language, speech=speech, event_audio=event, event_preprocessor=mel)
        before = {k: tensor_digest(m) for k, m in frozen.items()}
        torch.manual_seed(config["seed"])
        head = NativeCandidateReadout().eval().requires_grad_(False).to(device)
        if reference:
            head.load_state_dict(torch.load(args.replay_reference / "readout.pt",
                                            weights_only=True, map_location=device), strict=True)
        torch.save(head.cpu().state_dict(), args.output / "readout.pt")
        head.to(device)
        head_before = tensor_digest(head)
        counted = parameter_inventory(dict(**frozen, readout=head))

        def language_tokens(texts, pairs=None):
            encoded = tokenizer(texts, text_pair=pairs, padding=True, truncation=False,
                                return_tensors="pt").to(device)
            if encoded["input_ids"].shape[1] > 512:
                raise ValueError("language input exceeds512; no truncation")
            with torch.inference_mode():
                hidden = language.deberta(**encoded).last_hidden_state
            return hidden, encoded["attention_mask"]

        def block(kind, values):
            values = torch.as_tensor(np.array(values, copy=True), device=device,
                                     dtype=torch.float32)
            arrays[kind + "-tokens"] = values.cpu().numpy().copy()
            return NativeTokenBlock(kind, values, positions(kind, len(values)).to(device),
                                    torch.ones(len(values), dtype=torch.bool, device=device))

        started = time.perf_counter()
        text, mask = language_tokens([native["text"]])
        blocks = {"text": [block("text", text[0, mask[0].bool()].cpu().numpy())]}
        with zipfile.ZipFile(native["audio_archive"]) as archive:
            content = archive.read(native["audio_member"])
        if hashlib.sha256(content).hexdigest() != native["audio_sha256"]:
            raise ValueError("raw audio member differs")
        pcm, rate = sf.read(io.BytesIO(content), dtype="float32")
        if rate != 16000 or pcm.ndim != 1 or not 0 < len(pcm) <= processor.n_samples:
            raise ValueError("fixed mono16k complete waveform required")
        with torch.inference_mode():
            features = processor(pcm, sampling_rate=rate, return_tensors="pt").input_features
            acoustic = speech(features.to(device)).last_hidden_state
            valid = math.ceil(len(pcm) / 320)
            waveform = torchaudio.functional.resample(torch.from_numpy(pcm)[None].to(device),
                                                     rate, 32000)
            _, events = event(mel(waveform).unsqueeze(1))
        blocks["audio"] = [block("speech", acoustic[0, :valid].cpu().numpy()),
                           block("event_audio", events.cpu().numpy())]
        result["text_audio_encode_seconds"] = time.perf_counter() - started
        requests = [dict(key=k, modality=k, assets=[dict(path=str(p), sha256=pins[str(p)])])
                    for k, p in [("image", image_path), ("video", video_path)]]
        write(args.output / "requests.json", requests)
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), JAX_PLATFORMS="cpu",
                   TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="2")
        command = [str(R1 / "videoprism-base-r1/venv/Scripts/python.exe"), "-m",
                   "scripts.run_shared_visual_requests", "--requests",
                   str(args.output / "requests.json"), "--report",
                   str(args.output / "visual-worker.json"), "--hold-for-release"]
        log = (args.output / "visual-worker.log").open("wb")
        worker = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=log,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        result["visual_command"] = command
        messages = queue.Queue()

        def receive_tokens():
            try:
                for kind, shape in [("image", (1, 17, 768)), ("video", (1, 2048, 768))]:
                    messages.put((kind, read_tokens(worker.stdout, shape)))
            except Exception as error:
                messages.put(error)

        threading.Thread(target=receive_tokens, daemon=True).start()
        peak_sum = 0
        for expected in ["image", "video"]:
            while True:
                live = [psutil.Process(), *psutil.Process().children(recursive=True)]
                peak_sum = max(peak_sum, sum(p.memory_info().rss for p in live if p.is_running()))
                if (time.perf_counter() - tick > 900 or peak_sum > 12 * 2**30
                        or shutil.disk_usage(args.output).free < 2**30):
                    raise RuntimeError("native integration resource guard")
                try:
                    message = messages.get(timeout=.5)
                    break
                except queue.Empty:
                    if worker.poll() is not None:
                        raise RuntimeError("visual worker exited; inspect log") from None
            if isinstance(message, Exception):
                raise message
            kind, payload = message
            if kind != expected or payload is None or payload[0] != expected:
                raise ValueError("native worker payload order differs")
            blocks[kind] = [block(kind, payload[1][0])]
        if worker.poll() is not None:
            raise RuntimeError("visual worker must stay resident for combined inference")
        result["visual_process_live_during_combined_snapshot"] = worker.pid
        result["observed_process_tree_peak_rss_bytes"] = peak_sum
        outputs = []
        reloaded = NativeCandidateReadout().eval().requires_grad_(False).to(device)
        reloaded.load_state_dict(torch.load(args.output / "readout.pt", weights_only=True,
                                           map_location=device), strict=True)
        # Extra head is verification-only and explicitly counted in actual smoke residency.
        reload_parameters = sum(p.numel() for p in reloaded.parameters())
        for modality, native_blocks in blocks.items():
            query = config["queries"][modality]
            for index, question in enumerate(query["questions"]):
                hidden, mask = language_tokens([question] * len(query["candidates"]),
                                                query["candidates"])
                q = masked_mean(hidden, mask)
                key = f"{modality}-{index}"
                arrays[key + "-queries"] = q.cpu().numpy()
                with torch.inference_mode():
                    scores = head(native_blocks, q)
                    repeat = head(native_blocks, q)
                    reordered = head(native_blocks, q.flip(0)).flip(0)
                    loaded = reloaded(native_blocks, q)
                torch.testing.assert_close(scores, repeat, atol=0, rtol=0)
                for value in [loaded, reordered]:
                    torch.testing.assert_close(scores, value, atol=1e-6, rtol=0)
                arrays[key + "-scores"] = scores.cpu().numpy()
                decision = candidate_json(scores, [f"option-{i}" for i in range(4)])
                outputs.append(dict(modality=modality, question=question,
                                    candidates=query["candidates"],
                                    decision=decision,
                                    reload_max_abs=float((scores - loaded).abs().max()),
                                    permutation_max_abs=float((scores - reordered).abs().max())))
        worker.stdin.write(b"release\n")
        worker.stdin.flush()
        worker.wait(timeout=60)
        if worker.returncode != 0:
            raise RuntimeError("visual terminal exit differs")
        visual = read(args.output / "visual-worker.json")
        if visual["status"] != "PASS_SHARED_WORKER" or visual["parameters"] != 114365184:
            raise ValueError("full shared visual inventory required")
        total = counted["total_parameters"] + visual["parameters"]
        if total > config["max_parameters"]:
            raise ValueError("complete candidate exceeds200M")
        for name, model in frozen.items():
            if (tensor_digest(model) != before[name]
                    or any(p.grad is not None for p in model.parameters())):
                raise ValueError("frozen parameters changed")
        if tensor_digest(head) != head_before:
            raise ValueError("readout changed without authorization")
        verify(pins)
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError("nonfinite saved payload")
            np.save(args.output / (name + ".npy"), value, allow_pickle=False)
        replay = {}
        if reference:
            for name, value in arrays.items():
                old = np.load(args.replay_reference / (name + ".npy"), allow_pickle=False)
                np.testing.assert_allclose(value, old, rtol=0, atol=config["replay_atol"])
                replay[name] = float(np.max(np.abs(value - old)))
        result.update(status="PASS_NATIVE_CANDIDATE_INTEGRATION", decisions=outputs,
                      torch_inventory=counted, visual_parameters=visual["parameters"],
                      candidate_total_parameters=total, reload_helper_parameters=reload_parameters,
                      smoke_resident_parameters_including_reload=total + reload_parameters,
                      buffer_elements=sum(b.numel() for m in frozen.values() for b in m.buffers()),
                      buffer_bytes=sum(b.numel() * b.element_size()
                                       for m in frozen.values() for b in m.buffers()),
                      candidate_tensor_bytes=counted["tensor_bytes"] + visual["parameters"] * 4,
                      frozen_state_sha256=before, head_state_sha256=head_before,
                      validated_hashes=pins, input_config=config,
                      observations=dict(text=1, image=1, audio=1, video=1,
                                        independent_media_source_groups=2),
                      outputs=len(outputs), frozen_and_sources_unchanged=True,
                      replay_max_abs=replay, device=torch.cuda.get_device_name(),
                      versions=dict(python=sys.version, torch=torch.__version__,
                                    numpy=np.__version__, torchaudio=torchaudio.__version__,
                                    transformers=__import__("transformers").__version__),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      main_peak_rss_bytes=getattr(psutil.Process().memory_info(),
                                                 "peak_wset", None))
    except Exception as error:
        result.update(status="FAIL_INCOMPLETE", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if worker is not None and worker.poll() is None:
            try:
                worker.stdin.write(b"release\n")
                worker.stdin.flush()
                worker.wait(timeout=60)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                # Keep a timed-out handle identifiable; never silently launch a duplicate.
                result["worker_still_live_or_release_failed"] = worker.pid
        if log is not None:
            log.close()
        result.update(finished_utc=datetime.now(UTC).isoformat(),
                      seconds=time.perf_counter() - tick,
                      artifact_hashes={p.name: sha(p) for p in args.output.iterdir()
                                       if p.is_file() and p.name != "result.json"})
        write(args.output / "result.json", result)
        print(json.dumps({k: result[k] for k in ["status", "seconds", "source_commit"]}))


if __name__ == "__main__":
    main()
