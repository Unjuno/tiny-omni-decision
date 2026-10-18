"""Fixed CPU collection-cue ASR/null diagnostic, with a fresh-process replay mode."""

import argparse
import copy
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.diagnose_robomnist_lengths import sha, write, write_arrays
from scripts.fetch_robomnist_input_smoke import GROUP, peak_bytes
from scripts.probe_robomnist_container_features import (
    RAW,
    RAW_REPLAY,
    RAW_SHA,
    add_pin,
    failure_peak,
    model_state,
)
from scripts.robomnist_audio_cue import (
    describe_numbers,
    read_bound_json,
    recheck,
    require_sequence,
    resample_native,
)
from scripts.robomnist_container_features import same_arrays
from scripts.robomnist_input_arrays import read_wav

REFERENCE = Path("C:/CodexArtifacts/pretrained-reuse-r2/native-speech-eval-01/config.json")
REFERENCE_SHA = "897add18c70b5d67a0811ae6b2dc4391e124c8a3113e550847a1a59e8b38dac8"
WEIGHT_SHA = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"
CLOCK = Path("docs/pretrained_reuse/R2_ROBOMNIST_CLOCK_SOURCE_20261015.json")
CLOCK_SHA = "cdea6567d160a58df291ff18a9896bef9391c9b76719560ea27651bb9ba52868"
PROTOCOL = Path("docs/pretrained_reuse/R2_ROBOMNIST_AUDIO_CUE_PROTOCOL_20261015.md")
GENERATION = dict(
    language="english",
    task="transcribe",
    return_timestamps=False,
    return_dict_in_generate=True,
    max_new_tokens=128,
    num_beams=1,
    do_sample=False,
    temperature=0.0,
    compression_ratio_threshold=None,
    logprob_threshold=None,
    no_speech_threshold=None,
)


def require_audio_scope(r):
    if (
        r["parameters"] != 37760640
        or r["trainable_parameters"] != 0
        or r["generated_conditions"] != 6
        or r["state_before"] != r["state_after"]
        or r["forward_cpu_rng_unchanged"] is not True
        or any(
            r[k] is not False for k in ["model_training", "gradients_present", "cuda_initialized"]
        )
    ):
        raise ValueError("complete frozen/eval CPU six-condition scope required")


def bind_sources():
    raw = read_bound_json(RAW_REPLAY, RAW_SHA)
    pins = raw["validated_hashes"].copy()
    add_pin(pins, RAW_REPLAY, RAW_SHA)
    recheck(pins)
    previous = read_bound_json(REFERENCE, REFERENCE_SHA)
    release = previous["model_files"]
    if release.get("model.safetensors") != WEIGHT_SHA or len(release) != 11:
        raise ValueError("independently fixed released Whisper weight/files required")
    model = Path(previous["argv"]["model"])
    if set(p.name for p in model.iterdir() if p.is_file()) != set(release):
        raise ValueError("exact original11file model/tokenizer directory required")
    for name, value in release.items():
        add_pin(pins, model / name, value)
    add_pin(pins, REFERENCE, REFERENCE_SHA)
    clock = read_bound_json(CLOCK, CLOCK_SHA)
    for entry in clock["files"]:
        add_pin(pins, entry["local"], entry["sha256"])
    add_pin(pins, CLOCK, CLOCK_SHA)
    for p in [
        Path(__file__),
        Path("scripts/robomnist_audio_cue.py"),
        Path("scripts/robomnist_input_arrays.py"),
        Path("scripts/probe_robomnist_container_features.py"),
        Path("scripts/robomnist_container_features.py"),
        Path("tests/test_robomnist_audio_cue.py"),
        PROTOCOL,
    ]:
        add_pin(pins, p, sha(p))
    recheck(pins)
    return pins, model


def guard(output, tick):
    seconds, peak = time.perf_counter() - tick, peak_bytes()
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if (
        seconds > 600
        or peak > 3 * 2**30
        or size > 63 * 2**20
        or shutil.disk_usage(output).free < 2**30
    ):
        raise RuntimeError("fixed CPU/time/output/disk guard exhausted")
    return dict(elapsed_seconds=seconds, peak_cpu_bytes=peak, output_bytes_before_terminal=size)


def require_products(output, hashes, config, records, arrays):
    recheck({str(output / name): h for name, h in hashes.items()})
    if (
        json.loads((output / "config.json").read_bytes()) != config
        or json.loads((output / "transcripts.json").read_bytes()) != records
    ):
        raise ValueError("saved effective config/complete transcripts changed")
    with np.load(output / "arrays.npz", allow_pickle=False) as bank:
        same_arrays(arrays, {k: bank[k] for k in bank.files})


def generation_paths():
    from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
    from transformers.generation.utils import GenerationMixin
    from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration

    return sorted(
        {
            Path(inspect.getfile(actual))
            for value in [
                WhisperForConditionalGeneration.generate,
                GenerationMixin.generate,
                GenerationConfig,
                LogitsProcessorList,
                StoppingCriteriaList,
            ]
            for actual in [value, inspect.unwrap(value)]
        }
    )


def generate_conditions(model_path, pins, check):
    import av
    import av.audio.frame
    import av.audio.resampler
    import torch
    import transformers
    from transformers import (
        StoppingCriteria,
        StoppingCriteriaList,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperTokenizer,
    )

    if torch.cuda.is_initialized():
        raise ValueError("new CPU-only process required")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    # Capture library/source hashes before the long load/decoder work.
    for p in [
        Path(torch.__file__),
        Path(torch._C.__file__),
        Path(transformers.__file__),
        Path(av.__file__),
        Path(av.audio.resampler.__file__),
        Path(av.audio.frame.__file__),
        Path(inspect.getfile(WhisperForConditionalGeneration)),
        Path(inspect.getfile(WhisperTokenizer)),
        Path(inspect.getfile(WhisperFeatureExtractor)),
    ] + generation_paths():
        add_pin(pins, p, sha(p))
    recheck(pins)
    model, loading = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        output_loading_info=True,
        attn_implementation="eager",
    )
    if any(loading.values()):
        raise ValueError(f"exact released load required: {loading}")
    model.cpu().eval().requires_grad_(False)
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("exact CPU float32 weights required")
    tokenizer = WhisperTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        language="english",
        task="transcribe",
        predict_timestamps=False,
    )
    processor = WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    if (
        tokenizer.prefix_tokens != [50258, 50259, 50359, 50363]
        or model.config._attn_implementation != "eager"
    ):
        raise ValueError("fixed English prefix/eager backend required")
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.forced_decoder_ids = None
    state_before, rng_before = model_state(model), torch.get_rng_state().clone()
    before_config = model.generation_config.to_dict()
    baseline = json.loads((RAW / "result.json").read_bytes())
    arrays, records, forward_count = {}, [], []

    class Bounds(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def finite_forward(module, args, result):
        if not torch.isfinite(result.logits).all():
            raise ValueError("nonfinite decoder logits")
        forward_count.append(1)

    hook = model.register_forward_hook(finite_forward)
    with np.load(RAW / "native-inputs.npz", allow_pickle=False) as bank:
        for receiver in [1, 2, 3]:
            pcm, meta = read_wav(RAW / f"microphone-{receiver}.wav")
            if dict(receiver=receiver, **meta) != baseline["audio"][receiver - 1]:
                raise ValueError("fresh native PCM metadata differs")
            same_arrays({"pcm": pcm}, {"pcm": bank[f"microphone_{receiver}_pcm"]})
            wave = resample_native(pcm, meta["sample_rate"])
            arrays[f"receiver_{receiver}_resampled_pcm"] = wave
            for name, value in [("real", wave), ("silence", np.zeros_like(wave))]:
                check()
                inputs = processor(
                    value, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
                )
                if (
                    inputs.input_features.shape != (1, 80, 3000)
                    or not torch.isfinite(inputs.input_features).all()
                ):
                    raise ValueError("complete finite native mel80x3000 required")
                prefix = f"receiver_{receiver}_{name}"
                arrays[f"{prefix}_mel"] = inputs.input_features.numpy().copy()
                arrays[f"{prefix}_attention_mask"] = inputs.attention_mask.numpy().copy()
                started, calls = time.perf_counter(), len(forward_count)
                with torch.inference_mode():
                    generated = model.generate(
                        input_features=inputs.input_features,
                        attention_mask=inputs.attention_mask,
                        generation_config=copy.deepcopy(generation_config),
                        stopping_criteria=StoppingCriteriaList([Bounds()]),
                        **GENERATION,
                    )
                    ids = generated.sequences[0].cpu().numpy().copy()
                    del generated
                require_sequence(ids)
                text = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
                arrays[f"{prefix}_tokens"] = ids
                row = dict(
                    receiver=receiver,
                    condition=name,
                    native_sample_frames=meta["sample_frames"],
                    native_sample_rate=44100,
                    resampled_samples=len(value),
                    resampled_rate=16000,
                    pcm_peak_abs=float(np.abs(value).max()),
                    text=text,
                    token_ids=ids.tolist(),
                    number_observations=describe_numbers(text),
                    decoder_forward_calls=len(forward_count) - calls,
                    generation_seconds=time.perf_counter() - started,
                )
                records.append(row)
                print("CPU ASR", receiver, name, repr(text), flush=True)
    hook.remove()
    scope = dict(
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        model_training=model.training,
        gradients_present=any(p.grad is not None for p in model.parameters()),
        cuda_initialized=torch.cuda.is_initialized(),
        state_before=state_before,
        state_after=model_state(model),
        forward_cpu_rng_unchanged=torch.equal(rng_before, torch.get_rng_state()),
        generated_conditions=len(records),
        decoder_forward_calls=len(forward_count),
        model_generation_config_unchanged=before_config == model.generation_config.to_dict(),
        generation_config=generation_config.to_dict(),
        explicit_generation_kwargs=GENERATION,
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
        av=av.__version__,
        ffmpeg_libraries={name: list(version) for name, version in av.library_versions.items()},
        processor_class=type(processor).__name__,
        tokenizer_class=type(tokenizer).__name__,
        device="cpu",
        dtype="float32",
        attention="eager",
        threads=2,
        seed=17,
    )
    require_audio_scope(scope)
    if scope["model_generation_config_unchanged"] is not True:
        raise ValueError("base generation config changed by diagnostic")
    return records, arrays, scope


def stable_records(records):
    return [{k: v for k, v in row.items() if k != "generation_seconds"} for row in records]


def run_body(output, tick, reference=None, reference_sha=None):
    def check():
        return guard(output, tick)

    check()
    pins, model = bind_sources()
    old = None
    if reference is not None:
        old = read_bound_json(reference / "result.json", reference_sha)
        if old["status"] != "PASS_FIXED_ACOUSTIC_CUE_MECHANICS_ONLY" or old["group"] != GROUP:
            raise ValueError("complete fixed acoustic diagnostic reference required")
        pins.update({str(reference / "result.json"): reference_sha})
        for name, h in old["artifacts"].items():
            add_pin(pins, reference / name, h)
        recheck(old["validated_hashes"])
        recheck(pins)
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        execution_arguments=sys.argv,
        model_id="openai/whisper-tiny",
        model_revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        model_path=str(model),
        group=GROUP,
        receivers=[1, 2, 3],
        conditions=["real", "silence"],
        generation=GENERATION,
        cpu_profile="float32/eager/2threads/seed17/fullyfrozen",
        resampler="PyAV flt/mono16000/default/flush; complete44.1kHzPCM",
        max_seconds=600,
        max_cpu_bytes=3 * 2**30,
        max_output_bytes=64 * 2**20,
        disk_reserve_bytes=2**30,
        source_hashes=pins.copy(),
        reference_result_sha256=reference_sha,
    )
    artifacts = {"config.json": write(output / "config.json", config)}
    records, arrays, scope = generate_conditions(model, pins, check)
    artifacts["transcripts.json"] = write(output / "transcripts.json", records)
    artifacts["arrays.npz"] = write_arrays(output / "arrays.npz", arrays)
    require_products(output, artifacts, config, records, arrays)
    if old is not None:
        if (
            any(pins.get(p) != h for p, h in old["validated_hashes"].items())
            or scope != old["scope"]
        ):
            raise ValueError("fresh model/environment/state/pins must match main")
        previous_records = json.loads((reference / "transcripts.json").read_bytes())
        if stable_records(records) != stable_records(previous_records):
            raise ValueError("all six fresh transcript/number/token/accounting records differ")
        with np.load(reference / "arrays.npz", allow_pickle=False) as bank:
            same_arrays(arrays, {k: bank[k] for k in bank.files})
        recheck(old["validated_hashes"])
    require_products(output, artifacts, config, records, arrays)
    recheck(pins)
    return dict(
        status="PASS_FIXED_ACOUSTIC_CUE_MECHANICS_ONLY",
        source_commit=config["source_commit"],
        group=GROUP,
        unique_recordings=1,
        native_audio_files=3,
        generated_conditions=6,
        scope=scope,
        transcripts=records,
        artifacts=artifacts,
        arrays=len(arrays),
        validated_hashes=pins,
        source_pins_unchanged=True,
        fresh_replay=old is not None,
        reference_result_sha256=reference_sha,
        shared_asr_implementation=True,
        speech_ground_truth="UNVERIFIED",
        collection_counter_active="UNKNOWN",
        common_clock="UNVERIFIED",
        training_corpus_admission="NOT_ADMITTED",
        decision_quality_measured=False,
        teacher_training=False,
        gpu_used=False,
        new_source_payloads=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        **check(),
    )


def execute(output, reference=None, reference_sha=None):
    if output.exists() or (
        reference is not None and output.resolve().is_relative_to(reference.resolve())
    ):
        raise ValueError("new isolated output outside reference required")
    output.mkdir()
    tick, started = time.perf_counter(), datetime.now(UTC).isoformat()
    try:
        r = run_body(output, tick, reference, reference_sha)
        r.update(guard(output, tick))
    except Exception as exc:
        r = dict(
            status="FAIL_INCOMPLETE_ACOUSTIC_CUE_DIAGNOSTIC",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - tick,
            peak_cpu_bytes=failure_peak(),
            decision_quality_measured=False,
        )
    r.update(pid=os.getpid(), started_utc=started, ended_utc=datetime.now(UTC).isoformat())
    write(output / "result.json", r)
    print(r["status"], r.get("error", ""), r["elapsed_seconds"], flush=True)
    return r["status"] == "PASS_FIXED_ACOUSTIC_CUE_MECHANICS_ONLY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.reference is None) != (args.reference_sha256 is None):
        parser.error("fresh replay requires both reference path and exact result SHA")
    if not execute(args.output, args.reference, args.reference_sha256):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
