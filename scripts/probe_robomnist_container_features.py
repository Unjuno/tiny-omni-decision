"""Bounded native decoder/frozen CPU frontend; no Teacher or quality evaluation."""

import argparse
import hashlib
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

from scripts.diagnose_robomnist_lengths import ARMS, load_rows, sha, write, write_arrays
from scripts.fetch_robomnist_input_smoke import GROUP, peak_bytes
from scripts.robomnist_container_features import (
    append_free_atom,
    digest_arrays,
    same_arrays,
    stereo_tiles,
)
from scripts.robomnist_input_arrays import read_video
from scripts.robomnist_length_probe import centroid_logp

RAW = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-smoke-01")
RAW_REPLAY = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-replay-02.json")
RAW_SHA = "5cd718a66f53d843ce30d9f1ba3519aa534ca977319491d90798367096576d51"
LENGTH = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-length-diagnostic-01")
LENGTH_REPLAY = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-length-replay-02.json")
LENGTH_SHA = "15aab2045b13c6ee64ff811d709ab77b3174658f52a0a06355354ea4e05f0182"
MODEL = Path("C:/CodexArtifacts/pretrained-reuse-r1/tinyclip-39m19m-audit-01/model")
R1 = Path("docs/pretrained_reuse/R1_SMALL_ALIGNED_MODELS_20261012.json")
R1_SHA = "f63a3cac64d8ab84320b70368eef543742cfc54485ba656bf905f3486de984fc"
WEIGHT_SHA = "112bdfb07ebc8994e9afba7a2a314e278493a675033595cf48e54cf8a17ae587"
PROTOCOL = Path("docs/pretrained_reuse/R2_ROBOMNIST_CONTAINER_FEATURE_PROTOCOL_20261015.md")
PLAN = Path("docs/superpowers/plans/2026-10-15-robomnist-container-features.md")
FORMAT_SOURCE = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/robomnist-container-format-source-01/free_space_atom.md"
)
FORMAT_SHA = "febb055c776e072edffdfaa118c1306a3d57c4eccbd080564c92b89401ca679a"


def failure_peak():
    try:
        return peak_bytes()
    except Exception:
        return None


def released_r1(body):
    if hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest() != R1_SHA:
        raise ValueError("immutable historical R1 report required before parsing")
    report = json.loads(body)
    release = report["audit"]["input_sha256"]
    if (
        release.get("model.safetensors") != WEIGHT_SHA
        or report["download"]["revision"] != "07a4b0bc751cb64fecd2b661c048c1dd98d69444"
        or set(release)
        != {
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "README.md",
            "tokenizer.json",
        }
    ):
        raise ValueError("independently fixed released weight/revision/files required")
    return release


def add_pin(source, path, value):
    key = str(Path(path).resolve())
    if key in source and source[key] != value:
        raise ValueError("existing immutable source pin cannot be replaced")
    source[key] = value


def require_frozen_scope(r):
    if (
        r["parameters"] != 83101185
        or r["trainable_parameters"] != 0
        or any(
            r[k] is not False for k in ["gradients_present", "model_training", "cuda_initialized"]
        )
        or any(
            r[k] is not True
            for k in [
                "state_unchanged",
                "forward_cpu_rng_unchanged",
                "features_container_invariant",
                "pixels_container_invariant",
                "uniform_sensitive",
            ]
        )
    ):
        raise ValueError("complete frozen CPU/content-invariant and sensitive frontend required")


def bind_sources():
    source = {}
    for p, expected in [(RAW_REPLAY, RAW_SHA), (LENGTH_REPLAY, LENGTH_SHA)]:
        if sha(p) != expected:
            raise ValueError("immutable input/length replay required")
        r = json.loads(p.read_bytes())
        for name, value in {**r["validated_hashes"], str(p): expected}.items():
            if name in source and source[name] != value or sha(name) != value:
                raise ValueError("source binding changed or conflicts")
            source[name] = value
    release = released_r1(R1.read_bytes())
    if set(p.name for p in MODEL.iterdir() if p.is_file()) != set(release):
        raise ValueError("exact existing five-file released checkpoint required")
    for name, value in release.items():
        p = MODEL / name
        if sha(p) != value:
            raise ValueError("released model source hash mismatch")
        source[str(p)] = value
    if sha(FORMAT_SOURCE) != FORMAT_SHA:
        raise ValueError("pinned official atom format source required")
    for p in [
        Path(__file__),
        Path("scripts/robomnist_container_features.py"),
        Path("scripts/robomnist_input_arrays.py"),
        Path("tests/test_robomnist_container_features.py"),
        PROTOCOL,
        PLAN,
        R1,
        FORMAT_SOURCE,
    ]:
        add_pin(source, p, sha(p))
    return source


def guard(output, tick):
    seconds, peak = time.perf_counter() - tick, peak_bytes()
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if (
        seconds > 600
        or peak > 3 * 2**30
        or size > 127 * 2**20
        or shutil.disk_usage(output).free < 2**30
    ):
        raise RuntimeError("frozen frontend CPU/time/output/disk bound exhausted")
    return dict(elapsed_seconds=seconds, peak_cpu_bytes=peak, output_bytes_before_terminal=size)


def decode_derivatives(output, check):
    baseline = json.loads((RAW / "result.json").read_bytes())
    original, derived = {}, {}
    rows = []
    with np.load(RAW / "native-inputs.npz", allow_pickle=False) as bank:
        for receiver in [1, 2, 3]:
            print("CPU derivative camera", receiver, flush=True)
            source, destination = (
                RAW / f"camera-{receiver}.mp4",
                output / f"camera-{receiver}-free.mp4",
            )
            row = append_free_atom(source, destination, 2**20)
            if (
                sha(source) != row["source_sha256"]
                or sha(destination) != row["derived_sha256"]
                or destination.stat().st_size != row["derived_bytes"]
            ):
                raise ValueError("copied source prefix/derived creation checksum differs")
            values, meta = read_video(destination, check)
            expected_meta = baseline["video"][receiver - 1]
            if dict(receiver=receiver, **meta) != expected_meta:
                raise ValueError("every native frame RGB digest/PTS/metadata must match source")
            for key, a in values.items():
                name = f"camera_{receiver}_{key}"
                original[name] = bank[name]
                derived[name] = a
            rows.append(
                dict(
                    receiver=receiver,
                    source_path=str(source),
                    path=str(destination),
                    **row,
                    decoded_frames=meta["decoded_frames"],
                    rgb_stream_sha256=meta["rgb_stream_sha256"],
                )
            )
    same_arrays(original, derived)
    return original, derived, rows


def length_controls(derivative_rows):
    rows, x, _ = load_rows()
    index = next(i for i, r in enumerate(rows) if r["group_id"] == GROUP)
    train = np.array([r["split"] == "train" for r in rows])
    if not train[index]:
        raise ValueError("fixed observed pilot must remain train")
    original_sizes = (
        [int((RAW / f"camera-{i}.mp4").stat().st_size) for i in [1, 2, 3]]
        + [int((RAW / f"microphone-{i}.wav").stat().st_size) for i in [1, 2, 3]]
        + [int((RAW / "csi.json").stat().st_size)]
    )
    a = np.log1p(np.array(original_sizes, np.float64))
    if not np.array_equal(a, x[index]):
        raise ValueError("actual pilot lengths differ from original diagnostic source")
    altered = original_sizes.copy()
    altered[:3] = [r["derived_bytes"] for r in derivative_rows]
    variants = np.stack([a, np.log1p(np.array(altered, np.float64))])
    y = np.array([r["class_target"] for r in rows], np.int64)
    result, arrays = {}, {}
    with np.load(LENGTH / "arrays.npz", allow_pickle=False) as reference:
        for arm in ["camera_length", "all_lengths"]:
            cols = ARMS[arm]
            logp = centroid_logp(x[train][:, cols], y[train], variants[:, cols], 60)
            if not np.allclose(logp[0], reference[arm][index], rtol=0, atol=1e-12):
                raise ValueError("original frozen length estimator posterior differs")
            p = np.exp(logp)
            arrays[f"{arm}_logp"] = logp
            result[arm] = dict(
                original_prediction=int(p[0].argmax()),
                derivative_prediction=int(p[1].argmax()),
                probability_l1_change=float(np.abs(p[1] - p[0]).sum()),
                max_logp_change=float(np.abs(logp[1] - logp[0]).max()),
                original_target_probability=float(p[0, y[index]]),
                derivative_target_probability=float(p[1, y[index]]),
            )
    return result, arrays


def model_state(model):
    return digest_arrays(
        {
            **{f"parameter:{n}": p.detach().numpy() for n, p in model.named_parameters()},
            **{f"buffer:{n}": p.detach().numpy() for n, p in model.named_buffers()},
        }
    )


def encode_frontend(original, derived, source, check):
    import av
    import av.container.input
    import av.video.frame
    import PIL
    import PIL._imaging
    import torch
    import transformers
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if torch.cuda.is_initialized():
        raise ValueError("new CPU-only process required")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    model, info = CLIPModel.from_pretrained(
        MODEL, local_files_only=True, output_loading_info=True, attn_implementation="eager"
    )
    if any(info.values()):
        raise ValueError("released checkpoint loading mismatch")
    model.cpu().eval().requires_grad_(False)
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("exact frozen CPU float32 parameters required")
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    if model.vision_model.config._attn_implementation != "eager":
        raise ValueError("fixed eager attention backend required")
    for p in [
        Path(torch.__file__),
        Path(torch._C.__file__),
        Path(np.__file__),
        Path(PIL.__file__),
        Path(PIL._imaging.__file__),
        Path(av.__file__),
        Path(av.container.input.__file__),
        Path(av.video.frame.__file__),
        Path(transformers.__file__),
        Path(inspect.getfile(CLIPModel)),
        Path(inspect.getfile(CLIPProcessor)),
        Path(inspect.getfile(type(processor.image_processor))),
    ]:
        add_pin(source, p, sha(p))
    mean = processor.image_processor.image_mean
    tensors = {}
    for name, values in [("original", original), ("derivative", derived)]:
        tiles = np.concatenate(
            [
                stereo_tiles(frame, mean)
                for receiver in [1, 2, 3]
                for frame in values[f"camera_{receiver}_rgb_samples"]
            ]
        )
        inputs = processor(
            images=[Image.fromarray(a) for a in tiles],
            do_resize=False,
            do_center_crop=False,
            return_tensors="pt",
        )
        if (
            set(inputs) != {"pixel_values"}
            or inputs["pixel_values"].shape != (18, 3, 224, 224)
            or inputs["pixel_values"].dtype != torch.float32
        ):
            raise ValueError("only18native-normalized pixel tensors may enter model")
        tensors[name] = inputs["pixel_values"]
    fill = np.rint(np.array(mean) * 255).astype(np.uint8)
    uniform = np.empty((224, 224, 3), np.uint8)
    uniform[:] = fill
    tensors["uniform"] = processor(
        images=[Image.fromarray(uniform)] * 18,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )["pixel_values"]
    if not torch.equal(tensors["original"], tensors["derivative"]):
        raise ValueError("preprocessed native pixel tensor changed with container padding")
    state_before = model_state(model)
    rng_before = torch.get_rng_state().clone()
    features = {}
    timings = {}
    for name, pixels in tensors.items():
        phase = time.perf_counter()
        vectors = []
        with torch.inference_mode():
            for offset in range(0, 18, 3):
                check()
                value = model.visual_projection(
                    model.vision_model(
                        pixel_values=pixels[offset : offset + 3], return_dict=True
                    ).pooler_output
                )
                if not torch.isfinite(value).all():
                    raise ValueError("nonfinite frozen visual features")
                vectors.append(value.numpy().copy())
        features[name] = np.concatenate(vectors).reshape(3, 3, 2, 512)
        timings[name] = time.perf_counter() - phase
        print("CPU encoded", name, "18images", timings[name], flush=True)
    same_arrays({"features": features["original"]}, {"features": features["derivative"]})
    distance = np.linalg.norm(features["original"] - features["uniform"], axis=-1)
    r = dict(
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        buffers=sum(b.numel() for b in model.buffers()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gradients_present=any(p.grad is not None for p in model.parameters()),
        model_training=model.training,
        cuda_initialized=torch.cuda.is_initialized(),
        state_unchanged=state_before == model_state(model),
        state_before=state_before,
        state_after=model_state(model),
        forward_cpu_rng_unchanged=torch.equal(rng_before, torch.get_rng_state()),
        features_container_invariant=True,
        pixels_container_invariant=True,
        uniform_sensitive=bool(np.any(distance > 0)),
        all_uniform_l2_distances=distance.tolist(),
        inference_seconds=timings,
        load_and_forward_seconds=time.perf_counter() - start,
        device="cpu",
        dtype="float32",
        threads=2,
        seed=17,
        attention_backend="eager",
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
        av=av.__version__,
        pillow=PIL.__version__,
        processor_class=type(processor.image_processor).__name__,
        image_mean=list(mean),
        image_std=list(processor.image_processor.image_std),
        feature_shape=[3, 3, 2, 512],
    )
    require_frozen_scope(r)
    arrays = {
        **{f"{name}_pixels": tensor.numpy().copy() for name, tensor in tensors.items()},
        **{f"{name}_features": a for name, a in features.items()},
    }
    return r, arrays


def require_products(output, hashes, config, arrays):
    if any(sha(output / name) != value for name, value in hashes.items()):
        raise ValueError("prepared output binding changed")
    if json.loads((output / "config.json").read_bytes()) != config:
        raise ValueError("saved effective config changed")
    with np.load(output / "arrays.npz", allow_pickle=False) as bank:
        same_arrays(arrays, {key: bank[key] for key in bank.files})


def run_body(output, tick):
    def check():
        return guard(output, tick)

    check()
    source = bind_sources()
    config = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        execution_arguments=sys.argv,
        source_hashes=source.copy(),
        group=GROUP,
        model_revision="07a4b0bc751cb64fecd2b661c048c1dd98d69444",
        model_id="wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M",
        cpu_profile="float32/eager/2threads/seed17/fullyfrozen",
        python=platform.python_version(),
        platform=platform.platform(),
        free_payload_bytes=2**20,
        max_seconds=600,
        max_cpu_bytes=3 * 2**30,
        max_output_bytes=128 * 2**20,
        disk_reserve_bytes=2**30,
        samples=[0, 224, 449],
        lenses=["left", "right"],
        preprocessing="whole-lens letterbox224/BICUBIC/mean-fill; native normalize; no resize/crop",
        length_reference_logp_absolute_tolerance=1e-12,
    )
    products = {"config.json": write(output / "config.json", config)}
    original, derived, rows = decode_derivatives(output, check)
    for row in rows:
        products[Path(row["path"]).name] = row["derived_sha256"]
    length, control_arrays = length_controls(rows)
    frontend, arrays = encode_frontend(original, derived, source, check)
    arrays.update(control_arrays)
    products["arrays.npz"] = write_arrays(output / "arrays.npz", arrays)
    require_products(output, products, config, arrays)
    for p, value in source.items():
        if sha(p) != value:
            raise ValueError("source/model/library pin changed during execution")
    report = dict(
        status="PASS_NATIVE_CONTAINER_AND_FROZEN_IMAGE_FEATURE_MECHANICS_ONLY",
        source_commit=config["source_commit"],
        config_sha256=products["config.json"],
        group=GROUP,
        unique_source_recordings=1,
        camera_rows=rows,
        projected_image_observations=18,
        encoded_image_instances=54,
        encoder_forward_calls=18,
        frontend=frontend,
        nuisance_length_controls=length,
        decoded_frames=1350,
        raw_arrays_digest=digest_arrays(original),
        derived_arrays_digest=digest_arrays(derived),
        arrays_digest=digest_arrays(arrays),
        artifacts=products,
        validated_hashes=source,
        source_pins_unchanged=True,
        model_training=False,
        teacher_training=False,
        new_source_payloads=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        quality_measured=False,
        native_temporal_model=False,
        training_corpus_admission="NOT_ADMITTED",
        common_clock="UNVERIFIED",
        gpu_used=False,
        cloud_cost=0,
    )
    require_products(output, products, config, arrays)
    for p, value in source.items():
        if sha(p) != value:
            raise ValueError("terminal source/model/library pin changed")
    report.update(check())
    return report


def execute(output):
    if output.exists():
        raise ValueError("new isolated output required; originals protected")
    output.mkdir()
    tick, started = time.perf_counter(), datetime.now(UTC).isoformat()
    try:
        r = run_body(output, tick)
        r.update(guard(output, tick))
    except Exception as exc:
        r = dict(
            status="FAIL_INCOMPLETE_CONTAINER_FEATURE_PROBE",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - tick,
            peak_cpu_bytes=failure_peak(),
            quality_measured=False,
            training_corpus_admission="NOT_ADMITTED",
        )
    r.update(pid=os.getpid(), started_utc=started, ended_utc=datetime.now(UTC).isoformat())
    write(output / "result.json", r)
    print(r["status"], r.get("error", ""), "seconds", r["elapsed_seconds"], flush=True)
    return r["status"] == "PASS_NATIVE_CONTAINER_AND_FROZEN_IMAGE_FEATURE_MECHANICS_ONLY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not execute(args.output):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
