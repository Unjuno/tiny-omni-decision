"""Actual fine spatial tokens for all known train scenes; no labels or learning."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.extract_spatial_context_corpus import (
    AUDIT,
    ROOT,
    require_bitwise,
    sha,
    state_sha,
    write,
)
from scripts.prepare_grid_role_corpus import admit_train_rows, peak_bytes
from tiny_omni_decision.fine_spatial_inputs import (
    fine_array_bytes,
    fine_context,
    require_disk,
    require_pinned_file,
    require_reference,
    require_smoke,
)
from tiny_omni_decision.spatial_context_features import (
    pooled_spatial_context,
    spatial_context_positions,
)

PINS = dict(
    zip(
        (
            "spatial-context-corpus-01/result.json",
            "spatial-context-corpus-01/config.json",
            "perception-occlusion-train-media-01/manifest.json",
            "perception-occlusion-trajectory-features-01/result.json",
            "perception-temporal-appearance-01/result.json",
        ),
        (
            "a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da",
            "d17c1f1f9e55f17d58983603c1c8bf27ac171bed9d2ee04108535c84b15e35ff",
            "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",
            "30910583657724619e82334fd7effca128b9e5415834422c8f4804c75ffe3d68",
            "a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4",
        ),
        strict=True,
    )
)
STATE = "3c92ad70854f549bc46fde282f4248e949d733fadffcfa2b060afada2ea6dd4a"


def verify(pins):
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError("protected input changed: " + str(path))


def load_contract():
    pins = {str(ROOT / p): h for p, h in PINS.items()}
    verify(pins)
    coarse = json.loads((ROOT / "spatial-context-corpus-01/result.json").read_text())
    cfg = json.loads((ROOT / "spatial-context-corpus-01/config.json").read_text())
    require_pinned_file(Path("scripts/extract_spatial_context_corpus.py"), cfg["runner_sha256"])
    require_pinned_file(
        Path("src/tiny_omni_decision/spatial_context_features.py"), cfg["pooling_sha256"]
    )
    manifest = json.loads((ROOT / "perception-occlusion-train-media-01/manifest.json").read_text())
    prior = json.loads(
        (ROOT / "perception-occlusion-trajectory-features-01/result.json").read_text()
    )
    references = json.loads((ROOT / "perception-temporal-appearance-01/result.json").read_text())
    if (
        coarse["status"] != "PASS_COMPLETE_SPATIAL_CONTEXT_CORPUS_NOT_QUALITY"
        or coarse["config_sha256"] != PINS["spatial-context-corpus-01/config.json"]
        or cfg["grid_size"] != 16
        or cfg["frames"] != 32
        or prior["status"] != "PASS_FEATURE_EXTRACTION_NOT_QUALITY"
        or references["status"] != "PASS_SPATIOTEMPORAL_APPEARANCE_EXTRACTION_ONLY"
    ):
        raise ValueError("complete pinned native input parents required")
    train, media = admit_train_rows(coarse["features"], manifest["clips"])
    plans = []
    for row in train:
        vid = row["video_id"]
        source = media[vid]
        frames = np.array(row["frame_indices"], np.int64)
        if (
            row["original_frames"] < 32
            or (source["cut_frame"] != -1 and row["original_frames"] > source["cut_frame"])
            or not np.array_equal(
                frames, np.rint(np.linspace(0, row["original_frames"] - 1, 32)).astype(np.int64)
            )
        ):
            raise ValueError("changed source cuts/native sampling")
        pins[source["path"]] = source["sha256"]
        for name, shape, dtype in [
            ("appearance", (32, 256, 384), np.float32),
            ("positions", (32, 256, 3), np.float32),
            ("frames", (32,), np.int64),
        ]:
            info = row["outputs"][name]
            pins[info["path"]] = info["sha256"]
            if sha(info["path"]) != info["sha256"]:
                raise ValueError("protected native array changed")
            a = np.load(info["path"], mmap_mode="r", allow_pickle=False)
            if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
                raise ValueError("native shape/dtype/finiteness mismatch")
        require_bitwise(frames, np.load(row["outputs"]["frames"]["path"], allow_pickle=False))
        require_bitwise(
            spatial_context_positions(frames, source_frames=row["original_frames"]),
            np.load(row["outputs"]["positions"]["path"], allow_pickle=False),
        )
        plans.append(
            dict(
                video_id=vid,
                split="train",
                media_path=source["path"],
                media_sha256=source["sha256"],
                frames=row["original_frames"],
                frame_indices=row["frame_indices"],
                selected_rgb_sha256=row["selected_rgb_sha256"],
                coarse_outputs=row["outputs"],
                cut_frame=source["cut_frame"],
            )
        )
    weight = AUDIT / "tapir_checkpoint_panning.pt"
    pins[str(weight)] = prior["weight_sha256"]
    if prior["weight_sha256"] != "628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa":
        raise ValueError("released weight identity changed")
    pins.update(prior["source_hashes"])
    ref = {r["video_id"]: r for r in references["rows"]}
    if len(ref) != 8 or not set(ref).issubset(media):
        raise ValueError("exact eight train raw-grid reference clips required")
    for vid, r in ref.items():
        pins[str(ROOT / "perception-temporal-appearance-01" / (vid + ".npz"))] = r["sha256"]
    for p in [
        Path(__file__),
        Path("scripts/extract_spatial_context_corpus.py"),
        Path("scripts/prepare_grid_role_corpus.py"),
        Path("src/tiny_omni_decision/fine_spatial_inputs.py"),
        Path("src/tiny_omni_decision/spatial_context_features.py"),
        Path("docs/pretrained_reuse/R2_FINE_SPATIAL_CORPUS_PROTOCOL_20261014.md"),
    ]:
        pins[str(p.resolve())] = sha(p)
    verify(pins)
    contract = dict(
        plans=plans,
        source_hashes=pins,
        frames=32,
        grid=32,
        raster=256,
        dtype="float32",
        seed=17,
        tf32=False,
        precision="highest",
        parameters=31072327,
        state_sha256=STATE,
        source_revision="730cda1c730877cfedbe01bf87fb1cadb78a565d",
        hub_revision="5d3fb48e76c5422841e38501514121e251beabb7",
        max_seconds=1800,
        peak_gpu_bytes=4 * 2**30,
        peak_process_bytes=4 * 2**30,
        disk_reserve_bytes=8 * 2**30,
    )
    return contract, ref, weight


def check_limits(start, output, torch, remaining):
    if (
        time.perf_counter() - start > 1800
        or torch.cuda.max_memory_allocated() > 4 * 2**30
        or peak_bytes() > 4 * 2**30
    ):
        raise RuntimeError("fixed fine extraction time/GPU/process cap exceeded")
    if remaining:
        require_disk(shutil.disk_usage(output).free, remaining)
    elif shutil.disk_usage(output).free < 8 * 2**30:
        raise OSError("protected final disk reserve breached")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if (not args.smoke and args.smoke_reference is None) or (
        args.reference is not None and not args.smoke
    ):
        parser.error("main requires --smoke-reference; --reference only for a fresh smoke")
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        smoke=args.smoke,
        features=[],
        training=False,
        annotations_loaded=False,
        validation_arrays_or_annotations_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
    )

    def save():
        report["seconds"] = time.perf_counter() - start
        write(args.output / "progress.json", report)

    try:
        contract, references, weight = load_contract()
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        report["contract_sha256"] = digest
        plans = contract["plans"][:1] if args.smoke else contract["plans"]
        require_disk(shutil.disk_usage(args.output).free, len(plans))
        smoke = None
        extra_pins = {}
        if not args.smoke:
            smoke = require_smoke(args.smoke_reference, digest)
            measured = smoke["features"][0]["seconds"] * len(plans)
            if measured > 1500:
                raise RuntimeError("measured smoke projects beyond1500s margin for1800s cap")
            report["projected_clip_body_seconds_from_smoke"] = measured
            extra_pins.update(
                {
                    str(args.smoke_reference / "result.json"): sha(
                        args.smoke_reference / "result.json"
                    ),
                    str(args.smoke_reference / "config.json"): smoke["config_sha256"],
                }
            )
            for info in smoke["features"][0]["outputs"].values():
                extra_pins[info["path"]] = info["sha256"]
        fresh = None
        if args.reference:
            fresh = require_reference(args.reference, digest)
            extra_pins.update(
                {
                    str(args.reference / "result.json"): sha(args.reference / "result.json"),
                    str(args.reference / "config.json"): fresh["config_sha256"],
                }
            )
            report["reference_result_sha256"] = extra_pins[str(args.reference / "result.json")]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import av
        import torch

        sys.path[:0] = [str(AUDIT / "deps"), str(AUDIT / "source")]
        from tapnet.torch.tapir_model import TAPIR

        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            raise MemoryError("12GiB free GPU required")
        model = TAPIR(pyramid_level=0, extra_convs=False).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )
        model.cuda()
        report.update(
            parameters=sum(p.numel() for p in model.parameters()),
            state_before=state_sha(model),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            av=av.__version__,
        )
        if report["parameters"] != 31072327 or report["state_before"] != STATE:
            raise ValueError("frozen model inventory/state differs")
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            contract=contract,
            contract_sha256=digest,
            smoke=args.smoke,
            expected_array_bytes=fine_array_bytes(len(plans)),
            selected_train_ids=[p["video_id"] for p in plans],
        )
        write(args.output / "config.json", config)
        report.update(
            status="RUNNING",
            config_sha256=sha(args.output / "config.json"),
            source_commit=config["source_commit"],
        )
        save()
        for index, p in enumerate(plans):
            check_limits(start, args.output, torch, len(plans) - index)
            clip_start = time.perf_counter()
            times = np.array(p["frame_indices"], np.int64)
            selected = []
            found = []
            with av.open(p["media_path"]) as container:
                for frame_id, frame in enumerate(container.decode(video=0)):
                    if frame_id >= p["frames"]:
                        break
                    if frame_id in times:
                        selected.append(
                            frame.reformat(width=256, height=256, format="rgb24").to_ndarray()
                        )
                        found.append(frame_id)
            if found != times.tolist():
                raise ValueError("native decode/sample routing differs")
            rgb = np.stack(selected)
            rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            if rgb_hash != p["selected_rgb_sha256"]:
                raise ValueError("native sampledRGB changed")
            video = torch.from_numpy(rgb).float()[None].cuda() / 255 * 2 - 1
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            torch.cuda.synchronize()
            encode_start = time.perf_counter()
            with torch.inference_mode():
                grids = model.get_feature_grids(video, is_training=False)
                torch.cuda.synchronize()
                encode_seconds = time.perf_counter() - encode_start
                low, high = grids.lowres[0][0].cpu().numpy(), grids.hires[0][0].cpu().numpy()
                context = fine_context(low, high)
                require_bitwise(
                    pooled_spatial_context(low, high),
                    np.load(p["coarse_outputs"]["appearance"]["path"], allow_pickle=False),
                )
                raw_reference = None
                if p["video_id"] in references:
                    with np.load(
                        ROOT / "perception-temporal-appearance-01" / (p["video_id"] + ".npz"),
                        allow_pickle=False,
                    ) as old:
                        require_bitwise(low, old["lowres"])
                        require_bitwise(high, old["hires"])
                        require_bitwise(times, old["frame_indices"])
                    raw_reference = True
                if index == 0:
                    repeat = model.get_feature_grids(video, is_training=False)
                    require_bitwise(low, repeat.lowres[0][0].cpu().numpy())
                    require_bitwise(high, repeat.hires[0][0].cpu().numpy())
                    require_bitwise(
                        context,
                        fine_context(
                            repeat.lowres[0][0].cpu().numpy(), repeat.hires[0][0].cpu().numpy()
                        ),
                    )
                    report["first_repeat_bitwise"] = True
                    del repeat
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(
                torch.equal(a, b)
                for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
            ):
                raise ValueError("frozen feature forward changed RNG")
            positions = spatial_context_positions(times, source_frames=p["frames"], grid_size=32)
            outputs = {}
            for name, value in [
                ("appearance", context),
                ("positions", positions),
                ("frames", times),
            ]:
                dest = args.output / (p["video_id"] + "-" + name + ".npy")
                if dest.exists():
                    raise FileExistsError(dest)
                np.save(dest, value, allow_pickle=False)
                require_bitwise(value, np.load(dest, allow_pickle=False))
                if index == 0 and smoke:
                    require_bitwise(
                        value,
                        np.load(smoke["features"][0]["outputs"][name]["path"], allow_pickle=False),
                    )
                if fresh:
                    original = fresh["features"][0]
                    if original["video_id"] != p["video_id"]:
                        raise ValueError("fresh first identity changed")
                    info = original["outputs"][name]
                    extra_pins[info["path"]] = info["sha256"]
                    if sha(info["path"]) != info["sha256"]:
                        raise ValueError("fresh reference array changed")
                    require_bitwise(value, np.load(info["path"], allow_pickle=False))
                outputs[name] = dict(
                    path=str(dest),
                    sha256=sha(dest),
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    file_bytes=dest.stat().st_size,
                )
            report["features"].append(
                dict(
                    video_id=p["video_id"],
                    split="train",
                    media_sha256=p["media_sha256"],
                    selected_rgb_sha256=rgb_hash,
                    original_frames=p["frames"],
                    frame_indices=p["frame_indices"],
                    outputs=outputs,
                    coarse_reference_bitwise=True,
                    raw_grids_reference_bitwise=raw_reference,
                    rng_unchanged=True,
                    encoder_seconds=encode_seconds,
                    seconds=time.perf_counter() - clip_start,
                )
            )
            save()
            print(index + 1, "/", len(plans), p["video_id"], "SAVED", flush=True)
            del grids, low, high, context, positions, video, rgb, selected, value
            torch.cuda.empty_cache()
        report["state_after"] = state_sha(model)
        if report["state_after"] != STATE or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise ValueError("frozen parameter or gradient state changed")
        verify(contract["source_hashes"] | extra_pins)
        if sha(args.output / "config.json") != report["config_sha256"]:
            raise ValueError("effective config changed")
        if len(report["features"]) != (1 if args.smoke else 99):
            raise ValueError("incomplete train feature corpus")
        check_limits(start, args.output, torch, 0)
        report.update(
            status="PASS_TRAIN_FINE_SPATIAL_SMOKE_NOT_QUALITY"
            if args.smoke
            else "PASS_ALL_TRAIN_FINE_SPATIAL_CORPUS_NOT_QUALITY",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_working_set_bytes=peak_bytes(),
            fresh_first_clip_bitwise=fresh is not None,
            ended_utc=datetime.now(UTC).isoformat(),
        )
        save()
        write(args.output / "result.json", report)
        print(report["status"], report["seconds"], flush=True)
    except Exception:
        report.update(
            status="FAIL_INCOMPLETE",
            error=traceback.format_exc(),
            ended_utc=datetime.now(UTC).isoformat(),
        )
        save()
        write(args.output / "result.json", report)
        raise


if __name__ == "__main__":
    main()
