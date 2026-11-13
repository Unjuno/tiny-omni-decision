"""Frozen whole-grid appearance at exactly the existing corpus time samples."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tiny_omni_decision.spatial_context_features import (
    pooled_spatial_context,
    spatial_context_positions,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
AUDIT = Path("C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01")
PRIOR_PIN = "30910583657724619e82334fd7effca128b9e5415834422c8f4804c75ffe3d68"
MEDIA_PINS = {
    "train": "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",
    "valid": "213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546",
}
RESERVE_BYTES = 8 * 1024**3
CLIP_ARRAY_BYTES = 32 * 256 * (384 + 3) * 4 + 32 * 8 + 3 * 128


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def require_bitwise(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise ValueError("bitwise replay mismatch")


def validate_plans(prior, manifests, *, expected_counts=None):
    expected_counts = expected_counts or {"train": 99, "valid": 257}
    media, media_hashes = {}, set()
    for split, manifest in manifests.items():
        if split not in expected_counts or len(manifest["clips"]) != expected_counts[split]:
            raise ValueError("unexpected media split/count")
        for row in manifest["clips"]:
            if row["video_id"] in media or row["sha256"] in media_hashes:
                raise ValueError("cross-split or duplicate media identity/content")
            media[row["video_id"]] = dict(row, split=split)
            media_hashes.add(row["sha256"])
    if set(manifests) != set(expected_counts):
        raise ValueError("missing media split")
    seen, prepared = set(), {s: set() for s in expected_counts}
    plans = []
    for row in prior["features"]:
        vid, split = row["video_id"], row["split"]
        source = media.get(vid)
        if (
            re.fullmatch(r"video_\d+", vid) is None
            or vid in seen
            or source is None
            or split != source["split"]
            or row["media_sha256"] != source["sha256"]
            or row["shape"] != [256, 512]
            or type(row["frames"]) is not int
            or row["frames"] < 32
            or (source["cut_frame"] != -1 and row["frames"] > source["cut_frame"])
        ):
            raise ValueError("invalid feature/media/time binding")
        frames = np.asarray(row["frame_indices"])
        expected = np.rint(np.linspace(0, row["frames"] - 1, 32)).astype(np.int64)
        if frames.dtype.kind not in "iu" or not np.array_equal(frames, expected):
            raise ValueError("changed frame sampling")
        seen.add(vid)
        prepared[split].add(row["prepared_sha256"])
        plans.append(
            dict(
                video_id=vid,
                split=split,
                media_path=source["path"],
                media_sha256=source["sha256"],
                cut_frame=source["cut_frame"],
                frames=row["frames"],
                frame_indices=row["frame_indices"],
                old_feature_path=row["path"],
                old_feature_sha256=row["sha256"],
                old_prepared_sha256=row["prepared_sha256"],
            )
        )
    if seen != set(media):
        raise ValueError("incomplete or extra feature corpus")
    if prepared["train"] & prepared["valid"]:
        raise ValueError("cross-split prepared-window overlap")
    return plans


def load_plans():
    path = ROOT / "perception-occlusion-trajectory-features-01/result.json"
    if sha(path) != PRIOR_PIN:
        raise ValueError("prior corpus result changed")
    prior = json.loads(path.read_text())
    if (
        prior["status"] != "PASS_FEATURE_EXTRACTION_NOT_QUALITY"
        or prior["manifest_hashes"] != MEDIA_PINS
    ):
        raise ValueError("complete pinned prior corpus mandatory")
    if prior["settings"]["grid"] != 16 or prior["settings"]["samples"] != 32:
        raise ValueError("prior geometry changed")
    manifests = {}
    for split, pin in MEDIA_PINS.items():
        path = ROOT / f"perception-occlusion-{split}-media-01/manifest.json"
        if sha(path) != pin:
            raise ValueError("media manifest changed")
        manifests[split] = json.loads(path.read_text())
    return prior, validate_plans(prior, manifests)


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        smoke=args.smoke,
        started_utc=datetime.now(UTC).isoformat(),
        features=[],
        annotations_loaded=False,
        training=False,
        sealed_audit_opened=False,
        cloud_cost=0,
    )

    def save():
        report["seconds"] = time.perf_counter() - start
        write(args.output / "progress.json", report)

    try:
        prior, plans = load_plans()
        if args.smoke:
            plans = plans[:1]
        reference_root = ROOT / "perception-temporal-appearance-01"
        if (
            sha(reference_root / "result.json")
            != "a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4"
        ):
            raise ValueError("prior spatial reference changed")
        references = {
            r["video_id"]: r
            for r in json.loads((reference_root / "result.json").read_text())["rows"]
        }
        weight = AUDIT / "tapir_checkpoint_panning.pt"
        if sha(weight) != prior["weight_sha256"]:
            raise ValueError("checkpoint changed")
        for path, pin in prior["source_hashes"].items():
            if sha(path) != pin:
                raise ValueError("upstream implementation changed")
        expected_bytes = CLIP_ARRAY_BYTES * len(plans)
        if shutil.disk_usage(args.output).free < RESERVE_BYTES + expected_bytes + 64 * 1024**2:
            raise OSError("insufficient disk while protecting8GiB reserve")
        for p in plans:
            if (
                sha(p["media_path"]) != p["media_sha256"]
                or sha(p["old_feature_path"]) != p["old_feature_sha256"]
            ):
                raise ValueError("media or prior feature changed")
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
        if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            raise MemoryError("12GiB free GPU preflight mandatory")
        model = TAPIR(pyramid_level=0, extra_convs=False).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(weight, map_location="cpu", weights_only=True), strict=True
        )
        model.cuda()
        report.update(
            parameters=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            state_before=state_sha(model),
        )
        assert report["parameters"] == 31072327
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            pooling_sha256=sha(Path("src/tiny_omni_decision/spatial_context_features.py")),
            weight_sha256=sha(weight),
            source_hashes=prior["source_hashes"],
            prior_result_sha256=PRIOR_PIN,
            media_manifest_sha256=MEDIA_PINS,
            plans=plans,
            frames=32,
            grid_size=16,
            output_dtype="float32",
            pooling="all2x2 lowres/all4x4 hires cells; FP64 mean then FP32",
            seed=17,
            tf32=False,
            precision="FP32",
            smoke=args.smoke,
            max_seconds=1800,
            torch_peak_cap_bytes=4 * 1024**3,
            expected_array_bytes=expected_bytes,
            disk_reserve_bytes=RESERVE_BYTES,
            prior_spatial_reference_sha256=sha(reference_root / "result.json"),
        )
        write(args.output / "config.json", config)
        report.update(status="RUNNING", config_sha256=sha(args.output / "config.json"))
        save()
        prepared = {"train": set(), "valid": set()}
        for index, p in enumerate(plans):
            if (
                time.perf_counter() - start > 1800
                or torch.cuda.max_memory_allocated() > 4 * 1024**3
            ):
                raise RuntimeError("predeclared time/memory cap exceeded")
            remaining = CLIP_ARRAY_BYTES * (len(plans) - index)
            if shutil.disk_usage(args.output).free < RESERVE_BYTES + remaining + 64 * 1024**2:
                raise OSError("disk reserve violated")
            clip_start = time.perf_counter()
            times = np.array(p["frame_indices"], dtype=np.int64)
            selected, found = [], []
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
                raise ValueError("decode/frame routing changed")
            rgb = np.stack(selected)
            rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            prepared[p["split"]].add(rgb_hash)
            video = torch.from_numpy(rgb).float()[None].cuda() / 255 * 2 - 1
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            torch.cuda.synchronize()
            encode_start = time.perf_counter()
            with torch.inference_mode():
                grids = model.get_feature_grids(video, is_training=False)
                torch.cuda.synchronize()
                encode_seconds = time.perf_counter() - encode_start
                low, high = grids.lowres[0][0].cpu().numpy(), grids.hires[0][0].cpu().numpy()
                assert low.shape == (32, 32, 32, 256) and high.shape == (32, 64, 64, 128)
                context = pooled_spatial_context(low, high)
                assert context.shape == (32, 256, 384)
                reference_exact = None
                if p["video_id"] in references:
                    archive = reference_root / f"{p['video_id']}.npz"
                    assert sha(archive) == references[p["video_id"]]["sha256"]
                    with np.load(archive, allow_pickle=False) as old:
                        require_bitwise(low, old["lowres"])
                        require_bitwise(high, old["hires"])
                        require_bitwise(times, old["frame_indices"])
                        require_bitwise(
                            context, pooled_spatial_context(old["lowres"], old["hires"])
                        )
                    reference_exact = True
                if index == 0:
                    repeat = model.get_feature_grids(video, is_training=False)
                    require_bitwise(low, repeat.lowres[0][0].cpu().numpy())
                    require_bitwise(high, repeat.hires[0][0].cpu().numpy())
                    require_bitwise(
                        context,
                        pooled_spatial_context(
                            repeat.lowres[0][0].cpu().numpy(), repeat.hires[0][0].cpu().numpy()
                        ),
                    )
                    report["first_repeat_bitwise"] = True
                    del repeat
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(
                torch.equal(a, b)
                for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
            ):
                raise ValueError("spatial extraction changed RNG")
            positions = spatial_context_positions(times, source_frames=p["frames"])
            outputs = {}
            for name, value in [
                ("appearance", context),
                ("positions", positions),
                ("frames", times),
            ]:
                dest = args.output / f"{p['split']}-{p['video_id']}-{name}.npy"
                assert not dest.exists()
                np.save(dest, value, allow_pickle=False)
                require_bitwise(value, np.load(dest, allow_pickle=False))
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
                    split=p["split"],
                    media_sha256=p["media_sha256"],
                    selected_rgb_sha256=rgb_hash,
                    original_frames=p["frames"],
                    frame_indices=p["frame_indices"],
                    outputs=outputs,
                    reference_grids_bitwise=reference_exact,
                    rng_unchanged=True,
                    encoder_seconds=encode_seconds,
                    seconds=time.perf_counter() - clip_start,
                )
            )
            save()
            print(p["split"], p["video_id"], index + 1, "/", len(plans), "SAVED", flush=True)
            del grids, low, high, context, positions, rgb, video, selected, value
            torch.cuda.empty_cache()
        if prepared["train"] & prepared["valid"]:
            raise ValueError("cross-split selected RGB overlap")
        report["state_after"] = state_sha(model)
        if report["state_before"] != report["state_after"] or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise ValueError("frozen parameter state changed")
        for p in plans:
            assert sha(p["media_path"]) == p["media_sha256"]
            assert sha(p["old_feature_path"]) == p["old_feature_sha256"]
        assert sha(weight) == config["weight_sha256"]
        for path, pin in config["source_hashes"].items():
            assert sha(path) == pin
        assert len(report["features"]) == (1 if args.smoke else 356)
        if time.perf_counter() - start > 1800 or torch.cuda.max_memory_allocated() > 4 * 1024**3:
            raise RuntimeError("final time/memory cap exceeded")
        report.update(
            status="PASS_TRAIN_ONLY_SPATIAL_SMOKE"
            if args.smoke
            else "PASS_COMPLETE_SPATIAL_CONTEXT_CORPUS_NOT_QUALITY",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
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
