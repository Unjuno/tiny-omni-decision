"""New CPU VideoPrism caches with full spatial extent; never overwrites historical files."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil

from scripts.extract_frozen_videoprism_clevr4_features import _preprocess_rgb_image
from scripts.extract_frozen_videoprism_features import _decode_video, _preprocess
from scripts.native_full_extent import POLICY, encoder_distribution_versions, preprocess_full_extent
from scripts.native_video_cache import sha
from tiny_omni_decision.dinov2_features import require_image_splits
from tiny_omni_decision.video_token_grid import token_grid
from tiny_omni_decision.videoprism_spatial_features import spatial_cells

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/pretrained_reuse/native_full_extent_cache_v1.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    if (args.phase == "full") != bool(args.reference) or bool(args.reference) != bool(
        args.reference_sha256
    ):
        raise ValueError("full extraction requires completed smoke plus receipt SHA")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
        raise ValueError("clean committed source required")
    config = read(CONFIG)
    if config["policy"] != POLICY:
        raise ValueError("fixed full-extent policy required")
    args.output.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(
        status="RUNNING",
        phase=args.phase,
        pid=psutil.Process().pid,
        started_utc=datetime.now(UTC).isoformat(),
        command=sys.argv,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        config_sha256=sha(CONFIG),
        quality_evaluation=False,
        encoder_updates=0,
        calibration_loaded=False,
        sealed_audit_loaded=False,
        validation_media_loaded=args.phase == "full",
        cloud_cost=0,
        completed_images=0,
        completed_videos=0,
    )
    write(args.output / "result.json", result)
    pins, verified = dict(config["input_pins"]), {}

    def verify(path, expected):
        path = Path(path)
        if sha(path) != expected:
            raise ValueError("input/source hash mismatch: " + str(path))
        verified[str(path)] = expected

    def guard():
        result["peak_working_set_bytes"] = psutil.Process().memory_info().peak_wset
        if (
            time.perf_counter() - started > config["max_seconds"]
            or result["peak_working_set_bytes"] > config["max_rss_bytes"]
            or shutil.disk_usage(args.output).free < config["minimum_disk_bytes"]
        ):
            raise RuntimeError("fixed extraction resource budget exceeded")

    def save_array(name, values):
        path = args.output / name
        with path.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
        np.testing.assert_array_equal(values, np.load(path, allow_pickle=False))
        return dict(path=str(path), sha256=sha(path), shape=list(values.shape))

    try:
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=REPO, text=True
        ).split("\0")
        pins.update({str(REPO / name): sha(REPO / name) for name in names if name})
        pins[str(CONFIG)] = sha(CONFIG)
        for path, digest in pins.items():
            verify(path, digest)
        source = Path(config["visual_source"])
        if subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip() != config["visual_revision"] or subprocess.check_output(
            ["git", "diff", "HEAD", "--"], cwd=source
        ):
            raise ValueError("upstream source revision or contents changed")
        image_rows = read(config["image_records"])
        require_image_splits(image_rows["train"], image_rows["val"])
        if {k: len(v) for k, v in image_rows.items()} != {"train": 8424, "val": 512}:
            raise ValueError("complete original image cohort required")
        video_receipt = read(config["video_receipt"])
        videos = video_receipt["rows"]
        if (
            video_receipt["status"] != "PASS_COMPLETE_NATIVE_VIDEO_TOKEN_CACHE"
            or len(videos) != 456
            or sum(v["split"] == "train" for v in videos) != 256
            or sum(v["split"] == "validation" for v in videos) != 200
        ):
            raise ValueError("complete original video cohort required")
        observations = read(Path(config["corpus"]) / "observations.json")
        old_banks = read(Path(config["corpus"]) / "banks.json")
        image_plan = []
        for split, rows in image_rows.items():
            for i, row in enumerate(rows):
                expected = dict(
                    kind="image", bank=f"image/{split}-cells.npy", index=i, offset=0, tokens=17
                )
                if observations["image:" + row["id"]]["features"] != [expected]:
                    raise ValueError("original image feature order changed")
                image_plan.append(
                    dict(
                        id=row["id"],
                        split=split,
                        index=i,
                        sha256=row["sha256"],
                        path=str(
                            Path(config["image_roots"][split])
                            / split
                            / "images"
                            / (row["id"] + ".png")
                        ),
                    )
                )
        for row in videos:
            expected = dict(kind="video", bank="video/" + row["file"], offset=0, tokens=2048)
            if observations["video:" + row["media_sha256"]]["features"] != [expected]:
                raise ValueError("original video feature binding changed")
        plan = dict(images=image_plan, videos=videos)
        write(args.output / "media-plan.json", plan)
        write(args.output / "effective-config.json", config)
        reference = None
        if args.reference:
            verify(args.reference / "result.json", args.reference_sha256)
            reference = read(args.reference / "result.json")
            if (
                reference["status"] != "PASS_FULL_EXTENT_SMOKE"
                or reference["source_pins"] != pins
                or reference["config_sha256"] != sha(CONFIG)
                or reference["artifacts"]["media-plan.json"] != sha(args.output / "media-plan.json")
                or reference["projected_full_seconds"] > config["max_seconds"]
            ):
                raise ValueError("matching practical completed smoke required")
            for name, digest in reference["artifacts"].items():
                verify(args.reference / name, digest)
        sys.path.insert(0, str(source))
        import jax
        import jax.numpy as jnp
        import tensorflow as tf
        from videoprism import models as vp

        if any(device.platform != "cpu" for device in jax.devices()):
            raise RuntimeError("CPU encoder extraction required")
        tf.config.set_visible_devices([], "GPU")
        model = vp.get_model("videoprism_public_v1_base")
        variables = vp.load_pretrained_weights(
            "videoprism_public_v1_base", checkpoint_path=config["checkpoint"]
        )
        parameters = sum(int(v.size) for v in jax.tree_util.tree_leaves(variables))
        if parameters != 114365184:
            raise ValueError("frozen encoder parameter count differs")
        encode_image = jax.jit(
            lambda x: model.apply(
                variables, x, train=False, return_intermediate={"spatial_features"}
            )[1]["spatial_features"]
        )
        encode_video = jax.jit(lambda x: model.apply(variables, x, train=False))
        versions = encoder_distribution_versions()
        if reference and reference["versions"] != versions:
            raise ValueError("smoke/full dependency environment differs")
        result.update(
            parameters=parameters,
            versions=versions,
            backend=jax.default_backend(),
            python=sys.version,
            source_pins=pins,
            dtype="float32",
            reference_sha256=args.reference_sha256,
        )

        def image_encode(values):
            result = encode_image(jnp.asarray(values))
            jax.block_until_ready(result)
            return spatial_cells(np.asarray(result)).astype(np.float32)

        def video_encode(values):
            result = encode_video(jnp.asarray(values[None]))
            jax.block_until_ready(result)
            tokens = result[0] if isinstance(result, tuple) else result
            return token_grid(np.asarray(tokens)).reshape(2048, 768).astype(np.float32)

        guard()
        geometry = {}
        image_seconds, video_seconds = [], []
        banks = {}
        allowed_images = image_plan if args.phase == "full" else image_plan[:8]
        allowed_videos = (
            videos if args.phase == "full" else [v for v in videos if v["split"] == "train"][:2]
        )
        if args.phase == "smoke" and any(r["split"] != "train" for r in allowed_images):
            raise ValueError("smoke must use TRAIN media only")
        with (args.output / "progress.jsonl").open("x", encoding="utf-8") as log:
            for split in ("train", "val"):
                rows = [r for r in allowed_images if r["split"] == split]
                if not rows:
                    continue
                path = args.output / (split + "-cells.npy")
                values = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(len(rows), 17, 768)
                )
                for start in range(0, len(rows), 8):
                    guard()
                    tick = time.perf_counter()
                    images, originals = [], []
                    for row in rows[start : start + 8]:
                        verify(row["path"], row["sha256"])
                        pixels = np.asarray(
                            tf.io.decode_png(tf.io.read_file(row["path"]), channels=3)
                        )
                        transformed, shape = preprocess_full_extent(pixels, tf)
                        geometry["image:" + str(pixels.shape[:2])] = shape
                        images.append(transformed[None])
                        originals.append(pixels)
                    batch = np.stack(images)
                    if args.phase == "smoke":
                        bank = old_banks["image/train-cells.npy"]
                        verify(bank["path"], bank["sha256"])
                        old = image_encode(
                            np.stack([_preprocess_rgb_image(p, tf)[:1] for p in originals])
                        )
                        np.testing.assert_array_equal(old, np.load(bank["path"], mmap_mode="r")[:8])
                        save_array("control-image.npy", old)
                        result["old_image_parity_max_abs"] = 0.0
                    extracted = image_encode(batch)
                    if args.phase == "smoke":
                        result["image_feature_delta_max_abs"] = float(
                            np.max(np.abs(extracted - old))
                        )
                        if result["image_feature_delta_max_abs"] == 0:
                            raise ValueError("full-extent image features unexpectedly unchanged")
                    values[start : start + len(images)] = extracted
                    values.flush()
                    if args.phase == "smoke":
                        warm = time.perf_counter()
                        np.testing.assert_array_equal(extracted, image_encode(batch))
                        result["warm_image_batch_seconds"] = time.perf_counter() - warm
                    if reference and split == "train" and start == 0:
                        np.testing.assert_array_equal(
                            extracted,
                            np.load(args.reference / "train-cells.npy", allow_pickle=False),
                        )
                    elapsed = time.perf_counter() - tick
                    image_seconds.append(elapsed)
                    result["completed_images"] += len(images)
                    log.write(
                        json.dumps(
                            dict(
                                kind="image",
                                split=split,
                                completed=start + len(images),
                                seconds=elapsed,
                            )
                        )
                        + "\n"
                    )
                    log.flush()
                    if start % 128 == 0:
                        write(args.output / "result.json", result)
                        print("images", split, start + len(images), "/", len(rows), flush=True)
                del values
                bank = dict(path=str(path), sha256=sha(path), shape=[len(rows), 17, 768])
                banks["image/" + path.name] = bank
                check = np.load(path, mmap_mode="r", allow_pickle=False)
                if check.shape != tuple(bank["shape"]) or not np.isfinite(check).all():
                    raise ValueError("saved image bank incomplete or nonfinite")
            for i, row in enumerate(allowed_videos):
                guard()
                tick = time.perf_counter()
                path = Path(config["video_root"]) / row["media_path"]
                verify(path, row["media_sha256"])
                frames = _decode_video(path)
                if frames.shape != (8, 320, 480, 3):
                    raise ValueError("unchanged eight-frame source geometry required")
                processed, shape = preprocess_full_extent(frames, tf)
                geometry["video:320x480"] = shape
                if args.phase == "smoke":
                    bank = old_banks["video/" + row["file"]]
                    verify(bank["path"], bank["sha256"])
                    old = video_encode(_preprocess(frames, tf))
                    np.testing.assert_array_equal(old, np.load(bank["path"], allow_pickle=False))
                    save_array(f"control-video-{i}.npy", old)
                    result["old_video_parity_max_abs"] = 0.0
                extracted = video_encode(processed)
                if args.phase == "smoke":
                    difference = float(np.max(np.abs(extracted - old)))
                    result.setdefault("video_feature_delta_max_abs", []).append(difference)
                    if difference == 0:
                        raise ValueError("full-extent video features unexpectedly unchanged")
                banks["video/" + row["file"]] = save_array(row["file"], extracted)
                if args.phase == "smoke":
                    warm = time.perf_counter()
                    np.testing.assert_array_equal(extracted, video_encode(processed))
                    result.setdefault("warm_video_seconds", []).append(time.perf_counter() - warm)
                if reference and (args.reference / row["file"]).is_file():
                    np.testing.assert_array_equal(
                        extracted, np.load(args.reference / row["file"], allow_pickle=False)
                    )
                elapsed = time.perf_counter() - tick
                video_seconds.append(elapsed)
                result["completed_videos"] += 1
                log.write(
                    json.dumps(
                        dict(
                            kind="video",
                            media_sha256=row["media_sha256"],
                            completed=i + 1,
                            seconds=elapsed,
                        )
                    )
                    + "\n"
                )
                log.flush()
                if i % 16 == 0:
                    write(args.output / "result.json", result)
                    print("videos", i + 1, "/", len(allowed_videos), flush=True)
        write(args.output / "geometry.json", geometry)
        write(args.output / "banks.json", banks)
        if args.phase == "smoke":
            result.update(
                repeat_max_abs=0.0,
                projected_full_seconds=1.5
                * (
                    result["warm_image_batch_seconds"] * (8936 / 8)
                    + np.median(result["warm_video_seconds"]) * 456
                )
                + 300,
            )
        elif result["completed_images"] != 8936 or result["completed_videos"] != 456:
            raise ValueError("full extraction incomplete")
        for path, digest in verified.copy().items():
            verify(path, digest)
        guard()
        result.update(
            status="PASS_FULL_EXTENT_" + args.phase.upper(),
            validated_hashes=verified,
            image_seconds=sum(image_seconds),
            video_seconds=sum(video_seconds),
            output_bytes=sum(p.stat().st_size for p in args.output.glob("*.npy")),
            artifacts={p.name: sha(p) for p in args.output.iterdir() if p.name != "result.json"},
        )
    except BaseException as exc:
        result.update(status="INCOMPLETE", failure=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            seconds=time.perf_counter() - started, finished_utc=datetime.now(UTC).isoformat()
        )
        write(args.output / "result.json", result)
    print(result["status"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
