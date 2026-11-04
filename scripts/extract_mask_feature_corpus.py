"""CPU area statistics for every admitted native survivor; no source labels."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from tiny_omni_decision.mask_feature_corpus import (
    assemble_mask_statistics,
    require_aligned_feature_times,
)
from tiny_omni_decision.mask_feature_pool import mask_feature_statistics

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
NATIVE_PIN = "ebe85d5a33cea86a4b045e2244e52b4c8dbc03a3e7ab80b037d77762a238bcda"
AUDIT_PIN = "e3bed6dc93aa299f2904aaad00fa8b00eb771b1d4e1fe1668c4799963f6750cc"
FINE_PIN = "a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4"
PILOT_PIN = "e39dc4962da48ef28a6646c14e77d47c4c34f9cc63ccb66f5b591db4c1390379"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def require_config_binding(path, report):
    if sha(path) != report["config_sha256"]:
        raise ValueError("protected parent effective config changed")


def require_resource_limits(elapsed, free_bytes):
    if not np.isfinite(elapsed) or elapsed < 0 or elapsed > 600 or free_bytes < 8 * 2**30:
        raise RuntimeError("fixed CPU body/disk cap exceeded")


def require_arrays(a, b):
    if set(a) != set(b):
        raise ValueError("array inventory changed")
    for key in a:
        if (
            a[key].shape != b[key].shape
            or a[key].dtype != b[key].dtype
            or a[key].tobytes() != b[key].tobytes()
        ):
            raise ValueError("saved/replayed array changed: " + key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        native_dir = ROOT / "native-dynamic-corpus-eight-01"
        fine_dir = ROOT / "perception-temporal-appearance-01"
        audit_path = ROOT / "native-dynamic-corpus-full-audit-01/result.json"
        native_path, fine_path = native_dir / "result.json", fine_dir / "result.json"
        pilot_path = ROOT / "mask-feature-pool-pilot-01/result.json"
        for path, pin in [
            (native_path, NATIVE_PIN),
            (audit_path, AUDIT_PIN),
            (fine_path, FINE_PIN),
            (pilot_path, PILOT_PIN),
        ]:
            if sha(path) != pin:
                raise ValueError("completed admitted parent result changed: " + str(path))
        native, fine, audit, pilot = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in (native_path, fine_path, audit_path, pilot_path)
        ]
        native_cfg = json.loads((native_dir / "effective-config.json").read_text(encoding="utf-8"))
        fine_cfg = json.loads((fine_dir / "effective-config.json").read_text(encoding="utf-8"))
        if (
            native["status"] != "PASS_NATIVE_EIGHT_CLIP_PROPOSAL_CORPUS"
            or audit["status"] != "PASS_NATIVE_FULL_CORPUS_INPUT_ADMISSION_NOT_QUALITY"
            or not audit["full_corpus_complete"]
            or audit["input_report_sha256"] != NATIVE_PIN
            or audit["frames_checked"] != 256
            or audit["media_assets_checked"] != 8
            or fine["status"] != "PASS_SPATIOTEMPORAL_APPEARANCE_EXTRACTION_ONLY"
            or fine_cfg["raster"] != 256
            or fine_cfg["gt_inputs"] is not False
            or pilot["status"] != "PASS_MASK_STATISTICS_INPUT_PROBE_NOT_QUALITY"
        ):
            raise ValueError("full native input admission and fixed frozen raster required")
        hashes = {
            str(p.resolve()): sha(p)
            for p in (
                native_path,
                audit_path,
                fine_path,
                pilot_path,
                native_dir / "effective-config.json",
                fine_dir / "effective-config.json",
                ROOT / "mask-feature-pool-pilot-01/effective-config.json",
                Path(pilot["path"]),
                Path(__file__),
                Path("src/tiny_omni_decision/mask_feature_pool.py"),
                Path("src/tiny_omni_decision/mask_feature_corpus.py"),
            )
        }
        for directory, report in [
            (native_dir, native),
            (fine_dir, fine),
            (ROOT / "mask-feature-pool-pilot-01", pilot),
        ]:
            require_config_binding(directory / "effective-config.json", report)
        if sha(pilot["path"]) != pilot["sha256"]:
            raise ValueError("protected pilot statistics changed")
        plans = native_cfg["plans"]
        if (
            [p["video_id"] for p in plans] != [p["video_id"] for p in fine_cfg["plans"]]
            or len(plans) != 8
            or len(native["rows"]) != 256
            or len(fine["rows"]) != 8
        ):
            raise ValueError("complete same ordered eight train assets required")
        dense = {}
        for plan, feature_plan, record in zip(plans, fine_cfg["plans"], fine["rows"], strict=True):
            video = plan["video_id"]
            path = fine_dir / (video + ".npz")
            if record["video_id"] != video or sha(path) != record["sha256"]:
                raise ValueError("frozen feature bank differs")
            with np.load(path, allow_pickle=False) as data:
                require_aligned_feature_times(plan, feature_plan, data["frame_indices"])
            dense[video] = path
            hashes[str(path)] = record["sha256"]
            hashes[plan["media_path"]] = plan["media_sha256"]
        for row in native["rows"]:
            hashes[row["path"]] = row["sha256"]
        if any(sha(p) != h for p, h in hashes.items()):
            raise ValueError("protected input source changed")
        if shutil.disk_usage(args.output).free < 8 * 2**30 + 256 * 2**20:
            raise RuntimeError("preserve 8GiB disk reserve plus corpus allowance")
        config = dict(
            source_hashes=hashes,
            plans=plans,
            native_result_sha256=NATIVE_PIN,
            full_audit_sha256=AUDIT_PIN,
            feature_result_sha256=FINE_PIN,
            projection="piecewise-constant cells; foreground area",
            stats="mean and two-pass population std",
            grid_channels=[256, 128],
            saved_dtype="float32",
            accumulation="float64",
            seed=17,
            expected_proposals=18441,
            expected_frames=256,
            golden_prefix_per_clip=8,
            golden_absolute_tolerance=1e-5,
            golden_relative_tolerance=0,
            body_cap_seconds=600,
            training=False,
            annotations_loaded=False,
            gpu_or_model_loaded=False,
            registered_parameters_added=0,
        )
        cfg_path = args.output / "effective-config.json"
        write(cfg_path, config)
        cfg_pin = sha(cfg_path)
        previous, reference_pins = None, {}
        if args.reference:
            ref_path = args.reference / "result.json"
            previous = json.loads(ref_path.read_text(encoding="utf-8"))
            if (
                previous["status"] != "PASS_EIGHT_CLIP_MASK_FEATURE_INPUT_CORPUS_NOT_QUALITY"
                or previous["config_sha256"] != cfg_pin
                or sha(args.reference / "effective-config.json") != cfg_pin
            ):
                raise ValueError("completed identical primary feature corpus required")
            reference_pins[str(ref_path)] = sha(ref_path)
            reference_pins[str(args.reference / "effective-config.json")] = cfg_pin
            for r in previous["rows"]:
                reference_pins[r["path"]] = r["sha256"]
            if len(previous["rows"]) != 8 or any(sha(p) != h for p, h in reference_pins.items()):
                raise ValueError("protected primary outputs changed")
        np.random.seed(17)
        rng = np.random.get_state()
        rows = []
        for plan in plans:
            tick = time.perf_counter()
            video = plan["video_id"]
            with np.load(dense[video], allow_pickle=False) as data:
                low, high, times = data["lowres"], data["hires"], data["frame_indices"]
            if (
                low.shape != (32, 32, 32, 256)
                or high.shape != (32, 64, 64, 128)
                or low.dtype != np.float32
                or high.dtype != np.float32
                or not np.isfinite(low).all()
                or not np.isfinite(high).all()
            ):
                raise ValueError("complete finite aligned native feature grids required")
            source_rows = [r for r in native["rows"] if r["video_id"] == video]
            if [r["frame_index"] for r in source_rows] != times.tolist():
                raise ValueError("complete identical native mask/dense times required")
            frames, error_mean, error_std = [], 0.0, 0.0
            for index, row in enumerate(source_rows):
                with np.load(row["path"], allow_pickle=False) as bank:
                    ids = bank["nms_indices"].copy()
                    masks = np.unpackbits(
                        bank["masks_packed"][ids], axis=1, bitorder="big"
                    ).reshape(len(ids), 256, 256)
                    boxes, scores = bank["boxes"][ids].copy(), bank["scores"][ids].copy()
                if len(ids):
                    mean, std, counts = mask_feature_statistics(masks, [low[index], high[index]])
                else:
                    mean, std = np.empty((0, 384), np.float32), np.empty((0, 384), np.float32)
                    counts = np.empty(0, np.int64)
                if index == 0:
                    if len(ids) < 8:
                        raise ValueError("complete fixed golden prefix required in every clip")
                    for i in range(8):
                        ys, xs = np.nonzero(masks[i])
                        pixels = np.concatenate(
                            [low[0, ys // 8, xs // 8], high[0, ys // 4, xs // 4]], 1
                        ).astype(np.float64)
                        m = (
                            pixels.mean(0).astype(np.float32)
                            if len(pixels)
                            else np.zeros(384, np.float32)
                        )
                        s = (
                            pixels.std(0).astype(np.float32)
                            if len(pixels)
                            else np.zeros(384, np.float32)
                        )
                        error_mean = max(
                            error_mean, float(np.abs(mean[i].astype(np.float64) - m).max())
                        )
                        error_std = max(
                            error_std, float(np.abs(std[i].astype(np.float64) - s).max())
                        )
                    if max(error_mean, error_std) > 1e-5:
                        raise ValueError("unchanged independent pixel reference guard exceeded")
                frames.append(
                    dict(
                        mean=mean,
                        std=std,
                        foreground_counts=counts,
                        proposal_ids=ids,
                        boxes=boxes,
                        scores=scores,
                    )
                )
            arrays = assemble_mask_statistics(times, frames)
            if video == "video_10903":
                with np.load(pilot["path"], allow_pickle=False) as data:
                    require_arrays(data, arrays)
            path = args.output / (video + ".npz")
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                require_arrays(saved, arrays)
            if previous:
                ref = next(r for r in previous["rows"] if r["video_id"] == video)
                with np.load(ref["path"], allow_pickle=False) as saved:
                    require_arrays(saved, arrays)
            rows.append(
                dict(
                    video_id=video,
                    path=str(path),
                    sha256=sha(path),
                    file_bytes=path.stat().st_size,
                    frames=32,
                    proposals=len(arrays["mean"]),
                    empty_masks=int((arrays["foreground_counts"] == 0).sum()),
                    golden_objects_checked=8,
                    golden_mean_max_abs=error_mean,
                    golden_std_max_abs=error_std,
                    seconds=time.perf_counter() - tick,
                )
            )
            print(video, len(arrays["mean"]), "SAVED_ALL_REGION_STATISTICS", flush=True)
            del low, high, frames, arrays, masks
            require_resource_limits(
                time.perf_counter() - start, shutil.disk_usage(args.output).free
            )
        after = np.random.get_state()
        if not (rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]):
            raise ValueError("CPU corpus unexpectedly consumed RNG")
        if (
            sum(r["proposals"] for r in rows) != 18441
            or any(sha(p) != h for p, h in {**hashes, **reference_pins}.items())
            or sha(cfg_path) != cfg_pin
        ):
            raise ValueError("complete corpus or protected sources/config changed")
        seconds = time.perf_counter() - start
        require_resource_limits(seconds, shutil.disk_usage(args.output).free)
        result = dict(
            status="PASS_EIGHT_CLIP_MASK_FEATURE_INPUT_CORPUS_NOT_QUALITY",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            config_sha256=cfg_pin,
            rows=rows,
            media_assets=8,
            observation_frames=256,
            proposals=18441,
            feature_channels=384,
            added_registered_parameters=0,
            annotations_loaded=False,
            quality_evaluated=False,
            training=False,
            gpu_or_model_loaded=False,
            validation_opened=False,
            sealed_audit_opened=False,
            fresh_replay_bitwise=bool(previous),
            all_readback_exact=True,
            pilot_all_nine_arrays_exact=True,
            numpy_rng_unchanged=True,
            reference_result_sha256=reference_pins.get(str(args.reference / "result.json"))
            if args.reference
            else None,
            protected_sources_unchanged=True,
            cloud_cost=0,
            seconds=seconds,
        )
        write(args.output / "result.json", result)
        print(result["status"], result["seconds"], flush=True)
    except Exception:
        write(
            args.output / "result.json",
            dict(
                status="FAIL_INCOMPLETE",
                error=traceback.format_exc(),
                seconds=time.perf_counter() - start,
                training=False,
                gpu_or_model_loaded=False,
                annotations_loaded=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
