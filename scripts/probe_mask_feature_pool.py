"""CPU masked statistics over every native survivor in a completed known train clip."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from tiny_omni_decision.mask_feature_pool import mask_feature_statistics

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
DYNAMIC_PIN = "191796c3e85ffea628c55d45f15cd1a09ecdba546551476e57dd55ce09a7ae0e"
FINE_PIN = "a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4"
FINE_ARRAY_PIN = "e79ff7d73ff3b58f04f0d058666768f603823140829141311d63e3cdf18158bb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_arrays(a, b):
    if set(a) != set(b):
        raise ValueError("array inventory changed")
    for k in a:
        if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or a[k].tobytes() != b[k].tobytes():
            raise ValueError("saved/replayed array differs: " + k)


def write(path, obj):
    path.write_bytes((json.dumps(obj, indent=2) + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    try:
        native_dir = ROOT / "dynamic-discovery-fixed32-01"
        fine_dir = ROOT / "perception-temporal-appearance-01"
        fine_path = fine_dir / "video_10903.npz"
        if (
            sha(native_dir / "result.json") != DYNAMIC_PIN
            or sha(fine_dir / "result.json") != FINE_PIN
        ):
            raise ValueError("completed pinned known train parents required")
        native = json.loads((native_dir / "result.json").read_text())
        fine = json.loads((fine_dir / "result.json").read_text())
        native_cfg = json.loads((native_dir / "config.json").read_text())
        fine_cfg = json.loads((fine_dir / "effective-config.json").read_text())
        if (
            native["status"] != "PASS_FIXED32_DYNAMIC_PROPOSALS_NOT_QA"
            or fine["status"] != "PASS_SPATIOTEMPORAL_APPEARANCE_EXTRACTION_ONLY"
            or sha(native_dir / "config.json") != native["config_sha256"]
            or sha(fine_dir / "effective-config.json") != fine["config_sha256"]
            or sha(fine_path) != FINE_ARRAY_PIN
        ):
            raise ValueError("parent status/config/feature array changed")
        if fine_cfg["raster"] != 256 or fine_cfg["gt_inputs"] is not False:
            raise ValueError("unaltered full256 raster with no GT feature inputs required")
        plan = native_cfg["input_plan"]
        source = next(p for p in fine_cfg["plans"] if p["video_id"] == "video_10903")
        if (
            plan["split"] != "train"
            or plan["video_id"] != source["video_id"]
            or any(
                plan[k] != source[k] for k in ("media_path", "media_sha256", "frames", "cut_frame")
            )
        ):
            raise ValueError("mask and feature native source identities differ")
        hashes = {
            str(p): sha(p)
            for p in (
                native_dir / "result.json",
                native_dir / "config.json",
                fine_dir / "result.json",
                fine_dir / "effective-config.json",
                fine_path,
                Path(__file__).resolve(),
                Path("src/tiny_omni_decision/mask_feature_pool.py").resolve(),
            )
        }
        hashes[plan["media_path"]] = plan["media_sha256"]
        for row in native["rows"]:
            if sha(row["path"]) != row["sha256"]:
                raise ValueError("frozen full native proposal bank changed")
            hashes[row["path"]] = row["sha256"]
        if any(sha(p) != h for p, h in hashes.items()):
            raise ValueError("source hash changed")
        config = dict(
            source_hashes=hashes,
            input_plan=plan,
            frame_records=32,
            expected_survivors=1070,
            grid_channels=[256, 128],
            projection="piecewise-constant cells; foreground area",
            stats="mean and two-pass population std",
            accumulation="float64",
            saved_dtype="float32",
            empty_masks="keep zero descriptors and explicit count",
            seed=17,
            golden_prefix_objects=8,
            golden_absolute_tolerance=1e-5,
            golden_relative_tolerance=0,
            max_body_seconds=300,
            training=False,
            annotations_loaded=False,
            gpu_or_model_loaded=False,
            registered_parameters_added=0,
        )
        write(args.output / "effective-config.json", config)
        config_sha = sha(args.output / "effective-config.json")
        previous = None
        reference_hash = None
        if args.reference:
            reference_path = args.reference / "result.json"
            reference_hash = sha(reference_path)
            previous = json.loads(reference_path.read_text())
            if (
                previous["status"] != "PASS_MASK_STATISTICS_INPUT_PROBE_NOT_QUALITY"
                or previous["config_sha256"] != config_sha
                or sha(args.reference / "effective-config.json") != config_sha
                or sha(previous["path"]) != previous["sha256"]
            ):
                raise ValueError("completed identical primary statistics contract required")
        np.random.seed(17)
        rng = np.random.get_state()
        collections = {
            k: []
            for k in (
                "mean",
                "std",
                "foreground_counts",
                "frame_indices",
                "proposal_ids",
                "boxes",
                "scores",
            )
        }
        rows, offsets = [], [0]
        golden_mean_error = golden_std_error = None
        with np.load(fine_path, allow_pickle=False) as data:
            low, high, times = data["lowres"], data["hires"], data["frame_indices"]
        if (
            low.shape != (32, 32, 32, 256)
            or high.shape != (32, 64, 64, 128)
            or low.dtype != np.float32
            or high.dtype != np.float32
            or times.tolist() != plan["frame_indices"]
            or [r["frame_index"] for r in native["rows"]] != times.tolist()
        ):
            raise ValueError("complete aligned native32 mask/dense feature times required")
        for index, row in enumerate(native["rows"]):
            with np.load(row["path"], allow_pickle=False) as saved:
                ids = saved["nms_indices"].copy()
                masks = np.unpackbits(saved["masks_packed"][ids], axis=1, bitorder="big").reshape(
                    len(ids), 256, 256
                )
                boxes, scores = saved["boxes"][ids].copy(), saved["scores"][ids].copy()
            if len(ids):
                mean, std, count = mask_feature_statistics(masks, [low[index], high[index]])
            else:
                mean, std = np.empty((0, 384), np.float32), np.empty((0, 384), np.float32)
                count = np.empty(0, np.int64)
            if (
                mean.shape != (len(ids), 384)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
            ):
                raise ValueError("all retained descriptor outputs must be finite and aligned")
            if index == 0:
                if len(ids) < 8:
                    raise ValueError("all fixed eight golden prefix objects required")
                # Different reference algorithm: repeat feature cells to all256x256 pixels.
                expanded = np.concatenate(
                    (
                        np.repeat(np.repeat(low[0], 8, 0), 8, 1),
                        np.repeat(np.repeat(high[0], 4, 0), 4, 1),
                    ),
                    2,
                )
                errors_mean, errors_std = [], []
                for i in range(min(8, len(ids))):
                    pixels = expanded[masks[i].astype(bool)].astype(np.float64)
                    expected_mean = (
                        pixels.mean(0).astype(np.float32) if len(pixels) else np.zeros(384)
                    )
                    expected_std = (
                        pixels.std(0).astype(np.float32) if len(pixels) else np.zeros(384)
                    )
                    errors_mean.append(
                        float(np.abs(mean[i].astype(np.float64) - expected_mean).max())
                    )
                    errors_std.append(float(np.abs(std[i].astype(np.float64) - expected_std).max()))
                    if len(pixels) != count[i]:
                        raise ValueError("golden foreground mass differs")
                golden_mean_error = max(errors_mean, default=0.0)
                golden_std_error = max(errors_std, default=0.0)
                if max(golden_mean_error, golden_std_error) > 1e-5:
                    raise ValueError("fixed expanded-pixel golden reference tolerance exceeded")
                del expanded, pixels
            values = dict(
                mean=mean,
                std=std,
                foreground_counts=count,
                frame_indices=np.full(len(ids), row["frame_index"], np.int64),
                proposal_ids=ids,
                boxes=boxes,
                scores=scores,
            )
            for key, a in values.items():
                collections[key].append(a)
            offsets.append(offsets[-1] + len(ids))
            rows.append(
                dict(
                    frame_index=row["frame_index"],
                    proposals=len(ids),
                    empty_masks=int((count == 0).sum()),
                    foreground_pixels=int(count.sum()),
                )
            )
        arrays = {k: np.concatenate(v) for k, v in collections.items()}
        arrays["frame_offsets"] = np.array(offsets, np.int64)
        arrays["observation_times"] = times.copy()
        if offsets[-1] != 1070:
            raise ValueError("complete original all1070 survivor inventory required")
        path = args.output / "mask-statistics.npz"
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as data:
            require_arrays(data, arrays)
        if previous:
            if (
                sha(args.reference / "result.json") != reference_hash
                or sha(args.reference / "effective-config.json") != config_sha
                or sha(previous["path"]) != previous["sha256"]
            ):
                raise ValueError("protected primary result/config/archive changed")
            with np.load(previous["path"], allow_pickle=False) as data:
                require_arrays(data, arrays)
        after = np.random.get_state()
        if not (rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]):
            raise ValueError("CPU statistics unexpectedly consumed RNG")
        if (
            any(sha(p) != h for p, h in hashes.items())
            or sha(args.output / "effective-config.json") != config_sha
        ):
            raise ValueError("protected sources/config changed")
        seconds = time.perf_counter() - tick
        if seconds > 300:
            raise RuntimeError("fixed CPU body cap exceeded; no silent retries")
        result = dict(
            status="PASS_MASK_STATISTICS_INPUT_PROBE_NOT_QUALITY",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            config_sha256=config_sha,
            path=str(path),
            sha256=sha(path),
            file_bytes=path.stat().st_size,
            rows=rows,
            proposals=1070,
            observation_times=32,
            feature_channels=384,
            mean_shape=list(arrays["mean"].shape),
            std_shape=list(arrays["std"].shape),
            golden_mean_max_abs=golden_mean_error,
            golden_std_max_abs=golden_std_error,
            golden_objects_checked=min(8, offsets[1]),
            rng_unchanged=True,
            save_reload_bitwise=True,
            fresh_replay_bitwise=bool(previous),
            reference_result_sha256=reference_hash,
            seconds=seconds,
            training=False,
            quality_evaluated=False,
            annotations_loaded=False,
            gpu_or_model_loaded=False,
            unique_media_assets=1,
            unique_physical_objects_known=False,
            main_process_intervened=False,
            registered_parameters_added=0,
            cloud_cost=0,
        )
        write(args.output / "result.json", result)
        print(result["status"], seconds, golden_mean_error, golden_std_error, flush=True)

    except Exception:
        write(
            args.output / "result.json",
            dict(
                status="FAIL_INCOMPLETE",
                seconds=time.perf_counter() - tick,
                error=traceback.format_exc(),
                training=False,
                gpu_or_model_loaded=False,
                annotations_loaded=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
