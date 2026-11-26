"""CPU-only fine-grid sampling along fixed model tracks; no QA or fitting."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from scripts.extract_spatial_context_corpus import PRIOR_PIN, ROOT, require_bitwise, sha
from tiny_omni_decision.track_appearance_features import track_appearance_features

REFERENCE_PIN = "a000672b4500a8b4ae9bfde3f15e06d8103c56e986c2d7d3c17a6a77addf98f4"
IDS = ["video_" + str(v) for v in (10103, 10186, 10212, 10387, 1087, 10886, 10894, 10903)]


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def validate_binding(point, plan, row, times):
    if (
        point["video_id"] != plan["video_id"]
        or point["video_id"] != row["video_id"]
        or point["split"] != "train"
        or plan["split"] != "train"
        or point["shape"] != [256, 512]
        or point["media_sha256"] != plan["media_sha256"]
        or point["frames"] != plan["frames"]
        or point["frames"] != row["original_clip_frames"]
        or row["frames_encoded"] != 32
        or times.shape != (32,)
        or times.dtype.kind not in "iu"
        or not np.array_equal(times, point["frame_indices"])
        or not np.array_equal(
            times, np.rint(np.linspace(0, point["frames"] - 1, 32)).astype(np.int64)
        )
    ):
        raise ValueError("training identity/media/time/point binding changed")


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def official_track_appearance(low, high, query_xy, utils):
    import torch

    n, t, _ = query_xy.shape
    q = (
        torch.stack(
            (
                torch.arange(t)[None].expand(n, -1),
                torch.from_numpy(query_xy.copy()[..., 1]) * 256,
                torch.from_numpy(query_xy.copy()[..., 0]) * 256,
            ),
            -1,
        )
        .float()
        .reshape(1, -1, 3)
    )
    blocks = []
    with torch.no_grad():
        for grid in (low, high):
            coords = utils.convert_grid_coordinates(
                q, (t, 256, 256), grid.shape[:3], coordinate_format="tyx"
            )
            blocks.append(utils.map_coordinates_3d(torch.from_numpy(grid[None]), coords.clone()))
    return torch.cat(blocks, -1)[0].reshape(n, t, -1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        rows=[],
        training=False,
        annotations_loaded=False,
        validation_feature_arrays_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        device="cpu",
        new_parameters=0,
        models_loaded=False,
    )
    try:
        fine_root = ROOT / "perception-temporal-appearance-01"
        if sha(fine_root / "result.json") != REFERENCE_PIN:
            raise ValueError("complete fine-grid reference changed")
        fine = json.loads((fine_root / "result.json").read_text())
        fine_config = json.loads((fine_root / "effective-config.json").read_text())
        if sha(fine_root / "effective-config.json") != fine["config_sha256"]:
            raise ValueError("fine-grid effective config changed")
        prior_path = ROOT / "perception-occlusion-trajectory-features-01/result.json"
        if sha(prior_path) != PRIOR_PIN:
            raise ValueError("complete point bank changed")
        points = json.loads(prior_path.read_text())
        if points["status"] != "PASS_FEATURE_EXTRACTION_NOT_QUALITY":
            raise ValueError("complete original point bank mandatory")
        point_rows = {r["video_id"]: r for r in points["features"]}
        if [p["video_id"] for p in fine_config["plans"]] != IDS or [
            r["video_id"] for r in fine["rows"]
        ] != IDS:
            raise ValueError("fixed eight training references required")
        for path, pin in fine_config["source_hashes"].items():
            if sha(path) != pin:
                raise ValueError("official sampler source changed")
        reference = None
        if args.reference:
            reference = json.loads((args.reference / "result.json").read_text())
            if reference["status"] != "PASS_TRACK_APPEARANCE_PRIMITIVE_NOT_QA_QUALITY":
                raise ValueError("complete primitive reference mandatory")
            if [r["video_id"] for r in reference["rows"]] != IDS:
                raise ValueError("reference identities changed")
        expected_bytes = (
            0
            if reference
            else 8 * (2 * 256 * 32 * 384 * 4 + 2 * 256 * 32 + 256 * 32 * 2 * 4 + 32 * 8 + 6 * 128)
        )
        if shutil.disk_usage(args.output).free < 8 * 1024**3 + expected_bytes + 64 * 1024**2:
            raise OSError("8GiB disk reserve mandatory")
        audit = Path("C:/CodexArtifacts/pretrained-reuse-r1/tapir-audit-01")
        sys.path[:0] = [str(audit / "deps"), str(audit / "source")]
        import torch
        from tapnet.torch import utils

        from scripts.train_spatial_trajectory_readout import process_peak_bytes

        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        cpu_rng = torch.get_rng_state()
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            helper_sha256=sha(Path("src/tiny_omni_decision/track_appearance_features.py")),
            fine_result_sha256=REFERENCE_PIN,
            point_result_sha256=PRIOR_PIN,
            fine_config_sha256=fine["config_sha256"],
            source_hashes=fine_config["source_hashes"],
            ids=IDS,
            points=256,
            times=32,
            channels=384,
            device="cpu",
            torch=torch.__version__,
            numpy=np.__version__,
            dtype="float32",
            accumulation="float64",
            official_absolute_gate=1e-5,
            source_coordinate_domain="normalized raster; closed[0,1]",
            padding="border",
            new_parameters=0,
            training=False,
            max_seconds=300,
            time_limit_policy="admission checks between clips and at completion; no watchdog",
            disk_reserve_bytes=8 * 1024**3,
            expected_output_bytes=expected_bytes,
            reference_result_sha256=sha(args.reference / "result.json") if reference else None,
        )
        write(args.output / "config.json", config)
        report.update(
            status="RUNNING",
            config_sha256=sha(args.output / "config.json"),
            source_commit=config["source_commit"],
        )

        def cap():
            if time.perf_counter() - start > 300:
                raise TimeoutError("300-second body admission gate exceeded")

        for index, (plan, row) in enumerate(zip(fine_config["plans"], fine["rows"], strict=True)):
            cap()
            vid = plan["video_id"]
            point = point_rows[vid]
            archive = fine_root / f"{vid}.npz"
            if (
                sha(archive) != row["sha256"]
                or sha(point["path"]) != point["sha256"]
                or sha(plan["media_path"]) != plan["media_sha256"]
            ):
                raise ValueError("fine grid/point/media bytes changed")
            with np.load(archive, allow_pickle=False) as saved:
                low, high, times = (saved[k].copy() for k in ("lowres", "hires", "frame_indices"))
            validate_binding(point, plan, row, times)
            motion = np.load(point["path"], allow_pickle=False)
            if (
                motion.shape != (256, 512)
                or motion.dtype != np.float32
                or not np.isfinite(motion).all()
            ):
                raise ValueError("invalid original point tokens")
            if low.shape != (32, 32, 32, 256) or high.shape != (32, 64, 64, 128):
                raise ValueError("fine-grid shape changed")
            xy = motion[:, :128].reshape(256, 32, 4)[..., :2].copy()
            before = {
                k: array_sha(v) for k, v in dict(low=low, high=high, xy=xy, motion=motion).items()
            }
            actual, support = track_appearance_features(low, high, xy)
            fixed_xy = np.broadcast_to(xy[:, :1], xy.shape)
            initial, initial_support = track_appearance_features(low, high, fixed_xy)
            require_bitwise(actual, track_appearance_features(low, high, xy)[0])
            require_bitwise(actual[::-1], track_appearance_features(low, high, xy[::-1])[0])

            errors = {}
            for name, value, coords in (("actual", actual, xy), ("initial", initial, fixed_xy)):
                expected = official_track_appearance(low, high, coords, utils)
                errors[name] = float(np.abs(expected - value).max())
                if errors[name] > 1e-5:
                    raise ValueError("official fine-grid sampler gate failed")
            if any(
                before[k] != array_sha(v)
                for k, v in dict(low=low, high=high, xy=xy, motion=motion).items()
            ):
                raise ValueError("source array or coordinates mutated")
            arrays = dict(
                actual_appearance=actual,
                initial_position_appearance=initial,
                actual_support=support,
                initial_position_support=initial_support,
                estimated_xy=xy,
                frame_indices=times,
            )
            outputs = {}
            for name, value in arrays.items():
                if reference:
                    info = reference["rows"][index]["outputs"][name]
                    if sha(info["path"]) != info["sha256"]:
                        raise ValueError("primitive output changed")
                    require_bitwise(value, np.load(info["path"], allow_pickle=False))
                    outputs[name] = info
                else:
                    path = args.output / f"{vid}-{name}.npy"
                    np.save(path, value, allow_pickle=False)
                    require_bitwise(value, np.load(path, allow_pickle=False))
                    outputs[name] = dict(
                        path=str(path),
                        sha256=sha(path),
                        shape=list(value.shape),
                        dtype=str(value.dtype),
                        bytes=path.stat().st_size,
                    )
            a, b = actual.astype(np.float64), initial.astype(np.float64)
            norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            cosine = np.divide((a * b).sum(-1), norms, out=np.zeros_like(norms), where=norms > 0)
            report["rows"].append(
                dict(
                    video_id=vid,
                    media_sha256=plan["media_sha256"],
                    point_sha256=point["sha256"],
                    fine_grid_sha256=row["sha256"],
                    frame_indices=times.tolist(),
                    outputs=outputs,
                    official_maximum_errors=errors,
                    support_count=int(support.sum()),
                    total_point_times=int(support.size),
                    initial_position_support_count=int(initial_support.sum()),
                    actual_initial_position_descriptor_max_difference=float(
                        np.abs(actual - initial).max()
                    ),
                    mean_actual_initial_position_cosine=float(cosine.mean()),
                    zero_norm_pairs=int((norms == 0).sum()),
                    source_arrays_unchanged=True,
                    point_permutation_bitwise=True,
                    repeat_bitwise=True,
                )
            )
            write(args.output / "progress.json", report)
            print(vid, "SAVED" if not reference else "FRESH_REPLAY_EXACT", errors, flush=True)
            del (
                low,
                high,
                xy,
                motion,
                actual,
                initial,
                support,
                initial_support,
                arrays,
                value,
                expected,
                a,
                b,
            )
        for plan, row in zip(fine_config["plans"], fine["rows"], strict=True):
            if (
                sha(plan["media_path"]) != plan["media_sha256"]
                or sha(fine_root / f"{row['video_id']}.npz") != row["sha256"]
                or sha(point_rows[row["video_id"]]["path"]) != point_rows[row["video_id"]]["sha256"]
            ):
                raise ValueError("protected inputs changed during execution")
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise ValueError("sampling changed CPU RNG")
        for path, pin in fine_config["source_hashes"].items():
            if sha(path) != pin:
                raise ValueError("official implementation changed")
        cap()
        report.update(
            status="PASS_TRACK_APPEARANCE_PRIMITIVE_NOT_QA_QUALITY",
            fresh_replay=bool(reference),
            cpu_rng_unchanged=True,
            peak_process_working_set_bytes=process_peak_bytes(),
        )
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        write(args.output / "result.json", report)


if __name__ == "__main__":
    main()
