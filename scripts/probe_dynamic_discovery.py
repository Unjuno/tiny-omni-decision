"""Frozen per-time model proposals; labels are absent from the inference process."""

import argparse
import ast
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.probe_track_appearance import REFERENCE_PIN, ROOT, sha, write

R1 = Path("C:/CodexArtifacts/pretrained-reuse-r1/sam2.1-hiera-tiny-audit-01")
WEIGHT_PIN = "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"
LEGACY_PIN = "b9918adcdf5aceed646b6d404a611618fe3026a9edec822beea382dd37e943c0"
OLD_RESULT_PIN = "75665a07c986dd0a97bff4854912b7a265e91baf49c0c5d45637886e880fa3d1"
OLD_PROPOSAL_PIN = "9d4f50c0722a93589d05471f1b94e9c05b72dc7fb49edfc5deb7df63c528f075"


def input_plan(source, times):
    if (
        source["video_id"] != "video_10903"
        or source["split"] != "train"
        or source["frames"] < 32
        or source["cut_frame"] not in (-1, source["frames"])
        or times.shape != (32,)
        or times.dtype.kind not in "iu"
        or not np.array_equal(
            times, np.rint(np.linspace(0, source["frames"] - 1, 32)).astype(np.int64)
        )
    ):
        raise ValueError("fixed training scene/cut/32 uniform times required")
    plan = {
        k: source[k]
        for k in ("video_id", "split", "media_path", "media_sha256", "cut_frame", "frames")
    }
    return dict(**plan, frame_indices=times.tolist())


def verify_cache_policy(current, previous):
    for key in (
        "input_plan",
        "source_hashes",
        "cache_policy",
        "model_revision",
        "grid_side",
        "chunk_objects",
        "mask_threshold",
        "nms_iou",
        "top_k",
        "dtype",
        "attention",
        "tf32",
        "fine_time_archive_sha256",
    ):
        if current[key] != previous[key]:
            raise ValueError("cache smoke/replay contract changed: " + key)


def legacy_functions():
    """Reuse only four pinned pure functions, never execute the old experiment."""
    path = R1 / "auto_init02.py"
    if sha(path) != LEGACY_PIN:
        raise ValueError("reference geometry/state implementation changed")
    source = ast.parse(path.read_text())
    selected = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"ious", "nms", "boxes_from_masks", "state_sha"}
    ]
    if len(selected) != 4:
        raise ValueError("reference helper inventory changed")
    scope = dict(np=np, hashlib=hashlib)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path)
    parser.add_argument("--replay-last", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import av
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    from scripts.train_spatial_trajectory_readout import process_peak_bytes

    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_at=datetime.now(UTC).isoformat(),
        rows=[],
        training=False,
        annotations_loaded=False,
        validation_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        smoke=args.smoke,
        replay_last=bool(args.replay_last),
    )
    try:
        helpers = legacy_functions()
        fine_root = ROOT / "perception-temporal-appearance-01"
        if sha(fine_root / "result.json") != REFERENCE_PIN:
            raise ValueError("completed fine-grid reference changed")
        fine = json.loads((fine_root / "result.json").read_text())
        inherited = json.loads((fine_root / "effective-config.json").read_text())
        if sha(fine_root / "effective-config.json") != fine["config_sha256"]:
            raise ValueError("reference config changed")
        source = next(p for p in inherited["plans"] if p["video_id"] == "video_10903")
        row = next(r for r in fine["rows"] if r["video_id"] == "video_10903")
        with np.load(fine_root / "video_10903.npz", allow_pickle=False) as saved:
            times = saved["frame_indices"].copy()
        if sha(fine_root / "video_10903.npz") != row["sha256"]:
            raise ValueError("reference time bank changed")
        plan = input_plan(source, times)
        weight = R1 / "model/model.safetensors"
        if sha(weight) != WEIGHT_PIN or sha(plan["media_path"]) != plan["media_sha256"]:
            raise ValueError("checkpoint/media changed")
        old_root = ROOT / "perception-sam2-auto-init-02"
        if sha(old_root / "result.json") != OLD_RESULT_PIN:
            raise ValueError("pinned original result changed")
        old_result = json.loads((old_root / "result.json").read_text())
        old_row = next(r for r in old_result["rows"] if r["video_id"] == "video_10903")
        old_proposal = old_root / "video_10903-proposals.npz"
        if source["bank_sha256"] != OLD_PROPOSAL_PIN:
            raise ValueError("original bank/fine-grid provenance disagrees")
        if (
            old_result["status"] != "PASS_AUTO_INIT_EXECUTION_ONLY"
            or sha(old_proposal) != OLD_PROPOSAL_PIN
            or old_row["proposal_sha256"] != OLD_PROPOSAL_PIN
        ):
            raise ValueError("complete pinned original frame0 proposals mandatory")
        if sha(old_root / "effective-config.json") != old_result["config_sha256"]:
            raise ValueError("original proposal config changed")
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            raise MemoryError("12GiB free local GPU required")
        if shutil.disk_usage(args.output).free < 8 * 1024**3 + 384 * 1024**2:
            raise OSError("8GiB disk reserve plus bounded output required")
        if not args.smoke:
            if not args.smoke_reference:
                raise ValueError("completed separate cache smoke required")
            smoke = json.loads((args.smoke_reference / "result.json").read_text())
            if (
                smoke["status"] != "PASS_SAME_FRAME_CACHE_SMOKE"
                or not smoke["old_frame0_proposals_exact"]
            ):
                raise ValueError("cache smoke did not pass")
            if smoke["runner_sha256"] != sha(Path(__file__)):
                raise ValueError("smoke execution code changed")
        reference = None
        if args.replay_last:
            reference = json.loads((args.replay_last / "result.json").read_text())
            if (
                reference["status"] != "PASS_FIXED32_DYNAMIC_PROPOSALS_NOT_QA"
                or len(reference["rows"]) != 32
            ):
                raise ValueError("complete32-frame replay reference mandatory")
        selected = times[:1] if args.smoke else times[-1:] if reference else times
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        model, info = Sam2VideoModel.from_pretrained(
            R1 / "model",
            local_files_only=True,
            output_loading_info=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        if any(
            info.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError("strict checkpoint loading failed")
        model = model.eval().requires_grad_(False).cuda()
        processor = Sam2VideoProcessor.from_pretrained(R1 / "model", local_files_only=True)
        before = helpers["state_sha"](model)
        if sum(p.numel() for p in model.parameters()) != 38962498:
            raise ValueError("full registered parameter inventory changed")
        source_files = {
            str(Path(__file__)): sha(Path(__file__)),
            str(R1 / "auto_init02.py"): LEGACY_PIN,
            str(weight): WEIGHT_PIN,
            str(Path(plan["media_path"])): plan["media_sha256"],
        }
        for path in (
            fine_root / "result.json",
            fine_root / "effective-config.json",
            fine_root / "video_10903.npz",
            old_root / "result.json",
            old_root / "effective-config.json",
            old_proposal,
        ):
            source_files[str(path)] = sha(path)
        for p in (R1 / "model").glob("*.json"):
            source_files[str(p)] = sha(p)
        for cls in (
            type(model),
            type(processor),
            type(processor.image_processor),
            type(processor.video_processor),
        ):
            p = Path(inspect.getfile(cls))
            source_files[str(p)] = sha(p)
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            input_plan=plan,
            executed_frame_indices=selected.tolist(),
            model_revision="de431c4043854a71d8101e17995dfe596bf101a5",
            old_proposal_sha256=old_row["proposal_sha256"],
            fine_result_sha256=REFERENCE_PIN,
            fine_config_sha256=fine["config_sha256"],
            fine_time_archive_sha256=row["sha256"],
            old_result_sha256=sha(old_root / "result.json"),
            old_config_sha256=old_result["config_sha256"],
            source_hashes=source_files,
            grid_side=32,
            chunk_objects=64,
            mask_threshold=0.0,
            nms_iou=0.7,
            top_k=None,
            cache_policy="reset_tracking_data within one frame; new session for every source time",
            seed=17,
            torch=torch.__version__,
            transformers=__import__("transformers").__version__,
            dtype="float32",
            attention="sdpa",
            tf32=False,
            max_seconds=120 if args.smoke else 1200,
            peak_torch_cap_bytes=6 * 1024**3,
            disk_reserve_bytes=8 * 1024**3,
            training=False,
            smoke_result_sha256=sha(args.smoke_reference / "result.json")
            if args.smoke_reference
            else None,
            replay_result_sha256=sha(args.replay_last / "result.json") if reference else None,
        )
        if not args.smoke:
            if sha(args.smoke_reference / "config.json") != smoke["config_sha256"]:
                raise ValueError("cache smoke config changed")
            verify_cache_policy(
                config, json.loads((args.smoke_reference / "config.json").read_text())
            )
        if reference:
            if sha(args.replay_last / "config.json") != reference["config_sha256"]:
                raise ValueError("full replay reference config changed")
            verify_cache_policy(config, json.loads((args.replay_last / "config.json").read_text()))
        write(args.output / "config.json", config)
        report.update(
            status="RUNNING",
            config_sha256=sha(args.output / "config.json"),
            source_commit=config["source_commit"],
            runner_sha256=sha(Path(__file__)),
            parameters=38962498,
            model_state_before=before,
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            encoder_calls_total=0,
        )
        rng = torch.get_rng_state().clone()
        cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
        counter = [0]

        def count_encoder(_module, _inputs):
            counter[0] += 1

        hook = model.vision_encoder.register_forward_pre_hook(count_encoder)

        def cap():
            if time.perf_counter() - start > config["max_seconds"]:
                raise TimeoutError("body admission gate exceeded; no hard watchdog")
            if torch.cuda.max_memory_allocated() > config["peak_torch_cap_bytes"]:
                raise MemoryError("6GiB peak Torch allocation cap exceeded")

        def session(frame):
            s = processor.init_video_session(
                video=frame[None],
                inference_device="cuda",
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.float32,
            )
            if s.num_frames != 1 or s.cache.get_vision_features(0) is not None:
                raise ValueError("new source-frame session has inherited state")
            return s

        wanted = set(selected.tolist())
        with av.open(plan["media_path"]) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for original_index, frame in enumerate(container.decode(stream)):
                if original_index >= plan["frames"]:
                    break
                if original_index not in wanted:
                    continue
                cap()
                tick = time.perf_counter()
                image = frame.to_ndarray(format="rgb24")
                rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
                h, w = image.shape[:2]
                points = np.array(
                    [
                        [(x + 0.5) * w / 32, (y + 0.5) * h / 32]
                        for y in range(32)
                        for x in range(32)
                    ],
                    np.float32,
                )
                s = session(image)
                pixels_before = s.processed_frames[0].clone()
                masks = []
                scores = []
                cached_seconds = 0.0
                fresh_seconds = 0.0
                cached_calls = 0
                fresh_calls = 0
                prefixes = {}
                with torch.inference_mode():
                    for begin in range(0, 1024, 64):
                        cap()
                        s.reset_tracking_data()
                        if (
                            s.get_obj_num() != 0
                            or s.obj_ids
                            or s.output_dict_per_obj
                            or s.obj_with_new_inputs
                        ):
                            raise ValueError("tracking history survived reset")
                        if begin and s.cache.get_vision_features(0) is None:
                            raise ValueError("same-frame cache disappeared")
                        processor.add_inputs_to_inference_session(
                            s,
                            frame_idx=0,
                            obj_ids=list(range(1, 65)),
                            input_points=points[begin : begin + 64][None, :, None, :].tolist(),
                            input_labels=[[[1]] * 64],
                        )
                        count_before = counter[0]
                        torch.cuda.synchronize()
                        forward_start = time.perf_counter()
                        output = model(s, frame_idx=0, run_mem_encoder=False)
                        torch.cuda.synchronize()
                        cached_seconds += time.perf_counter() - forward_start
                        cached_calls += counter[0] - count_before
                        raw = output.pred_masks.detach().cpu().numpy()[:, 0]
                        score = output.object_score_logits.detach().cpu().numpy().reshape(-1)
                        if (
                            raw.shape != (64, 256, 256)
                            or score.shape != (64,)
                            or not np.isfinite(raw).all()
                            or not np.isfinite(score).all()
                        ):
                            raise ValueError("invalid native proposal outputs")
                        masks.extend(raw > 0)
                        scores.extend(score)
                        if args.smoke and begin < 128:
                            fresh = session(image)
                            processor.add_inputs_to_inference_session(
                                fresh,
                                frame_idx=0,
                                obj_ids=list(range(1, 65)),
                                input_points=points[begin : begin + 64][None, :, None, :].tolist(),
                                input_labels=[[[1]] * 64],
                            )
                            count_before = counter[0]
                            torch.cuda.synchronize()
                            forward_start = time.perf_counter()
                            other = model(fresh, frame_idx=0, run_mem_encoder=False)
                            torch.cuda.synchronize()
                            fresh_seconds += time.perf_counter() - forward_start
                            fresh_calls += counter[0] - count_before
                            other_raw = other.pred_masks.detach().cpu().numpy()[:, 0]
                            other_score = (
                                other.object_score_logits.detach().cpu().numpy().reshape(-1)
                            )
                            for key, a, b in (
                                ("raw", raw, other_raw),
                                ("score", score, other_score),
                            ):
                                if (
                                    a.shape != b.shape
                                    or a.dtype != b.dtype
                                    or a.tobytes() != b.tobytes()
                                ):
                                    raise ValueError(
                                        "fresh-session/cached raw output parity failed"
                                    )
                                prefixes[f"{begin}_{key}_cached"] = a.copy()
                                prefixes[f"{begin}_{key}_fresh"] = b.copy()
                            del fresh, other, other_raw, other_score
                        del output
                if cached_calls != 1 or (args.smoke and fresh_calls != 2):
                    raise ValueError("encoder cache call inventory violated")
                if not torch.equal(pixels_before, s.processed_frames[0]):
                    raise ValueError("processor source frame mutated")
                masks = np.stack(masks)
                scores = np.array(scores, np.float32)
                boxes = helpers["boxes_from_masks"](masks)
                kept = np.array(helpers["nms"](boxes, scores), np.int64)
                values = dict(
                    masks_packed=np.packbits(masks.reshape(1024, -1), axis=1, bitorder="big"),
                    scores=scores,
                    boxes=boxes,
                    points=points,
                    nms_indices=kept,
                )
                if original_index == 0:
                    with np.load(old_proposal, allow_pickle=False) as old:
                        for name, value in values.items():
                            if (
                                old[name].shape != value.shape
                                or old[name].dtype != value.dtype
                                or old[name].tobytes() != value.tobytes()
                            ):
                                raise ValueError("original all1024 frame0 archive parity failed")
                    report["old_frame0_proposals_exact"] = True
                if reference:
                    oldrow = reference["rows"][-1]
                    if (
                        oldrow["frame_index"] != original_index
                        or oldrow["rgb_sha256"] != rgb_sha
                        or sha(oldrow["path"]) != oldrow["sha256"]
                    ):
                        raise ValueError("last-frame replay binding changed")
                    with np.load(oldrow["path"], allow_pickle=False) as old:
                        if set(old.files) != set(values):
                            raise ValueError("replay keys changed")
                        for name, value in values.items():
                            if (
                                old[name].shape != value.shape
                                or old[name].dtype != value.dtype
                                or old[name].tobytes() != value.tobytes()
                            ):
                                raise ValueError("last-frame fresh model reload changed bytes")
                    report["last_frame_reload_exact"] = True
                path = args.output / f"{original_index:04d}-proposals.npz"
                np.savez_compressed(path, **values)
                with np.load(path, allow_pickle=False) as saved:
                    for name, value in values.items():
                        if (
                            saved[name].dtype != value.dtype
                            or saved[name].shape != value.shape
                            or saved[name].tobytes() != value.tobytes()
                        ):
                            raise ValueError("proposal readback changed")
                if prefixes:
                    prefix_path = args.output / "smoke-raw-prefix.npz"
                    np.savez_compressed(prefix_path, **prefixes)
                    report["prefix_sha256"] = sha(prefix_path)
                    report["raw_prefix_cached_fresh_bitwise"] = True
                report["rows"].append(
                    dict(
                        frame_index=original_index,
                        rgb_sha256=rgb_sha,
                        input_shape=list(image.shape),
                        path=str(path),
                        sha256=sha(path),
                        file_bytes=path.stat().st_size,
                        raw_proposals=1024,
                        empty_proposals=int((~masks.any((1, 2))).sum()),
                        nms_count=len(kept),
                        encoder_calls_cached=cached_calls,
                        encoder_calls_fresh=fresh_calls,
                        cached_forward_seconds=cached_seconds,
                        fresh_prefix_forward_seconds=fresh_seconds,
                        frame_seconds=time.perf_counter() - tick,
                        readback_bitwise=True,
                    )
                )
                report["seconds"] = time.perf_counter() - start
                write(args.output / "progress.json", report)
                print(
                    original_index,
                    "SAVED_ALL1024",
                    len(kept),
                    "NMS",
                    "encoder",
                    cached_calls,
                    flush=True,
                )
                del (
                    s,
                    image,
                    pixels_before,
                    masks,
                    scores,
                    boxes,
                    kept,
                    values,
                    raw,
                    score,
                    prefixes,
                    value,
                )
                torch.cuda.empty_cache()
                cap()
        hook.remove()
        if [r["frame_index"] for r in report["rows"]] != selected.tolist():
            raise ValueError("selected source time coverage incomplete")
        after = helpers["state_sha"](model)
        if after != before or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise ValueError("frozen model state changed")
        if not torch.equal(rng, torch.get_rng_state()) or any(
            not torch.equal(a, b)
            for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        ):
            raise ValueError("inference changed CPU/CUDA RNG")
        for path, pin in source_files.items():
            if sha(path) != pin:
                raise ValueError("protected source changed")
        cap()
        report.update(
            status="PASS_SAME_FRAME_CACHE_SMOKE"
            if args.smoke
            else "PASS_LAST_FRAME_FRESH_RELOAD"
            if reference
            else "PASS_FIXED32_DYNAMIC_PROPOSALS_NOT_QA",
            encoder_calls_total=counter[0],
            model_state_after=after,
            frozen_state_unchanged=True,
            cpu_cuda_rng_unchanged=True,
            source_files_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_working_set_bytes=process_peak_bytes(),
        )
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.perf_counter() - start, ended_at=datetime.now(UTC).isoformat())
        write(args.output / "result.json", report)


if __name__ == "__main__":
    main()
