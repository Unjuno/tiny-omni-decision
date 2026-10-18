"""Eight known train clips,32 native proposal times; labels never enter this process."""

import argparse
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

from scripts.probe_dynamic_discovery import (
    OLD_RESULT_PIN,
    R1,
    WEIGHT_PIN,
    legacy_functions,
)
from scripts.probe_timed_mask_tracking import require_arrays, verify_sources
from scripts.probe_track_appearance import REFERENCE_PIN, ROOT, sha, write
from tiny_omni_decision.native_proposal_corpus import (
    PROPOSAL_KEYS,
    plan_native_corpus,
    require_reference_config,
    validate_native_proposal_arrays,
    validate_native_smoke_report,
)
from tiny_omni_decision.sam2_chunked_masks import extend_pinned_sources

DYNAMIC_PIN = "191796c3e85ffea628c55d45f15cd1a09ecdba546551476e57dd55ce09a7ae0e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-reference", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.smoke and args.replay:
        parser.error("smoke and replay are separate processes")
    args.output.mkdir(exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import av
    import torch
    import transformers
    from transformers import Sam2VideoModel, Sam2VideoProcessor

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
        replay=bool(args.replay),
    )
    try:
        fine_dir = ROOT / "perception-temporal-appearance-01"
        old_dir = ROOT / "perception-sam2-auto-init-02"
        dynamic_dir = ROOT / "dynamic-discovery-fixed32-01"
        for p, pin in (
            (fine_dir / "result.json", REFERENCE_PIN),
            (old_dir / "result.json", OLD_RESULT_PIN),
            (dynamic_dir / "result.json", DYNAMIC_PIN),
        ):
            if sha(p) != pin:
                raise ValueError("protected parent result changed: " + str(p))
        fine, old, dynamic = [
            json.loads((p / "result.json").read_text()) for p in (fine_dir, old_dir, dynamic_dir)
        ]
        if (
            fine["status"] != "PASS_SPATIOTEMPORAL_APPEARANCE_EXTRACTION_ONLY"
            or old["status"] != "PASS_AUTO_INIT_EXECUTION_ONLY"
            or dynamic["status"] != "PASS_FIXED32_DYNAMIC_PROPOSALS_NOT_QA"
        ):
            raise ValueError("complete native/fine parent experiments required")
        parent_cfgs = [
            json.loads((p / name).read_text())
            for p, name in (
                (fine_dir, "effective-config.json"),
                (old_dir, "effective-config.json"),
                (dynamic_dir, "config.json"),
            )
        ]
        for p, name, r in (
            (fine_dir, "effective-config.json", fine),
            (old_dir, "effective-config.json", old),
            (dynamic_dir, "config.json", dynamic),
        ):
            if sha(p / name) != r["config_sha256"]:
                raise ValueError("protected parent configuration changed")
        if (
            parent_cfgs[2]["grid_side"] != 32
            or parent_cfgs[2]["chunk_objects"] != 64
            or parent_cfgs[2]["top_k"] is not None
            or parent_cfgs[2]["dtype"] != "float32"
            or parent_cfgs[2]["nms_iou"] != 0.7
            or parent_cfgs[2]["mask_threshold"] != 0.0
            or parent_cfgs[2]["model_revision"] != "de431c4043854a71d8101e17995dfe596bf101a5"
        ):
            raise ValueError("unchanged native full-grid policy required")
        hashes = dict(parent_cfgs[2]["source_hashes"])
        verify_sources(hashes)
        times = {}
        old_rows = {r["video_id"]: r for r in old["rows"]}
        fine_rows = {r["video_id"]: r for r in fine["rows"]}
        reused = {}
        for source in parent_cfgs[0]["plans"]:
            video = source["video_id"]
            bank = fine_dir / (video + ".npz")
            parent = old_dir / (video + "-proposals.npz")
            if (
                sha(bank) != fine_rows[video]["sha256"]
                or sha(parent) != old_rows[video]["proposal_sha256"]
                or source["bank_sha256"] != old_rows[video]["proposal_sha256"]
            ):
                raise ValueError("full fine-time/initial-proposal ancestry changed")
            with np.load(bank, allow_pickle=False) as data:
                times[video] = data["frame_indices"].copy()
            reused[(video, 0)] = dict(path=str(parent), sha256=sha(parent), kind="initial_parent")
            hashes = extend_pinned_sources(
                hashes,
                {
                    str(bank): sha(bank),
                    str(parent): sha(parent),
                    source["media_path"]: source["media_sha256"],
                },
            )
        plans = plan_native_corpus(parent_cfgs[0]["plans"], times)
        plan10903 = next(p for p in plans if p["video_id"] == "video_10903")
        if parent_cfgs[2]["input_plan"] != plan10903:
            raise ValueError("existing32-time native parent plan differs")
        if [r["frame_index"] for r in dynamic["rows"]] != plan10903["frame_indices"]:
            raise ValueError("complete existing32-time native records required")
        for r in dynamic["rows"]:
            if sha(r["path"]) != r["sha256"]:
                raise ValueError("existing dynamic proposal changed")
            reused[("video_10903", r["frame_index"])] = dict(
                path=r["path"],
                sha256=r["sha256"],
                kind="dynamic_parent",
                rgb_sha256=r["rgb_sha256"],
            )
            hashes = extend_pinned_sources(hashes, {r["path"]: r["sha256"]})
        for p in (
            fine_dir / "result.json",
            fine_dir / "effective-config.json",
            old_dir / "result.json",
            old_dir / "effective-config.json",
            dynamic_dir / "result.json",
            dynamic_dir / "config.json",
            Path(__file__).resolve(),
            Path("src/tiny_omni_decision/native_proposal_corpus.py").resolve(),
            Path("src/tiny_omni_decision/sam2_chunked_masks.py").resolve(),
        ):
            hashes = extend_pinned_sources(hashes, {str(p): sha(p)})
        verify_sources(hashes)
        if (
            not torch.cuda.is_available()
            or torch.cuda.mem_get_info()[0] < 12 * 1024**3
            or shutil.disk_usage(args.output).free < 8 * 1024**3 + 512 * 1024**2
        ):
            raise MemoryError("12GiB free GPU/8GiB+512MiB disk admission required")
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
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
            raise ValueError("strict native checkpoint required")
        model.eval().requires_grad_(False).cuda()
        processor = Sam2VideoProcessor.from_pretrained(R1 / "model", local_files_only=True)
        for cls in (type(model), type(processor), type(processor.video_processor)):
            p = Path(inspect.getfile(cls))
            hashes = extend_pinned_sources(hashes, {str(p): sha(p)})
        if sha(R1 / "model/model.safetensors") != WEIGHT_PIN:
            raise ValueError("unchanged native checkpoint required")
        config = dict(
            plans=plans,
            reused_records=[dict(video_id=v, frame_index=t, **r) for (v, t), r in reused.items()],
            source_hashes=hashes,
            model_revision=parent_cfgs[2]["model_revision"],
            weight_sha256=WEIGHT_PIN,
            parameters=38962498,
            grid_side=32,
            objects_per_native_session=64,
            decoder_object_batch=1,
            mask_threshold=0.0,
            nms_iou=0.7,
            top_k=None,
            dtype="float32",
            attention="sdpa",
            seed=17,
            tf32=False,
            image_cache="same-frame only; new session every time",
            training=False,
            annotations_loaded=False,
            peak_torch_cap_bytes=6 * 1024**3,
            full_body_cap_seconds=7200,
            disk_reserve_bytes=8 * 1024**3,
            batched_head_path_used=False,
            parent_artifacts_changed=False,
        )
        write(args.output / "effective-config.json", config)
        report["config_sha256"] = sha(args.output / "effective-config.json")
        replay = None
        reference_pins = {}
        if args.smoke_reference:
            smoke = json.loads((args.smoke_reference / "result.json").read_text())
            validate_native_smoke_report(smoke, plans)
            for row in smoke["rows"]:
                if sha(row["path"]) != row["sha256"]:
                    raise ValueError("smoke proposal archive changed")
                with np.load(row["path"], allow_pickle=False) as saved:
                    validate_native_proposal_arrays({k: saved[k] for k in PROPOSAL_KEYS})
                prefix = row["cache_prefix"]
                if sha(prefix["path"]) != prefix["sha256"]:
                    raise ValueError("smoke cache prefix changed")
                with np.load(prefix["path"], allow_pickle=False) as saved:
                    require_arrays(
                        dict(raw=saved["cached_raw"], score=saved["cached_score"]),
                        dict(raw=saved["fresh_raw"], score=saved["fresh_score"]),
                    )
            if (
                smoke["status"] != "PASS_NATIVE_CORPUS_CACHE_SMOKE"
                or sha(args.smoke_reference / "effective-config.json") != report["config_sha256"]
                or smoke["config_sha256"] != report["config_sha256"]
                or not smoke["initial_parent_byte_exact"]
                or not smoke["cached_fresh_raw_exact"]
            ):
                raise ValueError("same-code/config native cache smoke required")
            report["smoke_result_sha256"] = sha(args.smoke_reference / "result.json")
            reference_pins.update(
                {
                    str(args.smoke_reference / "result.json"): report["smoke_result_sha256"],
                    str(args.smoke_reference / "effective-config.json"): report["config_sha256"],
                }
            )
            for row in smoke["rows"]:
                reference_pins.update(
                    {
                        row["path"]: row["sha256"],
                        row["cache_prefix"]["path"]: row["cache_prefix"]["sha256"],
                    }
                )
        elif not args.smoke:
            raise ValueError("separate passing cache smoke mandatory")
        if args.replay:
            replay = json.loads((args.replay / "result.json").read_text())
            require_reference_config(
                args.replay / "effective-config.json",
                replay["config_sha256"],
                report["config_sha256"],
            )
            if (
                replay["status"] != "PASS_NATIVE_EIGHT_CLIP_PROPOSAL_CORPUS"
                or replay["config_sha256"] != report["config_sha256"]
            ):
                raise ValueError("complete identical replay reference required")
            expected_rows = [(p["video_id"], t) for p in plans for t in p["frame_indices"]]
            if [(r["video_id"], r["frame_index"]) for r in replay["rows"]] != expected_rows:
                raise ValueError("complete ordered replay corpus reference required")
            report["replay_result_sha256"] = sha(args.replay / "result.json")
            reference_pins.update(
                {
                    str(args.replay / "result.json"): report["replay_result_sha256"],
                    str(args.replay / "effective-config.json"): report["config_sha256"],
                }
            )
            for row in replay["rows"]:
                reference_pins[row["path"]] = row["sha256"]
            verify_sources(reference_pins)
        helpers = legacy_functions()
        before = helpers["state_sha"](model)
        if sum(p.numel() for p in model.parameters()) != 38962498:
            raise ValueError("native model parameter inventory changed")
        report.update(
            status="RUNNING",
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            model_state_before=before,
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            parameters=38962498,
        )
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [a.clone() for a in torch.cuda.get_rng_state_all()]
        counter = [0]

        def count_encoder(_module, _inputs):
            counter[0] += 1

        hook = model.vision_encoder.register_forward_pre_hook(count_encoder)

        def cap():
            if (
                time.perf_counter() - start > (180 if args.smoke else 7200)
                or torch.cuda.max_memory_allocated() > 6 * 1024**3
                or shutil.disk_usage(args.output).free < 8 * 1024**3
            ):
                raise RuntimeError("fixed time/memory/disk safety admission exceeded")

        def new_session(image):
            s = processor.init_video_session(
                video=image[None],
                inference_device="cuda",
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.float32,
            )
            if s.cache.get_vision_features(0) is not None or s.num_frames != 1:
                raise ValueError("independent source-frame session required")
            return s

        executed = plans[:1] if args.smoke or replay else plans
        for plan in executed:
            video = plan["video_id"]
            selected = (
                plan["frame_indices"][:2]
                if args.smoke
                else ([plan["frame_indices"][1]] if replay else plan["frame_indices"])
            )
            wanted = set(selected)
            clip_dir = args.output / video
            clip_dir.mkdir()
            with av.open(plan["media_path"]) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                for index, frame in enumerate(container.decode(stream)):
                    if index >= plan["frames"]:
                        break
                    if index not in wanted:
                        continue
                    cap()
                    tick = time.perf_counter()
                    image = frame.to_ndarray(format="rgb24")
                    rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
                    parent = reused.get((video, index))
                    cache_prefix = None
                    dest = clip_dir / f"{index:04d}-proposals.npz"
                    if parent and not args.smoke:
                        if parent.get("rgb_sha256", rgb_sha) != rgb_sha:
                            raise ValueError("reused source-frame RGB mismatch")
                        with np.load(parent["path"], allow_pickle=False) as data:
                            values = {k: data[k].copy() for k in PROPOSAL_KEYS}
                        forward_seconds, encoded = 0.0, 0
                    else:
                        h, w = image.shape[:2]
                        points = np.array(
                            [
                                [(x + 0.5) * w / 32, (y + 0.5) * h / 32]
                                for y in range(32)
                                for x in range(32)
                            ],
                            np.float32,
                        )
                        s = new_session(image)
                        pixels = s.processed_frames[0].clone()
                        masks, scores, prefixes = [], [], {}
                        forward_seconds = 0.0
                        encode_before = counter[0]
                        with torch.inference_mode():
                            for begin in range(0, 1024, 64):
                                cap()
                                s.reset_tracking_data()
                                if (
                                    s.obj_ids
                                    or s.output_dict_per_obj
                                    or s.obj_with_new_inputs
                                    or (begin and s.cache.get_vision_features(0) is None)
                                ):
                                    raise ValueError("tracking reset/cache boundary violated")
                                processor.add_inputs_to_inference_session(
                                    s,
                                    frame_idx=0,
                                    obj_ids=list(range(1, 65)),
                                    input_points=points[begin : begin + 64][None, :, None].tolist(),
                                    input_labels=[[[1]] * 64],
                                )
                                torch.cuda.synchronize()
                                forward_start = time.perf_counter()
                                o = model(s, frame_idx=0, run_mem_encoder=False)
                                torch.cuda.synchronize()
                                forward_seconds += time.perf_counter() - forward_start
                                raw = o.pred_masks.detach().cpu().numpy()[:, 0].copy()
                                score = (
                                    o.object_score_logits.detach().cpu().numpy().reshape(-1).copy()
                                )
                                if (
                                    raw.shape != (64, 256, 256)
                                    or raw.dtype != np.float32
                                    or score.shape != (64,)
                                    or score.dtype != np.float32
                                    or not np.isfinite(raw).all()
                                    or not np.isfinite(score).all()
                                ):
                                    raise ValueError("full finite native64 outputs required")
                                masks.append(raw > 0)
                                scores.append(score)
                                if args.smoke and begin == 64:
                                    fresh = new_session(image)
                                    processor.add_inputs_to_inference_session(
                                        fresh,
                                        frame_idx=0,
                                        obj_ids=list(range(1, 65)),
                                        input_points=points[begin : begin + 64][
                                            None, :, None
                                        ].tolist(),
                                        input_labels=[[[1]] * 64],
                                    )
                                    other = model(fresh, frame_idx=0, run_mem_encoder=False)
                                    pair = dict(
                                        raw=other.pred_masks.detach().cpu().numpy()[:, 0],
                                        score=other.object_score_logits.detach()
                                        .cpu()
                                        .numpy()
                                        .reshape(-1),
                                    )
                                    require_arrays(pair, dict(raw=raw, score=score))
                                    prefixes = dict(
                                        cached_raw=raw,
                                        cached_score=score,
                                        fresh_raw=pair["raw"],
                                        fresh_score=pair["score"],
                                    )
                                    del other, fresh
                                del o
                        encoded = counter[0] - encode_before
                        if encoded != (2 if args.smoke else 1):
                            raise ValueError(
                                "one cached source encoder call plus smoke replay required"
                            )
                        if not torch.equal(pixels, s.processed_frames[0]):
                            raise ValueError("native processor pixels changed")
                        masks = np.concatenate(masks)
                        scores = np.concatenate(scores)
                        boxes = helpers["boxes_from_masks"](masks)
                        values = dict(
                            masks_packed=np.packbits(masks.reshape(1024, -1), axis=1),
                            scores=scores,
                            boxes=boxes,
                            points=points,
                            nms_indices=np.asarray(helpers["nms"](boxes, scores), np.int64),
                        )
                        if prefixes:
                            prefix_path = clip_dir / f"{index:04d}-cache-prefix.npz"
                            np.savez_compressed(prefix_path, **prefixes)
                            with np.load(prefix_path, allow_pickle=False) as saved:
                                require_arrays(saved, prefixes)
                            cache_prefix = dict(path=str(prefix_path), sha256=sha(prefix_path))
                            report["cached_fresh_raw_exact"] = True
                        del s, pixels, masks, raw, score, scores, boxes, prefixes
                    validate_native_proposal_arrays(values)
                    if parent and args.smoke:
                        with np.load(parent["path"], allow_pickle=False) as data:
                            require_arrays(values, {k: data[k] for k in PROPOSAL_KEYS})
                        report["initial_parent_byte_exact"] = True
                    np.savez_compressed(dest, **values)
                    with np.load(dest, allow_pickle=False) as data:
                        require_arrays(data, values)
                    if replay:
                        row = next(
                            r
                            for r in replay["rows"]
                            if r["video_id"] == video and r["frame_index"] == index
                        )
                        if sha(row["path"]) != row["sha256"] or row["rgb_sha256"] != rgb_sha:
                            raise ValueError("replay output/source hash changed")
                        with np.load(row["path"], allow_pickle=False) as data:
                            require_arrays(data, values)
                        report["fresh_replay_bitwise"] = True
                    report["rows"].append(
                        dict(
                            video_id=video,
                            frame_index=index,
                            rgb_sha256=rgb_sha,
                            input_shape=list(image.shape),
                            path=str(dest),
                            sha256=sha(dest),
                            file_bytes=dest.stat().st_size,
                            raw_proposals=1024,
                            nms_count=len(values["nms_indices"]),
                            encoder_calls=encoded,
                            cached_forward_seconds=forward_seconds,
                            parent=parent if parent and not args.smoke else None,
                            readback_bitwise=True,
                            cache_prefix=cache_prefix,
                            frame_seconds=time.perf_counter() - tick,
                        )
                    )
                    report.update(
                        seconds=time.perf_counter() - start, encoder_calls_total=counter[0]
                    )
                    write(args.output / "progress.json", report)
                    print(
                        video,
                        index,
                        "SAVED_ALL1024",
                        len(values["nms_indices"]),
                        "REUSE" if parent and not args.smoke else "NATIVE",
                        flush=True,
                    )
                    del image, values
                    torch.cuda.empty_cache()
            observed = [r["frame_index"] for r in report["rows"] if r["video_id"] == video]
            if observed != selected:
                raise ValueError("incomplete source time coverage")
        hook.remove()
        if helpers["state_sha"](model) != before or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise ValueError("frozen model state or flags changed")
        if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
            not torch.equal(a, b)
            for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        ):
            raise ValueError("native inference consumed CPU/CUDA RNG")
        verify_sources(hashes)
        verify_sources(reference_pins)
        if sha(args.output / "effective-config.json") != report["config_sha256"]:
            raise ValueError("saved current config changed")
        cap()
        report.update(
            status="PASS_NATIVE_CORPUS_CACHE_SMOKE"
            if args.smoke
            else "PASS_NATIVE_CORPUS_FRESH_FRAME_REPLAY"
            if replay
            else "PASS_NATIVE_EIGHT_CLIP_PROPOSAL_CORPUS",
            model_state_after=before,
            frozen_state_unchanged=True,
            cpu_cuda_rng_unchanged=True,
            sources_unchanged=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.perf_counter() - start, ended_at=datetime.now(UTC).isoformat())
        write(args.output / "result.json", report)


if __name__ == "__main__":
    main()
