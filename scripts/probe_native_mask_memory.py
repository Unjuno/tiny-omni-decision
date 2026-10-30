"""Frozen native video-memory execution; annotations never read by this process."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import av
import numpy as np
import torch
import transformers
from transformers import Sam2VideoModel, Sam2VideoProcessor

from scripts.cover_point_metrics import state_hash
from scripts.extract_spatial_context_corpus import sha, write
from scripts.train_spatial_trajectory_readout import process_peak_bytes
from tiny_omni_decision.mask_memory_probe import pack_frame, seed_masks, verify_times, video_control

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
MODEL = Path("C:/CodexArtifacts/pretrained-reuse-r1/sam2.1-hiera-tiny-audit-01/model")
IDS = [
    "video_10103",
    "video_10186",
    "video_10212",
    "video_10387",
    "video_1087",
    "video_10886",
    "video_10894",
    "video_10903",
]
COUNTS = [55, 186, 43, 168, 67, 64, 73, 40]
ARMS = ("actual", "initial_repeat")
HISTORICAL_REFERENCE = Path("docs/pretrained_reuse/R1_SAM2_TINY_20261014.json")
HISTORICAL_SHA = "905166e514fbac11b2489cc36ee448443e4f660a7ea1a90049097ec1928a9e3b"
PINS = {
    "perception-sam2-auto-init-02/result.json": "75665a07c986dd0a97bff4854912b7a265e91baf49c0c5d45637886e880fa3d1",  # noqa: E501
    "perception-occlusion-train-media-01/manifest.json": "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3",  # noqa: E501
    "spatial-context-corpus-01/result.json": "a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da",  # noqa: E501
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def limits(start, output, seconds):
    if time.perf_counter() - start > seconds:
        raise TimeoutError("fixed body limit; incomplete")
    if torch.cuda.max_memory_allocated() > 6 * 2**30 or process_peak_bytes() > 4 * 2**30:
        raise MemoryError("fixed6GiB GPU/4GiB host cap exceeded")
    if shutil.disk_usage(output).free < 8 * 2**30:
        raise OSError("8GiB disk reserve exhausted")


def verify_inherited(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("inherited protected model/settings/source changed")
    return pins


def load_contract():
    pins = {str(ROOT / n): p for n, p in PINS.items()}
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError("protected parent changed")
    auto = json.loads((ROOT / "perception-sam2-auto-init-02/result.json").read_text())
    if (
        auto["status"] != "PASS_AUTO_INIT_EXECUTION_ONLY"
        or auto["training"]
        or auto["annotation_data_loaded"]
        or auto["parameters"] != 38962498
    ):
        raise ValueError("native automatic-mask source not admitted")
    media = {
        r["video_id"]: r
        for r in json.loads(
            (ROOT / "perception-occlusion-train-media-01/manifest.json").read_text()
        )["clips"]
    }
    frames = {
        r["video_id"]: r
        for r in json.loads((ROOT / "spatial-context-corpus-01/result.json").read_text())[
            "features"
        ]
        if r["split"] == "train"
    }
    rows = {r["video_id"]: r for r in auto["rows"]}
    plans = []
    for vid, count in zip(IDS, COUNTS, strict=True):
        m, f, a = media[vid], frames[vid], rows[vid]
        verify_times(np.array(f["frame_indices"], np.int64), f["original_frames"], m["cut_frame"])
        path = ROOT / "perception-sam2-auto-init-02" / f"{vid}-proposals.npz"
        if (
            sha(path) != a["proposal_sha256"]
            or a["nms_kept"] != count
            or a["input_sha256"] != m["sha256"]
            or f["media_sha256"] != m["sha256"]
            or sha(m["path"]) != m["sha256"]
        ):
            raise ValueError("source mask/media/time binding changed")
        with np.load(path, allow_pickle=False) as b:
            if b["masks_packed"].shape != (1024, 8192):
                raise ValueError("all native proposals required")
            masks = seed_masks(b["masks_packed"], b["nms_indices"])
            if len(masks) != count:
                raise ValueError("complete original NMS cohort required")
            indices = b["nms_indices"].tolist()
        pins[str(path)] = a["proposal_sha256"]
        pins[m["path"]] = m["sha256"]
        plans.append(
            dict(
                video_id=vid,
                media_path=m["path"],
                media_sha256=m["sha256"],
                cut=m["cut_frame"],
                source_frames=f["original_frames"],
                selected_rgb_sha256=f["selected_rgb_sha256"],
                frame_indices=f["frame_indices"],
                proposal_path=str(path),
                proposal_sha256=a["proposal_sha256"],
                seed_indices=indices,
            )
        )
    verify_inherited({str(HISTORICAL_REFERENCE): HISTORICAL_SHA})
    historical = json.loads(HISTORICAL_REFERENCE.read_text())["admission"]
    inherited = {str(MODEL / n): v["sha256"] for n, v in historical["files"].items()}
    inherited.update(historical["installed_source_sha256"])
    pins.update(verify_inherited(inherited))
    pins[str(HISTORICAL_REFERENCE.resolve())] = HISTORICAL_SHA
    if {p.name for p in MODEL.iterdir() if p.is_file()} != set(historical["files"]):
        raise ValueError("inherited model directory file cohort changed")
    if (
        pins[str(MODEL / "model.safetensors")]
        != "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"
    ):
        raise ValueError("released checkpoint changed")
    processing = Sam2VideoProcessor.from_pretrained(MODEL, local_files_only=True)
    paths = [
        Path(__file__),
        Path("src/tiny_omni_decision/mask_memory_probe.py"),
        Path("scripts/cover_point_metrics.py"),
        Path("scripts/extract_spatial_context_corpus.py"),
        Path("scripts/train_spatial_trajectory_readout.py"),
        Path("docs/pretrained_reuse/R2_NATIVE_MASK_MEMORY_PROTOCOL_20261014.md"),
        Path(inspect.getfile(Sam2VideoModel)),
        Path(inspect.getfile(Sam2VideoProcessor)),
        Path(inspect.getfile(type(processing.video_processor))),
        Path(inspect.getfile(type(processing.image_processor))),
    ]
    pins.update({str(p.resolve()): sha(p) for p in paths})
    return dict(
        plans=plans,
        source_hashes=pins,
        frames=32,
        raster=256,
        seeds=696,
        seed=17,
        model_id="facebook/sam2.1-hiera-tiny",
        revision="de431c4043854a71d8101e17995dfe596bf101a5",
        parameters=38962498,
        trainable_parameters=0,
        arms=list(ARMS),
        backend="sdpa",
        weights_outputs_dtype="float32",
        native_mask_memory_dtype="bfloat16",
        tf32=False,
        cache_reference=1,
        cache_admitted=32,
        state_device="cuda",
        video_device="cpu",
        max_main_seconds=7200,
        smoke_fresh_seconds=600,
        peak_gpu_bytes=6 * 2**30,
        peak_host_bytes=4 * 2**30,
        disk_reserve_bytes=8 * 2**30,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
        av=av.__version__,
        python=platform.python_version(),
        cuda=torch.version.cuda,
    )


def decode(plan):
    wanted = set(plan["frame_indices"])
    frames, verification = [], []
    with av.open(plan["media_path"]) as c:
        c.streams.video[0].codec_context.thread_count = 1
        for i, f in enumerate(c.decode(video=0)):
            if i in wanted:
                verification.append(f.reformat(width=256, height=256, format="rgb24").to_ndarray())
                frames.append(f.to_ndarray(format="rgb24"))
            if i == plan["frame_indices"][-1]:
                break
    if len(frames) != 32:
        raise ValueError("incomplete fixed observation sequence")
    rgb = np.stack(frames)
    if hashlib.sha256(np.stack(verification).tobytes()).hexdigest() != plan["selected_rgb_sha256"]:
        raise ValueError("decoded source pixels differ from admitted source times")
    return rgb


def empty_tracking(session):
    session.reset_tracking_data()
    fields = [
        "_obj_id_to_idx",
        "_obj_idx_to_id",
        "obj_ids",
        "point_inputs_per_obj",
        "mask_inputs_per_obj",
        "output_dict_per_obj",
        "frames_tracked_per_obj",
        "obj_with_new_inputs",
    ]
    if any(getattr(session, k) for k in fields):
        raise ValueError("object memory not cleared")


def propagate(model, processor, session, mask, start, output, cap, retain_raw):
    empty_tracking(session)
    processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=[1], input_masks=[mask])
    packed, boxes, scores, rawm, raws = [], [], [], [], []
    h = hashlib.sha256()
    tick = time.perf_counter()
    for expected, v in enumerate(
        model.propagate_in_video_iterator(
            session, start_frame_idx=0, max_frame_num_to_track=32, show_progress_bar=False
        )
    ):
        limits(start, output, cap)
        if int(v.frame_idx) != expected or expected >= 32:
            raise ValueError("native propagation frame routing changed")
        a = v.pred_masks.detach().cpu().numpy()
        s = v.object_score_logits.detach().cpu().numpy()
        p, b = pack_frame(a, s)
        h.update(a.tobytes())
        h.update(s.tobytes())
        packed.append(p)
        boxes.append(b)
        scores.append(float(s.reshape(-1)[0]))
        if retain_raw:
            rawm.append(a.copy())
            raws.append(s.copy())
    torch.cuda.synchronize()
    if len(packed) != 32:
        raise ValueError("incomplete native memory stream")
    values = dict(
        masks_packed=np.stack(packed),
        boxes=np.stack(boxes),
        scores=np.array(scores, np.float32),
        empty_masks=np.array([not p.any() for p in packed]),
        raw_sha256=h.hexdigest(),
    )
    if retain_raw:
        values.update(raw_masks=np.stack(rawm), raw_scores=np.stack(raws))
    return values, time.perf_counter() - tick


def execute_arm(
    model, processor, rgb, masks, arm, cache_size, start, output, cap, retain_raw, reference=None
):
    inputs = video_control(rgb, arm)
    s = processor.init_video_session(
        video=inputs,
        inference_device="cuda",
        inference_state_device="cuda",
        processing_device="cpu",
        video_storage_device="cpu",
        max_vision_features_cache_size=cache_size,
        dtype=torch.float32,
    )
    del inputs
    if s.num_frames != 32:
        raise ValueError("processor changed fixed frame count")
    calls = []
    hook = model.vision_encoder.register_forward_hook(lambda *args: calls.append(1))
    values = []
    timers = []
    try:
        for i, mask in enumerate(masks):
            write(
                output / "current-seed.json",
                dict(
                    pid=os.getpid(),
                    arm=arm,
                    cache_size=cache_size,
                    seed_ordinal=i + 1,
                    total_seeds=len(masks),
                    completed_seeds=len(values),
                ),
            )
            v, seconds = propagate(model, processor, s, mask, start, output, cap, retain_raw)
            values.append(v)
            timers.append(seconds)
            if reference is not None and i < 2 and v["raw_sha256"] != reference[i]:
                raise ValueError("native raw prefix differs from admitted original smoke")
            print(arm, "seed", i + 1, "/", len(masks), "vision_calls", len(calls), flush=True)
        if len(calls) != (32 if cache_size == 32 else 32 * len(masks)):
            raise ValueError("actual vision cache reuse/call count differs")
        # Native implementation's internal encoded memory dtype is recorded, not replaced.
        memory_dtypes = sorted(
            {
                str(v["maskmem_features"].dtype)
                for d in s.output_dict_per_obj.values()
                for group in d.values()
                for v in group.values()
                if v.get("maskmem_features") is not None
            }
        )
        if memory_dtypes != ["torch.bfloat16"]:
            raise ValueError("native memory dtype changed")
        return values, dict(
            seconds_by_seed=timers,
            vision_encoder_calls=len(calls),
            cache_frames=len(s.cache._vision_features),
            memory_dtypes=memory_dtypes,
            single_object_per_forward=True,
            tracking_reset_checked=True,
        )
    except Exception:
        if values:
            np.savez_compressed(
                output / f"incomplete-{arm}-cache{cache_size}.npz",
                **{k: np.stack([v[k] for v in values]) for k in values[0]},
            )
        raise
    finally:
        hook.remove()
        del s
        gc.collect()
        torch.cuda.empty_cache()


def save_arm(path, values, seed_indices, frames):
    arrays = {k: np.stack([v[k] for v in values]) for k in values[0]}
    arrays.update(
        seed_indices=np.array(seed_indices, np.int64), frame_indices=np.array(frames, np.int64)
    )
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as r:
        if set(r.files) != set(arrays):
            raise ValueError("archive keys changed")
        for k, a in arrays.items():
            if a.shape != r[k].shape or a.dtype != r[k].dtype or a.tobytes() != r[k].tobytes():
                raise ValueError("producer save/readback changed")
    return dict(
        path=str(path),
        sha256=sha(path),
        file_bytes=path.stat().st_size,
        shapes={k: list(a.shape) for k, a in arrays.items()},
        dtypes={k: str(a.dtype) for k, a in arrays.items()},
    )


def evidence(folder, result, mode):
    cfg = folder / "config.json"
    c = json.loads(cfg.read_text())
    if (
        sha(cfg) != result["config_sha256"]
        or c["mode"] != mode
        or c["contract"] != result["contract"]
        or result["contract_sha256"] != digest(result["contract"])
    ):
        raise ValueError("actual reference config not bound")
    pins = {
        str(cfg): result["config_sha256"],
        str(folder / "result.json"): sha(folder / "result.json"),
    }
    for row in result["rows"]:
        p = row["artifact"]["path"]
        pin = row["artifact"]["sha256"]
        if sha(p) != pin:
            raise ValueError("actual reference artifact changed")
        pins[p] = pin
    return pins


def admit_smoke(folder, contract):
    prior = json.loads((folder / "result.json").read_text())
    project = prior.get("projected_main_seconds", float("nan"))
    variants = {(r["arm"], r["cache_size"]) for r in prior["rows"]}
    if (
        prior["status"] != "PASS_NATIVE_MEMORY_SMOKE_ONLY"
        or prior["contract"] != contract
        or not np.isfinite(project)
        or not 0 < project < 7200
        or len(prior["rows"]) != 3
        or variants != {("actual", 1), ("actual", 32), ("initial_repeat", 32)}
        or not prior.get("parameters_frozen")
        or not prior.get("rng_unchanged")
        or prior.get("state_before") != prior.get("state_after")
    ):
        raise ValueError("complete finite same-contract cache/scale admission required")
    pins = evidence(folder, prior, "smoke")
    banks = {}
    for row in prior["rows"]:
        expected = 32 if row["cache_size"] == 32 else 64
        if (
            row["video_id"] != IDS[0]
            or row["seeds"] != 2
            or row["native_object_frames"] != 64
            or row["vision_encoder_calls"] != expected
            or not row["tracking_reset_checked"]
        ):
            raise ValueError("actual smoke cohort/cache trace incomplete")
        with np.load(row["artifact"]["path"], allow_pickle=False) as a:
            if (
                a["raw_masks"].shape != (2, 32, 1, 1, 256, 256)
                or a["raw_masks"].dtype != np.float32
                or a["raw_scores"].shape != (2, 32, 1, 1)
                or a["raw_scores"].dtype != np.float32
            ):
                raise ValueError("complete raw prefix arrays required")
            np.testing.assert_array_equal(
                a["seed_indices"], contract["plans"][0]["seed_indices"][:2]
            )
            np.testing.assert_array_equal(a["frame_indices"], contract["plans"][0]["frame_indices"])
            banks[(row["arm"], row["cache_size"])] = {
                k: a[k].copy() for k in ("raw_masks", "raw_scores")
            }
    for k in ("raw_masks", "raw_scores"):
        if banks[("actual", 1)][k].tobytes() != banks[("actual", 32)][k].tobytes():
            raise ValueError("actual default/cache32 raw predictions differ")
    return prior, pins


def admit_main(folder, contract):
    prior = json.loads((folder / "result.json").read_text())
    expected = [(v, a) for v in IDS for a in ARMS]
    if (
        prior["status"] != "PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY"
        or prior["contract"] != contract
        or [(r["video_id"], r["arm"]) for r in prior["rows"]] != expected
        or not prior.get("parameters_frozen")
        or not prior.get("rng_unchanged")
        or prior.get("state_before") != prior.get("state_after")
    ):
        raise ValueError("complete ordered same-contract main required")
    pins = evidence(folder, prior, "main")
    for i, row in enumerate(prior["rows"]):
        plan = contract["plans"][i // 2]
        count = COUNTS[i // 2]
        if (
            row["cache_size"] != 32
            or row["seeds"] != count
            or row["native_object_frames"] != count * 32
        ):
            raise ValueError("complete native seed/frame cohort required")
        specs = dict(
            masks_packed=((count, 32, 8192), "uint8"),
            boxes=((count, 32, 4), "float64"),
            scores=((count, 32), "float32"),
            empty_masks=((count, 32), "bool"),
            raw_sha256=((count,), "<U64"),
            seed_indices=((count,), "int64"),
            frame_indices=((32,), "int64"),
        )
        with np.load(row["artifact"]["path"], allow_pickle=False) as bank:
            if set(bank.files) != set(specs):
                raise ValueError("complete main archive schema required")
            for k, (shape, dtype) in specs.items():
                if bank[k].shape != shape or bank[k].dtype != np.dtype(dtype):
                    raise ValueError("main archive shape/dtype changed")
            np.testing.assert_array_equal(bank["seed_indices"], plan["seed_indices"])
            np.testing.assert_array_equal(bank["frame_indices"], plan["frame_indices"])
            if bank["raw_sha256"][:2].tolist() != row["raw_prefix_sha256"]:
                raise ValueError("main prefix digests not bound to archive")
    return prior, pins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--smoke-from", type=Path)
    modes.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        rows=[],
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
    )
    cap = 7200 if args.smoke_from else 600
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            raise MemoryError("12GiB free GPU required")
        limits(start, args.output, cap)
        contract = load_contract()
        result.update(contract=contract, contract_sha256=digest(contract))
        write(
            args.output / "config.json",
            dict(
                contract=contract,
                mode="smoke" if args.smoke else "replay" if args.replay_from else "main",
                command=subprocess.list2cmdline(
                    ["python", "-m", "scripts.probe_native_mask_memory"] + os.sys.argv[1:]
                ),
            ),
        )
        result["config_sha256"] = sha(args.output / "config.json")
        references = {}
        prefix = {}
        if args.smoke_from:
            prior, references = admit_smoke(args.smoke_from, contract)
            if shutil.disk_usage(args.output).free < 8 * 2**30 + 450 * 2**20:
                raise OSError("full packed-output budget plus8GiB reserve required")
            prefix = {
                r["arm"]: r["raw_prefix_sha256"] for r in prior["rows"] if r["cache_size"] == 32
            }
        if args.replay_from:
            prior, references = admit_main(args.replay_from, contract)
            prefix = {
                r["arm"]: r["raw_prefix_sha256"] for r in prior["rows"] if r["video_id"] == IDS[0]
            }
        model, info = Sam2VideoModel.from_pretrained(
            MODEL,
            local_files_only=True,
            output_loading_info=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        if any(
            info.get(k)
            for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
        ):
            raise ValueError("strict released weights required")
        model = model.eval().requires_grad_(False).cuda()
        if sum(p.numel() for p in model.parameters()) != 38962498:
            raise ValueError("registered parameter count changed")
        processor = Sam2VideoProcessor.from_pretrained(MODEL, local_files_only=True)
        before = state_hash(model)
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all()
        numpy_rng = np.random.get_state()
        result.update(state_before=before, gpu=torch.cuda.get_device_name())
        for plan in contract["plans"][:1] if args.smoke or args.replay_from else contract["plans"]:
            write(args.output / "current-input-plan.json", plan)
            rgb = decode(plan)
            with np.load(plan["proposal_path"], allow_pickle=False) as bank:
                indices = bank["nms_indices"].copy()
                masks = seed_masks(bank["masks_packed"], indices)
            if args.smoke or args.replay_from:
                masks, indices = masks[:2], indices[:2]
            variants = (
                [("actual", 1), ("actual", 32), ("initial_repeat", 32)]
                if args.smoke
                else [(a, 32) for a in ARMS]
            )
            raw_reference = None
            projected = 0
            for arm, cache in variants:
                limits(start, args.output, cap)
                values, metrics = execute_arm(
                    model,
                    processor,
                    rgb,
                    masks,
                    arm,
                    cache,
                    start,
                    args.output,
                    cap,
                    args.smoke or bool(args.replay_from),
                    prefix.get(arm) if plan["video_id"] == IDS[0] else None,
                )
                path = args.output / f"{plan['video_id']}-{arm}-cache{cache}.npz"
                artifact = save_arm(path, values, indices, plan["frame_indices"])
                row = dict(
                    video_id=plan["video_id"],
                    arm=arm,
                    cache_size=cache,
                    seeds=len(indices),
                    native_object_frames=len(indices) * 32,
                    artifact=artifact,
                    **metrics,
                    raw_prefix_sha256=[v["raw_sha256"] for v in values[:2]],
                )
                result["rows"].append(row)
                result.update(seconds=time.perf_counter() - start)
                write(args.output / "progress.json", result)
                if args.smoke and cache == 1:
                    raw_reference = values
                elif args.smoke and arm == "actual":
                    for a, b in zip(raw_reference, values, strict=True):
                        for k in ["raw_masks", "raw_scores"]:
                            if (
                                a[k].shape != b[k].shape
                                or a[k].dtype != b[k].dtype
                                or a[k].tobytes() != b[k].tobytes()
                            ):
                                raise ValueError(
                                    "public cache32 differs; saved failure not admitted"
                                )
                    row["cache_reference_shape_dtype_bytes_identical"] = True
                if args.smoke and cache == 32:
                    projected += (
                        metrics["seconds_by_seed"][-1] * 696 * 1.25
                        + metrics["seconds_by_seed"][0] * 8
                    )
                if args.replay_from:
                    original = next(
                        r
                        for r in prior["rows"]
                        if r["video_id"] == plan["video_id"] and r["arm"] == arm
                    )
                    with np.load(original["artifact"]["path"], allow_pickle=False) as saved:
                        for k in ["masks_packed", "boxes", "scores", "empty_masks", "raw_sha256"]:
                            a = np.stack([v[k] for v in values])
                            b = saved[k][:2]
                            if (
                                a.shape != b.shape
                                or a.dtype != b.dtype
                                or a.tobytes() != b.tobytes()
                            ):
                                raise ValueError("fresh prefix array bytes differ")
            del rgb, masks
            gc.collect()
        if args.smoke:
            result["projected_main_seconds"] = projected
            result["status"] = "PASS_NATIVE_MEMORY_SMOKE_ONLY"
        elif args.replay_from:
            result["status"] = "PASS_FRESH_FIXED_NATIVE_MEMORY_PREFIX_ONLY"
        else:
            if len(result["rows"]) != 16 or sum(r["seeds"] for r in result["rows"]) != 1392:
                raise ValueError("incomplete full fixed696seed/two-arm cohort")
            result["status"] = "PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY"
        if state_hash(model) != before or any(
            p.requires_grad or p.grad is not None for p in model.parameters()
        ):
            raise ValueError("frozen model state/flags/gradients changed")
        if not torch.equal(rng, torch.get_rng_state()) or any(
            not torch.equal(a, b)
            for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        ):
            raise ValueError("native inference consumed Torch RNG")
        n = np.random.get_state()
        if n[0] != numpy_rng[0] or not np.array_equal(n[1], numpy_rng[1]) or n[2:] != numpy_rng[2:]:
            raise ValueError("NumPy RNG changed")
        result["reference_hashes"] = references
        for path, pin in {**contract["source_hashes"], **references}.items():
            limits(start, args.output, cap)
            if sha(path) != pin:
                raise ValueError("protected source/reference changed")
        if sha(args.output / "config.json") != result["config_sha256"]:
            raise ValueError("effective config changed")
        after = state_hash(model)
        limits(start, args.output, cap)
        result.update(
            state_after=after,
            rng_unchanged=True,
            parameters_frozen=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_working_set_bytes=process_peak_bytes(),
        )
    except Exception:
        result.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - start, ended_utc=datetime.now(UTC).isoformat())
        write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
