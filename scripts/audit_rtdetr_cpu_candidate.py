"""Pinned compact semantic-detector CPU mechanics; no quality labels or GPU."""

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml

REVISION = "5650961749fa93567c0d46fc7f43ea4f9e914107"
MODEL_ID = "PekingU/rtdetr_v2_r18vd"
WEIGHT_SHA = "d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115"
RELEASE_BLOBS = {
    "README.md": "37d358fb4e1d91881c4081fc8df0b75d0c2cd99e",
    "config.json": "6f984c30c65857419deeeebfdf7dc4e7d7bce6f8",
    "preprocessor_config.json": "fcdff16b42e5ebc51d28f59184490c6eff91a88b",
}
ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r1/rtdetr-v2-r18-audit-01")
MEDIA = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-train-media-01/manifest.json"
)
MEDIA_SHA = "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3"
SOURCE_SHA = "e9914a811679e99f26246ab0c1f89c582914e37433189bf925f06f5793ca38ed"
PROTOCOL = Path("docs/pretrained_reuse/R1_RTDETR_V2_R18_CPU_PROTOCOL_20261014.md")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def read_model_card(path):
    text = Path(path).read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("explicit model-card frontmatter required")
    metadata = yaml.safe_load(parts[1])
    if not isinstance(metadata, dict) or metadata.get("license") != "apache-2.0":
        raise ValueError("explicit Apache2 released-model license required")
    return metadata


def verify_git_blob(path, expected):
    data = Path(path).read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != expected:
        raise ValueError("immutable released config/card/processor Git blob mismatch")
    return hashlib.sha256(data).hexdigest()


def inspect_checkpoint(path, expected):
    if sha(path) != expected:
        raise ValueError("published weight SHA mismatch")
    total = Path(path).stat().st_size
    with Path(path).open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        if not 0 < length < min(total - 8, 8 * 2**20):
            raise ValueError("invalid safetensors header extent")
        header = json.loads(f.read(length))
    widths = {
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
        "F64": 8,
    }
    spans = []
    dtype_counts = {}
    inventory = {}
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        shape = tensor["shape"]
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid tensor shape")
        elements = math.prod(shape)
        dtype = tensor["dtype"]
        if dtype not in widths:
            raise ValueError("unsupported explicit checkpoint dtype")
        first, last = tensor["data_offsets"]
        if first < 0 or last - first != elements * widths[dtype]:
            raise ValueError("tensor data span differs from shape/dtype")
        spans.append((first, last))
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + elements
        inventory[key] = dict(shape=shape, dtype=dtype, elements=elements)
    end = 0
    for first, last in sorted(spans):
        if first != end:
            raise ValueError("overlap or gap in safetensors payload")
        end = last
    if end != total - 8 - length:
        raise ValueError("incomplete or extra tensor payload")
    return dict(
        tensor_elements=sum(dtype_counts.values()),
        tensor_data_bytes=end,
        dtype_elements=dtype_counts,
        inventory=inventory,
        header_bytes=length,
    )


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tick = time.perf_counter()
    result = dict(
        status="PREFLIGHT",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        gpu_seconds=0,
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        decision_quality_evaluated=False,
        product_adopted=False,
        cloud_cost=0,
    )
    try:
        import inspect

        import av
        import numpy as np
        import torch
        import transformers
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        from scripts.train_spatial_trajectory_readout import process_peak_bytes

        if torch.cuda.is_initialized():
            raise ValueError("CPU-only process must not initialize CUDA")
        torch.set_num_threads(4)
        torch.manual_seed(17)
        torch.use_deterministic_algorithms(True)
        model_dir = ROOT / "model"
        files = {
            str(model_dir / n): verify_git_blob(model_dir / n, h) for n, h in RELEASE_BLOBS.items()
        }
        files[str(model_dir / "model.safetensors")] = sha(model_dir / "model.safetensors")
        release_manifest = ROOT / "model-source-manifest.json"
        release = json.loads(release_manifest.read_bytes())
        if (
            release["revision"] != REVISION
            or release["id"] != MODEL_ID
            or release["license"] != "apache-2.0"
        ):
            raise ValueError("immutable release metadata differs")
        for name, pin in RELEASE_BLOBS.items():
            if release["files"][name]["git_blob_sha1"] != pin:
                raise ValueError("release metadata file pins differ")
        if release["files"]["model.safetensors"]["lfs_sha256"] != WEIGHT_SHA:
            raise ValueError("release metadata weight pin differs")
        files[str(release_manifest)] = sha(release_manifest)
        card = read_model_card(model_dir / "README.md")
        if (
            files[str(model_dir / "model.safetensors")] != WEIGHT_SHA
            or card["license"] != "apache-2.0"
        ):
            raise ValueError("pinned released weight/Apache2 card required")
        checkpoint = inspect_checkpoint(model_dir / "model.safetensors", WEIGHT_SHA)
        if sha(MEDIA) != MEDIA_SHA or shutil.disk_usage(args.output).free < 8 * 2**30:
            raise ValueError("source manifest or8GiB reserve failed")
        source = next(
            r for r in json.loads(MEDIA.read_bytes())["clips"] if r["video_id"] == "video_10103"
        )
        if source["sha256"] != SOURCE_SHA or sha(source["path"]) != SOURCE_SHA:
            raise ValueError("fixed approved TRAIN input media changed")
        files[str(MEDIA)] = MEDIA_SHA
        files[source["path"]] = SOURCE_SHA
        for p in [Path(__file__), PROTOCOL]:
            files[str(p.resolve())] = sha(p)
        (args.output / "execution-source.py").write_bytes(Path(__file__).read_bytes())
        result.update(
            model_id=MODEL_ID,
            revision=REVISION,
            weight_sha256=WEIGHT_SHA,
            source_base_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            source_committed=False,
            execution_source_sha256=sha(__file__),
            checkpoint=checkpoint,
            python=os.sys.version,
            torch=torch.__version__,
            transformers=transformers.__version__,
            av=av.__version__,
            released_weight_license="apache-2.0",
            project_weight_policy="ALLOW",
            original_training_media_rights_independently_verified=False,
            released_git_blob_pins=RELEASE_BLOBS,
        )
        model, info = RTDetrV2ForObjectDetection.from_pretrained(
            model_dir,
            local_files_only=True,
            output_loading_info=True,
            dtype=torch.float32,
            disable_custom_kernels=True,
        )
        keys = ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
        if any(info.get(k) for k in keys):
            raise ValueError("strict released checkpoint load failed")
        model = model.eval().requires_grad_(False).cpu()
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("frozen CPU FP32 parameters required")
        parameters = sum(p.numel() for p in model.parameters())
        if not 0 < parameters <= 200_000_000:
            raise ValueError("candidate parameter bound failed")
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        for cls in [type(model), type(processor)]:
            p = Path(inspect.getfile(cls))
            files[str(p)] = sha(p)
        with av.open(source["path"]) as container:
            container.streams.video[0].codec_context.thread_count = 1
            rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
        inputs = processor(images=rgb, return_tensors="pt")
        before = state_hash(model)
        rng = torch.get_rng_state().clone()
        arrays = {f"input_{k}": v.cpu().numpy() for k, v in inputs.items()}
        times = []
        raw = []
        for _ in range(2):
            s = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs)
            times.append(time.perf_counter() - s)
            a = dict(logits=outputs.logits.cpu().numpy(), boxes=outputs.pred_boxes.cpu().numpy())
            if (
                a["logits"].shape != (1, 300, 80)
                or a["boxes"].shape != (1, 300, 4)
                or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in a.values())
            ):
                raise ValueError("all native queries/classes/boxes must be retained finite FP32")
            raw.append(a)
        for k in raw[0]:
            if raw[0][k].tobytes() != raw[1][k].tobytes():
                raise ValueError("CPU repeat is not byte-identical")
        arrays.update(raw[0])
        np.savez_compressed(args.output / "all-query-outputs.npz", **arrays)
        with np.load(args.output / "all-query-outputs.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                saved[k].shape != v.shape
                or saved[k].dtype != v.dtype
                or saved[k].tobytes() != v.tobytes()
                for k, v in arrays.items()
            ):
                raise ValueError("saved all-query outputs differ")
        after = state_hash(model)
        if (
            after != before
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
            or not torch.equal(rng, torch.get_rng_state())
            or torch.cuda.is_initialized()
        ):
            raise ValueError("CPU state/freeze/RNG/CUDA guard failed")
        for p, h in files.items():
            if sha(p) != h:
                raise ValueError("model/source/code changed during CPU audit")
        if (
            time.perf_counter() - tick > 120
            or process_peak_bytes() > 3 * 2**30
            or shutil.disk_usage(args.output).free < 8 * 2**30
        ):
            raise ValueError("bounded CPU resource guard failed")
        result.update(
            status="PASS_R1_CPU_MECHANICS_ONLY",
            parameters=parameters,
            registered_buffer_elements=sum(b.numel() for b in model.buffers()),
            parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            processor=json.loads(processor.to_json_string()),
            effective_disable_custom_kernels=True,
            input=dict(
                video_id="video_10103",
                split="known TRAIN",
                frame_index=0,
                media_sha256=SOURCE_SHA,
                shape=list(rgb.shape),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            ),
            output_shapes={k: list(v.shape) for k, v in arrays.items()},
            queries_retained=300,
            classes_retained=80,
            outputs_sha256=sha(args.output / "all-query-outputs.npz"),
            cpu_forward_seconds=times,
            state_before=before,
            state_after=after,
            state_scope="parameters and persistent buffers",
            parameters_frozen=True,
            cpu_rng_unchanged=True,
            cuda_initialized=False,
            repeat_shape_dtype_bytes_equal=True,
            save_reload_equal=True,
            peak_working_set_bytes=process_peak_bytes(),
            protected_hashes=files,
        )
    except Exception:
        result.update(status="FAIL_R1_CPU_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        result.update(seconds=time.perf_counter() - tick, ended_utc=datetime.now(UTC).isoformat())
        write(args.output / "result.json", result)


if __name__ == "__main__":
    main()
