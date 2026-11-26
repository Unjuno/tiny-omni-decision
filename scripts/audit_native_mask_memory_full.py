"""Complete independent packed-mask/source/replay audit, without annotations or GPU."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from scripts.audit_native_mask_memory_smoke import verify_raw_archive

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
ARMS = ["actual", "initial_repeat"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def require_reference_bindings(expected, references):
    if not expected or any(references.get(p) != h for p, h in expected.items()):
        raise ValueError("mandatory main/fresh reference bindings incomplete or changed")
    return expected


def verify_packed_archive(a, seeds, frames):
    n, t = len(seeds), len(frames)
    specs = dict(
        masks_packed=((n, t, 8192), "uint8"),
        boxes=((n, t, 4), "float64"),
        scores=((n, t), "float32"),
        empty_masks=((n, t), "bool"),
        raw_sha256=((n,), "<U64"),
        seed_indices=((n,), "int64"),
        frame_indices=((t,), "int64"),
    )
    if set(a) != set(specs):
        raise ValueError("complete packed archive schema required")
    for k, (shape, dtype) in specs.items():
        if a[k].shape != shape or a[k].dtype != np.dtype(dtype):
            raise ValueError("packed archive shape/dtype differs")
    if (
        a["seed_indices"].tolist() != seeds
        or a["frame_indices"].tolist() != frames
        or not np.isfinite(a["scores"]).all()
        or not np.isfinite(a["boxes"]).all()
        or not all(re.fullmatch("[0-9a-f]{64}", str(h)) for h in a["raw_sha256"])
    ):
        raise ValueError("packed identity/numerics/raw digest format differs")
    empty = 0
    for i in range(n):
        for j in range(t):
            mask = np.unpackbits(a["masks_packed"][i, j], bitorder="big").reshape(256, 256)
            yy, xx = np.where(mask)
            box = np.zeros(4, np.float64)
            if len(xx):
                box[:] = [min(xx) / 256, min(yy) / 256, (max(xx) + 1) / 256, (max(yy) + 1) / 256]
            else:
                empty += 1
            if box.tobytes() != a["boxes"][i, j].tobytes() or bool(a["empty_masks"][i, j]) != (
                len(xx) == 0
            ):
                raise ValueError("complete independently derived mask geometry/empty count differs")
    return dict(native_object_frames=n * t, empty_masks=empty)


def report(folder, mode, status):
    p = folder / "result.json"
    r = json.loads(p.read_bytes())
    cfg = folder / "config.json"
    c = json.loads(cfg.read_bytes())
    if (
        r["status"] != status
        or c["mode"] != mode
        or c["contract"] != r["contract"]
        or sha(cfg) != r["config_sha256"]
        or digest(r["contract"]) != r["contract_sha256"]
        or not r["parameters_frozen"]
        or not r["rng_unchanged"]
        or r["state_before"] != r["state_after"]
        or r["training"]
        or r["annotations_loaded"]
        or r["sealed_audit_opened"]
        or not np.isfinite(r["seconds"])
        or r["seconds"] > r["resource_body_limit_seconds"]
        or r["peak_allocated_bytes"] > 6 * 2**30
        or r["peak_working_set_bytes"] > 4 * 2**30
    ):
        raise ValueError("terminal exact effective config/freeze/resource binding required")
    return r, {str(p): sha(p), str(cfg): r["config_sha256"]}


def load_row(row, pins):
    p = row["artifact"]["path"]
    pin = row["artifact"]["sha256"]
    if sha(p) != pin:
        raise ValueError("actual archive hash differs")
    pins[p] = pin
    with np.load(p, allow_pickle=False) as b:
        a = {k: b[k] for k in b.files}
    if (
        {k: list(v.shape) for k, v in a.items()} != row["artifact"]["shapes"]
        or {k: str(v.dtype) for k, v in a.items()} != row["artifact"]["dtypes"]
        or Path(p).stat().st_size != row["artifact"]["file_bytes"]
    ):
        raise ValueError("actual archive inventory differs")
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new audit output required")
    tick = time.perf_counter()
    r, pins = report(args.main, "main", "PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY")
    fresh, fpins = report(args.fresh, "replay", "PASS_FRESH_FIXED_NATIVE_MEMORY_PREFIX_ONLY")
    main_evidence = {
        **pins,
        **{row["artifact"]["path"]: row["artifact"]["sha256"] for row in r["rows"]},
    }
    require_reference_bindings(main_evidence, fresh["reference_hashes"])
    fresh_evidence = {
        **fpins,
        **{row["artifact"]["path"]: row["artifact"]["sha256"] for row in fresh["rows"]},
    }
    pins.update(fpins)
    c = r["contract"]
    if (
        fresh["contract"] != c
        or [(p["video_id"], len(p["seed_indices"])) for p in c["plans"]]
        != list(zip(IDS, COUNTS, strict=True))
        or [(v["video_id"], v["arm"]) for v in r["rows"]] != [(v, a) for v in IDS for a in ARMS]
        or [(v["video_id"], v["arm"]) for v in fresh["rows"]] != [(IDS[0], a) for a in ARMS]
        or c["max_main_seconds"] != 14400
        or c["resource_amendment"]["original_scale_gate_passed"]
        or not c["resource_amendment"]["amended_scale_gate_passed"]
        or c["resource_amendment"]["numerical_conditions_changed"]
    ):
        raise ValueError("complete frozen cohort/resource amendment/replay scope required")
    pins.update(c["source_hashes"])
    pins.update(r["reference_hashes"])
    pins.update(fresh["reference_hashes"])
    original_path = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/native-mask-memory-smoke-02/result.json"
    )
    original = json.loads(original_path.read_bytes())
    old = original["contract"]
    if r["state_before"] != original["state_before"] or fresh["state_before"] != r["state_before"]:
        raise ValueError("cold model state differs from original admitted checkpoint state")
    for k, v in old.items():
        if k not in {"source_commit", "source_hashes", "max_main_seconds"} and c.get(k) != v:
            raise ValueError("original numerical/input condition changed")
    if any(c["source_hashes"].get(p) != h for p, h in old["source_hashes"].items()):
        raise ValueError("inherited pins changed")
    rows = []
    for i, row in enumerate(r["rows"]):
        plan = c["plans"][i // 2]
        if (
            row["cache_size"] != 32
            or row["seeds"] != COUNTS[i // 2]
            or row["native_object_frames"] != COUNTS[i // 2] * 32
            or row["vision_encoder_calls"] != 32
            or row["cache_frames"] != 32
            or not row["tracking_reset_checked"]
            or row["memory_dtypes"] != ["torch.bfloat16"]
        ):
            raise ValueError("complete native cache/reset/memory/cohort trace required")
        a = load_row(row, pins)
        counts = verify_packed_archive(a, plan["seed_indices"], plan["frame_indices"])
        if a["raw_sha256"][:2].tolist() != row["raw_prefix_sha256"]:
            raise ValueError("main prefix not bound to actual archive")
        if i < 2:
            frow = fresh["rows"][i]
            fa = load_row(frow, pins)
            verify_raw_archive(fa, plan["seed_indices"][:2], plan["frame_indices"])
            for k in a:
                if fa[k].tobytes() != a[k][:2].tobytes() and k not in {"frame_indices"}:
                    raise ValueError("fresh prefix bytes differ")
            if fa["frame_indices"].tobytes() != a["frame_indices"].tobytes():
                raise ValueError("fresh time routing differs")
            prefix = next(
                v["raw_prefix_sha256"]
                for v in original["rows"]
                if v["arm"] == row["arm"] and v["cache_size"] == 32
            )
            if row["raw_prefix_sha256"] != prefix or frow["raw_prefix_sha256"] != prefix:
                raise ValueError("original smoke prefix does not match main/fresh")
        rows.append(dict(video_id=row["video_id"], arm=row["arm"], **counts))
        if time.perf_counter() - tick > 600:
            raise TimeoutError("CPU independent audit cap600s")
    for p, h in pins.items():
        if sha(p) != h:
            raise ValueError("source/reference changed")
    if time.perf_counter() - tick > 600:
        raise TimeoutError("CPU independent audit cap600s")
    out = dict(
        status="PASS_COMPLETE_INDEPENDENT_NATIVE_MASK_AUDIT_ONLY",
        rows=rows,
        native_object_frames=sum(v["native_object_frames"] for v in rows),
        fresh_numerical_scope=dict(videos=1, seeds=2, arms=2, frames=32),
        main_full_raw_logits_reconstructed=False,
        main_raw_digest_format_and_prefix_checked=True,
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        gpu_seconds=0,
        protected_hashes=pins,
        validated_main_evidence=main_evidence,
        validated_fresh_evidence=dict(folder=str(args.fresh), hashes=fresh_evidence),
        auditor_sha256=sha(__file__),
        seconds=time.perf_counter() - tick,
    )
    args.output.open("x", encoding="utf-8").write(json.dumps(out, indent=2) + "\n")
    print(out["status"], out["native_object_frames"], flush=True)


if __name__ == "__main__":
    main()
