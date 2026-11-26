"""Independent NumPy audit of every saved smoke logit; no model or annotation load."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_archive(a, seeds, frames):
    specs = dict(
        raw_masks=((2, 32, 1, 1, 256, 256), "float32"),
        raw_scores=((2, 32, 1, 1), "float32"),
        masks_packed=((2, 32, 8192), "uint8"),
        boxes=((2, 32, 4), "float64"),
        scores=((2, 32), "float32"),
        empty_masks=((2, 32), "bool"),
        raw_sha256=((2,), "<U64"),
        seed_indices=((2,), "int64"),
        frame_indices=((32,), "int64"),
    )
    if set(a) != set(specs):
        raise ValueError("complete smoke schema required")
    for k, (shape, dtype) in specs.items():
        if a[k].shape != shape or a[k].dtype != np.dtype(dtype):
            raise ValueError("smoke shape/dtype differs")
    if (
        a["seed_indices"].tolist() != seeds
        or a["frame_indices"].tolist() != frames
        or not np.isfinite(a["raw_masks"]).all()
        or not np.isfinite(a["raw_scores"]).all()
    ):
        raise ValueError("source routing/numerics differ")
    if a["scores"].tobytes() != a["raw_scores"].reshape(2, 32).tobytes():
        raise ValueError("presence scores differ")
    empty = 0
    for i in range(2):
        h = hashlib.sha256()
        for t in range(32):
            raw, score = a["raw_masks"][i, t], a["raw_scores"][i, t]
            h.update(raw.tobytes())
            h.update(score.tobytes())
            binary = raw.reshape(256, 256) > 0
            if (
                np.packbits(binary.reshape(-1), bitorder="big").tobytes()
                != a["masks_packed"][i, t].tobytes()
            ):
                raise ValueError("packed mask differs from native raw logit threshold0")
            yy, xx = np.where(binary)
            box = np.zeros(4, np.float64)
            if len(xx):
                box[:] = [min(xx) / 256, min(yy) / 256, (max(xx) + 1) / 256, (max(yy) + 1) / 256]
            else:
                empty += 1
            if box.tobytes() != a["boxes"][i, t].tobytes() or bool(a["empty_masks"][i, t]) != (
                len(xx) == 0
            ):
                raise ValueError("mask geometry/empty accounting differs")
        if h.hexdigest() != a["raw_sha256"][i]:
            raise ValueError("raw stream digest differs")
    return dict(seeds=2, frames=64, empty=empty)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    tick = time.perf_counter()
    r = json.loads((args.smoke / "result.json").read_text())
    c = json.loads((args.smoke / "config.json").read_text())
    contract = r["contract"]
    if (
        r["status"] != "PASS_NATIVE_MEMORY_SMOKE_ONLY"
        or c["mode"] != "smoke"
        or c["contract"] != contract
        or sha(args.smoke / "config.json") != r["config_sha256"]
        or hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != r["contract_sha256"]
        or not r["parameters_frozen"]
        or not r["rng_unchanged"]
        or r["state_before"] != r["state_after"]
    ):
        raise ValueError("actual config/freeze binding differs")
    if [(v["arm"], v["cache_size"]) for v in r["rows"]] != [
        ("actual", 1),
        ("actual", 32),
        ("initial_repeat", 32),
    ]:
        raise ValueError("complete ordered smoke variants required")
    pins = {
        str(args.smoke / "result.json"): sha(args.smoke / "result.json"),
        str(args.smoke / "config.json"): r["config_sha256"],
        **contract["source_hashes"],
    }
    rows, banks = [], {}
    for row in r["rows"]:
        if (
            row["video_id"] != contract["plans"][0]["video_id"]
            or row["seeds"] != 2
            or row["native_object_frames"] != 64
            or not row["tracking_reset_checked"]
            or row["memory_dtypes"] != ["torch.bfloat16"]
            or row["vision_encoder_calls"] != (64 if row["cache_size"] == 1 else 32)
        ):
            raise ValueError("native trace/cohort differs")
        p = row["artifact"]["path"]
        pins[p] = row["artifact"]["sha256"]
        if sha(p) != pins[p]:
            raise ValueError("artifact hash differs")
        with np.load(p, allow_pickle=False) as b:
            a = {k: b[k] for k in b.files}
        counts = verify_raw_archive(
            a, contract["plans"][0]["seed_indices"][:2], contract["plans"][0]["frame_indices"]
        )
        if row["raw_prefix_sha256"] != a["raw_sha256"].tolist():
            raise ValueError("reported prefix differs")
        banks[(row["arm"], row["cache_size"])] = a
        rows.append(dict(arm=row["arm"], cache_size=row["cache_size"], **counts))
    for k in banks[("actual", 1)]:
        if banks[("actual", 1)][k].tobytes() != banks[("actual", 32)][k].tobytes():
            raise ValueError("actual reference/cache32 archive bytes differ")
    projected = sum(
        row["seconds_by_seed"][-1] * 696 * 1.25 + row["seconds_by_seed"][0] * 8
        for row in r["rows"]
        if row["cache_size"] == 32
    )
    if projected != r["projected_main_seconds"]:
        raise ValueError("runtime projection differs")
    for p, pin in pins.items():
        if sha(p) != pin:
            raise ValueError("protected source/report changed")
    result = dict(
        status="PASS_INDEPENDENT_RAW_SMOKE_AUDIT_ONLY",
        rows=rows,
        predictions=192,
        parameters=r["contract"]["parameters"],
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        gpu_seconds=0,
        cache_shape_dtype_bytes_equal=True,
        projected_main_seconds=projected,
        scale_gate_passed=projected < contract["max_main_seconds"],
        protected_hashes=pins,
        seconds=time.perf_counter() - tick,
    )
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(result["status"], "scale gate", result["scale_gate_passed"], flush=True)


if __name__ == "__main__":
    main()
