"""Fresh native/model reload of the fixed probe; no learning or quality claim."""

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np

from scripts.probe_robomnist_container_features import (
    GROUP,
    RAW,
    bind_sources,
    encode_frontend,
    failure_peak,
    guard,
    length_controls,
    require_frozen_scope,
    require_products,
    sha,
    write,
)
from scripts.robomnist_container_features import digest_arrays, same_arrays
from scripts.robomnist_input_arrays import read_video

MAIN_SHA = "12a20f436a487e4e21a91956d58bf1fb9bd231ef1494eb9abc8ca7346425ddba"


def read_report(path, expected):
    body = path.read_bytes()
    actual = hashlib.sha256(body).hexdigest()
    if actual != expected:
        raise ValueError("fixed main report hash required before parsing")
    return json.loads(body), actual


def recheck(pins):
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("source/model/library/product/report changed during replay")


def consumer_snapshot(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def require_cohort(required, recorded):
    if any(recorded.get(p) != h for p, h in required.items()):
        raise ValueError("mandatory exact source cohort omitted or replaced")


def require_scope(r):
    if (
        r["status"] != "PASS_NATIVE_CONTAINER_AND_FROZEN_IMAGE_FEATURE_MECHANICS_ONLY"
        or r["group"] != GROUP
        or r["unique_source_recordings"] != 1
        or r["decoded_frames"] != 1350
        or r["projected_image_observations"] != 18
        or r["encoded_image_instances"] != 54
        or r["encoder_forward_calls"] != 18
        or r["training_corpus_admission"] != "NOT_ADMITTED"
        or r["common_clock"] != "UNVERIFIED"
        or any(
            r[k] is not False
            for k in [
                "model_training",
                "teacher_training",
                "new_source_payloads",
                "position_gt_opened",
                "sealed_audit_opened",
                "quality_measured",
                "native_temporal_model",
                "gpu_used",
            ]
        )
        or not 0 < r["elapsed_seconds"] <= 600
        or not 0 < r["peak_cpu_bytes"] <= 3 * 2**30
    ):
        raise ValueError("complete bounded native/frozen mechanics-only scope required")


def verify_atom(source, derivative, row):
    if (
        source.stat().st_size != row["source_bytes"]
        or derivative.stat().st_size != row["derived_bytes"]
        or row["derived_bytes"] != row["source_bytes"] + 8 + 2**20
        or sha(source) != row["source_sha256"]
        or sha(derivative) != row["derived_sha256"]
    ):
        raise ValueError("actual source/derivative sizes and hashes differ")
    with source.open("rb") as a, derivative.open("rb") as b:
        while piece := a.read(2**20):
            if b.read(len(piece)) != piece:
                raise ValueError("derivative source prefix differs")
        if b.read(8) != struct.pack(">I4s", 2**20 + 8, b"free") or b.read() != bytes(2**20):
            raise ValueError("exact fixed free header/zero payload required")


def audit(root, tick):
    consumer = consumer_snapshot([Path(__file__), Path("tests/test_robomnist_container_replay.py")])

    def check():
        return guard(root, tick)

    check()
    r, report_sha = read_report(root / "result.json", MAIN_SHA)
    require_scope(r)
    require_frozen_scope(r["frontend"])
    c = json.loads((root / "config.json").read_bytes())
    if (
        c["source_commit"] != r["source_commit"]
        or c["cpu_profile"] != "float32/eager/2threads/seed17/fullyfrozen"
        or c["free_payload_bytes"] != 2**20
        or c["samples"] != [0, 224, 449]
        or c["lenses"] != ["left", "right"]
        or c["model_revision"] != "07a4b0bc751cb64fecd2b661c048c1dd98d69444"
        or c["length_reference_logp_absolute_tolerance"] != 1e-12
        or sha(root / "config.json") != r["config_sha256"]
    ):
        raise ValueError("fixed original effective config required")
    source = bind_sources()
    require_cohort(source, c["source_hashes"])
    require_cohort(source, r["validated_hashes"])
    pins = dict(r["validated_hashes"])
    pins.update(consumer)
    pins.update({str(root / name): h for name, h in r["artifacts"].items()})
    pins[str(root / "result.json")] = report_sha
    recheck(pins)
    with np.load(root / "arrays.npz", allow_pickle=False) as bank:
        saved = {k: bank[k] for k in bank.files}
    require_products(root, r["artifacts"], c, saved)
    original, derived = {}, {}
    original_metadata = json.loads((RAW / "result.json").read_bytes())["video"]
    if [row["receiver"] for row in r["camera_rows"]] != [1, 2, 3]:
        raise ValueError("all three cameras in fixed order required")
    for row in r["camera_rows"]:
        receiver = row["receiver"]
        actual_source, actual_derivative = (
            RAW / f"camera-{receiver}.mp4",
            root / f"camera-{receiver}-free.mp4",
        )
        if (
            Path(row["source_path"]).resolve() != actual_source.resolve()
            or Path(row["path"]).resolve() != actual_derivative.resolve()
        ):
            raise ValueError("actual source and derivative paths must match fixed roots")
        verify_atom(actual_source, actual_derivative, row)
        for path, store in [(actual_source, original), (actual_derivative, derived)]:
            print("CPU fresh decode", path.name, flush=True)
            values, metadata = read_video(path, check)
            if dict(receiver=receiver, **metadata) != original_metadata[receiver - 1]:
                raise ValueError("fresh full450frame RGB/PTS/metadata differs")
            store.update({f"camera_{receiver}_{k}": v for k, v in values.items()})
    same_arrays(original, derived)
    if (
        digest_arrays(original) != r["raw_arrays_digest"]
        or digest_arrays(derived) != r["derived_arrays_digest"]
    ):
        raise ValueError("all native retained RGB/PTS/index/times arrays differ")
    frontend, arrays = encode_frontend(original, derived, source, check)
    require_cohort(source, r["validated_hashes"])
    for k, expected in r["frontend"].items():
        if k not in {"inference_seconds", "load_and_forward_seconds"} and frontend[k] != expected:
            raise ValueError(f"fresh frontend state/environment/scope differs: {k}")
    controls, logs = length_controls(r["camera_rows"])
    if controls != r["nuisance_length_controls"]:
        raise ValueError("fresh fixed length-control posteriors differ")
    arrays.update(logs)
    same_arrays(saved, arrays)
    if digest_arrays(arrays) != r["arrays_digest"]:
        raise ValueError("all recomputed tensors/features/posteriors must match")
    require_products(root, r["artifacts"], c, arrays)
    recheck(pins)
    resources = check()
    return dict(
        status="PASS_FRESH_NATIVE_DECODER_AND_FROZEN_MODEL_RELOAD_ONLY",
        main_report_sha256=report_sha,
        main_source_commit=r["source_commit"],
        config_sha256=r["config_sha256"],
        source_recordings=1,
        decoded_frames=2700,
        arrays_exact=len(arrays),
        projected_image_observations=18,
        encoded_image_instances=54,
        encoder_forward_calls=18,
        all_features_and_pixels_byte_equal=True,
        all_length_control_logp_byte_equal=True,
        full_model_state_unchanged=True,
        frontend=frontend,
        nuisance_length_controls=controls,
        validated_hashes=pins,
        consumer_sha256=consumer[str(Path(__file__).resolve())],
        independent_model_reload=True,
        shared_frontend_implementation=True,
        training_corpus_admission="NOT_ADMITTED",
        common_clock="UNVERIFIED",
        quality_measured=False,
        native_temporal_model=False,
        teacher_training=False,
        gpu_used=False,
        new_source_payloads=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        **resources,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.main.resolve()):
        raise ValueError("new output outside protected producer root required")
    tick = time.perf_counter()
    try:
        r = audit(args.main, tick)
        r.update(guard(args.main, tick))
    except Exception as exc:
        r = dict(
            status="FAIL_INCOMPLETE_NATIVE_FEATURE_REPLAY",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - tick,
            peak_cpu_bytes=failure_peak(),
            quality_measured=False,
        )
    r["pid"] = os.getpid()
    write(args.output, r)
    print(r["status"], r.get("error", ""), r["elapsed_seconds"], flush=True)
    if r["status"] != "PASS_FRESH_NATIVE_DECODER_AND_FROZEN_MODEL_RELOAD_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
