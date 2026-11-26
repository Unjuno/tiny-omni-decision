"""Fresh CPU raw-input decoding and complete saved-array replay, never a model eval."""

import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np

from scripts.fetch_robomnist_input_smoke import (
    GROUP,
    PARENT,
    PARENT_SHA,
    PROTOCOL,
    fixed_plan,
    local_name,
    peak_bytes,
)
from scripts.robomnist_input_arrays import load_csi, read_video, read_wav


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_run(r):
    if (
        r["status"] != "PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY"
        or len(r["raw_files"]) != 7
        or any("pos.json" in v["source_member"] for v in r["raw_files"])
        or any(
            r[k] is not False
            for k in [
                "models_loaded",
                "training",
                "position_gt_opened",
                "sealed_audit_opened",
                "gpu_used",
                "quality_measured",
            ]
        )
        or r["config"]["group"]["group_id"] != GROUP
        or not r["source_pins_unchanged"]
        or not 0 <= r["elapsed_seconds"] <= 900
        or not 0 < r["actual_transfer_bytes"] <= r["reserved_transfer_bytes"] <= 64 * 2**20
        or not 0 < r["peak_cpu_bytes"] <= 2 * 2**30
        or not r["sync_assessment"].startswith("UNVERIFIED_COMMON_CLOCK:")
        or sorted(r["array_bank"]["alignment_only_keys"])
        != [f"module_{i}_timestamps_ns" for i in [1, 2, 3]]
    ):
        raise ValueError("complete fixed mechanics/clock-scope/resource admission required")


def compare_bank(expected, actual):
    if not expected or set(expected) != set(actual):
        raise ValueError("every native array key required")
    for key, a in expected.items():
        b = actual[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise ValueError("raw native decode and saved bank shape/dtype/bytes differ")


def require_pins(required, recorded):
    if not required or any(recorded.get(p) != h for p, h in required.items()):
        raise ValueError("mandatory parent/source/helper/decoder-library pin missing or changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new CPU replay output required")
    tick = time.perf_counter()

    def check():
        if time.perf_counter() - tick > 600 or peak_bytes() > 2 * 2**30:
            raise RuntimeError("fresh CPU replay time/process cap exhausted")

    r = json.loads((args.main / "result.json").read_bytes())
    c = json.loads((args.main / "config.json").read_bytes())
    require_run(r)
    if (
        c != r["config"]
        or sha(args.main / "config.json") != r["config_sha256"]
        or c["reserved_transfer_cap_bytes"] != 64 * 2**20
        or c["body_limit_seconds"] != 900
        or c["output_cap_bytes"] != 384 * 2**20
        or c["disk_reserve_bytes"] != 2**30
        or c["cpu_peak_cap_bytes"] != 2 * 2**30
        or c["video_sample_rule"] != "0/floor((n-1)/2)/(n-1)"
        or not c["native_csi_clock_role"].startswith("alignment audit only")
        or sha(PARENT) != PARENT_SHA
    ):
        raise ValueError("exact config/hash/source/clock roles required")
    parent = json.loads(PARENT.read_bytes())
    import av
    import av.container.input
    import av.video.frame

    required = {**parent["validated_hashes"], str(PARENT): PARENT_SHA}
    for p in [
        Path("scripts/fetch_robomnist_input_smoke.py").resolve(),
        Path("scripts/robomnist_input_arrays.py"),
        PROTOCOL,
        Path("docs/superpowers/plans/2026-10-14-robomnist-input-smoke.md"),
        Path("tests/test_robomnist_input_arrays.py"),
        Path(np.__file__),
        Path(av.__file__),
        Path(av.container.input.__file__),
        Path(av.video.frame.__file__),
    ]:
        required[str(p)] = sha(p)
    required[
        "C:\\CodexArtifacts\\pretrained-reuse-r2\\robomnist-input-source-code-01\\examples__wifi_csi_read.ipynb"
    ] = "f78b577856f015e2d972078f815a1c6d99f70b8cb9f5b9773bf91e75a06001b0"
    require_pins(required, c["source_hashes"])
    directory = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-01/directory-00.json"
    )
    b = json.loads(directory.read_bytes())
    g = fixed_plan(b)
    if g != c["group"] or b["source_archive"] != c["archive"]:
        raise ValueError("fixed physical source plan changed")
    if sorted(v["source_member"] for v in r["raw_files"]) != sorted(g["input_members"]):
        raise ValueError("seven actual native inputs/source identities required")
    pins = {
        **c["source_hashes"],
        **{
            str(args.main / n): sha(args.main / n)
            for n in ["result.json", "config.json", "execution-source.py"]
        },
    }
    if sha(args.main / "execution-source.py") != sha("scripts/fetch_robomnist_input_smoke.py"):
        raise ValueError("executed producer source snapshot differs")
    metadata = {i["name"]: i for i in b["members"]}
    for row in r["raw_files"]:
        p = args.main / local_name(row["source_member"])
        if (
            Path(row["path"]).resolve() != p.resolve()
            or sha(p) != row["sha256"]
            or p.stat().st_size != row["file_bytes"]
        ):
            raise ValueError("actual raw-file checksum/size/path differs")
        crc = 0
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                crc = zlib.crc32(chunk, crc)
        original = metadata[row["source_member"]]
        if (
            f"{crc:08x}" != row["crc32"]
            or row["crc32"] != original["crc32"]
            or row["file_bytes"] != original["size"]
        ):
            raise ValueError("actual complete payload CRC/size differs from admitted directory")
        pins[str(p)] = row["sha256"]
    arrays, videos, audio = {}, [], []
    for receiver in [1, 2, 3]:
        check()
        print("fresh CPU camera", receiver, flush=True)
        a, meta = read_video(args.main / f"camera-{receiver}.mp4", check)
        arrays.update({f"camera_{receiver}_{k}": v for k, v in a.items()})
        videos.append(dict(receiver=receiver, **meta))
        a, meta = read_wav(args.main / f"microphone-{receiver}.wav")
        arrays[f"microphone_{receiver}_pcm"] = a
        audio.append(dict(receiver=receiver, **meta))
    a, csi = load_csi(args.main / "csi.json")
    arrays.update(a)
    if videos != r["video"] or audio != r["audio"] or csi != r["csi"]:
        raise ValueError("all decoded native stream metadata/timing differs")
    path = args.main / "native-inputs.npz"
    if (
        Path(r["array_bank"]["path"]).resolve() != path.resolve()
        or sha(path) != r["array_bank"]["sha256"]
    ):
        raise ValueError("actual tensor bank file/hash binding required")
    with np.load(path, allow_pickle=False) as saved:
        actual = {key: saved[key] for key in saved.files}
    if {k: list(v.shape) for k, v in actual.items()} != r["array_bank"]["shapes"] or {
        k: str(v.dtype) for k, v in actual.items()
    } != r["array_bank"]["dtypes"]:
        raise ValueError("actual tensor inventories differ")
    compare_bank(arrays, actual)
    pins[str(path)] = r["array_bank"]["sha256"]
    for p, pin in pins.items():
        if sha(p) != pin:
            raise ValueError("source/library/raw/bank pin changed")
    files_bytes = sum(p.stat().st_size for p in args.main.rglob("*") if p.is_file())
    if files_bytes > 384 * 2**20:
        raise ValueError("actual input output cap exceeded")
    check()
    result = dict(
        status="PASS_FRESH_NATIVE_INPUT_ARRAY_REPLAY_ONLY",
        main_source_commit=c["source_commit"],
        fixed_physical_group=GROUP,
        raw_files=r["raw_files"],
        video=videos,
        audio=audio,
        csi=csi,
        native_arrays=len(arrays),
        arrays_byte_equal=True,
        actual_output_bytes=files_bytes,
        main_seconds=r["elapsed_seconds"],
        main_peak_cpu_bytes=r["peak_cpu_bytes"],
        source_transfer_bytes=r["actual_transfer_bytes"],
        reserved_transfer_bytes=r["reserved_transfer_bytes"],
        replay_seconds=time.perf_counter() - tick,
        replay_peak_cpu_bytes=peak_bytes(),
        sync_assessment=r["sync_assessment"],
        absolute_csi_clock_role=c["native_csi_clock_role"],
        training_corpus_admission="NOT_ADMITTED: clock/session/content/split/quality unverified",
        models_loaded=False,
        training=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        gpu_used=False,
        full_archive_md5_verified=False,
        cloud_cost=0,
        validated_hashes=pins,
        auditor_sha256=sha(__file__),
    )
    with args.output.open("x", encoding="utf8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(result["status"], "arrays", len(arrays), "seconds", result["replay_seconds"], flush=True)


if __name__ == "__main__":
    main()
