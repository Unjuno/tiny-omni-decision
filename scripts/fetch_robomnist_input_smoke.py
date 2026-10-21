"""Extract/decode exactly seven pinned input streams, with no GT/model/GPU."""

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.audit_robomnist_all_directories import Budget
from scripts.audit_robomnist_source import group_members, sha
from scripts.fetch_starss23_smoke import HttpRangeReader
from scripts.robomnist_input_arrays import load_csi, read_video, read_wav

PARENT = Path("C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-replay-02.json")
PARENT_SHA = "b34bc54f57d3c434947d673ed32cd9a9c6e4f82bfc10f9b02213b397a76a29a4"
GROUP = "Ar1_Vhigh_Ac0_Rp01_UNC062"
PROTOCOL = Path("docs/pretrained_reuse/R2_ROBOMNIST_INPUT_PROTOCOL_20261014.md")


def fixed_plan(bank):
    if bank["source_archive"]["name"] != "00_Ar1_Vhigh_Ac0.zip" or not bank["groups"]:
        raise ValueError("fixed archive00 source required")
    g = bank["groups"][0]
    if (
        g["group_id"] != GROUP
        or len(g["input_members"]) != 7
        or g["supervision_only_member"] in g["input_members"]
    ):
        raise ValueError("fixed first seven input streams required; positionGT forbidden")
    reconstructed = group_members(
        g["input_members"] + [g["supervision_only_member"]], expected_folder="00_Ar1_Vhigh_Ac0"
    )
    if reconstructed != [g]:
        raise ValueError("all fixed receiver streams and source grouping must match")
    infos = {i["name"]: i for i in bank["members"]}
    if any(n not in infos or infos[n]["size"] > 16 * 2**20 for n in g["input_members"]):
        raise ValueError("bounded metadata for every fixed input required")
    return g


def peak_bytes():
    if os.name != "nt":
        raise RuntimeError("this committed smoke's resource monitor requires Windows")
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (key, ctypes.c_size_t)
            for key in [
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            ]
        ]

    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    api = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    if not api(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError("process peak counter unavailable")
    return int(value.PeakWorkingSetSize)


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


class InputReader(HttpRangeReader):
    def __init__(self, archive, shared, check):
        super().__init__(archive["download_url"], archive["size"], budget=64 * 2**20, retries=0)
        self.shared, self.check = shared, check

    def read(self, size=-1):
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        self.check()
        self.shared.charge(count + 1 if count else 0, 0)
        a = super().read(size)
        self.check()
        return a


def local_name(name):
    tag = name.rsplit("_", 2)[-2:]
    receiver, kind = tag[0], tag[1]
    if kind == "csi.json" and receiver == "Rxall":
        return "csi.json"
    if receiver not in ["Rx1", "Rx2", "Rx3"]:
        raise ValueError("known input receiver required")
    if kind == "cam.mp4":
        return f"camera-{receiver[-1]}.mp4"
    if kind == "mic.wav":
        return f"microphone-{receiver[-1]}.wav"
    raise ValueError("only input members can be named; GT forbidden")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    budget = Budget(total=64 * 2**20, seconds=900)
    r = dict(
        status="INCOMPLETE",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        raw_files=[],
        video=[],
        audio=[],
        models_loaded=False,
        training=False,
        position_gt_opened=False,
        sealed_audit_opened=False,
        gpu_used=False,
        cloud_cost=0,
    )

    def check():
        budget.charge(0, time.perf_counter() - tick)
        if (
            shutil.disk_usage(args.output).free < 2**30
            or sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 383 * 2**20
        ):
            raise OSError("input-smoke output/disk budget exhausted; oldGPU8GiB guard unchanged")
        if peak_bytes() > 2 * 2**30:
            raise MemoryError("2GiB CPU process peak exceeded")

    try:
        import av
        import av.container.input
        import av.video.frame

        if sha(PARENT) != PARENT_SHA:
            raise ValueError("admitted complete-directory replay changed")
        parent = json.loads(PARENT.read_bytes())
        if (
            parent["status"] != "PASS_COMPLETE_DIRECTORY_GROUP_ACCOUNTING_ONLY"
            or parent["archives"] != 60
        ):
            raise ValueError("admitted60-directory source required")
        pins = {**parent["validated_hashes"], str(PARENT): PARENT_SHA}
        for p in [
            Path(__file__),
            Path("scripts/robomnist_input_arrays.py"),
            PROTOCOL,
            Path("docs/superpowers/plans/2026-10-14-robomnist-input-smoke.md"),
            Path("tests/test_robomnist_input_arrays.py"),
            Path(np.__file__),
            Path(av.__file__),
            Path(av.container.input.__file__),
            Path(av.video.frame.__file__),
            Path(
                "C:/CodexArtifacts/pretrained-reuse-r2/robomnist-input-source-code-01/examples__wifi_csi_read.ipynb"
            ),
        ]:
            pins[str(p)] = sha(p)
        for path, pin in pins.items():
            if sha(path) != pin:
                raise ValueError("source/protocol/library pin changed")
        directory = Path(
            "C:/CodexArtifacts/pretrained-reuse-r2/robomnist-all-directories-01/directory-00.json"
        )
        bank = json.loads(directory.read_bytes())
        g = fixed_plan(bank)
        infos = {i["name"]: i for i in bank["members"]}
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            group=g,
            archive=bank["source_archive"],
            source_hashes=pins,
            command=subprocess.list2cmdline(
                ["python", "-m", "scripts.fetch_robomnist_input_smoke"] + os.sys.argv[1:]
            ),
            reserved_transfer_cap_bytes=budget.total,
            body_limit_seconds=900,
            output_cap_bytes=384 * 2**20,
            terminal_record_reserve_bytes=2**20,
            disk_reserve_bytes=2**30,
            cpu_peak_cap_bytes=2 * 2**30,
            video_sample_rule="0/floor((n-1)/2)/(n-1)",
            native_csi_clock_role="alignment audit only; absolute epoch not a learned feature",
            python=platform.python_version(),
            numpy=np.__version__,
            av=av.__version__,
            ffmpeg=av.library_versions,
        )
        write(args.output / "config.json", config)
        (args.output / "execution-source.py").write_bytes(Path(__file__).read_bytes())
        r.update(config=config, config_sha256=sha(args.output / "config.json"))
        reader = InputReader(bank["source_archive"], budget, check)
        with zipfile.ZipFile(reader) as z:
            for ordinal, name in enumerate(
                sorted(g["input_members"], key=lambda n: infos[n]["offset"])
            ):
                check()
                expected = infos[name]
                i = z.getinfo(name)
                if (
                    i.file_size,
                    i.compress_size,
                    f"{i.CRC:08x}",
                    i.compress_type,
                    i.header_offset,
                ) != (
                    expected["size"],
                    expected["compressed_size"],
                    expected["crc32"],
                    expected["compression"],
                    expected["offset"],
                ) or i.flag_bits & 1:
                    raise ValueError("actual fixed member differs from pinned directory")
                path = args.output / local_name(name)
                print("input", ordinal + 1, "/7", path.name, flush=True)
                count, crc = 0, 0
                with z.open(i) as source, path.open("xb") as target:
                    while chunk := source.read(2**20):
                        check()
                        target.write(chunk)
                        count += len(chunk)
                        crc = zlib.crc32(chunk, crc)
                if count != i.file_size or f"{crc:08x}" != expected["crc32"]:
                    raise ValueError("complete raw member size/CRC differs")
                r["raw_files"].append(
                    dict(
                        source_member=name,
                        path=str(path),
                        file_bytes=count,
                        sha256=sha(path),
                        crc32=f"{crc:08x}",
                    )
                )
                write(
                    args.output / f"progress-input-{ordinal:02}.json",
                    dict(
                        pid=os.getpid(),
                        completed_inputs=ordinal + 1,
                        elapsed_seconds=time.perf_counter() - tick,
                    ),
                )
        if len(r["raw_files"]) != 7 or any(
            "pos.json" in v["source_member"] for v in r["raw_files"]
        ):
            raise ValueError("exactly seven allowed inputs required")
        arrays = {}
        for receiver in [1, 2, 3]:
            print("decode camera", receiver, flush=True)
            a, meta = read_video(args.output / f"camera-{receiver}.mp4", check)
            arrays.update({f"camera_{receiver}_{k}": v for k, v in a.items()})
            r["video"].append(dict(receiver=receiver, **meta))
            a, meta = read_wav(args.output / f"microphone-{receiver}.wav")
            arrays[f"microphone_{receiver}_pcm"] = a
            r["audio"].append(dict(receiver=receiver, **meta))
        a, meta = load_csi(args.output / "csi.json")
        arrays.update(a)
        r["csi"] = meta
        target = args.output / "native-inputs.npz"
        np.savez_compressed(target, **arrays)
        with np.load(target, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                saved[k].shape != v.shape
                or saved[k].dtype != v.dtype
                or saved[k].tobytes() != v.tobytes()
                for k, v in arrays.items()
            ):
                raise ValueError("native input NPZ producer/readback differs")
        r.update(
            array_bank=dict(
                path=str(target),
                sha256=sha(target),
                shapes={k: list(v.shape) for k, v in arrays.items()},
                dtypes={k: str(v.dtype) for k, v in arrays.items()},
                alignment_only_keys=[k for k in arrays if k.endswith("timestamps_ns")],
            )
        )
        for path, pin in pins.items():
            if sha(path) != pin:
                raise ValueError("protected source changed during input smoke")
        if sha(args.output / "config.json") != r["config_sha256"]:
            raise ValueError("effective configuration changed")
        check()
        r.update(
            status="PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY",
            source_pins_unchanged=True,
            sync_assessment="UNVERIFIED_COMMON_CLOCK: media PTS/WAV do not certify CSI mapping",
            peak_cpu_bytes=peak_bytes(),
            actual_transfer_bytes=reader.transferred,
        )
    except Exception:
        r.update(status="INCOMPLETE", error=traceback.format_exc())
    r.update(
        ended_utc=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - tick,
        reserved_transfer_bytes=budget.transferred,
        full_archive_md5_verified=False,
        training_corpus_admission="NOT_ADMITTED",
        quality_measured=False,
    )
    write(args.output / "result.json", r)
    print(r["status"], r["elapsed_seconds"], flush=True)
    if r["status"] != "PASS_FIXED_NATIVE_INPUT_MECHANICS_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
