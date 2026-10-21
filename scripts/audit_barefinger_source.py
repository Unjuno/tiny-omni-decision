"""Account for every pinned contact trial/photo without admitting training data."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from itertools import product
from pathlib import Path, PurePosixPath

from scripts.fetch_robomnist_input_smoke import peak_bytes
from scripts.robomnist_audio_cue import read_bound_json, recheck

TEXTURES = {"7t", "10t", "18t", "39t", "44t", "54t", "59t", "83t", "108t", "120t"}
TRIALS = {"std_0", "std_1", "std_2", "cte_force_3", "cte_speed_4"}
TRIAL = re.compile(
    r"datah5/subject_([0-9])/(\d+t)/subject_([0-9])_(\d+t)_"
    r"(std_[0-2]|cte_force_3|cte_speed_4)\.h5"
)
PHOTO = re.compile(r"img/(\d+t)/(\d+t)_([lr])_([0-9])\.jpg")


def names(rows):
    seen, files = set(), []
    for row in rows:
        name = row["name"]
        p = PurePosixPath(name)
        if (
            name in seen
            or not name
            or p.is_absolute()
            or ".." in p.parts
            or "\\" in name
            or ":" in name
            or "\x00" in name
        ):
            raise ValueError("duplicate or unsafe source member")
        seen.add(name)
        if not isinstance(row["directory"], bool):
            raise ValueError("explicit directory flag required")
        if not row["directory"]:
            if (
                not isinstance(row["size"], int)
                or row["size"] <= 0
                or not isinstance(row["compressed_size"], int)
                or row["compressed_size"] <= 0
                or row["compression"] not in (0, 8)
                or not re.fullmatch(r"[0-9a-f]{8}", row["crc32"])
            ):
                raise ValueError("invalid ZIP source metadata")
            files.append(name)
    return files


def account_directories(data_rows, image_rows):
    physical, photos = set(), set()
    calibration = 0
    for name in names(data_rows):
        if name == "datah5/texture_metadata.json":
            calibration += 1
            continue
        match = TRIAL.fullmatch(name)
        if not match:
            raise ValueError("unknown physical source member")
        subject, texture, file_subject, file_texture, trial = match.groups()
        if subject != file_subject or texture != file_texture or texture not in TEXTURES:
            raise ValueError("source path/physical identity disagreement")
        physical.add((subject, texture, trial))
    expected = set(product(map(str, range(10)), TEXTURES, TRIALS))
    if physical != expected or calibration != 1:
        raise ValueError("complete500 physical trials and calibration metadata required")
    for name in names(image_rows):
        match = PHOTO.fullmatch(name)
        if not match:
            raise ValueError("unknown texture photo")
        folder, texture, camera, shot = match.groups()
        if folder != texture or texture not in TEXTURES:
            raise ValueError("photo texture identity disagreement")
        photos.add((texture, camera, shot))
    if photos != set(product(TEXTURES, ["l", "r"], map(str, range(10)))):
        raise ValueError("complete200 texture photos required")
    return dict(
        physical_trials=len(physical),
        participants=10,
        textures=sorted(TEXTURES),
        participant_texture_groups=100,
        texture_photo_groups=10,
        trials_by_type=dict(sorted(Counter(r[2] for r in physical).items())),
        trials_by_participant=dict(sorted(Counter(r[0] for r in physical).items())),
        trials_by_texture=dict(sorted(Counter(r[1] for r in physical).items())),
        stereo_photos=len(photos),
        images_independent_by_participant=False,
        native_rgb_video_members=0,
        specimen_held_out_generalization="UNVERIFIED",
        independent_validation="NOT_DEFINED",
        normalized_content_overlap="UNVERIFIED",
        common_clock="UNVERIFIED",
        training_admission="NOT_ADMITTED",
        model_inference=False,
        training=False,
        quality_measured=False,
        grouping="participant/texture/trial; all channels together; photos grouped by texture",
        calibration_policy="texture-keyed force offsets cannot be selected using unknown target",
    )


def build(root, expected_sha):
    receipt_path = root / "result.json"
    receipt = read_bound_json(receipt_path, expected_sha)
    if (
        receipt["status"] != "PASS_IMMUTABLE_LICENSE_README_AND_DIRECTORY_ONLY"
        or receipt["media_members_opened"]
        or receipt["training"]
        or receipt["full_archive_md5_verified"]
    ):
        raise ValueError("directory-only immutable source receipt required")
    pins = {str(receipt_path.resolve()): expected_sha}
    for row in receipt["requests"]:
        path = Path(row.get("path", row.get("directory_path")))
        if path.resolve().parent != root.resolve():
            raise ValueError("immutable source paths must stay inside source root")
        if str(path.resolve()) in pins:
            raise ValueError("duplicate source receipt path")
        pins[str(path.resolve())] = row["sha256"]
    required = {
        str((root / name).resolve())
        for name in [
            "version-1.json",
            "README.md",
            "datah5.zip.directory.json",
            "img.zip.directory.json",
            "script.zip.directory.json",
        ]
    }
    if set(pins) != required | {str(receipt_path.resolve())}:
        raise ValueError("complete required pinned source files before parsing")
    for p in [
        Path(__file__),
        Path("tests/test_barefinger_source.py"),
        Path("scripts/robomnist_audio_cue.py"),
        Path("scripts/fetch_robomnist_input_smoke.py"),
        Path("scripts/diagnose_robomnist_lengths.py"),
        Path("docs/pretrained_reuse/R2_BAREFINGER_SOURCE_PROTOCOL_20261015.md"),
        Path("docs/superpowers/plans/2026-10-15-barefinger-source-audit.md"),
    ]:
        pins[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    recheck(pins)
    version = json.loads((root / "version-1.json").read_bytes())
    if (
        version["id"] != 26965267
        or version["version"] != 1
        or version["license"]["url"] != "https://creativecommons.org/licenses/by/4.0/"
    ):
        raise ValueError("explicit pinned dataset CC BY4.0 required")
    directories = {}
    for archive in ["datah5.zip", "img.zip", "script.zip"]:
        path = root / (archive + ".directory.json")
        if str(path.resolve()) not in pins:
            raise ValueError("missing pinned directory")
        data = json.loads(path.read_bytes())
        if (
            data["archive"] not in version["files"]
            or data["archive"]["name"] != archive
            or not 0 < data["transferred_bytes"] <= 2**20
        ):
            raise ValueError("directory/file revision or transfer budget mismatch")
        directories[archive] = data
    result = account_directories(
        directories["datah5.zip"]["members"], directories["img.zip"]["members"]
    )
    recheck(pins)
    result.update(
        source=version["doi"],
        license=version["license"],
        archive_sizes={f["name"]: f["size"] for f in version["files"]},
        directory_transferred_bytes=sum(d["transferred_bytes"] for d in directories.values()),
        full_archive_md5_verified=False,
        validated_hashes=pins,
        media_members_opened=False,
        upstream_code_executed=False,
        sealed_audit_opened=False,
        gpu_used=False,
        cloud_cost=0,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    return result


def execute(root, expected_sha, output):
    prepared = output.with_suffix(".accounting.json")
    if output.exists() or prepared.exists():
        raise ValueError("new source report path required")
    tick = time.perf_counter()
    result = build(root, expected_sha)
    result["status"] = "PREPARED_CONTACT_SOURCE_ACCOUNTING_UNQUALIFIED"
    prepared_body = (json.dumps(result, indent=2) + "\n").encode()
    if len(prepared_body) > 2**19:
        raise RuntimeError("bounded accounting report required")
    with prepared.open("xb") as f:
        f.write(prepared_body)
    if prepared.read_bytes() != prepared_body:
        raise ValueError("prepared canonical readback mismatch")
    recheck(result["validated_hashes"])
    elapsed, peak = time.perf_counter() - tick, peak_bytes()
    result.update(
        status="PASS_COMPLETE_CONTACT_SOURCE_ACCOUNTING_ONLY",
        elapsed_seconds=elapsed,
        peak_cpu_bytes=peak,
        prepared_accounting_sha256=hashlib.sha256(prepared_body).hexdigest(),
    )
    body = (json.dumps(result, indent=2) + "\n").encode()
    if (
        elapsed > 60
        or peak > 2**30
        or len(body) + len(prepared_body) > 2**20
        or shutil.disk_usage(output.parent).free < 2**30
    ):
        raise RuntimeError("60s/1GiB CPU/1MiB report/1GiB disk guard exhausted")
    recheck(result["validated_hashes"])
    with output.open("xb") as f:
        f.write(body)
    if output.read_bytes() != body:
        raise ValueError("report canonical readback mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.source_root, args.source_result_sha256, args.output)
    print(result["status"], result["physical_trials"], result["stereo_photos"])


if __name__ == "__main__":
    main()
