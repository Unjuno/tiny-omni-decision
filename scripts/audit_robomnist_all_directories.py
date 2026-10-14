"""Bounded directory-only inventory of every pinned RoboMNIST class archive."""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_robomnist_source import BASE, FOLDER, group_members, require_class, sha
from scripts.fetch_starss23_smoke import HttpRangeReader

VERSION_SHA = "18ef8b20c82056747c58572d24f262b2d12c9284e4ffd83c245ae64eca2ab120"
PROTOCOL = Path("docs/pretrained_reuse/R2_ROBOMNIST_DIRECTORY_PROTOCOL_20261014.md")


class Budget:
    def __init__(self, total=12 * 2**20, seconds=900):
        self.total, self.seconds, self.transferred = total, seconds, 0

    def charge(self, count, elapsed):
        if (
            not isinstance(count, int)
            or count < 0
            or self.transferred + count > self.total
            or not math.isfinite(elapsed)
            or elapsed < 0
            or elapsed > self.seconds
        ):
            raise ValueError("directory transfer/body budget exhausted or invalid")
        self.transferred += count


def inspect_infos(infos, archive_name):
    match = re.fullmatch(FOLDER + r"\.zip", archive_name)
    if not match:
        raise ValueError("known source archive required")
    m = match.groupdict()
    require_class(m["index"], m["robot"], m["velocity"], m["action"])
    if not infos or len({i["name"] for i in infos}) != len(infos):
        raise ValueError("unique nonempty ZIP directory required")
    for i in infos:
        if (
            not isinstance(i["size"], int)
            or i["size"] < 0
            or not isinstance(i["compressed_size"], int)
            or i["compressed_size"] < 0
            or not re.fullmatch("[0-9a-f]{8}", i["crc32"])
        ):
            raise ValueError("invalid ZIP member size/checksum metadata")
    names = [i["name"] for i in infos if not i["directory"]]
    groups = group_members(names, expected_folder=archive_name.removesuffix(".zip"))
    real = [n for n in names if not n.startswith("__MACOSX/")]
    return dict(
        archive=archive_name,
        class_index=int(m["index"]),
        real_files=len(real),
        macos_resource_forks=len(names) - len(real),
        repetition_groups=len(groups),
        groups=groups,
    )


def write(path, value):
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, separators=(",", ":"))
        f.write("\n")


def storage(folder):
    if shutil.disk_usage(folder).free < 2**30:
        raise OSError("1GiB metadata reserve exhausted; GPU8GiB guard is unchanged")
    if sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) > 16 * 2**20:
        raise OSError("16MiB metadata-output cap exceeded")


class DirectoryReader(HttpRangeReader):
    def __init__(self, url, size, budget, tick, output):
        super().__init__(url, size, budget=524288, retries=0)
        self.shared, self.tick, self.output = budget, tick, output

    def read(self, size=-1):
        count = self.size - self.position if size < 0 else min(size, self.size - self.position)
        # Reserve before transport, including the extra length-check byte. Failed
        # calls retain their reservation; actual successful body bytes are separate.
        self.shared.charge(count + 1 if count else 0, time.perf_counter() - self.tick)
        storage(self.output)
        data = super().read(size)
        self.shared.charge(0, time.perf_counter() - self.tick)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    tick = time.perf_counter()
    budget = Budget()
    result = dict(
        status="INCOMPLETE",
        pid=os.getpid(),
        started_utc=datetime.now(UTC).isoformat(),
        rows=[],
        media_members_opened=False,
        model_inference=False,
        training=False,
        sealed_audit_opened=False,
        cloud_cost=0,
    )
    try:
        source = BASE / "article-version-1.json"
        if sha(source) != VERSION_SHA:
            raise ValueError("immutable source version metadata changed")
        version = json.loads(source.read_bytes())
        if (
            version["id"] != 28179383
            or version["version"] != 1
            or version["license"]["url"] != "https://creativecommons.org/licenses/by/4.0/"
        ):
            raise ValueError("official dataset source/license changed")
        archives = sorted(version["files"], key=lambda f: f["name"])
        if len(archives) != 60 or [int(f["name"][:2]) for f in archives] != list(range(60)):
            raise ValueError("complete ordered60 source archives required")
        sources = [
            source,
            Path(__file__),
            Path("scripts/audit_robomnist_source.py"),
            Path("scripts/fetch_starss23_smoke.py"),
            PROTOCOL,
            Path("docs/superpowers/plans/2026-10-14-robomnist-directory-index.md"),
            Path("tests/test_robomnist_all_directories.py"),
        ]
        pins = {str(p): sha(p) for p in sources}
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_hashes=pins,
            source_doi=version["doi"],
            archives=archives,
            command=subprocess.list2cmdline(
                ["python", "-m", "scripts.audit_robomnist_all_directories"] + os.sys.argv[1:]
            ),
            per_archive_transfer_bytes=524288,
            overall_reserved_transfer_bytes=budget.total,
            body_seconds=budget.seconds,
            output_bytes_cap=16 * 2**20,
            metadata_disk_reserve_bytes=2**30,
            media_payload_reads_allowed=False,
            grouping="physical repetition/uncertainty; all views together; positionGT excluded",
        )
        write(args.output / "config.json", config)
        (args.output / "execution-source.py").write_bytes(Path(__file__).read_bytes())
        result.update(config_sha256=sha(args.output / "config.json"), config=config)
        storage(args.output)
        for ordinal, archive in enumerate(archives):
            budget.charge(0, time.perf_counter() - tick)
            print("directory", ordinal + 1, "/60", archive["name"], flush=True)
            reader = DirectoryReader(
                archive["download_url"], archive["size"], budget, tick, args.output
            )
            with zipfile.ZipFile(reader) as z:
                infos = [
                    dict(
                        name=i.filename,
                        directory=i.is_dir(),
                        size=i.file_size,
                        compressed_size=i.compress_size,
                        crc32=f"{i.CRC:08x}",
                        compression=i.compress_type,
                        offset=i.header_offset,
                    )
                    for i in z.infolist()
                ]
            summary = inspect_infos(infos, archive["name"])
            groups = summary.pop("groups")
            p = args.output / f"directory-{ordinal:02}.json"
            write(p, dict(source_archive=archive, members=infos, groups=groups))
            result["rows"].append(
                dict(
                    **summary,
                    path=str(p),
                    sha256=sha(p),
                    transferred_bytes=reader.transferred,
                    reserved_transfer_bytes=budget.transferred,
                )
            )
            write(
                args.output / f"progress-{ordinal:02}.json",
                dict(
                    pid=os.getpid(),
                    completed_archives=ordinal + 1,
                    elapsed_seconds=time.perf_counter() - tick,
                    current_archive=archive["name"],
                    repetition_groups=summary["repetition_groups"],
                ),
            )
            storage(args.output)
        for path, pin in pins.items():
            if sha(path) != pin:
                raise ValueError("source/protocol changed during directory audit")
        if sha(args.output / "config.json") != result["config_sha256"]:
            raise ValueError("effective config changed")
        budget.charge(0, time.perf_counter() - tick)
        result.update(status="PASS_COMPLETE_DIRECTORY_INVENTORY_ONLY", unchanged_source_pins=True)
    except Exception:
        result.update(status="INCOMPLETE", error=traceback.format_exc())
    result.update(
        ended_utc=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - tick,
        reserved_transfer_bytes=budget.transferred,
        completed_archives=len(result["rows"]),
        output_file_bytes_before_terminal=sum(
            p.stat().st_size for p in args.output.rglob("*") if p.is_file()
        ),
        full_archive_or_payload_checksums_verified=False,
        timing_content_session_split_integrity="UNVERIFIED",
        training_admission="NOT_ADMITTED",
    )
    write(args.output / "result.json", result)
    print(result["status"], len(result["rows"]), result["elapsed_seconds"], flush=True)
    if result["status"] != "PASS_COMPLETE_DIRECTORY_INVENTORY_ONLY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
