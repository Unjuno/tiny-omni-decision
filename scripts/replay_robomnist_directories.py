"""Independent lexical/group accounting of saved directory metadata, no network."""

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bank(bank):
    stem = bank["source_archive"]["name"].removesuffix(".zip")
    folder = stem.split("_")
    if (
        len(folder) != 4
        or folder[1] not in ["Ar1", "Ar2"]
        or folder[2] not in ["Vhigh", "Vmed", "Vlow"]
        or not re.fullmatch("Ac[0-9]", folder[3])
    ):
        raise ValueError("invalid source archive class")
    class_index = (
        {"Vhigh": 0, "Vmed": 20, "Vlow": 40}[folder[2]]
        + (int(folder[1][-1]) - 1) * 10
        + int(folder[3][-1])
    )
    if folder[0] != f"{class_index:02}":
        raise ValueError("source archive index differs from labels")
    groups, counts, seen = {}, Counter(), set()
    forks = 0
    for info in bank["members"]:
        name = info["name"]
        if (
            name in seen
            or info["size"] < 0
            or info["compressed_size"] < 0
            or not re.fullmatch("[0-9a-f]{8}", info["crc32"])
        ):
            raise ValueError("invalid or duplicate directory member")
        seen.add(name)
        if info["directory"]:
            continue
        if name.startswith("__MACOSX/"):
            forks += 1
            continue
        path = name.split("/")
        if len(path) != 2 or path[0] != stem:
            raise ValueError("actual member is not from inspected archive")
        parts = path[1].split("_")
        if (
            len(parts) != 7
            or [parts[3], parts[2], parts[1]] != folder[1:]
            or not re.fullmatch("Rp[0-9]{2}", parts[0])
            or not re.fullmatch("UNC[0-9]+", parts[4])
        ):
            raise ValueError("actual member identity/labels differ")
        gid = "_".join(folder[1:] + [parts[0], parts[4]])
        key = "_".join(parts[5:])
        g = groups.setdefault(gid, {})
        if key in g:
            raise ValueError("duplicate physical group stream")
        g[key] = name
        counts[parts[6].split(".")[0]] += 1
    streams = {
        f"Rx{i}_{kind}.{ext}" for i in [1, 2, 3] for kind, ext in [("cam", "mp4"), ("mic", "wav")]
    } | {"Rxall_csi.json", "Rxall_pos.json"}
    if not groups or any(set(g) != streams for g in groups.values()):
        raise ValueError("all eight streams required in every physical group")
    reconstructed = [
        dict(
            group_id=gid,
            input_members=sorted(
                name for key, name in groups[gid].items() if key != "Rxall_pos.json"
            ),
            supervision_only_member=groups[gid]["Rxall_pos.json"],
        )
        for gid in sorted(groups)
    ]
    if reconstructed != bank["groups"]:
        raise ValueError("saved group identities/inputs/supervision differ from actual members")
    return dict(
        class_index=class_index,
        repetition_groups=len(groups),
        real_files=sum(counts.values()),
        macos_resource_forks=forks,
        stream_counts=dict(counts),
        group_ids=sorted(groups),
    )


def require_complete(r):
    if (
        r["status"] != "PASS_COMPLETE_DIRECTORY_INVENTORY_ONLY"
        or r["completed_archives"] != 60
        or [v["class_index"] for v in r["rows"]] != list(range(60))
        or not math.isfinite(r["elapsed_seconds"])
        or not 0 <= r["elapsed_seconds"] <= 900
        or not 0 <= r["reserved_transfer_bytes"] <= 12 * 2**20
    ):
        raise ValueError("complete ordered60 directory cohort and bounded runtime required")


def verify_transfers(rows, final):
    actual, previous = 0, 0
    if not rows or not isinstance(final, int) or not 0 < final <= 12 * 2**20:
        raise ValueError("bounded final transfer reservation required")
    for row in rows:
        count, reserved = row["transferred_bytes"], row["reserved_transfer_bytes"]
        if (
            not isinstance(count, int)
            or not 0 < count <= 524288
            or not isinstance(reserved, int)
            or not previous + count <= reserved <= final
        ):
            raise ValueError("actual transfer/cumulative reservation accounting differs")
        actual += count
        if actual > reserved:
            raise ValueError("cumulative transfers exceed reserved total")
        previous = reserved
    if previous != final:
        raise ValueError("final reservation differs from the final complete row")
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new replay output required")
    tick = time.perf_counter()
    r = json.loads((args.main / "result.json").read_bytes())
    c = json.loads((args.main / "config.json").read_bytes())
    require_complete(r)
    actual_transferred = verify_transfers(r["rows"], r["reserved_transfer_bytes"])
    version_path = Path(
        "C:/CodexArtifacts/pretrained-reuse-r2/robomnist-source-audit-01/article-version-1.json"
    )
    if sha(version_path) != "18ef8b20c82056747c58572d24f262b2d12c9284e4ffd83c245ae64eca2ab120":
        raise ValueError("actual official source version changed")
    version = json.loads(version_path.read_bytes())
    if (
        r["config"] != c
        or sha(args.main / "config.json") != r["config_sha256"]
        or c["archives"] != sorted(version["files"], key=lambda f: f["name"])
        or not r["unchanged_source_pins"]
        or any(
            r[k] is not False
            for k in ["media_members_opened", "training", "model_inference", "sealed_audit_opened"]
        )
        or c["media_payload_reads_allowed"] is not False
        or c["per_archive_transfer_bytes"] != 524288
        or c["overall_reserved_transfer_bytes"] != 12 * 2**20
        or c["body_seconds"] != 900
    ):
        raise ValueError("exact effective configuration/source-only scope required")
    expected_sources = [
        version_path,
        Path("scripts/audit_robomnist_all_directories.py"),
        Path("scripts/audit_robomnist_source.py"),
        Path("scripts/fetch_starss23_smoke.py"),
        Path("docs/pretrained_reuse/R2_ROBOMNIST_DIRECTORY_PROTOCOL_20261014.md"),
        Path("docs/superpowers/plans/2026-10-14-robomnist-directory-index.md"),
        Path("tests/test_robomnist_all_directories.py"),
    ]
    if {Path(p).resolve() for p in c["source_hashes"]} != {p.resolve() for p in expected_sources}:
        raise ValueError("mandatory source pin coverage differs")
    pins = {
        **c["source_hashes"],
        **{
            str(args.main / n): sha(args.main / n)
            for n in ["result.json", "config.json", "execution-source.py"]
        },
    }
    if sha(args.main / "execution-source.py") != sha("scripts/audit_robomnist_all_directories.py"):
        raise ValueError("execution source snapshot differs")
    rows, gids = [], set()
    for i, row in enumerate(r["rows"]):
        p = args.main / f"directory-{i:02}.json"
        if Path(row["path"]).resolve() != p.resolve() or sha(p) != row["sha256"]:
            raise ValueError("saved actual directory file/pin differs")
        bank = json.loads(p.read_bytes())
        if bank["source_archive"] != c["archives"][i]:
            raise ValueError("actual directory/archive source binding differs")
        checked = verify_bank(bank)
        if (
            any(
                checked[k] != row[k]
                for k in ["class_index", "repetition_groups", "real_files", "macos_resource_forks"]
            )
            or row["archive"] != bank["source_archive"]["name"]
            or not 0 < row["transferred_bytes"] <= 524288
        ):
            raise ValueError("actual source grouping/accounting differs")
        if gids.intersection(checked["group_ids"]):
            raise ValueError("physical source group identity appears across archives")
        gids.update(checked.pop("group_ids"))
        rows.append(dict(archive=row["archive"], **checked))
        pins[str(p)] = row["sha256"]
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError("actual source/product changed during replay")
    output_bytes = sum(p.stat().st_size for p in args.main.rglob("*") if p.is_file())
    if output_bytes > 16 * 2**20:
        raise ValueError("metadata output cap exceeded")
    result = dict(
        status="PASS_COMPLETE_DIRECTORY_GROUP_ACCOUNTING_ONLY",
        source_commit=c["source_commit"],
        archives=60,
        rows=rows,
        unique_repetition_groups=len(gids),
        real_files=sum(v["real_files"] for v in rows),
        macos_resource_forks=sum(v["macos_resource_forks"] for v in rows),
        source_transferred_bytes=actual_transferred,
        reserved_transfer_bytes=r["reserved_transfer_bytes"],
        main_seconds=r["elapsed_seconds"],
        output_file_bytes=output_bytes,
        replay_seconds=time.perf_counter() - tick,
        full_archive_or_member_payload_checksums_verified=False,
        media_members_opened=False,
        model_inference=False,
        training=False,
        sealed_audit_opened=False,
        training_admission="NOT_ADMITTED: content/timing/session/split/complementarity unverified",
        validated_hashes=pins,
        auditor_sha256=sha(__file__),
    )
    with args.output.open("x", encoding="utf8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(result["status"], "groups", len(gids), "real_files", result["real_files"])


if __name__ == "__main__":
    main()
