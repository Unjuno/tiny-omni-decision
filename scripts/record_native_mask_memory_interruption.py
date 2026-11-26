"""Read-only integrity/accounting of interrupted inference; never admits quality."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.audit_native_mask_memory_full import (
    ARMS,
    COUNTS,
    IDS,
    digest,
    load_row,
    report,
    sha,
    verify_packed_archive,
)


def require_prefix(rows, plans):
    expected = [(p["video_id"], a) for p in plans for a in ARMS]
    actual = [(r["video_id"], r["arm"]) for r in rows]
    if not actual or len(actual) >= len(expected) or actual != expected[: len(actual)]:
        raise ValueError("interrupted rows must be a proper ordered complete-arm prefix")
    return len(actual)


def verify_partial(a, count):
    if set(a) != {"masks_packed", "boxes", "scores", "empty_masks", "raw_sha256"} or count <= 0:
        raise ValueError("known partial archive schema/count required")
    # Synthetic ordinal metadata only exercises the shared geometry verifier.
    # These IDs were NOT saved in this partial archive and are not evidence of identity.
    checks = verify_packed_archive(
        {**a, "seed_indices": np.arange(count), "frame_indices": np.arange(32)},
        list(range(count)),
        list(range(32)),
    )
    return {
        **checks,
        "independently_stored_seed_frame_identity": False,
        "raw_logits_reconstructed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new diagnostic output required")
    folder = args.main
    r = json.loads((folder / "result.json").read_bytes())
    config = json.loads((folder / "config.json").read_bytes())
    c = r["contract"]
    if (
        r["status"] != "FAIL_INCOMPLETE"
        or "OSError: 8GiB disk reserve exhausted" not in r["error"]
        or config["mode"] != "main"
        or config["contract"] != c
        or sha(folder / "config.json") != r["config_sha256"]
        or digest(c) != r["contract_sha256"]
        or [(p["video_id"], len(p["seed_indices"])) for p in c["plans"]]
        != list(zip(IDS, COUNTS, strict=True))
        or r["training"]
        or r["annotations_loaded"]
        or r["sealed_audit_opened"]
    ):
        raise ValueError("expected frozen disk-interrupted main/config required")
    n = require_prefix(r["rows"], c["plans"])
    pins = {
        str(folder / name): sha(folder / name)
        for name in ["result.json", "config.json", "current-input-plan.json", "current-seed.json"]
    }
    pins.update(c["source_hashes"])
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError("interrupted source/config/input pin changed")
    rows = []
    for row in r["rows"]:
        plan = next(p for p in c["plans"] if p["video_id"] == row["video_id"])
        a = load_row(row, pins)
        checks = verify_packed_archive(a, plan["seed_indices"], plan["frame_indices"])
        if (
            row["seeds"] != len(plan["seed_indices"])
            or row["native_object_frames"] != checks["native_object_frames"]
        ):
            raise ValueError("complete arm accounting differs")
        rows.append({"video_id": row["video_id"], "arm": row["arm"], **checks})
    current = json.loads((folder / "current-seed.json").read_bytes())
    plan = json.loads((folder / "current-input-plan.json").read_bytes())
    next_plan, next_arm = c["plans"][n // 2], ARMS[n % 2]
    if (
        plan != next_plan
        or current["pid"] != r["pid"]
        or current["arm"] != next_arm
        or current["total_seeds"] != len(plan["seed_indices"])
        or current["seed_ordinal"] != current["completed_seeds"] + 1
        or not 0 < current["completed_seeds"] < current["total_seeds"]
    ):
        raise ValueError("partial diagnostic context differs from next planned arm")
    partial_path = folder / f"incomplete-{next_arm}-cache32.npz"
    pins[str(partial_path)] = sha(partial_path)
    with np.load(partial_path, allow_pickle=False) as z:
        partial = verify_partial({k: z[k] for k in z.files}, current["completed_seeds"])
    try:
        report(folder, "main", "PASS_FULL_NATIVE_MEMORY_EXECUTION_ONLY")
    except ValueError as exc:
        gate_rejection = str(exc)
    else:
        raise ValueError("incomplete main incorrectly admitted by full quality barrier")
    for path, pin in pins.items():
        if sha(path) != pin:
            raise ValueError("pin changed during diagnostic")
    result = dict(
        status="PASS_SAVED_PARTIAL_GEOMETRY_INTEGRITY_ONLY",
        experiment_status="INCOMPLETE_ENVIRONMENT_DISK_RESERVE",
        checked_utc=datetime.now(UTC).isoformat(),
        source_commit=c["source_commit"],
        command=config["command"],
        started_utc=r["started_utc"],
        ended_utc=r["ended_utc"],
        elapsed_seconds=r["seconds"],
        error=r["error"],
        full_planned_object_frames=696 * 2 * 32,
        complete_arms=rows,
        complete_arm_object_frames=sum(v["native_object_frames"] for v in rows),
        partial_arm=dict(
            video_id=plan["video_id"],
            arm=next_arm,
            completed_seeds=current["completed_seeds"],
            **partial,
            identity_context="current plan/progress only; IDs absent in archive",
        ),
        source_pin_count=len(c["source_hashes"]),
        unchanged_source_pins=True,
        disk_free_bytes_at_diagnostic=shutil.disk_usage(folder).free,
        disk_reserve_bytes=c["disk_reserve_bytes"],
        outputs_total_file_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
        full_admission_rejected=gate_rejection,
        final_state_after="UNKNOWN: not recorded on exception",
        final_rng_freeze_resource_checks="NOT_REACHED",
        fresh_replay="NOT_RUN: full main admission required",
        group_proxy_scoring="NOT_RUN: full main/fresh admission required",
        training=False,
        annotations_loaded=False,
        sealed_audit_opened=False,
        cloud_cost=0,
        validated_hashes=pins,
        diagnostic_source_sha256=sha(__file__),
    )
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(result["status"], result["complete_arm_object_frames"], partial["native_object_frames"])


if __name__ == "__main__":
    main()
