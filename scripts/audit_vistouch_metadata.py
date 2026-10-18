"""Full native-source metadata accounting, never imports upstream implementation."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from scripts.fetch_robomnist_input_smoke import peak_bytes
from scripts.robomnist_audio_cue import read_bound_json, recheck

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2/vistouch-source-screen-01")
RECEIPT_SHA = "85b86f9b50e8067408db5e70ef326f92e238c41932a004fee12c7f0a3d6dce7e"
ROUNDING = Decimal("0.001")  # Source parent metadata has millisecond resolution.


def number(value):
    try:
        v = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("native finite decimal metadata required") from exc
    if not v.is_finite():
        raise ValueError("native finite decimal metadata required")
    return v


def indexed(rows, key):
    values = {row[key]: row for row in rows}
    if len(values) != len(rows) or any(not str(v) for v in values):
        raise ValueError("unique nonempty native IDs required")
    return values


def interval_union(values):
    ordered = sorted(values)
    if not ordered:
        return Decimal(0)
    start, end = ordered[0]
    total = Decimal(0)
    for a, b in ordered[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def audit_metadata(classes, sessions, samples, clips):
    if not all([classes, sessions, samples, clips]):
        raise ValueError("complete nonempty source cohorts required")
    labels = indexed(classes, "class_id")
    source = indexed(sessions, "session_id")
    parents = indexed(samples, "sample_id")
    indexed(clips, "clip_id")
    windows, session_splits, counts = defaultdict(list), defaultdict(set), Counter()
    for sample in samples:
        if sample["session_id"] not in source or sample["split"] not in {"train", "val", "test"}:
            raise ValueError("known native source session/split required")
        session = source[sample["session_id"]]
        try:
            label = labels[int(sample["class_id"])]
        except (KeyError, ValueError) as exc:
            raise ValueError("known native class identity required") from exc
        if (
            sample["material_english"] != label["material_english"]
            or sample["material_english"] != session["material_english"]
            or number(sample["force_n"]) != number(session["force_n"])
            or number(sample["start_epoch"]) >= number(sample["end_epoch"])
        ):
            raise ValueError("parent material/force/interval differs from native source")
        session_splits[sample["session_id"]].add(sample["split"])
    stored, represented, factors, strides = Decimal(0), Decimal(0), set(), set()
    identities, coverage_violations = [], []
    for row in clips:
        try:
            parent = parents[row["sample_id"]]
        except KeyError as exc:
            raise ValueError("every microclip must bind its native parent") from exc
        if any(
            row[k] != parent[k]
            for k in ["session_id", "class_id", "material_english", "force_n", "split"]
        ):
            raise ValueError("child source/class/force/split differs from parent")
        a, b = number(row["source_start_epoch"]), number(row["source_end_epoch"])
        duration, stretched, factor = (
            number(row[k]) for k in ["source_duration_s", "stored_duration_s", "stretch_factor"]
        )
        if (
            not b > a
            or abs(b - a - duration) > ROUNDING
            or duration <= 0
            or stretched <= 0
            or factor <= 0
            or abs(stretched - duration * factor) > ROUNDING
            or a < number(parent["start_epoch"]) - ROUNDING
            or b > number(parent["end_epoch"]) + ROUNDING
        ):
            raise ValueError("native parent/window duration and fixed stretch mismatch")
        identity = row["session_id"], a, b
        identities.append(identity)
        windows[row["session_id"]].append((a, b))
        counts[row["split"]] += 1
        represented += duration
        stored += stretched
        factors.add(factor)
        strides.add(number(row["window_stride_s"]))
        session = source[row["session_id"]]
        fields = [
            "audio_start_unix",
            "audio_end_unix",
            "tactile_first_ts_unix",
            "tactile_last_ts_unix",
        ]
        if all(k in session for k in fields):
            lo = max(number(session[fields[0]]), number(session[fields[2]]))
            hi = min(number(session[fields[1]]), number(session[fields[3]]))
            if a < lo - ROUNDING or b > hi + ROUNDING:
                coverage_violations.append(row["clip_id"])
    assets = defaultdict(set)
    for session in sessions:
        for modality in ["audio", "video", "tactile"]:
            name = session[f"raw_{modality}_filename"].replace("\\", "/").casefold()
            assets[(modality, name)].update(session_splits[session["session_id"]])
    total = sum((interval_union(v) for v in windows.values()), Decimal(0))
    return dict(
        clips=len(clips),
        source_samples=len(samples),
        source_sessions=len(sessions),
        classes=len(classes),
        split_counts=dict(counts),
        split_source_sessions={
            s: sorted(k for k, v in session_splits.items() if s in v) for s in sorted(counts)
        },
        split_force_levels={
            s: sorted({row["force_n"] for row in clips if row["split"] == s})
            for s in sorted(counts)
        },
        split_class_counts={
            s: dict(
                sorted(
                    Counter(row["material_english"] for row in clips if row["split"] == s).items()
                )
            )
            for s in sorted(counts)
        },
        cross_split_sessions=sorted(k for k, v in session_splits.items() if len(v) > 1),
        cross_split_source_assets=[
            dict(modality=k[0], asset=k[1], splits=sorted(v))
            for k, v in sorted(assets.items())
            if len(v) > 1
        ],
        metadata_unique_source_assets=len(assets),
        unique_source_windows=len(set(identities)),
        duplicate_source_windows=len(identities) - len(set(identities)),
        unique_source_seconds=float(total),
        source_window_seconds_sum=float(represented),
        stored_duration_seconds_sum=float(stored),
        source_coverage_multiplicity=float(represented / total),
        all_stretch_factors=[float(v) for v in sorted(factors)],
        all_window_strides_seconds=[float(v) for v in sorted(strides)],
        audio_tactile_coverage_violation_clip_ids=coverage_violations,
        validation_sessions_disjoint="val" in counts
        and not any(len(v) > 1 for v in session_splits.values()),
        independent_validation_present="UNVERIFIED" if "val" in counts else False,
        media_hash_overlap="UNVERIFIED",
        normalized_content_overlap="UNVERIFIED",
        specimen_identity="UNKNOWN",
        video_capture_clock="LINEAR_APPROXIMATION_DOCUMENTED",
        sensor_subsecond_clock="INTERPOLATED_FROM_INTEGER_SECONDS_DOCUMENTED",
        common_clock="UNVERIFIED",
        training_corpus_admission="NOT_ADMITTED",
        model_execution=False,
        media_opened=False,
        quality_measured=False,
    )


def build(output, tick):
    receipt = read_bound_json(ROOT / "receipt.json", RECEIPT_SHA)
    if (
        receipt["status"] != "PASS_FIXED_METADATA_SOURCE_INSPECTION_ONLY"
        or receipt["source_revision"] != "a569c7e572a78f232e819b652e0ebb718952e476"
        or len(receipt["files"]) != 12
    ):
        raise ValueError("exact complete pinned12file source receipt required")
    pins = {str(ROOT / "receipt.json"): RECEIPT_SHA}
    for row in receipt["files"]:
        p = Path(row["local"])
        if p.resolve() != (ROOT / row["path"].replace("/", "__")).resolve():
            raise ValueError("source paths differ from immutable independent root")
        data = p.read_bytes()
        if (
            len(data) != row["size"]
            or hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
            != row["git_blob_sha1"]
        ):
            raise ValueError("actual metadata/source Git bytes differ")
        pins[str(p)] = row["sha256"]
    for p, h in [
        (
            Path("C:/CodexArtifacts/pretrained-reuse-r2/vistouch-source-plan-01.json"),
            receipt["plan_sha256"],
        ),
        (
            Path("C:/CodexArtifacts/pretrained-reuse-r2/vistouch-source-fetch-01.py"),
            receipt["fetcher_sha256"],
        ),
    ]:
        pins[str(p)] = h
    for p in [
        Path(__file__),
        Path("tests/test_vistouch_metadata.py"),
        Path("scripts/robomnist_audio_cue.py"),
        Path("scripts/fetch_robomnist_input_smoke.py"),
        Path("scripts/diagnose_robomnist_lengths.py"),
        Path("docs/pretrained_reuse/R2_VISTOUCH_SOURCE_PROTOCOL_20261015.md"),
    ]:
        pins[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    recheck(pins)

    def rows(name):
        with (ROOT / f"metadata__{name}.csv").open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    classes = json.loads((ROOT / "metadata__classes.json").read_bytes())
    result = audit_metadata(classes, rows("sessions"), rows("samples"), rows("clips_index"))
    if (
        result["clips"],
        result["source_samples"],
        result["source_sessions"],
        result["classes"],
    ) != (10498, 312, 24, 8):
        raise ValueError("full original cohort required, no subset or source pruning")
    recheck(pins)
    result.update(
        status="PREPARED_FULL_SOURCE_ACCOUNTING_UNQUALIFIED",
        source_revision=receipt["source_revision"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_receipt_sha256=RECEIPT_SHA,
        source_bytes=receipt["total_source_bytes"],
        validated_hashes=pins,
        python=platform.python_version(),
        upstream_code_executed=False,
        sealed_audit_opened=False,
        gpu_used=False,
        cloud_cost=0,
    )
    return result


def guard(output, tick, reserve_bytes=0):
    seconds, peak = time.perf_counter() - tick, peak_bytes()
    files = [output, output.with_suffix(".accounting.json")]
    size = sum(p.stat().st_size for p in files if p.exists()) + reserve_bytes
    if (
        seconds > 60
        or peak > 2**30
        or size > 2**20
        or shutil.disk_usage(output.parent).free < 2**30
    ):
        raise RuntimeError("fixed60s/1GiB CPU/1MiB output/1GiB disk source-only guard exhausted")
    return dict(elapsed_seconds=seconds, peak_cpu_bytes=peak, reserved_output_bytes=size)


def execute(output):
    prepared = output.with_suffix(".accounting.json")
    if output.exists() or prepared.exists():
        raise ValueError("new isolated source-audit products required")
    tick = time.perf_counter()
    try:
        guard(output, tick)
        result = build(output, tick)
        body = (json.dumps(result, indent=2) + "\n").encode()
        prepared.write_bytes(body)
        if prepared.read_bytes() != body:
            raise ValueError("canonical full prepared accounting changed")
        recheck(result["validated_hashes"])
        resources = guard(output, tick)
        result.update(
            status="PASS_FULL_SOURCE_METADATA_AUDIT_ONLY",
            prepared_accounting_sha256=hashlib.sha256(body).hexdigest(),
            **resources,
        )
        # Reserve the prospective terminal plus room for updated resource fields.
        result.update(guard(output, tick, len(json.dumps(result, indent=2).encode()) + 4096))
    except Exception as exc:
        result = dict(
            status="FAIL_INCOMPLETE_SOURCE_METADATA_AUDIT",
            error=repr(exc),
            elapsed_seconds=time.perf_counter() - tick,
            quality_measured=False,
        )
    output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(
        result["status"],
        result.get("error", ""),
        result["elapsed_seconds"],
        flush=True,
    )
    return result["status"] == "PASS_FULL_SOURCE_METADATA_AUDIT_ONLY"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not execute(args.output):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
