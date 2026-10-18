"""All known99 train scenes: weak native-grid role-pair supervision, no fit."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.diagnose_timed_relations import grounded_roles
from tiny_omni_decision.grid_role_supervision import (
    cell_source_labels,
    grid_cells,
    pair_role_counts,
)
from tiny_omni_decision.mask_pair_association import association_windows, labelled_mask_pairs
from tiny_omni_decision.proposal_opportunity import source_box_at_time
from tiny_omni_decision.spatial_context_features import spatial_context_positions

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
PINS = {
    "spatial-context-corpus-01/result.json": (
        "a306930eb673a00c246b7b2e9caad3044315cb9f19b0f5ee2c43987d2f04f0da"
    ),
    "spatial-context-corpus-01/config.json": (
        "d17c1f1f9e55f17d58983603c1c8bf27ac171bed9d2ee04108535c84b15e35ff"
    ),
    "perception-role-supervision-audit-02/result.json": (
        "052f23f3282aafb3d7f4ed1a7775d6d9ad133954be465e6a1394898e7265d9b3"
    ),
    "perception-occlusion-train-media-01/manifest.json": (
        "469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3"
    ),
    "perception-test-source-audit-01/train_annotations.zip": (
        "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"
    ),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def verify_effective_config(path, expected):
    if sha(path) != expected:
        raise ValueError("effective config changed during preparation")


def peak_bytes():
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (k, ctypes.c_size_t)
                for k in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def limits(start, output):
    if (
        time.perf_counter() - start > 600
        or peak_bytes() > 4 * 2**30
        or (shutil.disk_usage(output).free < 8 * 2**30)
    ):
        raise RuntimeError("fixed body/memory/disk cap exceeded")


def admit_train_rows(rows, clips, *, expected=99):
    """Reject missing/duplicate/media mismatch; select train before opening arrays."""
    if (
        any(r["split"] not in ("train", "valid") for r in rows)
        or len({r["video_id"] for r in rows}) != len(rows)
        or len({r["media_sha256"] for r in rows}) != len(rows)
    ):
        raise ValueError("duplicate/cross-split metadata identity or content")
    train = [r for r in rows if r["split"] == "train"]
    if len(train) != expected or len(clips) != expected:
        raise ValueError("complete native train corpus required")
    ids, hashes = set(), set()
    lookup = {c["video_id"]: c for c in clips}
    if len(lookup) != len(clips):
        raise ValueError("duplicate train media identity")
    for row in train:
        vid, digest = row["video_id"], row["media_sha256"]
        if vid in ids or digest in hashes or vid not in lookup or lookup[vid]["sha256"] != digest:
            raise ValueError("duplicate identity/content or media hash mismatch")
        ids.add(vid)
        hashes.add(digest)
    if ids != set(lookup):
        raise ValueError("train media/feature membership differs")
    return train, lookup


def require_reference(directory, config_sha):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (
        result["status"] != "PASS_ALL_TRAIN_GRID_ROLE_CORPUS_NOT_QUALITY"
        or result["config_sha256"] != config_sha
        or sha(directory / "effective-config.json") != config_sha
        or result["clips"] != 99
        or len(result["rows"]) != 99
        or len({r["video_id"] for r in result["rows"]}) != 99
    ):
        raise ValueError("complete same-config fresh reference required")
    return result


def run(output, reference=None):
    start = time.perf_counter()
    result = dict(
        status="PREFLIGHT",
        started_utc=datetime.now(UTC).isoformat(),
        training=False,
        model_loaded=False,
        gpu_seconds=0,
        cloud_cost=0,
        validation_arrays_or_annotations_opened=False,
        sealed_audit_opened=False,
        rows=[],
    )
    try:
        limits(start, output)
        pins = {str(ROOT / p): h for p, h in PINS.items()}
        for path, pin in pins.items():
            if sha(path) != pin:
                raise ValueError("protected parent changed: " + path)
        bank = json.loads((ROOT / "spatial-context-corpus-01/result.json").read_text())
        cfg = json.loads((ROOT / "spatial-context-corpus-01/config.json").read_text())
        roles_report = json.loads(
            (ROOT / "perception-role-supervision-audit-02/result.json").read_text()
        )
        manifest = json.loads(
            (ROOT / "perception-occlusion-train-media-01/manifest.json").read_text()
        )
        if (
            bank["status"] != "PASS_COMPLETE_SPATIAL_CONTEXT_CORPUS_NOT_QUALITY"
            or bank["config_sha256"] != PINS["spatial-context-corpus-01/config.json"]
            or cfg["grid_size"] != 16
            or cfg["frames"] != 32
            or roles_report["clips"] != 99
        ):
            raise ValueError("completed unchanged native grid and role audit required")
        train, media = admit_train_rows(bank["features"], manifest["clips"])
        source_roles = {r["video_id"]: r["roles_supervision_only"] for r in roles_report["rows"]}
        if set(source_roles) != set(media):
            raise ValueError("role audit train identity mismatch")
        # Verify every train input BEFORE annotation read, never valid array paths.
        for row in train:
            limits(start, output)
            pins[media[row["video_id"]]["path"]] = row["media_sha256"]
            for name, shape, dtype in [
                ("appearance", (32, 256, 384), np.float32),
                ("positions", (32, 256, 3), np.float32),
                ("frames", (32,), np.int64),
            ]:
                info = row["outputs"][name]
                pins[info["path"]] = info["sha256"]
                if sha(info["path"]) != info["sha256"]:
                    raise ValueError("protected native train array changed")
                a = np.load(info["path"], mmap_mode="r", allow_pickle=False)
                if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
                    raise ValueError("native shape/dtype/finiteness mismatch")
            frames = np.load(row["outputs"]["frames"]["path"], allow_pickle=False)
            expected = np.rint(np.linspace(0, row["original_frames"] - 1, 32)).astype(np.int64)
            if not np.array_equal(frames, expected) or frames.tolist() != row["frame_indices"]:
                raise ValueError("changed native original-time sample order")
            positions = np.load(row["outputs"]["positions"]["path"], allow_pickle=False)
            if not np.array_equal(
                positions, spatial_context_positions(frames, source_frames=row["original_frames"])
            ):
                raise ValueError("changed native row-major input cell/time tags")
            if sha(media[row["video_id"]]["path"]) != row["media_sha256"]:
                raise ValueError("native train media changed")
        paths = [
            Path(__file__).resolve(),
            Path("src/tiny_omni_decision/grid_role_supervision.py"),
            Path("src/tiny_omni_decision/mask_pair_association.py"),
            Path("src/tiny_omni_decision/proposal_opportunity.py"),
            Path("src/tiny_omni_decision/spatial_context_features.py"),
            Path("scripts/diagnose_timed_relations.py"),
            Path("docs/pretrained_reuse/R2_GRID_ROLE_CORPUS_PROTOCOL_20261014.md"),
        ]
        pins.update({str(p.resolve()): sha(p) for p in paths})
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_hashes=pins,
            grid_size=16,
            frames=32,
            windows=association_windows(32),
            source_revision=roles_report["source_revision"],
            input_features="immutable whole-grid TAPIR384",
            rule="whole-cell-contained; no positive competing role-box overlap",
            seed=None,
            max_seconds=600,
            peak_bytes_cap=4 * 2**30,
            disk_reserve=8 * 2**30,
            train_ids=[r["video_id"] for r in train],
        )
        write(output / "effective-config.json", config)
        result["config_sha256"] = sha(output / "effective-config.json")
        result["source_commit"] = config["source_commit"]
        reference_rows, reference_pins = {}, {}
        if reference is not None:
            prior = require_reference(reference, result["config_sha256"])
            reference_rows = {r["video_id"]: r for r in prior["rows"]}
            if set(reference_rows) != set(media):
                raise ValueError("fresh reference train membership mismatch")
            reference_pins = {
                str(reference / "result.json"): sha(reference / "result.json"),
                str(reference / "effective-config.json"): result["config_sha256"],
            }
            result["reference_result_sha256"] = reference_pins[str(reference / "result.json")]
        random_before = np.random.get_state()
        annotation = ROOT / "perception-test-source-audit-01/train_annotations.zip"
        with zipfile.ZipFile(annotation) as archive:
            if archive.namelist() != ["all_train.json"]:
                raise ValueError("unexpected train-only annotation archive")
            source = json.loads(archive.read("all_train.json"))
        cells, windows = grid_cells(16), association_windows(32)
        total_roles = dict(
            total=0,
            same_hidden=0,
            same_group=0,
            different_hidden_group=0,
            different_group_group=0,
            same_cover=0,
        )
        total_ledger, total_nodes = None, dict(total=0, unique=0, unknown=0, ambiguous=0)
        for row in train:
            limits(start, output)
            vid = row["video_id"]
            s = source[vid]
            if s["metadata"]["split"] != "train" or s["metadata"]["video_id"] != vid:
                raise ValueError("source train metadata identity mismatch")
            roles = grounded_roles(s)
            if roles != source_roles[vid]:
                raise ValueError("original grounded role mapping changed")
            ids = roles["group"] + roles["hidden"]
            tracks = {i: [t for t in s["object_tracking"] if t["id"] == i] for i in ids}
            if any(len(v) != 1 for v in tracks.values()):
                raise ValueError("unique source task-role tracks required")
            times = np.array(row["frame_indices"], np.int64)
            boxes = np.full((32, len(ids), 4), np.nan, np.float64)
            flags = np.full((32, len(ids), 2), -1, np.int64)
            available = np.zeros((32, len(ids)), bool)
            exact = available.copy()
            distances = np.full(available.shape, -1, np.int64)
            labels, node_counts, frame_counts = (
                [],
                dict(total=0, unique=0, unknown=0, ambiguous=0),
                [],
            )
            support = {str(i): [] for i in ids}
            for t, frame in enumerate(times):
                selected = {}
                for j, identity in enumerate(ids):
                    proxy = source_box_at_time(tracks[identity][0], int(frame))
                    if proxy["box"] is not None:
                        boxes[t, j] = proxy["box"]
                        available[t, j] = True
                        exact[t, j] = proxy["source_frame_exact"]
                        distances[t, j] = proxy["nearest_annotation_distance_frames"]
                        if proxy["source_masked_flags"] is not None:
                            flags[t, j] = proxy["source_masked_flags"]
                        selected[identity] = proxy["box"]
                value, counts = cell_source_labels(cells, selected)
                labels.append(value)
                frame_counts.append(counts)
                for k, v in counts.items():
                    node_counts[k] += v
                for i in ids:
                    support[str(i)].append(int((value == i).sum()))
            labels = np.stack(labels)
            pairs, targets, strata, ledger = labelled_mask_pairs(
                labels.ravel(), np.arange(33, dtype=np.int64) * 256, windows
            )
            counts = pair_role_counts(
                labels.ravel(),
                pairs,
                hidden=roles["hidden"][0],
                group=roles["group"],
                cover=roles["cover"][0],
            )
            arrays = dict(
                source_ids=np.array(ids, np.int64),
                source_boxes=boxes,
                source_available=available,
                source_exact=exact,
                source_masked_flags=flags,
                source_annotation_distance=distances,
                labels=labels,
                cells=cells,
                frames=times,
                pairs=pairs,
                targets=targets,
                strata=strata,
            )
            path = output / (vid + ".npz")
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                for k, a in arrays.items():
                    if (
                        saved[k].dtype != a.dtype
                        or saved[k].shape != a.shape
                        or saved[k].tobytes() != a.tobytes()
                    ):
                        raise ValueError("output round-trip changed: " + k)
            if reference is not None:
                original = reference_rows[vid]
                if (
                    Path(original["path"]).resolve() != (reference / path.name).resolve()
                    or sha(original["path"]) != original["sha256"]
                ):
                    raise ValueError("protected fresh reference array changed")
                reference_pins[original["path"]] = original["sha256"]
                with np.load(original["path"], allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays):
                        raise ValueError("fresh array inventory mismatch")
                    for k, a in arrays.items():
                        if (
                            saved[k].dtype != a.dtype
                            or saved[k].shape != a.shape
                            or saved[k].tobytes() != a.tobytes()
                        ):
                            raise ValueError("fresh array mismatch: " + k)
            hidden_support = np.array(support[str(roles["hidden"][0])])
            result["rows"].append(
                dict(
                    video_id=vid,
                    roles_supervision_only=roles,
                    source_instances=len(ids),
                    input_arrays=row["outputs"],
                    path=str(path),
                    sha256=sha(path),
                    file_bytes=path.stat().st_size,
                    array_inventory={
                        k: dict(shape=list(a.shape), dtype=str(a.dtype)) for k, a in arrays.items()
                    },
                    nodes=node_counts,
                    frames=frame_counts,
                    support_by_instance=support,
                    source_available=int(available.sum()),
                    source_exact=int(exact.sum()),
                    source_interpolated=int((available & ~exact).sum()),
                    source_missing=int((~available).sum()),
                    pair_ledger=ledger,
                    pair_roles=counts,
                    eligible_unique_native_endpoints=len(np.unique(pairs)),
                    hidden_supported_times=int((hidden_support > 0).sum()),
                )
            )
            if total_ledger is None:
                total_ledger = json.loads(json.dumps(ledger))
            else:
                for k in ledger:
                    if k == "strata":
                        for group in ledger[k]:
                            for key in ledger[k][group]:
                                total_ledger[k][group][key] += ledger[k][group][key]
                    else:
                        total_ledger[k] += ledger[k]
            for k in total_nodes:
                total_nodes[k] += node_counts[k]
            for k in total_roles:
                total_roles[k] += counts[k]
            print(
                json.dumps(
                    dict(
                        video=vid,
                        eligible=len(pairs),
                        same_hidden=counts["same_hidden"],
                        hidden_times=int((hidden_support > 0).sum()),
                    )
                ),
                flush=True,
            )
        verify_effective_config(output / "effective-config.json", result["config_sha256"])
        for path, pin in (pins | reference_pins).items():
            if sha(path) != pin:
                raise ValueError("protected source changed at exit: " + path)
        if not all(
            np.array_equal(a, b) for a, b in zip(random_before, np.random.get_state(), strict=True)
        ):
            raise ValueError("global NumPy RNG changed")
        limits(start, output)
        result.update(
            status="PASS_ALL_TRAIN_GRID_ROLE_CORPUS_NOT_QUALITY",
            clips=len(train),
            nodes=total_nodes,
            pair_ledger=total_ledger,
            pair_roles=total_roles,
            scenes_with_same_hidden=sum(r["pair_roles"]["same_hidden"] > 0 for r in result["rows"]),
            source_instances=sum(r["source_instances"] for r in result["rows"]),
            distinct_input_assets=len(train),
            fresh_repeat_bitwise=reference is not None,
            source_visibility_verified=False,
            foreground_membership_verified=False,
            numpy_rng_unchanged=True,
            peak_working_set_bytes=peak_bytes(),
            seconds=time.perf_counter() - start,
            ended_utc=datetime.now(UTC).isoformat(),
            environment=dict(
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform()
            ),
        )
        write(output / "result.json", result)
        return result
    except Exception:
        result.update(
            status="FAIL_INCOMPLETE",
            error=traceback.format_exc(),
            seconds=time.perf_counter() - start,
        )
        write(output / "result.json", result)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    result = run(args.output, args.reference)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
