"""All-point input-only retrieval before separate training-source bbox diagnosis."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from scripts.probe_track_appearance import IDS, ROOT, array_sha, sha, write
from tiny_omni_decision.track_association import (
    association_summary,
    descriptor_matches,
    group_membership,
)

PRIMITIVE_PIN = "b395e18d6b57262197c8ac04133a0a10b12dd94698f0b1680bf1ef892088e7fb"
ANNOTATION_PIN = "9eaa6aad3fabdf4cde4955f333cfddf3b9e6839c5abcb271afa410870f640d78"


def run_matches(actual, initial):
    if actual.shape != initial.shape or actual.ndim != 3 or actual.shape[1] < 2:
        raise ValueError("aligned complete temporal descriptor banks required")
    query = actual[:, 0].copy()
    return dict(
        actual=descriptor_matches(query, actual),
        static=descriptor_matches(query, initial),
        misaligned=descriptor_matches(query, np.roll(actual, 64, axis=0)),
        fixed_index=dict(
            indices=np.broadcast_to(np.arange(len(actual))[:, None], actual.shape[:2])
        ),
    )


def verify_arrays(path, values):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(values):
            raise ValueError("saved prediction/label keys changed")
        for key, value in values.items():
            old = saved[key]
            if (
                old.shape != value.shape
                or old.dtype != value.dtype
                or old.tobytes() != value.tobytes()
            ):
                raise ValueError("saved prediction/label bytes changed: " + key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = dict(
        status="PREFLIGHT",
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        rows=[],
        annotations_opened_after_all_predictions=False,
        training=False,
        device="cpu",
        models_loaded=False,
        new_parameters=0,
        cloud_cost=0,
        validation_opened=False,
        sealed_audit_opened=False,
    )
    try:
        env = {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        }
        if any(v != "4" for v in env.values()):
            raise ValueError(
                "set all three BLAS thread environment variables to4 before process launch"
            )
        if shutil.disk_usage(args.output).free < 8 * 1024**3 + 32 * 1024**2:
            raise OSError("8GiB disk reserve mandatory")
        primitive_root = ROOT / "track-appearance-primitive-01"
        if sha(primitive_root / "result.json") != PRIMITIVE_PIN:
            raise ValueError("completed primitive changed")
        primitive = json.loads((primitive_root / "result.json").read_text())
        if (
            primitive["status"] != "PASS_TRACK_APPEARANCE_PRIMITIVE_NOT_QA_QUALITY"
            or [r["video_id"] for r in primitive["rows"]] != IDS
        ):
            raise ValueError("exact eight completed training references mandatory")
        if sha(primitive_root / "config.json") != primitive["config_sha256"]:
            raise ValueError("primitive config changed")
        annotation_path = ROOT / "perception-test-source-audit-01/train_annotations.zip"
        if sha(annotation_path) != ANNOTATION_PIN:
            raise ValueError("training annotations changed")
        reference = None
        config = dict(
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            runner_sha256=sha(Path(__file__)),
            helper_sha256=sha(Path("src/tiny_omni_decision/track_association.py")),
            primitive_result_sha256=PRIMITIVE_PIN,
            primitive_config_sha256=primitive["config_sha256"],
            annotation_sha256=ANNOTATION_PIN,
            ids=IDS,
            seed_search=False,
            threads_configured=env,
            actual_blas_thread_count_measured=False,
            numpy=np.__version__,
            python=platform.python_version(),
            platform=platform.platform(),
            query="all256 actual descriptors at sampled time0",
            candidates="all256 per time; no filter",
            retrieval="FP64 cosine; exact first-index ties; zero vectors retained",
            misalignment_roll=64,
            source_group="exact grounded similar/identical-object IDs; posthoc only",
            bbox_proxy="linear bbox interpolation inside annotated range; no extrapolation",
            group_coverage_policy="all group ranges required to resolve any point identity",
            primary_denominator="initial unique-group queries x31 times; future unknown retained",
            max_seconds=300,
            time_limit="between-clip/final admission checks; no watchdog",
            disk_reserve_bytes=8 * 1024**3,
            reference_result_sha256=sha(args.reference / "result.json") if args.reference else None,
        )
        write(args.output / "config.json", config)
        from scripts.train_spatial_trajectory_readout import process_peak_bytes

        report.update(
            status="INPUT_ONLY_PREDICTION",
            source_commit=config["source_commit"],
            config_sha256=sha(args.output / "config.json"),
        )
        rng = np.random.get_state()
        input_hashes = {
            str(primitive_root / "result.json"): PRIMITIVE_PIN,
            str(primitive_root / "config.json"): primitive["config_sha256"],
            str(annotation_path): ANNOTATION_PIN,
        }

        def cap():
            if time.perf_counter() - start > 300:
                raise TimeoutError("300s admission limit exceeded; diagnostic incomplete")

        for row in primitive["rows"]:
            cap()
            values = {}
            for name, info in row["outputs"].items():
                if sha(info["path"]) != info["sha256"]:
                    raise ValueError("primitive array changed")
                input_hashes[info["path"]] = info["sha256"]
                value = np.load(info["path"], allow_pickle=False)
                if list(value.shape) != info["shape"] or str(value.dtype) != info["dtype"]:
                    raise ValueError("primitive shape/dtype changed")
                values[name] = value
            actual, initial = values["actual_appearance"], values["initial_position_appearance"]
            if (
                actual.shape != (256, 32, 384)
                or actual.dtype != np.float32
                or initial.dtype != np.float32
            ):
                raise ValueError("fixed point/time/channel bank changed")
            if (
                actual[:, 0].tobytes() != initial[:, 0].tobytes()
                or values["estimated_xy"].shape != (256, 32, 2)
                or values["frame_indices"].tolist() != row["frame_indices"]
            ):
                raise ValueError("common initial queries/point/time binding changed")
            before = {k: array_sha(v) for k, v in values.items()}
            matches = run_matches(actual, initial)
            repeated = run_matches(actual, initial)
            arrays = {
                arm + "_" + key: value
                for arm, items in matches.items()
                for key, value in items.items()
            }
            if any(
                arrays[arm + "_" + key].tobytes() != value.tobytes()
                for arm, items in repeated.items()
                for key, value in items.items()
            ):
                raise ValueError("retrieval repeat changed")
            if any(array_sha(v) != before[k] for k, v in values.items()):
                raise ValueError("primitive source arrays mutated")
            path = args.output / (row["video_id"] + "-predictions.npz")
            np.savez(path, **arrays)
            verify_arrays(path, arrays)
            report["rows"].append(
                dict(
                    video_id=row["video_id"],
                    prediction_path=str(path),
                    prediction_sha256=sha(path),
                    all_query_point_times=8192,
                    predictions_saved_before_annotations=True,
                    zero_query_count=int(matches["actual"]["zero_query"].sum()),
                    tie_counts={
                        arm: int((items["ties"] > 1).sum())
                        for arm, items in matches.items()
                        if "ties" in items
                    },
                )
            )
            write(args.output / "progress.json", report)
            print(row["video_id"], "ALL_QUERY_PREDICTIONS_SAVED", flush=True)
            del values, actual, initial, matches, repeated, arrays, value
        prediction_seconds = time.perf_counter() - start
        # Replay also saves every new prediction before reading prior posthoc
        # group IDs, labels or metrics.
        if args.reference:
            reference = json.loads((args.reference / "result.json").read_text())
            if (
                reference["status"] != "PASS_TRAIN_ONLY_BBOX_ASSOCIATION_DIAGNOSIS"
                or [r["video_id"] for r in reference["rows"]] != IDS
            ):
                raise ValueError("complete same-identity reference mandatory")
            if sha(args.reference / "config.json") != reference["config_sha256"]:
                raise ValueError("reference config changed")
            reference_config = json.loads((args.reference / "config.json").read_text())
            for key in (
                "runner_sha256",
                "helper_sha256",
                "primitive_result_sha256",
                "annotation_sha256",
                "misalignment_roll",
                "bbox_proxy",
                "group_coverage_policy",
            ):
                if reference_config[key] != config[key]:
                    raise ValueError("reference execution policy changed")
            for old, current in zip(reference["rows"], report["rows"], strict=True):
                if sha(old["prediction_path"]) != old["prediction_sha256"]:
                    raise ValueError("reference prediction changed")
                with np.load(current["prediction_path"], allow_pickle=False) as saved:
                    verify_arrays(old["prediction_path"], {k: saved[k] for k in saved.files})
        report.update(status="POSTHOC_GEOMETRY", annotations_opened_after_all_predictions=True)
        with zipfile.ZipFile(annotation_path) as archive:
            if archive.namelist() != ["all_train.json"]:
                raise ValueError("unexpected training source archive")
            source = json.loads(archive.read("all_train.json"))
        for index, (saved_row, primitive_row) in enumerate(
            zip(report["rows"], primitive["rows"], strict=True)
        ):
            cap()
            vid = saved_row["video_id"]
            source_row = source[vid]
            if (
                source_row["metadata"]["split"] != "train"
                or source_row["metadata"]["video_id"] != vid
            ):
                raise ValueError("training source identity changed")
            groups = [
                q
                for q in source_row["grounded_question"]
                if q["question"].endswith("Track these similar or identical objects.")
            ]
            if len(groups) != 1 or not isinstance(groups[0]["answers"], list):
                raise ValueError("exact grounded group mandatory; no flag substitution")
            group_ids = groups[0]["answers"]
            xy = np.load(primitive_row["outputs"]["estimated_xy"]["path"], allow_pickle=False)
            times = np.load(primitive_row["outputs"]["frame_indices"]["path"], allow_pickle=False)
            actual_labels, available = group_membership(
                source_row["object_tracking"], group_ids, xy, times
            )
            static_labels, static_available = group_membership(
                source_row["object_tracking"],
                group_ids,
                np.broadcast_to(xy[:, :1], xy.shape),
                times,
            )
            if not np.array_equal(actual_labels[:, 0], static_labels[:, 0]) or not np.array_equal(
                available, static_available
            ):
                raise ValueError("posthoc common initial labels/source coverage changed")
            labels = dict(
                actual_labels=actual_labels,
                static_labels=static_labels,
                group_ids=np.asarray(group_ids, np.int64),
                box_range_available=available,
                frame_indices=times,
            )
            label_path = args.output / (vid + "-posthoc-labels.npz")
            np.savez(label_path, **labels)
            verify_arrays(label_path, labels)
            if reference:
                original = reference["rows"][index]
                if sha(original["label_path"]) != original["label_sha256"]:
                    raise ValueError("reference labels changed")
                verify_arrays(original["label_path"], labels)
            metrics, per_group = {}, {}
            with np.load(saved_row["prediction_path"], allow_pickle=False) as predictions:
                for arm in ("actual", "static", "misaligned", "fixed_index"):
                    indices = predictions[arm + "_indices"]
                    candidates = static_labels if arm == "static" else actual_labels
                    metrics[arm] = association_summary(actual_labels[:, 0], candidates, indices)
                    per_group[arm] = {}
                    for identity in group_ids:
                        select = actual_labels[:, 0] == identity
                        per_group[arm][str(identity)] = (
                            association_summary(
                                actual_labels[select, 0], candidates, indices[select]
                            )
                            if select.any()
                            else dict(initial_group_queries=0, future_denominator=0, future_hits=0)
                        )
            saved_row.update(
                group_ids=group_ids,
                source_range_available_counts=available.sum(1).tolist(),
                initial_group_queries={
                    str(i): int((actual_labels[:, 0] == i).sum()) for i in group_ids
                },
                actual_label_unknown_point_times=int((actual_labels == -1).sum()),
                actual_label_ambiguous_point_times=int((actual_labels == -2).sum()),
                label_path=str(label_path),
                label_sha256=sha(label_path),
                metrics=metrics,
                per_group=per_group,
            )
            if reference and any(
                saved_row[k] != reference["rows"][index][k]
                for k in ("metrics", "per_group", "group_ids", "initial_group_queries")
            ):
                raise ValueError("fresh posthoc metrics changed")
            write(args.output / "progress.json", report)
            print(
                vid,
                "POSTHOC_BBOX_RATES",
                {a: m["future_same_group_bbox_rate"] for a, m in metrics.items()},
                flush=True,
            )
        aggregate = {}
        for arm in ("actual", "static", "misaligned", "fixed_index"):
            values = [r["metrics"][arm] for r in report["rows"]]
            counts = {
                k: sum(v[k] for v in values)
                for k in (
                    "all_query_point_times",
                    "initial_group_queries",
                    "initial_unknown_queries",
                    "initial_ambiguous_queries",
                    "future_denominator",
                    "future_hits",
                    "future_resolved_predictions",
                    "future_unknown_predictions",
                    "future_ambiguous_predictions",
                    "time0_hits",
                )
            }
            valid = [
                v["future_same_group_bbox_rate"]
                for v in values
                if v["future_same_group_bbox_rate"] is not None
            ]
            aggregate[arm] = dict(
                **counts,
                future_same_group_bbox_rate=counts["future_hits"] / counts["future_denominator"]
                if counts["future_denominator"]
                else None,
                future_conditional_resolved_hit_rate=counts["future_hits"]
                / counts["future_resolved_predictions"]
                if counts["future_resolved_predictions"]
                else None,
                macro_clip_future_bbox_rate=float(np.mean(valid)) if valid else None,
                clips_with_initial_group_queries=len(valid),
                clips_total=8,
            )
        for path, pin in input_hashes.items():
            if sha(path) != pin:
                raise ValueError("protected source changed during diagnosis")
        rng_after = np.random.get_state()
        if (
            rng[0] != rng_after[0]
            or rng[2:] != rng_after[2:]
            or rng[1].tobytes() != rng_after[1].tobytes()
        ):
            raise ValueError("retrieval changed NumPy RNG")
        cap()
        report.update(
            status="PASS_TRAIN_ONLY_BBOX_ASSOCIATION_DIAGNOSIS",
            aggregate=aggregate,
            source_files_unchanged=True,
            numpy_rng_unchanged=True,
            fresh_replay=bool(reference),
            input_only_prediction_seconds=prediction_seconds,
            input_hashes=input_hashes,
            peak_process_working_set_bytes=process_peak_bytes(),
            limitation="Reused train scenes/sparse interpolated boxes; not semantic or QA accuracy",
        )
    except Exception:
        report.update(status="FAIL_INCOMPLETE", error=traceback.format_exc())
        raise
    finally:
        report.update(seconds=time.perf_counter() - start, ended_at=datetime.now(UTC).isoformat())
        write(args.output / "result.json", report)


if __name__ == "__main__":
    main()
