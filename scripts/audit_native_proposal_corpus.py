"""Independent native corpus CPU audit, including source RGB; never open labels."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tiny_omni_decision.native_proposal_audit import audit_native_arrays, audit_report_coverage

FIXED_IDS = (
    "video_10103",
    "video_10186",
    "video_10212",
    "video_10387",
    "video_1087",
    "video_10886",
    "video_10894",
    "video_10903",
)
CONFIG_PIN = "5d4417e13d8c2a4f27593d3149ac4167e7d8ee13ff9c87a076ff9780f60aa79f"
STATE_PIN = "658e3aaff3d08b6cc8337a1f125e5ee55451ed3bfec38f9def601a14bb6912d8"
WEIGHT_PIN = "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def main():
    if not __debug__:
        raise RuntimeError("audit checks require normal non-optimized Python")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--fresh-replay", type=Path)
    args = parser.parse_args()
    if not args.snapshot and not args.fresh_replay:
        parser.error("full admission requires completed fresh-frame replay")
    args.output.mkdir(exist_ok=False)
    import av

    tick = time.perf_counter()
    cfg_path = args.input / "effective-config.json"
    assert sha(cfg_path) == CONFIG_PIN, "fixed native corpus config changed"
    cfg = json.loads(cfg_path.read_text())
    source_path = args.input / ("progress.json" if args.snapshot else "result.json")
    capture = source_path.read_bytes()
    captured_at = datetime.now(UTC).isoformat()
    r = json.loads(capture)
    (args.output / "captured-input-report.json").write_bytes(capture)
    assert r["status"] == ("RUNNING" if args.snapshot else "PASS_NATIVE_EIGHT_CLIP_PROPOSAL_CORPUS")
    assert not r["smoke"] and not r["replay"] and not r["training"] and not r["annotations_loaded"]
    assert not r["validation_opened"] and not r["sealed_audit_opened"]
    assert r["config_sha256"] == CONFIG_PIN and r["parameters"] == 38962498
    assert r["model_state_before"] == STATE_PIN and cfg["weight_sha256"] == WEIGHT_PIN
    assert cfg["decoder_object_batch"] == 1 and not cfg["batched_head_path_used"]
    assert cfg["grid_side"] == 32 and cfg["objects_per_native_session"] == 64
    assert cfg["mask_threshold"] == 0 and cfg["nms_iou"] == 0.7 and cfg["top_k"] is None
    plans = cfg["plans"]
    assert [p["video_id"] for p in plans] == list(FIXED_IDS)
    for p in plans:
        assert p["split"] == "train" and p["cut_frame"] in (-1, p["frames"])
        assert (
            p["frame_indices"] == np.rint(np.linspace(0, p["frames"] - 1, 32)).astype(int).tolist()
        )
    expected = [(p["video_id"], t) for p in plans for t in p["frame_indices"]]
    observed = [(x["video_id"], x["frame_index"]) for x in r["rows"]]
    audit_report_coverage(r, cfg, snapshot=args.snapshot)
    if not args.snapshot:
        assert observed == expected and r["model_state_after"] == STATE_PIN
        assert (
            r["frozen_state_unchanged"] and r["cpu_cuda_rng_unchanged"] and r["sources_unchanged"]
        )
    for path, pin in cfg["source_hashes"].items():
        assert sha(path) == pin, "protected source changed: " + path
    parents = {
        (p["video_id"], p["frame_index"]): {
            k: v for k, v in p.items() if k not in ("video_id", "frame_index")
        }
        for p in cfg["reused_records"]
    }
    assert len(parents) == 39 and len(expected) == 256
    audit_rows = []
    for row in r["rows"]:
        pair = (row["video_id"], row["frame_index"])
        assert row["parent"] == parents.get(pair)
        assert row["raw_proposals"] == 1024 and row["readback_bitwise"]
        assert row["encoder_calls"] == (0 if row["parent"] else 1)
        assert sha(row["path"]) == row["sha256"]
        assert row["input_shape"][2] == 3
        with np.load(row["path"], allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
            checked = audit_native_arrays(
                arrays, height=row["input_shape"][0], width=row["input_shape"][1]
            )
            if row["parent"]:
                parent = row["parent"]
                assert sha(parent["path"]) == parent["sha256"]
                with np.load(parent["path"], allow_pickle=False) as original:
                    for key in arrays:
                        assert arrays[key].shape == original[key].shape
                        assert arrays[key].dtype == original[key].dtype
                        assert arrays[key].tobytes() == original[key].tobytes()
            assert checked["nms_count"] == row["nms_count"]
        audit_rows.append(dict(video_id=pair[0], frame_index=pair[1], **checked))
    assert r["encoder_calls_total"] == sum(x["encoder_calls"] for x in r["rows"])
    for plan in plans:
        rows = {x["frame_index"]: x for x in r["rows"] if x["video_id"] == plan["video_id"]}
        if not rows:
            continue
        assert sha(plan["media_path"]) == plan["media_sha256"]
        checked_times = []
        with av.open(plan["media_path"]) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for index, frame in enumerate(container.decode(stream)):
                if index > max(rows):
                    break
                if index in rows:
                    rgb = frame.to_ndarray(format="rgb24")
                    assert list(rgb.shape) == rows[index]["input_shape"]
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == rows[index]["rgb_sha256"]
                    checked_times.append(index)
        assert checked_times == list(rows)
    fresh_pins = None
    if not args.snapshot:
        fpath = args.fresh_replay / "result.json"
        fresh_bytes = fpath.read_bytes()
        fresh = json.loads(fresh_bytes)
        assert (
            sha(args.fresh_replay / "effective-config.json") == fresh["config_sha256"] == CONFIG_PIN
        )
        assert (
            fresh["status"] == "PASS_NATIVE_CORPUS_FRESH_FRAME_REPLAY"
            and fresh["fresh_replay_bitwise"]
        )
        assert fresh["replay_result_sha256"] == hashlib.sha256(capture).hexdigest()
        assert fresh["model_state_before"] == fresh["model_state_after"] == STATE_PIN
        assert (
            fresh["frozen_state_unchanged"]
            and fresh["cpu_cuda_rng_unchanged"]
            and fresh["sources_unchanged"]
        )
        assert len(fresh["rows"]) == 1
        fr = fresh["rows"][0]
        target = r["rows"][1]
        assert (fr["video_id"], fr["frame_index"]) == (FIXED_IDS[0], plans[0]["frame_indices"][1])
        assert fr["rgb_sha256"] == target["rgb_sha256"] and sha(fr["path"]) == fr["sha256"]
        with (
            np.load(fr["path"], allow_pickle=False) as a,
            np.load(target["path"], allow_pickle=False) as b,
        ):
            assert set(a.files) == set(b.files)
            for key in a:
                assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                assert a[key].tobytes() == b[key].tobytes()
        fresh_pins = dict(
            result_sha256=hashlib.sha256(fresh_bytes).hexdigest(),
            config_sha256=CONFIG_PIN,
            all_five_arrays_exact=True,
        )
        assert sha(fpath) == fresh_pins["result_sha256"]
        assert sha(args.fresh_replay / "effective-config.json") == CONFIG_PIN
        assert sha(source_path) == hashlib.sha256(capture).hexdigest()
    for row in r["rows"]:
        assert sha(row["path"]) == row["sha256"]
    for path, pin in cfg["source_hashes"].items():
        assert sha(path) == pin
    assert sha(cfg_path) == CONFIG_PIN
    write(
        args.output / "result.json",
        dict(
            status="PASS_NATIVE_SAVED_PREFIX_AUDIT_NOT_COMPLETE"
            if args.snapshot
            else "PASS_NATIVE_FULL_CORPUS_INPUT_ADMISSION_NOT_QUALITY",
            captured_at=captured_at,
            ended_at=datetime.now(UTC).isoformat(),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            snapshot=args.snapshot,
            snapshot_does_not_prove_process_liveness=True,
            input_report_sha256=hashlib.sha256(capture).hexdigest(),
            config_sha256=CONFIG_PIN,
            script_sha256=sha(__file__),
            helper_sha256=sha(Path(inspect_helper_path())),
            rows=audit_rows,
            frames_checked=len(observed),
            raw_proposals_checked=len(observed) * 1024,
            nms_survivors=sum(x["nms_count"] for x in audit_rows),
            media_assets_checked=len({v for v, _ in observed}),
            reused_rows=sum(x["parent"] is not None for x in r["rows"]),
            new_rows=sum(x["parent"] is None for x in r["rows"]),
            source_rgb_all_exact=True,
            source_hashes_checked=True,
            fresh_replay=fresh_pins,
            full_corpus_complete=not args.snapshot,
            unique_objects_known=False,
            quality_evaluated=False,
            annotations_loaded=False,
            gpu_or_model_loaded=False,
            training=False,
            seconds=time.perf_counter() - tick,
        ),
    )
    print(
        "AUDIT_SAVED", len(observed), "FRAMES", "SNAPSHOT" if args.snapshot else "FULL", flush=True
    )


def inspect_helper_path():
    import inspect

    return inspect.getfile(audit_native_arrays)


if __name__ == "__main__":
    main()
