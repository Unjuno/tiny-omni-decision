"""Build a training-only HARTH sensor-window smoke with separate target storage."""

import csv
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.range_generalization import sha, write
from tiny_omni_decision.observations import ObservationBatch
from tiny_omni_decision.sensor_windows import contiguous_windows

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SOURCE = ROOT / "r2m-source-readiness-01/harth-uci-v12"
ARCHIVE_SHA = "2ab54d6b11467ddbe998dfb60355f9f1ae181b0af7a89e7c78bb7af2cf16b64a"
SPLIT_SHA = "0259363d0dde5255a7c2ba2925959aad12df78007c60d26db855b009fca04695"
CHANNELS = ("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z")
CODES = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)


def main():
    output = ROOT / "harth-native-window-smoke-02"
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    archive = SOURCE / "harth.zip"
    split_file = SOURCE / "frozen-subject-splits.json"
    if sha(archive) != ARCHIVE_SHA or sha(split_file) != SPLIT_SHA:
        raise ValueError("immutable sensor source or split changed")
    splits = json.loads(split_file.read_text())["splits"]
    assert len(set(sum(splits.values(), []))) == 22
    member = splits["train"][0]
    with zipfile.ZipFile(archive) as z:
        raw = z.read(member)
    records = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    anchor = datetime.fromisoformat(records[0]["timestamp"])
    times = np.array(
        [(datetime.fromisoformat(r["timestamp"]) - anchor).total_seconds() * 1000 for r in records],
        dtype=np.float64,
    )
    values = np.array([[float(r[c]) for c in CHANNELS] for r in records], dtype=np.float32)
    labels = np.array([int(r["label"]) for r in records], dtype=np.int16)
    if not np.isfinite(values).all() or not set(labels.tolist()).issubset(CODES):
        raise ValueError("invalid source values or unknown label")
    windows = contiguous_windows(
        times.tolist(), [True] * len(times), length=128, stride=64, max_gap_ms=30.0
    )
    if not windows:
        raise ValueError("no valid windows")
    xs = np.stack([values[a:b] for a, b in windows])
    ts = np.stack([times[a:b] for a, b in windows])
    ys = np.stack([labels[a:b] for a, b in windows])
    # No label-based window exclusion; mixed-activity windows remain intact.
    assert xs.shape == (len(windows), 128, 6)
    assert (np.diff(ts, axis=1) <= 30).all()
    output.mkdir()
    feature = output / "observation-features.npz"
    target = output / "supervision.npz"
    np.savez_compressed(feature, values=xs, timestamps_ms=ts)
    np.savez_compressed(target, label_sequence=ys, source_intervals=np.array(windows))
    request = ObservationBatch.model_validate(
        dict(
            schema_version="1.0",
            observations=[
                dict(
                    modality="sensor",
                    source_id=member,
                    clock_id=f"{member}:relative",
                    payload_ref=str(feature),
                    payload_index=0,
                    payload_sha256=sha(feature),
                    timestamps_ms=ts[0].tolist(),
                    valid_mask=[True] * 128,
                    channels=[dict(name=c, unit="g", scale=1.0, offset=0.0) for c in CHANNELS],
                )
            ],
            options=[
                dict(option_id=f"activity-{i}", attributes=dict(activity_code=code))
                for i, code in enumerate(CODES)
            ],
        )
    )
    (output / "first-window-request.json").write_text(
        request.model_dump_json(indent=2), encoding="utf-8"
    )
    with np.load(feature, allow_pickle=False) as reloaded:
        assert set(reloaded.files) == {"values", "timestamps_ms"}
        assert np.array_equal(reloaded["values"], xs) and np.array_equal(
            reloaded["timestamps_ms"], ts
        )
    result = dict(
        status="PREPROCESSING_SMOKE_PASS",
        training=False,
        member=member,
        member_sha256=hashlib.sha256(raw).hexdigest(),
        archive_sha256=ARCHIVE_SHA,
        split_sha256=SPLIT_SHA,
        source_samples=len(values),
        windows=len(windows),
        window_length=128,
        stride=64,
        max_gap_ms=30.0,
        gap_count=int((np.diff(times) > 30).sum()),
        mixed_label_windows=int(sum(len(set(v.tolist())) > 1 for v in ys)),
        feature_shape=list(xs.shape),
        feature_sha256=sha(feature),
        target_sha256=sha(target),
        clock_origin=anchor.isoformat(),
        timezone="unspecified by source; no cross-subject clock alignment",
        input_units="g",
        validation_calibration_audit_opened=False,
        roundtrip_exact=True,
        seconds=time.perf_counter() - started,
        limitation="first train subject only; no accuracy or multimodal fusion claim",
    )
    write(output / "result.json", result)
    write(Path("docs/pretrained_reuse/R2M_SENSOR_WINDOW_SMOKE_V2_20261012.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
