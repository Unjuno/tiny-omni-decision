"""Build past-only dense windows for the fixed DROID train recordings."""

import argparse
import json
import runpy
import shutil
import time
from pathlib import Path

import numpy as np

from scripts.audit_droid_train_all_transitions import DATA, PARSER, PARSER_SHA, TRAIN
from scripts.audit_droid_train_temporal import (
    OPTIONS,
    RECEIPT_SHA,
    TRAIN_SHA,
    canonical_id,
    digest,
    direction,
)

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
CV = ROOT / "droid-train-temporal-cv-20261027-attempt-01"
CENSUS = ROOT / "droid-train-all-transitions-20261027-attempt-01/summary.json"
PINNED = {
    DATA / "download-receipt.json": RECEIPT_SHA,
    TRAIN: TRAIN_SHA,
    PARSER: PARSER_SHA,
    CV / "outer-train-predictions.jsonl": (
        "f020b2e3bcd0eb36ccc4117b2946a152cd98b5db27b9efc0e954a21214d7d6aa"
    ),
    CV / "result.json": "8b44d08afe358f5df2c37bca93b2a14998ede472acc8d031c2d2e1d11eb12867",
    CENSUS: "bcf23e7d2d92d908cd6181a38e9c3b96ea8566c8f4b1238d9a545f3e1adb18b3",
}
PROTOCOL = Path("docs/pretrained_reuse/R2_DROID_DENSE_TRAIN_WINDOWS_PROTOCOL_20261027.md")
REPO = Path(__file__).resolve().parents[1]


def make_windows(sensor, actions):
    sensor = np.asarray(sensor, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float64)
    if (sensor.ndim != 2 or sensor.shape[1] != 14 or actions.shape != (len(sensor), 6)
            or not np.isfinite(sensor).all() or not np.isfinite(actions).all()):
        raise ValueError("expected finite aligned sensor [steps,14] and action [steps,6]")
    n = max(len(sensor) - 8, 0)
    windows = np.stack([sensor[offset:offset + n] for offset in range(8)], axis=1)
    commands = np.asarray([direction(row[:3]) for row in actions], dtype=np.uint8)
    return windows, commands[7:7 + n], commands[8:8 + n]


def fixed_folds(rows, old_rows, folds):
    if len(rows) != len(old_rows) or len({row["group_id"] for row in rows}) != len(rows):
        raise ValueError("invalid group identities")
    if any(a["id"] != b["id"] or a["group_id"] != b["group_id"]
           for a, b in zip(rows, old_rows, strict=True)):
        raise ValueError("prior fold row identity mismatch")
    assignment = np.full(len(rows), 255, dtype=np.uint8)
    expected = set(range(len(rows)))
    for fold_id, fold in enumerate(folds):
        held, train = fold["held_indices"], fold["train_indices"]
        if (len(set(held)) != len(held) or len(set(train)) != len(train)
                or set(held) & set(train) or set(held) | set(train) != expected):
            raise ValueError("invalid outer group partition")
        if any(assignment[i] != 255 for i in held):
            raise ValueError("recording held out more than once")
        assignment[held] = fold_id
    if len(folds) != 5 or np.any(assignment == 255):
        raise ValueError("incomplete outer group assignment")
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(REPO):
        raise ValueError("new external output directory required")
    if any(digest(path) != value for path, value in PINNED.items()):
        raise ValueError("pinned input mismatch")
    if shutil.disk_usage(out.parent).free < 1024**3 + 128 * 1024**2:
        raise ValueError("insufficient disk reserve")
    train = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines()]
    old_rows = [json.loads(line) for line in
                (CV / "outer-train-predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    folds = json.loads((CV / "result.json").read_bytes())["folds"]
    if len(train) != 275 or any(row["split"] != "train" for row in train):
        raise ValueError("unexpected train cohort")
    assignments = fixed_folds(train, old_rows, folds)
    lengths = [max(int(row["source_step_count"]) - 8, 0) for row in train]
    offsets = np.cumsum([0, *lengths], dtype=np.int64)
    census = json.loads(CENSUS.read_bytes())
    if int(offsets[-1]) != census["transition_pairs"] or int(offsets[-1]) != 80998:
        raise ValueError("fixed train row count mismatch")
    api = runpy.run_path(str(PARSER), run_name="pinned_droid_parser")
    receipt = json.loads((DATA / "download-receipt.json").read_bytes())
    by_id = {row["id"]: i for i, row in enumerate(train)}
    if len(by_id) != len(train):
        raise ValueError("duplicate recording IDs")
    out.mkdir(parents=True)
    n = int(offsets[-1])
    specs = {"sensor": ((n, 8, 14), "float32"), "previous": ((n,), "uint8"),
             "target": ((n,), "uint8"), "group": ((n,), "uint16"),
             "step": ((n,), "uint32")}
    arrays = {name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+",
                                             dtype=dtype, shape=shape)
              for name, (shape, dtype) in specs.items()}
    fields = (("steps/observation/joint_position", 7),
              ("steps/observation/cartesian_position", 6),
              ("steps/observation/gripper_position", 1))
    seen, skipped, shard_count = set(), 0, 0
    started = time.perf_counter()
    for item in receipt["objects"]:
        if ".tfrecord-" not in item["name"]:
            continue
        path = DATA / item["name"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("source shard mismatch")
        shard_count += 1
        for raw in api["iter_tfrecord"](path):
            context = api["episode_example_context"](raw)
            identifier = canonical_id(context["episode_metadata/file_path"])
            if identifier not in by_id:
                skipped += 1
                continue
            if identifier in seen:
                raise ValueError("duplicate train episode")
            index = by_id[identifier]
            row = train[index]
            if canonical_id(context["episode_metadata/recording_folderpath"]) != row["group_id"]:
                raise ValueError("recording identity mismatch")
            steps = int(row["source_step_count"])
            parts = [np.asarray(api["episode_example_float_values"](raw, key), dtype=np.float32)
                     .reshape(-1, width) for key, width in fields]
            actions = np.asarray(api["episode_example_float_values"](
                raw, "steps/action_dict/cartesian_velocity"), dtype=np.float64).reshape(-1, 6)
            if any(len(part) != steps for part in parts) or len(actions) != steps:
                raise ValueError("source step count mismatch")
            if OPTIONS[direction(actions[row["target_index"], :3])] != row["target_option_id"]:
                raise ValueError("original fixed target mismatch")
            windows, previous, target = make_windows(np.concatenate(parts, axis=1), actions)
            sl = slice(int(offsets[index]), int(offsets[index + 1]))
            arrays["sensor"][sl] = windows
            arrays["previous"][sl] = previous
            arrays["target"][sl] = target
            arrays["group"][sl] = index
            arrays["step"][sl] = np.arange(8, steps, dtype=np.uint32)
            seen.add(identifier)
        print(f"processed {shard_count} shards; {len(seen)} train recordings", flush=True)
    if seen != set(by_id):
        raise ValueError("missing train recordings")
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (arrays["previous"], arrays["target"]), 1)
    if matrix.tolist() != census["transition_matrix_rows_previous_columns_next"]:
        raise ValueError("dense labels disagree with pinned prior transition census")
    for array in arrays.values():
        array.flush()
    del arrays
    if any(digest(path) != value for path, value in PINNED.items()):
        raise ValueError("pinned inputs changed")
    groups = [{"id": row["id"], "group_id": row["group_id"], "fold": int(assignments[i]),
               "start": int(offsets[i]), "end": int(offsets[i + 1])}
              for i, row in enumerate(train)]
    (out / "private-groups.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    hashes = {p.name: digest(p) for p in sorted(out.glob("*.npy"))}
    hashes["private-groups.json"] = digest(out / "private-groups.json")
    result = {"status": "TRAIN_WINDOWS_BUILT_NOT_EVALUATED", "rows": n,
              "recordings": len(groups), "changed_rows": n - int(np.trace(matrix)),
              "fold_recordings": [int(np.sum(assignments == i)) for i in range(5)],
              "skipped_opaque_records": skipped, "shards": shard_count,
              "input_hashes": {str(path): value for path, value in PINNED.items()},
              "output_hashes": hashes, "protocol_sha256": digest(PROTOCOL),
              "script_sha256": digest(Path(__file__)),
              "wall_seconds": time.perf_counter() - started,
              "limitations": "correlated train windows; validation and product quality unverified"}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
