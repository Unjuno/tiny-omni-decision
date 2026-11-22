"""Extract train-only HapTile instruction and causal robot-state features."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.pretrained_reuse.analyze_haptile_train_preflight import (  # noqa: E402
    validate_train_receipt,
)
from tiny_omni_decision.haptile_state import (  # noqa: E402
    STREAM_WIDTHS,
    causal_state_sequence,
)

PROTO_TOPICS = ("/instruction", "/end-effector-pose")
JSON_TOPICS = (
    "/end-effector-velocity.plot",
    "/joints.plot",
    "/joint-velocities.plot",
    "/gripper.plot",
    "/tactile.plot",
)
JSON_FIELDS = {
    "/end-effector-velocity.plot": ("vx", "vy", "vz", "wx", "wy", "wz"),
    "/joints.plot": ("base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"),
    "/joint-velocities.plot": (
        "base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3",
    ),
    "/gripper.plot": ("position",),
    "/tactile.plot": ("left", "right", "normalized"),
}
TOPIC_TO_STREAM = {
    "/end-effector-velocity.plot": "end_effector_velocity",
    "/joints.plot": "joints",
    "/joint-velocities.plot": "joint_velocities",
    "/gripper.plot": "gripper",
    "/tactile.plot": "tactile",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_one(path: Path, query_time_s: float) -> tuple[str, np.ndarray]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError("install the pinned MCAP decoder dependencies") from exc

    times: dict[str, list[float]] = {name: [] for name in STREAM_WIDTHS}
    values: dict[str, list[list[float]]] = {name: [] for name in STREAM_WIDTHS}
    instructions: list[tuple[float, str]] = []
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=list(PROTO_TOPICS)
        ):
            timestamp = message.log_time / 1_000_000_000
            if timestamp > query_time_s:
                continue
            if channel.topic == "/instruction":
                if schema is None or schema.name != "Instructions":
                    raise ValueError(f"unexpected instruction schema in {path}")
                instruction = str(decoded.data).strip()
                if instruction:
                    instructions.append((timestamp, instruction))
            elif channel.topic == "/end-effector-pose":
                if schema is None or schema.name != "foxglove.PoseInFrame":
                    raise ValueError(f"unexpected pose schema in {path}")
                point = decoded.pose.position
                times["pose"].append(timestamp)
                values["pose"].append([float(point.x), float(point.y), float(point.z)])
            else:
                raise ValueError(f"unexpected protobuf topic in {path}: {channel.topic}")

    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        for _, channel, message in reader.iter_messages(topics=list(JSON_TOPICS)):
            timestamp = message.log_time / 1_000_000_000
            if timestamp > query_time_s:
                continue
            fields = JSON_FIELDS[channel.topic]
            payload = json.loads(message.data)
            if not isinstance(payload, dict) or any(key not in payload for key in fields):
                raise ValueError(f"malformed numeric state payload on {channel.topic}: {path}")
            vector = [float(payload[key]) for key in fields]
            stream_name = TOPIC_TO_STREAM[channel.topic]
            times[stream_name].append(timestamp)
            values[stream_name].append(vector)

    if not instructions:
        raise ValueError(f"no causal task instruction is present in {path}")
    instruction = max(instructions, key=lambda row: row[0])[1]
    streams = {
        "pose": (times["pose"], values["pose"]),
        **{
            name: (times[name], values[name])
            for name in STREAM_WIDTHS if name != "pose"
        },
    }
    sequence = causal_state_sequence(streams, query_time_s)
    return instruction, sequence


def extract_train_state_text(
    *, receipt_path: Path, action_path: Path, split_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    try:
        from mcap.reader import make_reader  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("install the pinned MCAP decoder dependencies") from exc

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    files = validate_train_receipt(receipt, split, sha256_file(split_manifest_path))
    action = json.loads(action_path.read_text(encoding="utf-8"))
    if (action.get("format") != "haptile_train_only_action_preflight_v1"
            or action.get("input_receipt_sha256") != sha256_file(receipt_path)
            or action.get("split_manifest_sha256") != sha256_file(split_manifest_path)
            or action.get("validation_or_sealed_audit_opened") is not False):
        raise ValueError("train-only target preflight provenance is inconsistent")
    target_by_path = {row["path"]: row for row in action["samples"]}
    if len(target_by_path) != len(files) or len(action["samples"]) != len(files):
        raise ValueError("target rows and train media receipt do not align one-to-one")

    output_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__), output_dir / "extract_haptile_train_state_text.executed.py")
    task_order = list(split["development_splits"]["train"])
    records: list[dict[str, object]] = []
    state_sequences: list[np.ndarray] = []
    for index, item in enumerate(files):
        relative = Path(str(item["path"]))
        media = receipt_path.parent.joinpath(*relative.parts)
        actual_sha = sha256_file(media)
        row = target_by_path.get(str(relative).replace("\\", "/"))
        if actual_sha != item["sha256"] or row is None or row.get("sha256") != actual_sha:
            raise ValueError(f"media hash or target row mismatch: {relative}")
        instruction, state = _extract_one(media, float(row["query_time_log_seconds"]))
        state_sequences.append(state)
        records.append({
            "path": str(relative).replace("\\", "/"),
            "sha256": actual_sha,
            "task_group": item.get("task", relative.parts[1]),
            "instruction": instruction,
            "query_time_log_seconds": row["query_time_log_seconds"],
            "target": row["target"],
        })
        if (index + 1) % 100 == 0:
            print(f"extracted {index + 1}/{len(files)} train state rows", flush=True)

    state_path = output_dir / "state-sequences.npy"
    np.save(state_path, np.stack(state_sequences).astype(np.float32, copy=False))
    examples_path = output_dir / "examples.json"
    examples_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    result = {
        "format": "haptile_train_state_text_features_v1",
        "status": "complete_train_only_state_text_extraction",
        "input_receipt_sha256": sha256_file(receipt_path),
        "action_preflight_sha256": sha256_file(action_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "source_revision": split["conversion_revision"],
        "train_examples": len(records),
        "train_task_groups": task_order,
        "state_shape": list(np.load(state_path, mmap_mode="r").shape),
        "state_channels": list(STREAM_WIDTHS),
        "state_sequence_sha256": sha256_file(state_path),
        "examples_sha256": sha256_file(examples_path),
        "validation_or_sealed_audit_opened": False,
        "audio_present_in_source": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--action-preflight", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract_train_state_text(
        receipt_path=args.receipt,
        action_path=args.action_preflight,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
