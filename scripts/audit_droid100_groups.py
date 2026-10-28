"""Audit DROID-100 TFRecord episode and recording-group metadata without decoding media."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

METADATA_KEYS = {
    "episode_metadata/file_path",
    "episode_metadata/recording_folderpath",
}


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            break
    raise ValueError("invalid protobuf varint")


def _fields(data: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire_type = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("protobuf field number zero is invalid")
        if wire_type == 0:
            value, offset = _varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("truncated fixed64 protobuf field")
            value, offset = data[offset:end], end
        elif wire_type == 2:
            length, offset = _varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated length-delimited protobuf field")
            value, offset = data[offset:end], end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("truncated fixed32 protobuf field")
            value, offset = data[offset:end], end
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield number, wire_type, value


def _bytes_list(feature: bytes) -> list[bytes]:
    for number, wire_type, payload in _fields(feature):
        if number == 1 and wire_type == 2:
            values = [value for field, kind, value in _fields(payload) if field == 1 and kind == 2]
            return [value for value in values if isinstance(value, bytes)]
    return []


def episode_example_context(serialized: bytes) -> dict[str, str]:
    context = None
    for number, wire_type, payload in _fields(serialized):
        if number == 1 and wire_type == 2:
            context = payload
            break
    if context is None:
        raise ValueError("TFRecord payload has no Example feature map")

    result: dict[str, str] = {}
    for number, wire_type, entry in _fields(context):
        if number != 1 or wire_type != 2:
            continue
        key = None
        feature = None
        for field, kind, payload in _fields(entry):
            if field == 1 and kind == 2:
                key = payload.decode("utf-8")
            elif field == 2 and kind == 2:
                feature = payload
        if key not in METADATA_KEYS or feature is None:
            continue
        values = _bytes_list(feature)
        if len(values) != 1:
            raise ValueError(f"expected one scalar string for {key}, got {len(values)}")
        result[key] = values[0].decode("utf-8").strip()

    if set(result) != METADATA_KEYS or any(not value for value in result.values()):
        raise ValueError("episode is missing non-empty file/recording metadata")
    return result


def episode_example_instructions(serialized: bytes) -> list[str]:
    context = None
    for number, wire_type, payload in _fields(serialized):
        if number == 1 and wire_type == 2:
            context = payload
            break
    if context is None:
        return []

    instructions = set()
    for number, wire_type, entry in _fields(context):
        if number != 1 or wire_type != 2:
            continue
        key = None
        feature = None
        for field, kind, payload in _fields(entry):
            if field == 1 and kind == 2:
                key = payload.decode("utf-8")
            elif field == 2 and kind == 2:
                feature = payload
        if not key or "language_instruction" not in key or feature is None:
            continue
        for value in _bytes_list(feature):
            instruction = value.decode("utf-8", errors="strict").strip()
            if instruction:
                instructions.add(" ".join(instruction.split()))
    return sorted(instructions)


def episode_example_feature_counts(serialized: bytes) -> dict[str, dict[str, int | str]]:
    feature_map = None
    for number, wire_type, payload in _fields(serialized):
        if number == 1 and wire_type == 2:
            feature_map = payload
            break
    if feature_map is None:
        raise ValueError("TFRecord payload has no Example feature map")

    result = {}
    for number, wire_type, entry in _fields(feature_map):
        if number != 1 or wire_type != 2:
            continue
        key = None
        feature = None
        for field, kind, payload in _fields(entry):
            if field == 1 and kind == 2:
                key = payload.decode("utf-8")
            elif field == 2 and kind == 2:
                feature = payload
        if key is None or feature is None:
            continue

        feature_kind = None
        values = []
        for field, kind, payload in _fields(feature):
            if kind != 2 or field not in (1, 2, 3):
                continue
            feature_kind = {1: "bytes", 2: "float", 3: "int64"}[field]
            for value_field, value_wire, value in _fields(payload):
                if value_field != 1:
                    continue
                if feature_kind == "bytes" and value_wire == 2:
                    values.append(value)
                elif feature_kind == "float" and value_wire == 2:
                    if len(value) % 4:
                        raise ValueError(f"packed float bytes are not 4-byte aligned for {key}")
                    values.extend([None] * (len(value) // 4))
                elif feature_kind == "float" and value_wire == 5:
                    values.append(None)
                elif feature_kind == "int64" and value_wire == 2:
                    packed_offset = 0
                    packed_count = 0
                    while packed_offset < len(value):
                        _, packed_offset = _varint(value, packed_offset)
                        packed_count += 1
                    values.extend([None] * packed_count)
                elif feature_kind == "int64" and value_wire == 0:
                    values.append(None)

        row: dict[str, int | str] = {"kind": feature_kind, "count": len(values)}
        if feature_kind == "bytes":
            row["nonempty"] = sum(bool(value) for value in values)
        result[key] = row
    return result


def episode_example_float_values(serialized: bytes, target_key: str) -> list[float]:
    feature_map = None
    for number, wire_type, payload in _fields(serialized):
        if number == 1 and wire_type == 2:
            feature_map = payload
            break
    if feature_map is None:
        raise ValueError("TFRecord payload has no Example feature map")

    for number, wire_type, entry in _fields(feature_map):
        if number != 1 or wire_type != 2:
            continue
        key = None
        feature = None
        for field, kind, payload in _fields(entry):
            if field == 1 and kind == 2:
                key = payload.decode("utf-8")
            elif field == 2 and kind == 2:
                feature = payload
        if key != target_key or feature is None:
            continue

        float_list = None
        for field, kind, payload in _fields(feature):
            if field == 2 and kind == 2:
                float_list = payload
                break
        if float_list is None:
            raise ValueError(f"feature is not a FloatList: {target_key}")

        values = []
        for field, wire_type, payload in _fields(float_list):
            if field != 1:
                continue
            if wire_type == 2:
                if len(payload) % 4:
                    raise ValueError(f"packed FloatList is not 4-byte aligned: {target_key}")
                values.extend(value[0] for value in struct.iter_unpack("<f", payload))
            elif wire_type == 5:
                values.append(struct.unpack("<f", payload)[0])
            else:
                raise ValueError(f"unexpected FloatList value encoding: {target_key}")
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"FloatList is empty or non-finite: {target_key}")
        return values
    raise KeyError(target_key)


def episode_example_bytes_values(serialized: bytes, target_key: str) -> list[bytes]:
    feature_map = None
    for number, wire_type, payload in _fields(serialized):
        if number == 1 and wire_type == 2:
            feature_map = payload
            break
    if feature_map is None:
        raise ValueError("TFRecord payload has no Example feature map")

    for number, wire_type, entry in _fields(feature_map):
        if number != 1 or wire_type != 2:
            continue
        key = None
        feature = None
        for field, kind, payload in _fields(entry):
            if field == 1 and kind == 2:
                key = payload.decode("utf-8")
            elif field == 2 and kind == 2:
                feature = payload
        if key != target_key or feature is None:
            continue

        bytes_list = None
        for field, kind, payload in _fields(feature):
            if field == 1 and kind == 2:
                bytes_list = payload
                break
        if bytes_list is None:
            raise ValueError(f"feature is not a BytesList: {target_key}")
        values = [
            value
            for field, wire_type, value in _fields(bytes_list)
            if field == 1 and wire_type == 2 and isinstance(value, bytes)
        ]
        if not values:
            raise ValueError(f"BytesList is empty: {target_key}")
        return values
    raise KeyError(target_key)


def iter_tfrecord(path: Path) -> Iterator[bytes]:
    with path.open("rb") as stream:
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError(f"truncated TFRecord header in {path}")
            (length,) = struct.unpack("<Q", header[:8])
            if length > path.stat().st_size:
                raise ValueError(f"impossible TFRecord length in {path}")
            payload = stream.read(length)
            trailer = stream.read(4)
            if len(payload) != length or len(trailer) != 4:
                raise ValueError(f"truncated TFRecord payload in {path}")
            yield payload


def audit_dataset(root: Path, dataset_info: dict) -> dict:
    split = next((item for item in dataset_info["splits"] if item["name"] == "train"), None)
    if split is None:
        raise ValueError("dataset_info.json has no train split")
    expected_lengths = [int(value) for value in split["shardLengths"]]
    if len(expected_lengths) != 31:
        raise ValueError("expected exactly 31 pinned TFRecord shards")

    episodes = []
    instruction_counts = Counter()
    counts_by_shard = []
    for index, expected in enumerate(expected_lengths):
        name = f"r2d2_faceblur-train.tfrecord-{index:05d}-of-00031"
        shard = root / name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        shard_rows = []
        for record in iter_tfrecord(shard):
            shard_rows.append(episode_example_context(record))
            instruction_counts.update(episode_example_instructions(record))
        if len(shard_rows) != expected:
            raise ValueError(f"{name}: expected {expected} episodes, read {len(shard_rows)}")
        for row in shard_rows:
            row["shard"] = name
        episodes.extend(shard_rows)
        counts_by_shard.append(len(shard_rows))

    if len(episodes) != 100:
        raise ValueError(f"expected 100 episodes, read {len(episodes)}")
    if len({row["episode_metadata/file_path"] for row in episodes}) != len(episodes):
        raise ValueError("duplicate source file_path across DROID-100 episodes")

    group_counts = Counter(row["episode_metadata/recording_folderpath"] for row in episodes)
    return {
        "episode_count": len(episodes),
        "unique_source_files": len({row["episode_metadata/file_path"] for row in episodes}),
        "unique_recording_groups": len(group_counts),
        "episodes_per_recording_group": dict(sorted(group_counts.items())),
        "unique_instructions": len(instruction_counts),
        "instruction_episode_frequency": dict(sorted(instruction_counts.items())),
        "episodes_per_shard": counts_by_shard,
        "episodes": episodes,
        "split_warning": (
            "This release is train-only; any local split is exploratory, "
            "not a held-out benchmark."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info_path = args.data_dir / "dataset_info.json"
    dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    result = audit_dataset(args.data_dir, dataset_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "episode_count": result["episode_count"],
        "unique_source_files": result["unique_source_files"],
        "unique_recording_groups": result["unique_recording_groups"],
        "unique_instructions": result["unique_instructions"],
        "episodes_per_shard": result["episodes_per_shard"],
        "split_warning": result["split_warning"],
    }, indent=2))


if __name__ == "__main__":
    main()
