"""Build a frozen four-choice, same-room/event audio-to-video matching task."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: object) -> bytes:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def build_examples(rows: list[dict[str, object]], *, seed: int = 17):
    """Make one query per supported room/event/parent unit with hard negatives."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    by_cell_parent: dict[tuple[str, int, str], list[tuple[int, dict[str, object]]]] = defaultdict(
        list
    )
    for index, row in enumerate(rows):
        room = str(row["room_id"])
        event = int(row["event_class"])
        parent = str(row["parent_recording"])
        if not 0 <= event < 13 or not row.get("sample_id"):
            raise ValueError("invalid DCASE event class or sample ID")
        by_cell_parent[(room, event, parent)].append((index, row))
    cell_parents: dict[tuple[str, int], list[str]] = defaultdict(list)
    for room, event, parent in by_cell_parent:
        cell_parents[(room, event)].append(parent)
    eligible_cells = {
        cell: sorted(set(parents))
        for cell, parents in cell_parents.items()
        if len(set(parents)) >= 4
    }
    units = []
    for (room, event), parents in sorted(eligible_cells.items()):
        for parent in parents:
            source_rows = by_cell_parent[(room, event, parent)]
            representative_index, representative = min(
                source_rows,
                key=lambda item: _stable_key(seed, "representative", item[1]["sample_id"]),
            )
            units.append(
                {
                    "room_id": room,
                    "event_class": event,
                    "parent_recording": parent,
                    "query_row": representative_index,
                    "query_sample_id": str(representative["sample_id"]),
                    "candidate_parents": parents,
                }
            )
    units.sort(
        key=lambda unit: _stable_key(
            seed, "query", unit["room_id"], unit["event_class"], unit["parent_recording"]
        )
    )
    if not units:
        raise ValueError("no same-room/event cells have four independent parent recordings")
    positions = np.resize(np.arange(4, dtype=np.int64), len(units))
    rng = np.random.default_rng(seed)
    rng.shuffle(positions)
    examples = []
    for unit, target_index in zip(units, positions, strict=True):
        cell = (unit["room_id"], unit["event_class"])
        distractor_parents = [
            parent for parent in eligible_cells[cell] if parent != unit["parent_recording"]
        ]
        selected = rng.choice(distractor_parents, size=3, replace=False).tolist()
        option_parents = list(selected)
        option_parents.insert(int(target_index), unit["parent_recording"])
        options = []
        for option_parent in option_parents:
            source_rows = by_cell_parent[(unit["room_id"], unit["event_class"], option_parent)]
            row_index, row = min(
                source_rows,
                key=lambda item: _stable_key(seed, "representative", item[1]["sample_id"]),
            )
            sample_id = str(row["sample_id"])
            options.append(
                {
                    "candidate_id": "v_"
                    + sha256_bytes(_stable_key(seed, "candidate", sample_id))[:16],
                    "sample_id": sample_id,
                    "parent_recording": str(option_parent),
                    "feature_row": row_index,
                }
            )
        if options[int(target_index)]["parent_recording"] != unit["parent_recording"]:
            raise AssertionError("positive candidate was not placed at target index")
        if len({option["parent_recording"] for option in options}) != 4:
            raise AssertionError("candidate parents must be distinct")
        query_id = (
            "q_"
            + sha256_bytes(
                _stable_key(
                    seed, "query-id", unit["room_id"], unit["event_class"], unit["parent_recording"]
                )
            )[:16]
        )
        examples.append(
            {
                "query_id": query_id,
                "query_sample_id": unit["query_sample_id"],
                "query_row": unit["query_row"],
                "room_id": unit["room_id"],
                "event_class": unit["event_class"],
                "query_parent": unit["parent_recording"],
                "options": options,
                "target_index": int(target_index),
            }
        )
    return examples, eligible_cells


def write_task(features_dir: Path, output_dir: Path, *, seed: int = 17):
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite task output: {output_dir}")
    progress = json.loads((features_dir / "progress.json").read_text(encoding="utf-8"))
    preflight = json.loads((features_dir / "preflight.json").read_text(encoding="utf-8"))
    if (
        progress.get("status") != "PASS"
        or progress.get("valid_rows") != progress.get("total_queries")
        or preflight.get("status") != "PASS"
        or preflight.get("training_protocol")
        != "all official dev-train rooms; fixed budget; no validation selection"
        or preflight.get("devtest_label_content_opened") is not False
        or progress.get("identity", {}).get("query_ids_sha256")
        != preflight.get("selected_sample_id_order_sha256")
    ):
        raise ValueError("input must be the complete frozen DCASE dev-train feature cache")
    source_manifest = features_dir / "query-manifest.jsonl"
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
    query_ids_sha256 = sha256_bytes(
        "".join(f"{row['sample_id']}\n" for row in rows).encode("utf-8")
    )
    if (
        len(rows) != progress.get("total_queries")
        or len({row["sample_id"] for row in rows}) != len(rows)
        or query_ids_sha256 != progress.get("identity", {}).get("query_ids_sha256")
        or query_ids_sha256 != preflight.get("selected_sample_id_order_sha256")
    ):
        raise ValueError("source query manifest differs from frozen feature identity")
    audio = np.load(features_dir / "features_audio.npy", mmap_mode="r")
    video = np.load(features_dir / "features_video.npy", mmap_mode="r")
    if audio.shape != (len(rows), 768) or video.shape != audio.shape:
        raise ValueError("frozen audio/video feature shape mismatch")
    for name in ("valid_mask.npy", "processed_mask.npy"):
        mask = np.load(features_dir / name, mmap_mode="r")
        if mask.shape != (len(rows),) or not mask.all():
            raise ValueError(f"feature cache has invalid {name}")
    examples, eligible_cells = build_examples(rows, seed=seed)
    output_dir.mkdir(parents=True)
    task_manifest = output_dir / "task-manifest.jsonl"
    with task_manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(json.dumps(example, sort_keys=True) + "\n")
    position_counts = Counter(example["target_index"] for example in examples)
    parent_set = {example["query_parent"] for example in examples}
    room_counts = Counter(example["room_id"] for example in examples)
    event_counts = Counter(example["event_class"] for example in examples)
    config = {
        "task": "four-choice audiovisual parent-recording match; query audio, choose aligned video",
        "source_split": "DCASE official dev-train frozen features only",
        "feature_identity": progress["identity"],
        "source_preflight_sha256": sha256_file(features_dir / "preflight.json"),
        "source_manifest_sha256": sha256_file(source_manifest),
        "query_id_order_sha256": query_ids_sha256,
        "audio_features_sha256": sha256_file(features_dir / "features_audio.npy"),
        "video_features_sha256": sha256_file(features_dir / "features_video.npy"),
        "seed": seed,
        "construction": (
            "one representative sample per room/event/parent group; four options are "
            "same room and event, with distinct parent recordings; one exact aligned pair"
        ),
        "candidate_count": 4,
        "eligible_room_event_cells": len(eligible_cells),
        "target_position_counts": {str(index): int(position_counts[index]) for index in range(4)},
        "room_counts": dict(sorted(room_counts.items())),
        "event_class_counts": {
            str(index): int(event_counts[index]) for index in sorted(event_counts)
        },
        "query_units": len(examples),
        "unique_query_parents": len(parent_set),
        "metadata_only_predictability": (
            "uniform 25%; room and event class are constant within each choice set"
        ),
        "devtest_accessed": False,
        "sealed_audit_accessed": False,
    }
    (output_dir / "effective-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "examples": len(examples),
        "unique_query_parents": len(parent_set),
        "eligible_room_event_cells": len(eligible_cells),
        "target_position_counts": dict(sorted(position_counts.items())),
        "candidate_parent_unique_per_query": all(
            len({option["parent_recording"] for option in example["options"]}) == 4
            for example in examples
        ),
        "positive_matches_query_parent": all(
            example["options"][example["target_index"]]["parent_recording"]
            == example["query_parent"]
            for example in examples
        ),
        "task_manifest_sha256": sha256_file(task_manifest),
        "effective_config_sha256": sha256_file(output_dir / "effective-config.json"),
        "room_event_support_counts": {
            f"{room}|{event}": len(parents)
            for (room, event), parents in sorted(eligible_cells.items())
        },
    }
    (output_dir / "task-preflight.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = write_task(args.features, args.output, seed=args.seed)
    print(
        json.dumps(
            {key: summary[key] for key in ("status", "examples", "eligible_room_event_cells")}
        )
    )


if __name__ == "__main__":
    main()
