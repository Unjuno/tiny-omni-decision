"""Freeze metadata-derived CATER temporal Decision train/selector manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.audit_cater_temporal_task_feasibility import (
    EXPECTED_COLLISION_SHA256,
    EXPECTED_LIST_SHA256,
    EXPECTED_SCENES_SHA256,
    parse_video_id,
    sha256_file,
    temporal_query_windows,
)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).replace("_", " ").replace("-", " ")).strip().lower()


def _object_option(obj: dict[str, Any]) -> dict[str, str]:
    shape, color, material = (
        _clean(obj.get(key, "")) for key in ("shape", "color", "material")
    )
    option_id = f"{shape}|{color}|{material}"
    return {
        "id": option_id,
        "text": f"the {color} {material} {shape}",
    }


def _query_row(
    video_id: int,
    split: str,
    scene: dict[str, Any],
    interval: tuple[int, int],
) -> dict[str, Any]:
    start, end = interval
    scored = []
    for obj in scene["objects"]:
        loc = obj["locations"]
        first, last = loc[str(start)], loc[str(end)]
        distance = (
            sum((float(a) - float(b)) ** 2 for a, b in zip(first, last, strict=True))
            ** 0.5
        )
        scored.append((distance, _object_option(obj)))
    scored.sort(key=lambda row: row[0], reverse=True)
    winner = scored[0][1]["id"]
    options = sorted((_object_option(obj) for obj in scene["objects"]), key=lambda row: row["id"])
    question = (
        f"Between video frames {start} and {end}, which object had the greatest "
        "net displacement?"
    )
    return {
        "id": f"cater-{video_id:06d}-disp-{start:03d}-{end:03d}",
        "group_id": f"cater-video-{video_id:06d}",
        "video_id": f"CATER_new_{video_id:06d}.avi",
        "split": split,
        "question": question,
        "frame_interval": [start, end],
        "options": options,
        "target_option_id": winner,
        "num_options": len(options),
    }


def prepare(
    scenes_zip: Path,
    lists_dir: Path,
    collision_receipt: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(scenes_zip) != EXPECTED_SCENES_SHA256:
        raise ValueError("CATER scene archive hash mismatch")
    if sha256_file(collision_receipt) != EXPECTED_COLLISION_SHA256:
        raise ValueError("Clevr-4 collision receipt hash mismatch")
    collision_data = json.loads(collision_receipt.read_text(encoding="utf-8"))
    specs = {
        "train": ("train_subsetT.txt", "cater_train_subset_t"),
        "validation": ("train_subsetV.txt", "cater_selector_subset_v"),
    }
    result: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    with zipfile.ZipFile(scenes_zip) as archive:
        for split, (list_name, exclusion_key) in specs.items():
            list_path = lists_dir / list_name
            if sha256_file(list_path) != EXPECTED_LIST_SHA256[list_name]:
                raise ValueError(f"CATER list hash mismatch: {list_name}")
            excluded = set(collision_data[exclusion_key]["excluded_video_ids"])
            lines = [line for line in list_path.read_text(encoding="utf-8").splitlines() if line]
            video_ids = [parse_video_id(line) for line in lines]
            if len(video_ids) != len(set(video_ids)):
                raise ValueError(f"duplicate source video IDs in {list_name}")
            for video_id in video_ids:
                if video_id in excluded:
                    continue
                member = f"scenes/CATER_new_{video_id:06d}.json"
                try:
                    scene = json.loads(archive.read(member))
                except KeyError as exc:
                    raise ValueError(f"missing scene metadata: {member}") from exc
                if scene.get("image_index") != video_id:
                    raise ValueError(f"scene identity mismatch: {member}")
                for interval in temporal_query_windows(scene):
                    result[split].append(_query_row(video_id, split, scene, interval))

    train_groups = {row["group_id"] for row in result["train"]}
    validation_groups = {row["group_id"] for row in result["validation"]}
    if train_groups & validation_groups:
        raise ValueError("video/scene group overlap between train and selector")
    for split, rows in result.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate Decision sample IDs in {split}")
        for row in rows:
            option_ids = [option["id"] for option in row["options"]]
            if len(option_ids) != row["num_options"] or row["target_option_id"] not in option_ids:
                raise ValueError(f"invalid options/target for {row['id']}")

    summary = {
        "schema_version": 1,
        "scope": (
            "candidate train/selector labels from CATER scene metadata; "
            "no video bytes or model outputs"
        ),
        "source_hashes": {
            "scenes_zip_sha256": EXPECTED_SCENES_SHA256,
            "train_list_sha256": EXPECTED_LIST_SHA256["train_subsetT.txt"],
            "selector_list_sha256": EXPECTED_LIST_SHA256["train_subsetV.txt"],
            "collision_receipt_sha256": EXPECTED_COLLISION_SHA256,
        },
        "train_validation_video_group_overlap": 0,
        "splits": {},
    }
    for split, rows in result.items():
        summary["splits"][split] = {
            "rows": len(rows),
            "video_groups": len({row["group_id"] for row in rows}),
            "option_count": dict(sorted(Counter(str(row["num_options"]) for row in rows).items())),
            "interval_count": dict(
                sorted(Counter("-".join(map(str, row["frame_interval"])) for row in rows).items())
            ),
            "ordered_sample_id_sha256": hashlib.sha256(
                "\n".join(row["id"] for row in rows).encode("utf-8")
            ).hexdigest(),
        }
    return result["train"], result["validation"], summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-zip", required=True, type=Path)
    parser.add_argument("--lists-dir", required=True, type=Path)
    parser.add_argument("--collision-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    train, validation, summary = prepare(
        args.scenes_zip.resolve(strict=True),
        args.lists_dir.resolve(strict=True),
        args.collision_receipt.resolve(strict=True),
    )
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "validation.jsonl", validation)
    summary["manifest_sha256"] = {
        "train.jsonl": sha256_file(output_dir / "train.jsonl"),
        "validation.jsonl": sha256_file(output_dir / "validation.jsonl"),
    }
    (output_dir / "manifest-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
