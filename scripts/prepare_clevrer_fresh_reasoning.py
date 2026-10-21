"""Select immutable, content-disjoint CLEVRER binary-reasoning splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tiny_omni_decision.clevrer import (
    REASONING_TYPES,
    clevrer_binary_choice_content_key,
    expand_binary_reasoning_choices,
)

REVISION = "98b842082ba4f7c18b6b9e3f39145871782a65ef"
SOURCE_QUESTION_HASHES = {
    "train": "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4",
    "validation": "fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d",
}
EXPECTED_EXPOSURE = {
    "train": {
        "count": 9563,
        "sha256": "a1ff7fe0c1c67f37935e30387c2d1822ce768478bd9befb3d85a246f83d988ca",
    },
    "validation": {
        "count": 998,
        "sha256": "7c6b70ae0c55119939dc179f2ce50746b65bd5292eff25fd3d9c2090fb1f9a8a",
    },
}


def _ordered_ints_sha256(values: set[int]) -> str:
    return hashlib.sha256("\n".join(map(str, sorted(values))).encode()).hexdigest()


def source_scene_ids(row: dict[str, Any]) -> set[int]:
    """Recover only explicit CLEVRER scene tokens from an exposed record."""
    recovered: set[int] = set()
    for value in row.values():
        if not isinstance(value, str):
            continue
        recovered.update(int(match) for match in re.findall(r"video_(\d+)\.mp4", value))
        recovered.update(
            int(match) for match in re.findall(r"clevrer-(?:train|validation)-(\d+)", value)
        )
        recovered.update(
            int(match) for match in re.findall(r"clevrer:(?:train|validation):(\d+):", value)
        )
    if "scene_index" in row:
        recovered.add(int(row["scene_index"]))
    return {scene for scene in recovered if 0 <= scene < 15_000}


def _scene_rows(
    raw_scene: dict[str, Any], *, split: str, partition: str
) -> list[dict[str, Any]]:
    media = {
        "scene_index": int(raw_scene["scene_index"]),
        "video_filename": str(raw_scene["video_filename"]),
        "split": split,
    }
    rows = expand_binary_reasoning_choices([raw_scene], [media], split=split)
    for row in rows:
        row["partition"] = partition
    return rows


def _ordered_scenes(rows: list[dict[str, Any]], *, label: str, seed: int) -> list[int]:
    scenes = {int(row["scene_index"]) for row in rows}
    return sorted(
        scenes,
        key=lambda scene: hashlib.sha256(f"{seed}|{label}|{scene}".encode()).hexdigest(),
    )


def _select_partition(
    raw_rows: list[dict[str, Any]],
    *,
    split: str,
    partition: str,
    requested_scenes: int,
    excluded_scenes: set[int],
    used_content: set[tuple[str, tuple[str, ...]]],
    seed: int,
) -> list[dict[str, Any]]:
    by_scene: dict[int, dict[str, Any]] = {}
    for row in raw_rows:
        scene = int(row["scene_index"])
        if scene in by_scene:
            raise ValueError(f"duplicate source scene row: {scene}")
        by_scene[scene] = row

    eligible_rows = [
        row for scene, row in by_scene.items() if scene not in excluded_scenes
    ]
    selected: list[dict[str, Any]] = []
    content = set(used_content)
    for scene in _ordered_scenes(
        [
            {"scene_index": row["scene_index"]}
            for row in eligible_rows
        ],
        label=f"clevrer-fresh-reasoning-{partition}",
        seed=seed,
    ):
        scene_rows = _scene_rows(by_scene[scene], split=split, partition=partition)
        if not scene_rows:
            continue
        keys = [clevrer_binary_choice_content_key(row) for row in scene_rows]
        # Drop the entire scene when even one candidate duplicates prior content.
        if len(keys) != len(set(keys)) or any(key in content for key in keys):
            continue
        selected.extend(scene_rows)
        content.update(keys)
        if len({int(row["scene_index"]) for row in selected}) == requested_scenes:
            break

    selected_scenes = {int(row["scene_index"]) for row in selected}
    if len(selected_scenes) != requested_scenes:
        raise ValueError(
            f"not enough eligible scenes for {partition}: "
            f"selected {len(selected_scenes)}, need {requested_scenes}"
        )
    selected_types = {str(row["question_type"]) for row in selected}
    if selected_types != REASONING_TYPES:
        raise ValueError(
            f"{partition} partition is missing reasoning types: "
            f"{sorted(REASONING_TYPES - selected_types)}"
        )
    selected.sort(key=lambda row: (row["scene_index"], row["question_id"], row["choice_id"]))
    return selected


def select_fresh_splits(
    raw_train: list[dict[str, Any]],
    raw_validation: list[dict[str, Any]],
    *,
    excluded_train_scenes: set[int],
    excluded_validation_scenes: set[int],
    previously_used_content: set[tuple[str, tuple[str, ...]]],
    train_scene_count: int,
    development_scene_count: int,
    holdout_scene_count: int,
    seed: int = 17,
) -> dict[str, list[dict[str, Any]]]:
    """Select fresh whole-scene train/dev/holdout splits with content gates."""
    if min(train_scene_count, development_scene_count, holdout_scene_count) < 1:
        raise ValueError("requested scene counts must be positive")
    used_content = set(previously_used_content)
    train = _select_partition(
        raw_train,
        split="train",
        partition="train",
        requested_scenes=train_scene_count,
        excluded_scenes=excluded_train_scenes,
        used_content=used_content,
        seed=seed,
    )
    used_content.update(clevrer_binary_choice_content_key(row) for row in train)
    development = _select_partition(
        raw_validation,
        split="validation",
        partition="development",
        requested_scenes=development_scene_count,
        excluded_scenes=excluded_validation_scenes,
        used_content=used_content,
        seed=seed,
    )
    used_content.update(clevrer_binary_choice_content_key(row) for row in development)
    holdout = _select_partition(
        raw_validation,
        split="validation",
        partition="holdout",
        requested_scenes=holdout_scene_count,
        excluded_scenes=excluded_validation_scenes
        | {int(row["scene_index"]) for row in development},
        used_content=used_content,
        seed=seed,
    )

    scene_sets = [
        {int(row["scene_index"]) for row in split_rows}
        for split_rows in (train, development, holdout)
    ]
    if scene_sets[0] & (scene_sets[1] | scene_sets[2]) or scene_sets[1] & scene_sets[2]:
        raise ValueError("CLEVRER scene overlap across selected partitions")
    content_sets = [
        {clevrer_binary_choice_content_key(row) for row in split_rows}
        for split_rows in (train, development, holdout)
    ]
    if content_sets[0] & content_sets[1] or content_sets[0] & content_sets[2]:
        raise ValueError("normalized CLEVRER content overlap across partitions")
    if content_sets[1] & content_sets[2]:
        raise ValueError("normalized CLEVRER content overlap between development and holdout")
    return {"train": train, "development": development, "holdout": holdout}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["id"]) for row in rows).encode()).hexdigest()


def collect_exposure(screen_path: Path) -> dict[str, Any]:
    """Hash-check source-screen inputs and recover prior source scene identities."""
    screen_bytes = screen_path.read_bytes()
    screen = json.loads(screen_bytes)
    exposure_files = screen["exposure_audit"]["exposure_files"]
    exposed = {"train": set(), "validation": set()}
    historical_content: set[tuple[str, tuple[str, ...]]] = set()
    ambiguous_validation_aliases: set[int] = set()
    input_hashes = []
    for item in exposure_files:
        path = Path(item["path"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"prior exposure input changed: {path}")
        input_hashes.append({"path": str(path), "sha256": digest, "bytes": len(raw)})
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for scene in source_scene_ids(row):
                split = "train" if scene < 10_000 else "validation"
                exposed[split].add(scene)
            if "question" in row and "options" in row:
                try:
                    historical_content.add(clevrer_binary_choice_content_key(row))
                except (KeyError, TypeError):
                    pass
            # Legacy local validation rows sometimes stored source validation
            # scene N as local alias N; conservatively exclude official 10000+N.
            if row.get("split") == "validation" and "scene_index" in row:
                local_scene = int(row["scene_index"])
                if 0 <= local_scene < 10_000:
                    ambiguous_validation_aliases.add(10_000 + local_scene)

    actual = {
        split: {"count": len(values), "sha256": _ordered_ints_sha256(values)}
        for split, values in exposed.items()
    }
    if actual["train"] != EXPECTED_EXPOSURE["train"]:
        raise ValueError(f"train exposure reconstruction mismatch: {actual['train']}")
    return {
        "screen_sha256": hashlib.sha256(screen_bytes).hexdigest(),
        "source_revision": REVISION,
        "input_hashes": input_hashes,
        "exposed_scene_ids": {split: sorted(values) for split, values in exposed.items()},
        "actual_exposure": actual,
        "expected_exposure": EXPECTED_EXPOSURE,
        "validation_exposure_hash_reproduced": actual["validation"]
        == EXPECTED_EXPOSURE["validation"],
        "validation_known_identity_gap_count": max(
            0, EXPECTED_EXPOSURE["validation"]["count"] - actual["validation"]["count"]
        ),
        "conservative_validation_alias_ids": sorted(ambiguous_validation_aliases),
        "historical_normalized_content_key_count": len(historical_content),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def preflight(
    *,
    train_questions: Path,
    validation_questions: Path,
    exposure_screen: Path,
    output_dir: Path,
    train_scenes: int = 200,
    development_scenes: int = 200,
    holdout_scenes: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    source_bytes = {
        "train": train_questions.read_bytes(),
        "validation": validation_questions.read_bytes(),
    }
    source_hashes = {
        split: hashlib.sha256(value).hexdigest() for split, value in source_bytes.items()
    }
    if source_hashes != SOURCE_QUESTION_HASHES:
        raise ValueError(f"pinned CLEVRER question hashes differ: {source_hashes}")
    exposure = collect_exposure(exposure_screen)
    raw_train = json.loads(source_bytes["train"])
    raw_validation = json.loads(source_bytes["validation"])
    train_exposed = set(exposure["exposed_scene_ids"]["train"])
    validation_exposed = set(exposure["exposed_scene_ids"]["validation"])
    validation_exposed.update(exposure["conservative_validation_alias_ids"])
    selected = select_fresh_splits(
        raw_train,
        raw_validation,
        excluded_train_scenes=train_exposed,
        excluded_validation_scenes=validation_exposed,
        previously_used_content=set(),
        train_scene_count=train_scenes,
        development_scene_count=development_scenes,
        holdout_scene_count=holdout_scenes,
        seed=seed,
    )
    content_sets = {
        name: {clevrer_binary_choice_content_key(row) for row in rows}
        for name, rows in selected.items()
    }
    historical_content: set[tuple[str, tuple[str, ...]]] = set()
    for item in exposure["input_hashes"]:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"prior exposure input changed after scene selection: {path}")
        for row in _read_jsonl(path):
            if "question" in row and "options" in row:
                try:
                    historical_content.add(clevrer_binary_choice_content_key(row))
                except (KeyError, TypeError):
                    pass
    historical_overlap = {
        name: len(keys & historical_content) for name, keys in content_sets.items()
    }
    scene_sets = {
        name: sorted({int(row["scene_index"]) for row in rows})
        for name, rows in selected.items()
    }
    report = {
        "schema_version": 1,
        "status": "preflight_pass_clevrer_fresh_reasoning",
        "experiment_id": "clevrer-fresh-reasoning-v1",
        "seed": seed,
        "source_revision": REVISION,
        "source_question_sha256": source_hashes,
        "exposure": exposure,
        "exposure_scene_exclusion": {
            "train_exposed_scene_count": len(train_exposed),
            "validation_exposed_scene_count": len(exposure["exposed_scene_ids"]["validation"]),
            "validation_possible_alias_count": len(exposure["conservative_validation_alias_ids"]),
            "validation_total_excluded_scene_count": len(validation_exposed),
            "validation_exposure_hash_gap_disclosed": not exposure[
                "validation_exposure_hash_reproduced"
            ],
        },
        "partitions": {
            name: {
                "scene_ids": scenes,
                "scene_count": len(scenes),
                "choice_rows": len(selected[name]),
                "unique_parent_questions": len(
                    {row["parent_question_id"] for row in selected[name]}
                ),
                "choice_id_order_sha256": _ids_sha256(selected[name]),
                "canonical_rows_sha256": _canonical_rows_sha256(selected[name]),
                "question_type_counts": {
                    kind: sum(row["question_type"] == kind for row in selected[name])
                    for kind in sorted(REASONING_TYPES)
                },
            }
            for name, scenes in scene_sets.items()
        },
        "integrity": {
            "scene_overlap": 0,
            "normalized_content_overlap_across_new_partitions": 0,
            "historical_normalized_content_overlap_by_partition": historical_overlap,
            "validation_known_exposure_hash_reproduced": exposure[
                "validation_exposure_hash_reproduced"
            ],
            "sealed_audit_loaded": False,
            "source_media_hashes_checked": False,
        },
        "selection_policy": (
            "seeded sha256 ordering; whole scenes; all three task types per partition; "
            "exact normalized question/candidate content unique across the three new partitions"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in selected.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
    (output_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-questions", type=Path, required=True)
    parser.add_argument("--validation-questions", type=Path, required=True)
    parser.add_argument("--exposure-screen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=200)
    parser.add_argument("--development-scenes", type=int, default=200)
    parser.add_argument("--holdout-scenes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = preflight(
        train_questions=args.train_questions.resolve(strict=True),
        validation_questions=args.validation_questions.resolve(strict=True),
        exposure_screen=args.exposure_screen.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        train_scenes=args.train_scenes,
        development_scenes=args.development_scenes,
        holdout_scenes=args.holdout_scenes,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
