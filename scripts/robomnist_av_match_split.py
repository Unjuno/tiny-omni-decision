"""Deterministic group split and leakage-resistant AV correspondence pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

FORCED_TRAIN_GROUP = "Ar1_Vhigh_Ac0_Rp01_UNC062"
EXPECTED_ARCHIVES = 60
EXPECTED_GROUPS = 2135
BASE_GROUPS_PER_CONDITION = 16
EXTRA_CONDITIONS_PER_SPLIT = 20
QUESTION = "Do the audio and video come from the same recording repetition?"
OPTIONS = ["different repetition", "same repetition"]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_source_hashes(pins: Mapping[str, str], *, repo_root: Path) -> None:
    """Fail closed if the source-screen code/protocol hashes changed."""
    for raw_path, expected in pins.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root / path
        if not path.is_file():
            raise ValueError(f"pinned source file missing: {raw_path}")
        if sha256_file(path) != expected:
            raise ValueError(f"source hash mismatch: {raw_path}")


def _order_key(seed: int, purpose: str, value: str) -> str:
    payload = f"robomnist-av-match-v1:{seed}:{purpose}:{value}".encode()
    return _sha256_bytes(payload)


def _group_fields(
    groups: Sequence[Mapping[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    by_id: dict[str, dict[str, str]] = {}
    by_condition: dict[str, list[str]] = defaultdict(list)
    for row in groups:
        group_id = row.get("group_id")
        condition = row.get("condition")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("every group needs a non-empty group_id")
        if not isinstance(condition, str) or not condition:
            raise ValueError("every group needs a non-empty condition")
        if not group_id.startswith(f"{condition}_Rp"):
            raise ValueError("group_id must be nested under its primary condition")
        if group_id in by_id:
            raise ValueError(f"duplicate group_id: {group_id}")
        by_id[group_id] = {"group_id": group_id, "condition": condition}
        by_condition[condition].append(group_id)
    if not by_id:
        raise ValueError("at least one group is required")
    return by_id, by_condition


def build_group_split(
    groups: Sequence[Mapping[str, str]],
    *,
    seed: int,
    base_per_condition: int,
    extra_conditions_per_split: int,
    forced_train_group_ids: Iterable[str],
) -> dict[str, list[dict[str, str]]]:
    """Split full repetitions before pairing, adding capacity in even strata blocks."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if base_per_condition <= 0 or base_per_condition % 2:
        raise ValueError("base_per_condition must be positive and even")
    if extra_conditions_per_split < 0:
        raise ValueError("extra_conditions_per_split must be non-negative")

    by_id, by_condition = _group_fields(groups)
    forced = set(forced_train_group_ids)
    unknown = forced - by_id.keys()
    if unknown:
        raise ValueError(f"unknown forced train group: {min(unknown)}")

    train_counts = {condition: base_per_condition for condition in by_condition}
    validation_counts = {condition: base_per_condition for condition in by_condition}
    remaining_capacity = {
        condition: len(group_ids) - 2 * base_per_condition
        for condition, group_ids in by_condition.items()
    }
    if any(capacity < 0 for capacity in remaining_capacity.values()):
        raise ValueError("not enough groups to preserve even train/validation strata")

    for split_name, counts in (("train", train_counts), ("validation", validation_counts)):
        eligible = [
            condition for condition, capacity in remaining_capacity.items() if capacity >= 2
        ]
        eligible.sort(key=lambda condition: _order_key(seed, f"extra-{split_name}", condition))
        if len(eligible) < extra_conditions_per_split:
            raise ValueError("not enough capacity for even-pair split allocation")
        for condition in eligible[:extra_conditions_per_split]:
            counts[condition] += 2
            remaining_capacity[condition] -= 2

    forced_per_condition: dict[str, int] = defaultdict(int)
    for group_id in forced:
        forced_per_condition[by_id[group_id]["condition"]] += 1
    if any(forced_per_condition[c] > train_counts[c] for c in forced_per_condition):
        raise ValueError("forced train groups exceed their condition quota")

    result: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "reserve": []}
    for condition, group_ids in by_condition.items():
        ordered = sorted(group_ids, key=lambda group_id: _order_key(seed, "group", group_id))
        forced_here = sorted(forced.intersection(ordered))
        remaining = [group_id for group_id in ordered if group_id not in forced_here]
        train_count = train_counts[condition]
        validation_count = validation_counts[condition]
        train_ids = forced_here + remaining[: train_count - len(forced_here)]
        validation_start = train_count - len(forced_here)
        validation_ids = remaining[validation_start : validation_start + validation_count]
        reserve_ids = remaining[validation_start + validation_count :]
        if len(train_ids) != train_count or len(validation_ids) != validation_count:
            raise ValueError("split quotas exceed available groups")
        for split_name, selected_ids in (
            ("train", train_ids),
            ("validation", validation_ids),
            ("reserve", reserve_ids),
        ):
            result[split_name].extend(by_id[group_id] for group_id in selected_ids)

    for split_name in result:
        result[split_name].sort(key=lambda row: row["group_id"])
    split_ids = [{row["group_id"] for row in result[name]} for name in result]
    if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("group leakage across splits")
    if set.union(*split_ids) != set(by_id):
        raise AssertionError("split did not account for every source group")
    for split_name in ("train", "validation"):
        counts: dict[str, int] = defaultdict(int)
        for row in result[split_name]:
            counts[row["condition"]] += 1
        if any(count % 2 for count in counts.values()):
            raise ValueError("train and validation strata must have even pair counts")
    return result


def build_eligible_group_split(
    groups: Sequence[Mapping[str, str]],
    *,
    eligible_group_ids: Iterable[str],
    validation_group_count: int,
    seed: int,
    forced_train_group_ids: Iterable[str],
) -> dict[str, list[dict[str, str]]]:
    """Split only a pre-approved non-audit group pool into train and validation."""
    by_id, by_condition = _group_fields(groups)
    eligible = set(eligible_group_ids)
    unknown = eligible - by_id.keys()
    if unknown:
        raise ValueError(f"eligible group is absent from source inventory: {min(unknown)}")
    forced = set(forced_train_group_ids)
    if forced - eligible:
        raise ValueError("forced train group must be in the eligible group pool")
    if validation_group_count <= 0 or validation_group_count % 2:
        raise ValueError("validation group count must be positive and even")

    eligible_by_condition = {
        condition: [group_id for group_id in group_ids if group_id in eligible]
        for condition, group_ids in by_condition.items()
        if any(group_id in eligible for group_id in group_ids)
    }
    capacities = {
        condition: len(group_ids) - len(group_ids) % 2
        for condition, group_ids in eligible_by_condition.items()
    }
    if any(capacity < 4 for capacity in capacities.values()):
        raise ValueError(
            "eligible groups need capacity for train and validation pairs per condition"
        )
    capacity_total = sum(capacities.values())
    minimum_validation = 2 * len(capacities)
    maximum_validation = capacity_total - 2 * len(capacities)
    if validation_group_count < minimum_validation or validation_group_count > maximum_validation:
        raise ValueError("validation count exceeds eligible groups after preserving train pairs")

    validation_counts = dict.fromkeys(capacities, 2)
    remaining = validation_group_count - minimum_validation
    while remaining:
        candidates = [
            condition
            for condition, capacity in capacities.items()
            if validation_counts[condition] + 2 <= capacity - 2
        ]
        if not candidates:
            raise ValueError("eligible groups cannot satisfy validation quota")
        candidates.sort(
            key=lambda condition: (
                validation_counts[condition],
                _order_key(seed, "validation-capacity", condition),
            )
        )
        condition = candidates[0]
        validation_counts[condition] += 2
        remaining -= 2

    result: dict[str, list[dict[str, str]]] = {"train": [], "validation": []}
    for condition, group_ids in eligible_by_condition.items():
        capacity = capacities[condition]
        forced_here = sorted(forced.intersection(group_ids))
        ordered = sorted(
            (group_id for group_id in group_ids if group_id not in forced_here),
            key=lambda group_id: _order_key(seed, "eligible-group", group_id),
        )
        selected = forced_here + ordered[: capacity - len(forced_here)]
        validation_pool = sorted(
            (group_id for group_id in selected if group_id not in forced_here),
            key=lambda group_id: _order_key(seed, "eligible-validation", group_id),
        )
        validation_ids = set(validation_pool[: validation_counts[condition]])
        train_ids = set(selected) - validation_ids
        if len(train_ids) < 2 or len(train_ids) % 2:
            raise ValueError("eligible split did not preserve whole train pairs")
        if len(validation_ids) < 2 or len(validation_ids) % 2:
            raise ValueError("eligible split did not preserve whole validation pairs")
        result["train"].extend(by_id[group_id] for group_id in sorted(train_ids))
        result["validation"].extend(by_id[group_id] for group_id in sorted(validation_ids))

    result_ids = {name: {row["group_id"] for row in rows} for name, rows in result.items()}
    if result_ids["train"] & result_ids["validation"]:
        raise AssertionError("eligible group leakage across splits")
    if (result_ids["train"] | result_ids["validation"]) - eligible:
        raise AssertionError("eligible split used an excluded group")
    if len(result["validation"]) != validation_group_count:
        raise ValueError("validation quota could not be met exactly")
    return result


def _group_ids_in_rows(rows: object) -> set[str]:
    """Collect group IDs from non-audit manifest rows without opening audit data."""
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("group_id") and isinstance(child, str):
                    found.add(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(rows)
    return found


def load_prior_nonsealed_group_pool(prior_run: Path) -> tuple[list[str], dict[str, str]]:
    """Read only a prior run's summary, train and validation manifests."""
    summary_path = prior_run / "summary.json"
    train_path = prior_run / "train.json"
    validation_path = prior_run / "validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for name, path in (("train", train_path), ("validation", validation_path)):
        expected = summary.get(f"{name}_sha256")
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise ValueError(f"prior non-audit {name} manifest hash mismatch")
    required_false = (
        "payload_opened",
        "training",
        "model_inference",
        "sealed_audit_evaluated",
    )
    if any(summary.get(key) is not False for key in required_false):
        raise ValueError("prior run is not eligible as a metadata-only split reference")
    train_rows = json.loads(train_path.read_text(encoding="utf-8"))
    validation_rows = json.loads(validation_path.read_text(encoding="utf-8"))
    group_ids = _group_ids_in_rows(train_rows) | _group_ids_in_rows(validation_rows)
    if not group_ids:
        raise ValueError("prior non-audit manifests contain no source group IDs")
    return sorted(group_ids), {
        "summary_sha256": sha256_file(summary_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
    }


def build_pair_units(groups: Sequence[Mapping[str, str]], *, seed: int) -> list[dict[str, str]]:
    """Pair disjoint repetitions only inside a previously frozen split/condition."""
    _, by_condition = _group_fields(groups)
    units: list[dict[str, str]] = []
    for condition, group_ids in sorted(by_condition.items()):
        if len(group_ids) % 2:
            raise ValueError("each condition needs an even number of pairable groups")
        ordered = sorted(group_ids, key=lambda group_id: _order_key(seed, "pair", group_id))
        for index in range(0, len(ordered), 2):
            group_a, group_b = ordered[index : index + 2]
            pair_key = f"{seed}:{condition}:{group_a}:{group_b}"
            pair_id = _sha256_bytes(pair_key.encode("utf-8"))
            units.append(
                {
                    "pair_unit_id": pair_id,
                    "condition": condition,
                    "group_a": group_a,
                    "group_b": group_b,
                }
            )
    return sorted(units, key=lambda row: row["pair_unit_id"])


def expand_pair_unit(unit: Mapping[str, str]) -> list[dict[str, str | int]]:
    """Create matched/swapped rows so each clip occurs once under each target."""
    pair_id = unit.get("pair_unit_id")
    condition = unit.get("condition")
    group_a = unit.get("group_a")
    group_b = unit.get("group_b")
    values = (pair_id, condition, group_a, group_b)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("pair unit fields must be non-empty strings")
    if group_a == group_b:
        raise ValueError("pair unit must use two distinct repetition groups")
    if not group_a.startswith(f"{condition}_Rp") or not group_b.startswith(f"{condition}_Rp"):
        raise ValueError("pair groups must share the declared condition")
    rows = []
    for audio_group, video_group, target in (
        (group_a, group_a, 1),
        (group_a, group_b, 0),
        (group_b, group_b, 1),
        (group_b, group_a, 0),
    ):
        rows.append(
            {
                "pair_unit_id": pair_id,
                "condition": condition,
                "audio_group": audio_group,
                "video_group": video_group,
                "target": target,
            }
        )
    return rows


def _member_map(directory: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    members = directory.get("members")
    if not isinstance(members, list):
        raise ValueError("directory inventory must include members")
    return {
        member["name"]: member
        for member in members
        if isinstance(member, dict) and not member.get("directory")
    }


def _selected_member(member: Mapping[str, object]) -> dict[str, object]:
    fields = ("name", "size", "compressed_size", "crc32", "compression", "offset")
    if any(field not in member for field in fields):
        raise ValueError("selected archive member lacks exact integrity metadata")
    return {field: member[field] for field in fields}


def build_metadata_manifest(root: Path, prior_run: Path) -> dict[str, object]:
    """Build a small source-pinned split manifest from metadata only."""
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    archives = config.get("archives")
    if not isinstance(archives, list) or len(archives) != EXPECTED_ARCHIVES:
        raise ValueError("pinned source configuration must list all 60 archives")

    directories = sorted(root.glob("directory-*.json"))
    if len(directories) != EXPECTED_ARCHIVES:
        raise ValueError("all 60 pinned archive directory inventories are required")
    directory_by_archive: dict[str, tuple[Path, dict[str, object]]] = {}
    directory_hashes: dict[str, str] = {}
    for path in directories:
        directory = json.loads(path.read_text(encoding="utf-8"))
        source = directory.get("source_archive")
        if not isinstance(source, dict) or not isinstance(source.get("name"), str):
            raise ValueError("directory inventory must bind to a source archive")
        if source["name"] in directory_by_archive:
            raise ValueError("duplicate archive directory inventory")
        directory_by_archive[source["name"]] = (path, directory)
        directory_hashes[path.name] = sha256_file(path)

    archive_names = {archive.get("name") for archive in archives if isinstance(archive, dict)}
    if archive_names != set(directory_by_archive):
        raise ValueError("pinned archive and directory inventory names differ")

    group_rows: list[dict[str, str]] = []
    group_sources: dict[str, dict[str, object]] = {}
    for archive in archives:
        name = archive["name"]
        path, directory = directory_by_archive[name]
        source = directory["source_archive"]
        for key in ("id", "name", "size", "download_url"):
            if source.get(key) != archive.get(key):
                raise ValueError(f"source archive metadata mismatch for {name}: {key}")
        member_map = _member_map(directory)
        for group in directory.get("groups", []):
            group_id = group.get("group_id")
            if not isinstance(group_id, str):
                raise ValueError("every source repetition needs a group id")
            condition = group_id.split("_Rp", 1)[0]
            if not condition or len(group_id.split("_Rp", 1)) != 2:
                raise ValueError("malformed source repetition id")
            inputs = group.get("input_members", [])
            videos = [name for name in inputs if name.endswith("_Rx1_cam.mp4")]
            audios = [name for name in inputs if name.endswith("_Rx1_mic.wav")]
            if len(videos) != 1 or len(audios) != 1:
                raise ValueError("each repetition must have exactly one Rx1 video and microphone")
            if videos[0] not in member_map or audios[0] not in member_map:
                raise ValueError("selected AV member missing from pinned ZIP directory")
            row = {"group_id": group_id, "condition": condition}
            if group_id in group_sources:
                raise ValueError("repetition group appears in more than one source archive")
            group_rows.append(row)
            group_sources[group_id] = {
                "archive_name": source["name"],
                "archive_id": source["id"],
                "archive_size": source["size"],
                "archive_url": source["download_url"],
                "publisher_md5": source.get("supplied_md5"),
                "computed_md5": source.get("computed_md5"),
                "directory_metadata": path.name,
                "video_member": _selected_member(member_map[videos[0]]),
                "audio_member": _selected_member(member_map[audios[0]]),
            }

    if len(group_rows) != EXPECTED_GROUPS:
        raise ValueError(f"expected {EXPECTED_GROUPS} complete groups, found {len(group_rows)}")
    prior_group_ids, prior_hashes = load_prior_nonsealed_group_pool(prior_run)
    split = build_eligible_group_split(
        group_rows,
        eligible_group_ids=prior_group_ids,
        validation_group_count=1000,
        seed=17,
        forced_train_group_ids=(FORCED_TRAIN_GROUP,),
    )
    group_ids_by_split = {name: [row["group_id"] for row in rows] for name, rows in split.items()}
    pair_units: dict[str, list[dict[str, str]]] = {}
    examples: dict[str, list[dict[str, str | int]]] = {}
    for name in ("train", "validation"):
        pair_units[name] = build_pair_units(split[name], seed=17)
        examples[name] = [
            example for unit in pair_units[name] for example in expand_pair_unit(unit)
        ]

    training_ids = set(group_ids_by_split["train"])
    validation_ids = set(group_ids_by_split["validation"])
    if training_ids & validation_ids:
        raise AssertionError("source group leakage across split")
    for name in ("train", "validation"):
        allowed = set(group_ids_by_split[name])
        if any(
            row["audio_group"] not in allowed or row["video_group"] not in allowed
            for row in examples[name]
        ):
            raise AssertionError("pair uses a group outside its frozen split")

    source_hashes = config.get("source_hashes", {})
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("pinned source hashes are required")
    verify_pinned_source_hashes(source_hashes, repo_root=Path(__file__).resolve().parents[1])
    return {
        "schema_version": 1,
        "task": {
            "id": "robomnist-av-same-repetition-v1",
            "question": QUESTION,
            "options": OPTIONS,
            "target_encoding": {"different repetition": 0, "same repetition": 1},
            "pair_construction": (
                "two matched and two swapped rows per disjoint repetition-group dyad"
            ),
            "inference_unit": "pair_unit_id; cluster bootstrap over pair units",
        },
        "source": {
            "doi": config.get("source_doi"),
            "license": "CC BY 4.0",
            "metadata_only": True,
            "prior_metadata_exposure": True,
            "full_media_authentication_limit": (
                "one archive lacks a locally computed whole-archive MD5; a publisher-"
                "expected MD5 exists. Ranged members require exact size, compressed size, "
                "and CRC checks; publisher per-member SHA-256 values are unavailable"
            ),
            "config_sha256": sha256_file(config_path),
            "pinned_source_hashes": source_hashes,
            "directory_metadata_sha256": directory_hashes,
            "prior_nonsealed_pool": {
                "run_path": str(prior_run),
                "manifest_hashes": prior_hashes,
                "group_count": len(prior_group_ids),
                "group_ids_sha256": _sha256_bytes("\n".join(prior_group_ids).encode()),
                "groups": prior_group_ids,
                "eligibility_note": (
                    "IDs came only from the prior run's train and validation manifests; "
                    "its sealed-audit file was not opened. The prior summary records no "
                    "payload access, training, or inference."
                ),
            },
        },
        "split": {
            "seed": 17,
            "rule": (
                "seeded whole-repetition split only within the prior non-audit pool; "
                "even per-condition counts are paired before any media access"
            ),
            "group_counts": {name: len(ids) for name, ids in group_ids_by_split.items()},
            "group_ids_sha256": {
                name: _sha256_bytes("\n".join(ids).encode("utf-8"))
                for name, ids in group_ids_by_split.items()
            },
            "eligible_pool_group_count": len(prior_group_ids),
            "selected_group_count": len(training_ids | validation_ids),
            "excluded_group_count": len(group_rows) - len(training_ids | validation_ids),
            "unselected_prior_nonsealed_group_count": len(prior_group_ids)
            - len(training_ids | validation_ids),
            "excluded_source_groups_written": False,
        },
        "media_policy": (
            "whole-episode Rx1 video+microphone only; no CSI or PositionGT inputs; "
            "group IDs and source filenames are metadata only and never model inputs"
        ),
        "groups": {
            group_id: group_sources[group_id]
            for name in ("train", "validation")
            for group_id in group_ids_by_split[name]
        },
        "pair_unit_counts": {name: len(units) for name, units in pair_units.items()},
        "pair_units": pair_units,
        "example_counts": {name: len(rows) for name, rows in examples.items()},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    manifest = build_metadata_manifest(args.metadata_root.resolve(), args.prior_run.resolve())
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "execution-source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest_sha256": sha256_file(manifest_path),
                "groups": manifest["split"]["group_counts"],
                "pairs": manifest["pair_unit_counts"],
                "examples": manifest["example_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
