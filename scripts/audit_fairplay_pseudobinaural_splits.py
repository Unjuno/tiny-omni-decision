"""Audit FAIR-Play split metadata without opening media or test records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

_REQUIRED_FALSE_SCOPE_FLAGS = (
    "test_split_content_read",
    "media_bytes_downloaded",
    "models_loaded",
    "features_extracted",
    "training_or_evaluation",
)


def _normalized_ids(part: dict[str, Any], label: str) -> tuple[list[str], set[str]]:
    ids = part.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{label} must contain a nonempty ID list")
    if part.get("rows") != len(ids):
        raise ValueError(f"{label} row count does not match its ID list")

    normalized: list[str] = []
    for item in ids:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} contains an invalid ID")
        stem = Path(item).stem
        if not stem:
            raise ValueError(f"{label} contains an ID with an empty asset stem")
        normalized.append(stem)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} contains duplicate normalized asset IDs")
    return normalized, set(normalized)


def _order_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def audit_splits(metadata: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    """Check train/validation ID isolation and report limits without reading media."""
    scope = metadata.get("audit_scope")
    if not isinstance(scope, dict):
        raise ValueError("metadata-only scope flags are required")
    if scope.get("test_split_content_read") is not False:
        raise ValueError("test split content must remain unread")
    if any(scope.get(name) is not False for name in _REQUIRED_FALSE_SCOPE_FLAGS[1:]):
        raise ValueError("metadata-only scope flags must all be false")

    sources = metadata.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("metadata must include source split lists")
    fair_play = sources.get("fair_play", {}).get("original_random_splits_train_validation_only")
    pseudo_source = sources.get("pseudobinaural", {})
    pseudo = pseudo_source.get("train_validation_only")
    if not isinstance(fair_play, dict) or not isinstance(pseudo, dict):
        raise ValueError("FAIR-Play and PseudoBinaural train/validation lists are required")
    if pseudo_source.get("author_states_five_splits_non-overlapping_by_scene") is not True:
        raise ValueError("the pinned PseudoBinaural scene-split claim is missing")
    if set(pseudo) != {"1", "2", "3", "4", "5"}:
        raise ValueError("expected PseudoBinaural split variants 1 through 5")

    parsed: dict[str, dict[str, tuple[list[str], set[str]]]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    for split_id in sorted(pseudo, key=int):
        parts = pseudo[split_id]
        train_order, train = _normalized_ids(parts.get("train", {}), f"split {split_id} train")
        val_order, val = _normalized_ids(parts.get("val", {}), f"split {split_id} validation")
        overlap = train & val
        if overlap:
            raise ValueError(f"split {split_id} train/validation asset overlap: {len(overlap)}")
        parsed[split_id] = {"train": (train_order, train), "val": (val_order, val)}

    original = fair_play.get("1")
    if not isinstance(original, dict):
        raise ValueError("original FAIR-Play random split 1 is required for reconciliation")
    _, original_train = _normalized_ids(original.get("train", {}), "original split 1 train")
    _, original_val = _normalized_ids(original.get("val", {}), "original split 1 validation")

    preflight_ids = preflight.get("ids")
    preflight_split = preflight.get("split")
    if not isinstance(preflight_ids, list) or not preflight_ids:
        raise ValueError("preflight must contain train IDs")
    if (
        not isinstance(preflight_split, str)
        or "split1" not in preflight_split.lower()
        or "train" not in preflight_split.lower()
    ):
        raise ValueError("media preflight must be restricted to PseudoBinaural split1 train")
    normalized_preflight = [Path(str(item)).stem for item in preflight_ids]
    if len(set(normalized_preflight)) != len(normalized_preflight):
        raise ValueError("preflight contains duplicate IDs")
    preflight_set = set(normalized_preflight)
    split1_train = parsed["1"]["train"][1]
    split1_val = parsed["1"]["val"][1]
    if not preflight_set <= split1_train or preflight_set & split1_val:
        raise ValueError("preflight IDs must all belong to split1 train and not validation")

    for split_id in sorted(parsed, key=int):
        train_order, train = parsed[split_id]["train"]
        val_order, val = parsed[split_id]["val"]
        other_ids: set[str] = set()
        for other_id, other_parts in parsed.items():
            if other_id == split_id:
                continue
            other_ids.update(other_parts["train"][1] | other_parts["val"][1])
        split_reports[split_id] = {
            "train_count": len(train),
            "validation_count": len(val),
            "train_order_sha256": _order_sha256(train_order),
            "validation_order_sha256": _order_sha256(val_order),
            "train_validation_overlap_count": 0,
            "other_split_train_validation_asset_overlap_count": len(
                (train | val) & other_ids
            ),
            "preflight_ids_all_in_train": preflight_set <= train if split_id == "1" else None,
            "preflight_ids_in_validation": sorted(preflight_set & val) if split_id == "1" else None,
        }

    pairwise_overlap: dict[str, int] = {}
    pairwise_validation_overlap: dict[str, int] = {}
    validation_to_other_train_overlap: dict[str, int] = {}
    for left in sorted(parsed, key=int):
        left_ids = parsed[left]["train"][1] | parsed[left]["val"][1]
        for right in sorted(parsed, key=int):
            if int(right) <= int(left):
                continue
            right_ids = parsed[right]["train"][1] | parsed[right]["val"][1]
            pairwise_overlap[f"{left}-{right}"] = len(left_ids & right_ids)
            pairwise_validation_overlap[f"{left}-{right}"] = len(
                parsed[left]["val"][1] & parsed[right]["val"][1]
            )

    for validation_split, validation_parts in parsed.items():
        validation_ids = validation_parts["val"][1]
        for train_split, train_parts in parsed.items():
            if validation_split == train_split:
                continue
            validation_to_other_train_overlap[
                f"{validation_split}-val_in_{train_split}-train"
            ] = len(validation_ids & train_parts["train"][1])

    validation_union = set().union(*(parts["val"][1] for parts in parsed.values()))
    split1_train = parsed["1"]["train"][1]
    split1_known_train_validation = split1_train | parsed["1"]["val"][1]
    validation_union_in_split1_train = validation_union & split1_train

    pseudo_train = parsed["1"]["train"][1]
    pseudo_val = parsed["1"]["val"][1]
    original_union = original_train | original_val
    pseudo_union = pseudo_train | pseudo_val
    shared_union = original_union & pseudo_union
    cross_source = {
        "original_union_count": len(original_union),
        "pseudobinaural_union_count": len(pseudo_union),
        "shared_asset_id_count": len(shared_union),
        "original_only_asset_id_count": len(original_union - pseudo_union),
        "pseudobinaural_only_asset_id_count": len(pseudo_union - original_union),
        "original_train_in_pseudo_train_count": len(original_train & pseudo_train),
        "original_train_in_pseudo_val_count": len(original_train & pseudo_val),
        "original_val_in_pseudo_train_count": len(original_val & pseudo_train),
        "original_val_in_pseudo_val_count": len(original_val & pseudo_val),
    }
    return {
        "status": "PASS_ROW_LEVEL_SPLIT_AUDIT_WITH_SCENE_LIMIT",
        "source_revisions": {
            "fair_play": sources["fair_play"].get("revision"),
            "pseudobinaural": pseudo_source.get("revision"),
        },
        "test_content_read": False,
        "media_read_by_this_audit": False,
        "pseudobinaural_splits": split_reports,
        "pairwise_cross_split_train_validation_asset_overlap": pairwise_overlap,
        "pairwise_validation_id_overlap": pairwise_validation_overlap,
        "validation_to_other_split_train_id_overlap": validation_to_other_train_overlap,
        "validation_union_summary": {
            "total_validation_rows_across_variants": sum(
                len(parts["val"][1]) for parts in parsed.values()
            ),
            "unique_validation_asset_ids": len(validation_union),
            "unique_validation_ids_in_split1_train": len(validation_union_in_split1_train),
            "unique_validation_ids_outside_split1_train_and_validation": len(
                validation_union - split1_known_train_validation
            ),
            "split1_train_ids_after_excluding_validation_union": len(
                split1_train - validation_union
            ),
            "policy": (
                "descriptive only; do not pool without group and sealed-test reconciliation"
            ),
        },
        "cross_split_policy": (
            "overlapping train/validation assignments; not independent evaluation replicates"
        ),
        "original_random_split1_vs_pseudobinaural_split1": cross_source,
        "validation_group_gate": {
            "required_independent_groups": 200,
            "validation_record_count": split_reports["1"]["validation_count"],
            "independent_scene_group_count": None,
            "maximum_possible_independent_scene_groups": split_reports["1"][
                "validation_count"
            ],
            "passes_minimum_group_count": False
            if split_reports["1"]["validation_count"] < 200
            else None,
        },
        "scene_group_independence": "author_claim_not_independently_verified",
        "task_quality_or_both_input_necessity": "not_evaluated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    metadata_bytes = args.metadata.read_bytes()
    preflight_bytes = args.preflight.read_bytes()
    report = audit_splits(
        json.loads(metadata_bytes),
        json.loads(preflight_bytes),
    )
    report["input_sha256"] = {
        "metadata": hashlib.sha256(metadata_bytes).hexdigest(),
        "preflight": hashlib.sha256(preflight_bytes).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
