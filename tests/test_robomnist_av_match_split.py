from __future__ import annotations

from collections import Counter

import pytest

from scripts.robomnist_av_match_split import (
    build_eligible_group_split,
    build_group_split,
    build_pair_units,
    expand_pair_unit,
    load_prior_nonsealed_group_pool,
)


def groups_by_condition(count: int = 6) -> list[dict[str, str]]:
    return [
        {"group_id": f"{condition}_Rp{index:02d}_UNC062", "condition": condition}
        for condition in ("Ar1_Vhigh_Ac0", "Ar1_Vlow_Ac1", "Ar2_Vhigh_Ac0", "Ar2_Vmed_Ac1")
        for index in range(1, count + 1)
    ]


def test_split_is_group_disjoint_balanced_and_deterministic() -> None:
    groups = groups_by_condition()
    kwargs = {
        "seed": 17,
        "base_per_condition": 2,
        "extra_conditions_per_split": 1,
        "forced_train_group_ids": ("Ar1_Vhigh_Ac0_Rp01_UNC062",),
    }

    first = build_group_split(groups, **kwargs)
    second = build_group_split(groups, **kwargs)

    assert first == second
    assert len(first["train"]) == 10
    assert len(first["validation"]) == 10
    assert len(first["reserve"]) == 4
    assert "Ar1_Vhigh_Ac0_Rp01_UNC062" in {row["group_id"] for row in first["train"]}

    split_ids = [
        {row["group_id"] for row in first[name]} for name in ("train", "validation", "reserve")
    ]
    assert not (
        split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]
    )
    assert set.union(*split_ids) == {row["group_id"] for row in groups}

    for split_name in ("train", "validation"):
        counts = Counter(row["condition"] for row in first[split_name])
        assert set(counts.values()) <= {2, 4}


def test_pair_units_keep_groups_in_one_stratum_and_use_each_once() -> None:
    split = build_group_split(
        groups_by_condition(),
        seed=17,
        base_per_condition=2,
        extra_conditions_per_split=1,
        forced_train_group_ids=("Ar1_Vhigh_Ac0_Rp01_UNC062",),
    )

    units = build_pair_units(split["train"], seed=17)

    assert len(units) == 5
    used = [group_id for unit in units for group_id in (unit["group_a"], unit["group_b"])]
    assert len(used) == len(set(used)) == 10
    by_id = {row["group_id"]: row["condition"] for row in split["train"]}
    assert all(
        by_id[unit["group_a"]] == by_id[unit["group_b"]] == unit["condition"] for unit in units
    )


def test_pair_expansion_balances_each_audio_and_video_instance_across_labels() -> None:
    unit = {
        "pair_unit_id": "pair-1",
        "condition": "Ar1_Vhigh_Ac0",
        "group_a": "Ar1_Vhigh_Ac0_Rp01_UNC062",
        "group_b": "Ar1_Vhigh_Ac0_Rp02_UNC062",
    }

    rows = expand_pair_unit(unit)

    assert [(row["audio_group"], row["video_group"], row["target"]) for row in rows] == [
        (unit["group_a"], unit["group_a"], 1),
        (unit["group_a"], unit["group_b"], 0),
        (unit["group_b"], unit["group_b"], 1),
        (unit["group_b"], unit["group_a"], 0),
    ]
    for modality in ("audio_group", "video_group"):
        by_instance: dict[str, list[int]] = {}
        for row in rows:
            by_instance.setdefault(row[modality], []).append(row["target"])
        assert {group_id: sorted(targets) for group_id, targets in by_instance.items()} == {
            unit["group_a"]: [0, 1],
            unit["group_b"]: [0, 1],
        }


def test_eligible_split_excludes_out_of_pool_groups_and_meets_pair_unit_budget() -> None:
    groups = groups_by_condition(10)
    eligible_ids = {row["group_id"] for row in groups[:30]}
    split = build_eligible_group_split(
        groups,
        eligible_group_ids=eligible_ids,
        validation_group_count=14,
        seed=17,
        forced_train_group_ids=("Ar1_Vhigh_Ac0_Rp01_UNC062",),
    )

    train_ids = {row["group_id"] for row in split["train"]}
    validation_ids = {row["group_id"] for row in split["validation"]}
    assert len(train_ids) == 16
    assert len(validation_ids) == 14
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids <= eligible_ids
    assert "Ar1_Vhigh_Ac0_Rp01_UNC062" in train_ids
    assert len(build_pair_units(split["validation"], seed=17)) == 7


def test_eligible_split_requires_whole_pairs_and_two_sided_capacity() -> None:
    groups = groups_by_condition(5)
    eligible_ids = {row["group_id"] for row in groups}
    with pytest.raises(ValueError, match="even"):
        build_eligible_group_split(
            groups,
            eligible_group_ids=eligible_ids,
            validation_group_count=19,
            seed=17,
            forced_train_group_ids=(),
        )


def test_prior_nonsealed_pool_reads_only_hashed_train_validation_and_summary(tmp_path) -> None:
    import hashlib
    import json

    train = [{"source_refs": {"video": {"group_id": "Ar1_Vhigh_Ac0_Rp01_UNC062"}}}]
    validation = [{"audio_group_id": "Ar1_Vhigh_Ac0_Rp02_UNC062"}]
    train_bytes = json.dumps(train).encode()
    validation_bytes = json.dumps(validation).encode()
    (tmp_path / "train.json").write_bytes(train_bytes)
    (tmp_path / "validation.json").write_bytes(validation_bytes)
    summary = {
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
        "payload_opened": False,
        "training": False,
        "model_inference": False,
        "sealed_audit_evaluated": False,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    group_ids, hashes = load_prior_nonsealed_group_pool(tmp_path)

    assert group_ids == [
        "Ar1_Vhigh_Ac0_Rp01_UNC062",
        "Ar1_Vhigh_Ac0_Rp02_UNC062",
    ]
    assert set(hashes) == {"summary_sha256", "train_sha256", "validation_sha256"}
    assert not (tmp_path / "sealed_audit.json").exists()


def test_prior_pool_rejects_payload_exposed_or_changed_nonsealed_manifest(tmp_path) -> None:
    import json

    (tmp_path / "train.json").write_text("[]", encoding="utf-8")
    (tmp_path / "validation.json").write_text("[]", encoding="utf-8")
    summary = {
        "train_sha256": "0" * 64,
        "validation_sha256": "0" * 64,
        "payload_opened": False,
        "training": False,
        "model_inference": False,
        "sealed_audit_evaluated": False,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_prior_nonsealed_group_pool(tmp_path)


def test_split_rejects_insufficient_pair_capacity_and_unknown_forced_group() -> None:
    groups = groups_by_condition()
    with pytest.raises(ValueError, match="even"):
        build_group_split(
            groups_by_condition(5),
            seed=17,
            base_per_condition=2,
            extra_conditions_per_split=1,
            forced_train_group_ids=(),
        )

    with pytest.raises(ValueError, match="forced train group"):
        build_group_split(
            groups,
            seed=17,
            base_per_condition=2,
            extra_conditions_per_split=1,
            forced_train_group_ids=("missing-group",),
        )


def test_source_hash_pins_fail_closed_when_a_pinned_file_changes(tmp_path) -> None:
    from scripts.robomnist_av_match_split import verify_pinned_source_hashes

    source = tmp_path / "source.json"
    source.write_text("pinned", encoding="utf-8")
    expected = "a" * 64

    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_pinned_source_hashes({str(source): expected}, repo_root=tmp_path)
