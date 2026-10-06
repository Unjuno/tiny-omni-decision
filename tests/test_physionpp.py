from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tiny_omni_decision.corpus import source_asset_identity
from tiny_omni_decision.dataset import normalize_jsonl
from tiny_omni_decision.io import load_structured_file
from tiny_omni_decision.physionpp import (
    build_readout_rows,
    split_readout_examples,
)
from tiny_omni_decision.schema import DatasetManifest, DecisionExample, LicenseProvenance, MediaRef
from tiny_omni_decision.training import sampling_accounting

ROOT = Path(__file__).resolve().parents[1]


def test_physionpp_clean_dev_v3_candidate_keeps_e_long_training_settings() -> None:
    control = load_structured_file(ROOT / "configs/decision/teacher_quality_clean_dev_v2.yaml")
    candidate = load_structured_file(
        ROOT / "configs/decision/teacher_v2_physionpp_clean_dev_v3.yaml"
    )
    candidate["teacher_id"] = control["teacher_id"]

    assert candidate == control
    training = candidate["training"]
    assert training["seed"] == 17
    assert training["max_steps"] == 2048
    assert training["gradient_accumulation_steps"] == 4
    assert training["video_num_frames"] == 8
    assert training["lora_target_policy"] == "decoder_all_linear"
    assert training["rank"] == 16
    assert training["use_rslora"] is False
    assert training["early_stopping_patience"] == 17


def _row(name: str, seed: int, label: bool, path: str) -> dict[str, object]:
    return {
        "stimulus_name": name,
        "trial_seed": seed,
        "does_target_contact_zone": label,
        "scenario": path.split("/")[1],
        "zip_metadata_path": path,
        "num_frames": 50,
    }


def test_placeholder_rows_map_by_contiguous_media_index_and_keep_scene_group() -> None:
    metadata_path = "readout_data_v1/bouncy_wall_pp/config-a/metadata.json"
    records = [_row("None", 17, True, metadata_path), _row("None", 18, False, metadata_path)]
    members = {
        f"readout_data_v1/bouncy_wall_pp/config-a/{stem}{suffix}"
        for stem in ("0000", "0001")
        for suffix in ("_img.mp4", "_id.mp4", ".pkl")
    }

    rows, audit = build_readout_rows(
        records,
        members,
        source_revision="r1",
        split="dev",
    )

    assert [row["id"] for row in rows] == [
        "readout_data_v1/bouncy_wall_pp/config-a#0000",
        "readout_data_v1/bouncy_wall_pp/config-a#0001",
    ]
    assert rows[0]["target"] == "Yes"
    assert rows[1]["target"] == "No"
    assert rows[0]["task_group_id"] == rows[0]["id"]
    assert rows[1]["task_group_id"] == rows[1]["id"]
    assert rows[0]["source_asset_group_id"] == "bouncy_wall_pp:seed=17"
    assert rows[1]["source_asset_group_id"] == "bouncy_wall_pp:seed=18"
    assert rows[0]["media"][0]["num_frames"] == 50
    assert rows[0]["task_type"] == "temporal_descriptive"
    assert audit["placeholder_rows_mapped"] == 2


def test_placeholder_mapping_fails_closed_if_media_indices_are_not_contiguous() -> None:
    metadata_path = "readout_data_v1/bouncy_wall_pp/config-a/metadata.json"
    records = [_row("None", 17, True, metadata_path)]
    members = {
        f"readout_data_v1/bouncy_wall_pp/config-a/{stem}{suffix}"
        for stem in ("0000", "0002")
        for suffix in ("_img.mp4", "_id.mp4", ".pkl")
    }

    try:
        build_readout_rows(
            records,
            members,
            source_revision="r1",
            split="dev",
        )
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("ambiguous placeholder identity must fail closed")


def test_unmatched_named_rows_are_reported_and_duplicate_media_is_deduplicated() -> None:
    metadata_path = "readout_data_v1/friction_collision_pp/config-b/metadata.json"
    record = _row("friction_collision_0000", 41, True, metadata_path)
    records = [record, record.copy()]
    members = {
        f"readout_data_v1/friction_collision_pp/config-b/0000{suffix}"
        for suffix in ("_img.mp4", "_id.mp4", ".pkl")
    }
    records.append(_row("friction_collision_0001", 43, False, metadata_path))

    rows, audit = build_readout_rows(
        records,
        members,
        source_revision="r1",
        split="dev",
    )

    assert len(rows) == 1
    assert audit["duplicate_rows"] == 1
    assert audit["unmatched_rows"] == 1


def test_placeholder_and_named_rows_cannot_be_mixed_in_one_metadata_file() -> None:
    metadata_path = "readout_data_v1/bouncy_wall_pp/config-c/metadata.json"
    records = [
        _row("None", 17, True, metadata_path),
        _row("bouncy_wall_0001", 18, False, metadata_path),
    ]
    members = {
        f"readout_data_v1/bouncy_wall_pp/config-c/{stem}{suffix}"
        for stem in ("0000", "0001")
        for suffix in ("_img.mp4", "_id.mp4", ".pkl")
    }

    try:
        build_readout_rows(
            records,
            members,
            source_revision="r1",
            split="dev",
        )
    except ValueError as error:
        assert "mix" in str(error)
    else:
        raise AssertionError("mixed explicit and ordinal identities must fail closed")


def test_generic_normalization_preserves_physion_task_and_scene_group() -> None:
    manifest = DatasetManifest.model_validate(
        load_structured_file(ROOT / "manifests/candidates/physionpp-readout.yaml")
    )
    rows = [
        {
            "id": "readout/a#0000",
            "split": "readout-development",
            "state": "A simulated physical-scene video is supplied.",
            "question": "Will contact occur?",
            "options": ["No", "Yes"],
            "target": "Yes",
            "source_target": True,
            "task_type": "temporal_descriptive",
            "task_group_id": "record-a",
            "source_asset_group_id": "mass_dominoes_pp:seed=72",
            "media": [
                {
                    "kind": "video",
                    "uri": "source-ref://physionpp-readout/r1/readout_data_v1/a/0000_img.mp4",
                    "license": "MIT",
                    "num_frames": 50,
                }
            ],
        },
        {
            "id": "readout/b#0001",
            "split": "readout-development",
            "state": "A simulated physical-scene video is supplied.",
            "question": "Will contact occur?",
            "options": ["No", "Yes"],
            "target": "No",
            "source_target": False,
            "task_type": "temporal_descriptive",
            "task_group_id": "record-b",
            "source_asset_group_id": "mass_dominoes_pp:seed=72",
            "media": [
                {
                    "kind": "video",
                    "uri": "source-ref://physionpp-readout/r1/readout_data_v1/b/0001_img.mp4",
                    "license": "MIT",
                    "num_frames": 55,
                }
            ],
        },
    ]

    examples = list(normalize_jsonl(rows, manifest, "generic"))

    assert [item.source_target for item in examples] == [True, False]
    assert [item.task_type for item in examples] == [
        "temporal_descriptive",
        "temporal_descriptive",
    ]
    assert examples[0].task_group_id != examples[1].task_group_id
    assert examples[0].source_asset_group_id == examples[1].source_asset_group_id
    assert source_asset_identity(examples[0]) == source_asset_identity(examples[1])
    accounting = sampling_accounting(examples)
    assert accounting["unique_questions_by_task_type"] == {"temporal_descriptive": 2}
    assert accounting["unique_underlying_assets"] == 1


def test_physion_train_validation_gate_rejects_shared_scene_seed() -> None:
    from tiny_omni_decision.physionpp import assert_physionpp_train_validation_groups

    group = "mass_dominoes_pp:seed=72"
    row = {
        "task_group_id": "record-a",
        "source_asset_group_id": group,
        "media": [{"uri": "source-ref://physionpp-readout/r1/video.mp4"}],
    }

    with pytest.raises(ValueError, match="scene/seed groups cross"):
        assert_physionpp_train_validation_groups(
            [row],
            [
                row
                | {
                    "task_group_id": "record-b",
                    "media": [{"uri": "source-ref://physionpp-readout/r1/other.mp4"}],
                }
            ],
        )


def test_physion_group_split_is_deterministic_and_keeps_each_scene_seed_together() -> None:
    examples = [
        DecisionExample(
            id=f"physion:{group}:{index}",
            modality="video",
            state="Simulated physical scene.",
            question=f"Will event {index} happen?",
            options=["No", "Yes"],
            target="Yes" if index % 2 else "No",
            media=[MediaRef(kind="video", uri=f"source-ref://physion/{group}/{index}.mp4")],
            source="physionpp/readout",
            source_revision="a" * 40,
            source_record_id=f"{group}:{index}",
            split="readout-development",
            task_type="temporal_descriptive",
            task_group_id=f"{group}:{index}",
            source_asset_group_id=group,
            provenance=LicenseProvenance(
                license="MIT",
                commercial_use=True,
                derivative_model_training_allowed=True,
                redistribution_allowed=True,
                media_redistribution_allowed=True,
                trust_status="trusted",
            ),
        )
        for group in (f"scenario-{group}:seed={group}" for group in range(5))
        for index in range(2)
    ]

    train, validation, report = split_readout_examples(examples, seed=17, validation_fraction=0.4)
    train_again, validation_again, _ = split_readout_examples(
        examples, seed=17, validation_fraction=0.4
    )

    assert [item.id for item in train] == [item.id for item in train_again]
    assert [item.id for item in validation] == [item.id for item in validation_again]
    assert len({item.source_asset_group_id for item in validation}) == 2
    assert all(item.split == "train" for item in train)
    assert all(item.split == "validation" for item in validation)
    assert report["integrity"]["status"] == "disjoint"


def test_deterministic_asset_holdout_is_balanced_and_reproducible() -> None:
    from tiny_omni_decision.development import deterministic_asset_holdout

    provenance = LicenseProvenance(
        license="MIT",
        commercial_use=True,
        derivative_model_training_allowed=True,
        redistribution_allowed=True,
        media_redistribution_allowed=True,
        trust_status="trusted",
    )
    examples = [
        DecisionExample(
            id=f"{source}:{asset}:{index}",
            modality="text",
            state=f"state {asset}",
            question=f"question {index}",
            options=["no", "yes"],
            target="yes",
            source=source,
            source_record_id=f"{asset}:{index}",
            source_asset_group_id=str(asset),
            source_revision="a" * 40,
            split="train",
            provenance=provenance,
        )
        for source in ("text/a", "text/b")
        for asset in range(4)
        for index in range(2)
    ]

    selected, groups = deterministic_asset_holdout(examples, modality="text", count=4, seed=17)
    repeated, repeated_groups = deterministic_asset_holdout(
        examples, modality="text", count=4, seed=17
    )

    assert [item.id for item in selected] == [item.id for item in repeated]
    assert groups == repeated_groups
    assert len(groups) == len(selected) == 4
    assert len({(item.source, item.source_asset_group_id) for item in selected}) == 4
    assert {item.source for item in selected} == {"text/a", "text/b"}


def test_deterministic_asset_holdout_fails_when_unique_assets_are_insufficient() -> None:
    from tiny_omni_decision.development import deterministic_asset_holdout

    provenance = LicenseProvenance(
        license="MIT",
        commercial_use=True,
        derivative_model_training_allowed=True,
        redistribution_allowed=True,
        media_redistribution_allowed=True,
        trust_status="trusted",
    )
    example = DecisionExample(
        id="only-one",
        modality="video",
        state="state",
        question="question",
        options=["no", "yes"],
        target="yes",
        source="video/source",
        source_record_id="one",
        source_asset_group_id="scene-1",
        media=[MediaRef(kind="video", uri="source-ref://video/scene-1.mp4")],
        source_revision="a" * 40,
        split="train",
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="only 1 eligible"):
        deterministic_asset_holdout([example], modality="video", count=2, seed=17)


def test_media_end_frame_is_video_only_and_positive() -> None:
    with pytest.raises(ValidationError, match="only valid for video"):
        MediaRef(kind="image", path="image.png", end_frame=3)
    with pytest.raises(ValidationError):
        MediaRef(kind="video", path="clip.mp4", end_frame=0)
    with pytest.raises(ValidationError):
        MediaRef(kind="video", path="clip.mp4", end_frame="3")
    with pytest.raises(ValidationError, match="only valid for video"):
        MediaRef(kind="image", path="image.png", num_frames=3)


def test_physionpp_crop_contains_exactly_prefix_frames(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    numpy = pytest.importorskip("numpy")
    try:
        av.codec.Codec("libx264", "w")
    except av.codec.codec.UnknownCodecError:
        pytest.skip("PyAV libx264 encoder is unavailable")

    from tiny_omni_decision.media import _truncate_physionpp_video, _video_frame_count

    source = tmp_path / "source.mp4"
    cropped = tmp_path / "cropped.mp4"
    with av.open(str(source), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(7):
            pixels = numpy.full((32, 32, 3), index * 30, dtype=numpy.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    _truncate_physionpp_video(source, cropped, end_frame=4)

    assert _video_frame_count(source) == 7
    assert _video_frame_count(cropped) == 4
