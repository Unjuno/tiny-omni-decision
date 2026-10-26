import base64
import csv
import json

import pytest

from scripts.sample_openimages_v6_tasks import (
    build_manifest_from_files,
    extract_private_attribution_rows,
    require_disjoint_attribution_rows,
)
from tiny_omni_decision.openimages_options import build_split_task_manifest


def _group(image_id, target, negatives):
    return {
        "image_id": image_id,
        "positive_labels": {target},
        "negative_labels": set(negatives),
    }


def test_split_manifest_is_seeded_scoped_licensed_and_image_unique():
    sibling_groups = {
        "/car": ("/boat", "/bicycle", "/bus"),
        "/dog": ("/bird", "/cat", "/horse"),
    }
    display_names = {
        "/car": "Car",
        "/boat": "Boat",
        "/bicycle": "Bicycle",
        "/bus": "Bus",
        "/dog": "Dog",
        "/bird": "Bird",
        "/cat": "Cat",
        "/horse": "Horse",
    }
    groups = [
        _group("a", "/car", ("/boat", "/bicycle", "/bus")),
        _group("b", "/car", ("/boat", "/bicycle", "/bus")),
        _group("c", "/dog", ("/bird", "/cat", "/horse")),
        _group("d", "/dog", ("/bird", "/cat", "/horse")),
        _group("excluded", "/dog", ("/bird", "/cat", "/horse")),
    ]
    kwargs = {
        "grouped_labels": groups,
        "allowed_image_ids": {"a", "b", "c", "d"},
        "target_labels": ("/car", "/dog"),
        "sibling_groups": sibling_groups,
        "display_names": display_names,
        "split": "train",
        "per_target_limit": 2,
        "seed": 17,
    }

    manifest = build_split_task_manifest(**kwargs)

    assert manifest == build_split_task_manifest(**kwargs)
    assert manifest["split"] == "train"
    assert manifest["media_downloaded"] is False
    assert manifest["target_selection_source"] == "training-only frozen manifest"
    assert len(manifest["examples"]) == 4
    assert {row["image_id"] for row in manifest["examples"]} == {"a", "b", "c", "d"}
    assert all(row["split"] == "train" for row in manifest["examples"])
    assert {row["target_label"] for row in manifest["examples"]} == {"/car", "/dog"}
    assert all(
        row["options"][row["target_position"]] == row["target"]
        for row in manifest["examples"]
    )


def test_split_manifest_excludes_prior_image_ids_deterministically():
    sibling_groups = {"/car": ("/boat", "/bicycle", "/bus")}
    display_names = {
        "/car": "Car",
        "/boat": "Boat",
        "/bicycle": "Bicycle",
        "/bus": "Bus",
    }
    kwargs = {
        "grouped_labels": [
            _group(image_id, "/car", ("/boat", "/bicycle", "/bus"))
            for image_id in ("a", "b", "c", "d")
        ],
        "allowed_image_ids": {"a", "b", "c", "d"},
        "target_labels": ("/car",),
        "sibling_groups": sibling_groups,
        "display_names": display_names,
        "split": "validation",
        "per_target_limit": 4,
        "seed": 1729,
        "excluded_image_ids": {"a", "c"},
    }

    manifest = build_split_task_manifest(**kwargs)

    assert manifest == build_split_task_manifest(**kwargs)
    assert {row["image_id"] for row in manifest["examples"]} == {"b", "d"}
    assert manifest["allowed_image_id_count"] == 4
    assert manifest["eligible_image_id_count"] == 2
    assert manifest["excluded_image_id_count"] == 2


def test_split_manifest_rejects_unknown_split_and_empty_allowlist():
    kwargs = {
        "grouped_labels": [],
        "allowed_image_ids": set(),
        "target_labels": ("/car",),
        "sibling_groups": {"/car": ("/boat", "/bicycle", "/bus")},
        "display_names": {"/car": "Car"},
        "split": "test",
        "per_target_limit": 1,
        "seed": 17,
    }
    with pytest.raises(ValueError, match="train or validation"):
        build_split_task_manifest(**kwargs)
    kwargs["split"] = "train"
    with pytest.raises(ValueError, match="allowlist must not be empty"):
        build_split_task_manifest(**kwargs)


def test_file_builder_uses_only_frozen_targets_and_writes_hashable_task_rows(tmp_path):
    images = tmp_path / "images.csv"
    labels = tmp_path / "labels.csv"
    hierarchy = tmp_path / "hierarchy.json"
    descriptions = tmp_path / "classes.csv"
    train_scope = tmp_path / "train-scope.json"
    with images.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ImageID", "Subset", "License"))
        writer.writeheader()
        writer.writerow(
            {
                "ImageID": "a",
                "Subset": "train",
                "License": "https://creativecommons.org/licenses/by/2.0/",
            }
        )
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("ImageID", "Source", "LabelName", "Confidence"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ImageID": "a",
                    "Source": "verification",
                    "LabelName": label,
                    "Confidence": confidence,
                }
                for label, confidence in (
                    ("/car", 1),
                    ("/boat", 0),
                    ("/bicycle", 0),
                    ("/bus", 0),
                )
            ]
        )
    hierarchy.write_text(
        json.dumps(
            {
                "LabelName": "/root",
                "Subcategory": [
                    {"LabelName": label}
                    for label in ("/car", "/boat", "/bicycle", "/bus")
                ],
            }
        ),
        encoding="utf-8",
    )
    with descriptions.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("LabelName", "DisplayName"))
        writer.writeheader()
        writer.writerows(
            [
                {"LabelName": "/car", "DisplayName": "Car"},
                {"LabelName": "/boat", "DisplayName": "Boat"},
                {"LabelName": "/bicycle", "DisplayName": "Bicycle"},
                {"LabelName": "/bus", "DisplayName": "Bus"},
            ]
        )
    train_scope.write_text(
        json.dumps(
            {
                "purpose": "train-only Open Images V6 hierarchy-choice support audit",
                "selected_target_classes": [{"label_id": "/car"}],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest_from_files(
        split="train",
        image_metadata_path=images,
        labels_path=labels,
        hierarchy_path=hierarchy,
        class_descriptions_path=descriptions,
        frozen_train_scope_path=train_scope,
        per_target_limit=1,
        seed=17,
    )

    assert manifest["counts_by_target"] == {"/car": 1}
    assert manifest["examples"][0]["target"] == "Car"
    assert manifest["individual_rights_verified"] is False
    assert len(manifest["source_hashes"]["human_labels_sha256"]) == 64


def test_private_attribution_extract_requires_complete_licensed_source_rows(tmp_path):
    path = tmp_path / "images.csv"
    fields = (
        "ImageID",
        "Subset",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "AuthorProfileURL",
        "Author",
        "OriginalMD5",
        "Thumbnail300KURL",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "ImageID": "a",
                "Subset": "train",
                "OriginalURL": "https://media.example/a.jpg",
                "OriginalLandingURL": "https://www.flickr.com/photos/author/1",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "AuthorProfileURL": "https://www.flickr.com/people/author/",
                "Author": "A Person",
                "OriginalMD5": base64.b64encode(b"0123456789abcdef").decode("ascii"),
                "Thumbnail300KURL": "",
            }
        )
        writer.writerow(
            {
                "ImageID": "ignored",
                "Subset": "train",
                "License": "https://creativecommons.org/licenses/by/2.0/",
            }
        )

    receipt = extract_private_attribution_rows(
        path,
        expected_split="train",
        selected_image_ids={"a"},
    )

    assert len(receipt) == 1
    assert receipt[0]["ImageID"] == "a"
    assert receipt[0]["individual_rights_verified"] is False
    assert receipt[0]["Author"] == "A Person"
    assert receipt[0]["preferred_media_url"] == "https://media.example/a.jpg"
    assert receipt[0]["media_url_type"] == "original"


def test_split_source_md5_overlap_is_rejected():
    common_md5 = base64.b64encode(b"0123456789abcdef").decode("ascii")
    train = [{"ImageID": "train-a", "OriginalMD5": common_md5}]
    validation = [{"ImageID": "validation-b", "OriginalMD5": common_md5}]

    with pytest.raises(ValueError, match="OriginalMD5 overlap"):
        require_disjoint_attribution_rows(train, validation)
