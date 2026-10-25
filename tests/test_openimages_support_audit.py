import csv
import json

import pytest


def test_train_image_allowlist_keeps_only_train_cc_by_rows(tmp_path):
    from scripts.audit_openimages_v6_train_support import load_train_image_allowlist

    path = tmp_path / "images.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("ImageID", "Subset", "License"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ImageID": "train-allowed",
                    "Subset": "train",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                },
                {
                    "ImageID": "train-excluded",
                    "Subset": "train",
                    "License": "https://creativecommons.org/licenses/by-nc/2.0/",
                },
                {
                    "ImageID": "validation",
                    "Subset": "validation",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                },
            ]
        )

    allowed, counts = load_train_image_allowlist(path)

    assert allowed == {"train-allowed"}
    assert counts == {"rows": 3, "train_rows": 2, "cc_by_2_rows": 1}


def test_train_image_allowlist_rejects_duplicate_image_identity(tmp_path):
    from scripts.audit_openimages_v6_train_support import load_train_image_allowlist

    path = tmp_path / "images.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ImageID", "Subset", "License"))
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ImageID": "duplicate",
                    "Subset": "train",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                },
                {
                    "ImageID": "duplicate",
                    "Subset": "train",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                },
            ]
        )

    with pytest.raises(ValueError, match="duplicate ImageID"):
        load_train_image_allowlist(path)


def test_train_support_audit_uses_only_allowlisted_verification_rows(tmp_path):
    from scripts.audit_openimages_v6_train_support import audit_train_support

    images = tmp_path / "images.csv"
    labels = tmp_path / "labels.csv"
    hierarchy = tmp_path / "hierarchy.json"
    descriptions = tmp_path / "classes.csv"
    with images.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ImageID", "Subset", "License"))
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ImageID": "a",
                    "Subset": "train",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                },
                {
                    "ImageID": "b",
                    "Subset": "train",
                    "License": "https://creativecommons.org/licenses/by-nc/2.0/",
                },
            ]
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
                    "ImageID": image_id,
                    "Source": "verification",
                    "LabelName": label,
                    "Confidence": confidence,
                }
                for image_id in ("a", "b")
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

    result = audit_train_support(
        train_images=images,
        train_labels=labels,
        hierarchy_path=hierarchy,
        class_descriptions=descriptions,
    )

    assert result["images_supporting_any_target"] == 1
    assert result["total_positive_target_options"] == 1
    assert result["selected_target_class_count"] == 0
    assert result["validation_or_test_accessed"] is False
