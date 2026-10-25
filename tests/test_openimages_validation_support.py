import csv
import json


def test_validation_support_audit_uses_only_frozen_training_target_ids(tmp_path):
    from scripts.audit_openimages_v6_validation_support import audit_validation_support

    images = tmp_path / "validation-images.csv"
    labels = tmp_path / "validation-labels.csv"
    hierarchy = tmp_path / "hierarchy.json"
    with images.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("ImageID", "Subset", "License"))
        writer.writeheader()
        writer.writerows(
            [
                {
                    "ImageID": "v1",
                    "Subset": "validation",
                    "License": "https://creativecommons.org/licenses/by/2.0/",
                },
                {
                    "ImageID": "v2",
                    "Subset": "validation",
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
                for image_id in ("v1", "v2")
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

    result = audit_validation_support(
        validation_images=images,
        validation_labels=labels,
        hierarchy_path=hierarchy,
        frozen_target_labels=("/car",),
    )

    assert result["frozen_target_labels"] == ["/car"]
    assert result["support_by_target"] == {"/car": 1}
    assert result["unique_validation_images_with_task"] == 1
    assert result["target_selection_source"] == "training-only frozen manifest"
