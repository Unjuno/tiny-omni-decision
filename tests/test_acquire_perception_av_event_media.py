from copy import deepcopy

import pytest

from scripts.acquire_perception_av_event_media import validate_plan
from scripts.plan_perception_av_event_media import MEDIA_MANIFESTS, TRAIN_ARCHIVE_PIN


def plan_fixture():
    rows = []
    for index in range(400):
        rows.append(
            {
                "video_id": f"video_{index}",
                "bytes": 100,
                "metadata": {"split": "train", "video_id": f"video_{index}"},
                "model_split": "train" if index < 300 else "calibration",
            }
        )
    return {
        "status": "PASS_FRESH_EVENT_MEDIA_PLAN",
        "test_opened": False,
        "training": False,
        "gpu_used": False,
        "previous_media_manifest_sha256": MEDIA_MANIFESTS,
        "splits": {
            "train": {
                "pin": TRAIN_ARCHIVE_PIN,
                "rows": rows,
                "total_uncompressed_bytes": 40_000,
            }
        },
    }


def test_media_acquisition_accepts_separate_train_and_calibration_groups():
    validated = validate_plan(plan_fixture(), "train")
    assert len(validated["rows"]) == 400


def test_media_acquisition_rejects_test_scope_and_mutated_manifest_pins():
    test_plan = plan_fixture()
    test_plan["test_opened"] = True
    with pytest.raises(ValueError, match="train/valid-only"):
        validate_plan(test_plan, "train")
    changed_pins = plan_fixture()
    changed_pins["previous_media_manifest_sha256"] = {"changed": "0"}
    with pytest.raises(ValueError, match="inventory pins changed"):
        validate_plan(changed_pins, "train")


def test_media_acquisition_rejects_wrong_split_or_unbounded_download():
    wrong_split = plan_fixture()
    wrong_split["splits"]["train"]["rows"][0]["metadata"]["split"] = "valid"
    with pytest.raises(ValueError, match="invalid split"):
        validate_plan(wrong_split, "train")
    too_big = deepcopy(plan_fixture())
    too_big["splits"]["train"]["rows"][0]["bytes"] = 256_000_001
    with pytest.raises(ValueError, match="invalid split"):
        validate_plan(too_big, "train")
