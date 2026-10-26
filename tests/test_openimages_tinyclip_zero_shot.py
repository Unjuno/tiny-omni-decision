import math

import numpy as np
import pytest

from scripts.evaluate_openimages_tinyclip_zero_shot import (
    gather_choice_logits,
    probabilities_from_logits,
    quality_metrics,
    validate_rows,
    validate_task_media_rows,
)


def _row_bundle():
    task = {
        "image_id": "image-1",
        "sample_id": "sample-1",
        "split": "validation",
        "options": ["Cat", "Dog", "Boat", "Car"],
        "target": "Dog",
        "target_label": "/dog",
        "media_sha256": "a" * 64,
        "page_license_and_attribution_match": True,
    }
    feature = {
        "image_id": "image-1",
        "sample_id": "sample-1",
        "split": "validation",
        "target": "Dog",
        "media_sha256": "a" * 64,
    }
    media = {
        "image_id": "image-1",
        "sample_id": "sample-1",
        "split": "validation",
        "target": "Dog",
        "media_sha256": "a" * 64,
        "status": "saved",
        "page_license_and_attribution_match": True,
        "individual_rights_verified": False,
    }
    return [task], [feature], [media]


def test_validate_rows_accepts_aligned_page_matched_four_choice_inputs():
    validate_rows(*_row_bundle())


def test_validate_task_media_rows_does_not_require_another_encoder_cache():
    tasks, _, media = _row_bundle()
    tasks[0].pop("media_sha256")
    validate_task_media_rows(tasks, media)

    tasks[0]["media_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="task/media hash mismatch"):
        validate_task_media_rows(tasks, media)


def test_validate_rows_rejects_repeated_image_or_sample_ids():
    tasks, features, media = _row_bundle()
    tasks.append(tasks[0] | {"sample_id": "sample-2"})
    features.append(features[0] | {"sample_id": "sample-2"})
    media.append(media[0] | {"sample_id": "sample-2"})
    with pytest.raises(ValueError, match="image IDs must be nonempty and unique"):
        validate_rows(tasks, features, media)

    tasks, features, media = _row_bundle()
    tasks.append(tasks[0] | {"image_id": "image-2"})
    features.append(features[0] | {"image_id": "image-2"})
    media.append(media[0] | {"image_id": "image-2"})
    with pytest.raises(ValueError, match="sample IDs must be nonempty and unique"):
        validate_rows(tasks, features, media)


def test_validate_rows_rejects_wrong_split_rights_status_and_options():
    tasks, features, media = _row_bundle()
    media[0]["page_license_and_attribution_match"] = False
    with pytest.raises(ValueError, match="only saved media with a matching page declaration"):
        validate_rows(tasks, features, media)

    tasks, features, media = _row_bundle()
    tasks[0]["options"] = ["Dog"]
    with pytest.raises(ValueError, match="exactly four unique options"):
        validate_rows(tasks, features, media)

    tasks, features, media = _row_bundle()
    features[0]["split"] = "train"
    with pytest.raises(ValueError, match="all rows must belong to the requested split"):
        validate_rows(tasks, features, media)


def test_quality_metrics_compute_four_choice_calibration_values():
    result = quality_metrics(np.zeros((1, 4), dtype=np.float32), np.asarray([0]))
    assert result["count"] == 1
    assert result["accuracy"] == 1.0
    assert result["nll"] == pytest.approx(math.log(4))
    assert result["brier"] == pytest.approx(0.75)
    assert result["ece_15_bins"] == pytest.approx(0.75)
    assert sum(result["ece_bin_counts"]) == 1


def test_quality_metrics_reject_nonfinite_or_non_four_choice_logits():
    with pytest.raises(ValueError, match="expected four-choice logits"):
        quality_metrics(np.zeros((1, 3)), np.asarray([0]))
    with pytest.raises(ValueError, match="invalid values"):
        quality_metrics(np.asarray([[0.0, np.nan, 0.0, 0.0]]), np.asarray([0]))


def test_gathers_supplied_option_scores_and_normalizes_probabilities():
    task_rows = [{"options": ["Dog", "Cat", "Car", "Boat"]}]
    scores = np.asarray([[3.0, 1.0, 2.0, 0.0]], dtype=np.float32)
    gathered = gather_choice_logits(scores, task_rows, {"Cat": 1, "Dog": 0, "Boat": 3, "Car": 2})
    probabilities = probabilities_from_logits(gathered)
    assert gathered.tolist() == [[3.0, 1.0, 2.0, 0.0]]
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0])
    assert int(probabilities.argmax(axis=1)[0]) == 0
