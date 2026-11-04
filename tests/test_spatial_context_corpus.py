import copy

import pytest

from scripts.extract_spatial_context_corpus import validate_plans


def fixture():
    manifests = {
        s: {
            "clips": [
                {
                    "video_id": f"video_{i}",
                    "sha256": f"media{i}",
                    "path": f"{i}.mp4",
                    "cut_frame": 63,
                }
            ]
        }
        for i, s in enumerate(["train", "valid"], 1)
    }
    prior = {
        "features": [
            {
                "video_id": f"video_{i}",
                "split": s,
                "media_sha256": f"media{i}",
                "prepared_sha256": f"prepared{i}",
                "frames": 63,
                "frame_indices": [round(n * 2) for n in range(32)],
                "shape": [256, 512],
                "path": f"{i}.npy",
                "sha256": f"feature{i}",
            }
            for i, s in enumerate(["train", "valid"], 1)
        ]
    }
    return prior, manifests


def test_complete_pinned_media_feature_binding_and_no_input_mutation():
    prior, manifests = fixture()
    before = copy.deepcopy((prior, manifests))
    plans = validate_plans(prior, manifests, expected_counts={"train": 1, "valid": 1})
    assert [p["video_id"] for p in plans] == ["video_1", "video_2"]
    assert [p["split"] for p in plans] == ["train", "valid"]
    assert plans[0]["frames"] == plans[0]["cut_frame"] == 63
    assert (prior, manifests) == before


@pytest.mark.parametrize(
    "kind", ["media", "identity", "prepared", "missing", "time", "shape", "cut", "path"]
)
def test_corruption_cross_split_overlap_and_partial_corpora_fail_closed(kind):
    prior, manifests = fixture()
    if kind == "media":
        prior["features"][0]["media_sha256"] = "changed"
    elif kind == "identity":
        manifests["valid"]["clips"][0]["video_id"] = "video_1"
    elif kind == "prepared":
        prior["features"][1]["prepared_sha256"] = "prepared1"
    elif kind == "missing":
        prior["features"].pop()
    elif kind == "time":
        prior["features"][0]["frame_indices"][1] = 1
    elif kind == "shape":
        prior["features"][0]["shape"] = [255, 512]
    elif kind == "cut":
        manifests["train"]["clips"][0]["cut_frame"] = 62
    else:
        prior["features"][0]["video_id"] = "../outside"
    with pytest.raises(ValueError):
        validate_plans(prior, manifests, expected_counts={"train": 1, "valid": 1})
