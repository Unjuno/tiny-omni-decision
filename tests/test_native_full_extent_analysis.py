import numpy as np
import pytest

from scripts.analyze_native_full_extent import retention, visual_bootstrap


def metrics(nll=1.0, image=0.4, video=0.4, audio=0.6, text=0.4):
    return dict(
        macro={"nll": nll},
        modality_macro={
            k: {"accuracy": v}
            for k, v in dict(image=image, video=video, audio=audio, text=text).items()
        },
    )


def test_retention_requires_all_three_preregistered_conditions():
    control = metrics()
    assert retention(control, metrics(nll=0.9, image=0.5))["retain_for_research"]
    assert not retention(control, metrics(nll=1.0, image=0.5))["retain_for_research"]
    assert not retention(control, metrics(nll=0.9))["retain_for_research"]
    assert not retention(control, metrics(nll=0.9, image=0.5, audio=0.57))["retain_for_research"]


def test_paired_bootstrap_preserves_image_and_scene_clusters():
    rows = [
        dict(modality=m, observation=f"{m}:{i}", task=t)
        for m in ("image", "video")
        for i in range(3)
        for t in ("static_descriptive", "temporal_descriptive")
    ]
    first = {k: np.zeros(len(rows)) for k in ("accuracy", "nll", "brier")}
    second = {k: np.ones(len(rows)) * 0.25 for k in first}
    value = visual_bootstrap(rows, first, second, resamples=100)
    assert value == visual_bootstrap(rows, first, second, resamples=100)
    for label, item in value.items():
        assert item["assets"] == 3
        assert item["questions"] == (3 if label == "video_temporal" else 6)
        for name in first:
            assert item[name] == {"candidate_minus_control": 0.25, "low": 0.25, "high": 0.25}


def test_bootstrap_rejects_nonfinite_vectors():
    rows = [
        dict(modality=m, observation=f"{m}:{i}", task="temporal_descriptive")
        for m in ("image", "video")
        for i in range(2)
    ]
    first = {k: np.zeros(len(rows)) for k in ("accuracy", "nll", "brier")}
    second = {k: v.copy() for k, v in first.items()}
    second["nll"][0] = np.nan
    with pytest.raises(ValueError):
        visual_bootstrap(rows, first, second, resamples=10)
