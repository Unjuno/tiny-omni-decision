from copy import deepcopy
from math import log

import pytest

from scripts.report_shared_text_video import paired_video_metrics


def row(identity, scene, kind, correct):
    return dict(
        id=identity,
        scene_index=scene,
        parent_question_id=identity,
        question_type=kind,
        target_index=0,
        prediction_index=0 if correct else 1,
        probabilities=[0.8, 0.2] if correct else [0.2, 0.8],
    )


def test_scene_comparison_macros_question_types_not_question_counts():
    left, right = [], []
    for scene in [0, 1]:
        for kind in ["explanatory", "predictive", "counterfactual"]:
            n = 3 if scene == 0 and kind == "explanatory" else 1
            for i in range(n):
                identity = f"{scene}-{kind}-{i}"
                left.append(row(identity, scene, kind, scene == 1))
                right.append(
                    row(identity, scene, kind, scene == 0 if kind == "explanatory" else scene == 1)
                )
    result = paired_video_metrics(left, right, draws=100)
    assert result["metrics"]["accuracy"]["difference"] == pytest.approx(1 / 6)
    assert result["metrics"]["nll"]["difference"] == pytest.approx(-log(4) / 6)
    assert result["metrics"]["brier"]["difference"] == pytest.approx(-0.2)
    assert result["scenes"] == 2
    assert paired_video_metrics(left, left, draws=100)["metrics"]["nll"]["difference"] == 0
    with pytest.raises(ValueError):
        paired_video_metrics(left, right[::-1], draws=100)
    invalid = deepcopy(right)
    invalid[0]["probabilities"] = [0.8, 0.8]
    with pytest.raises(ValueError):
        paired_video_metrics(left, invalid, draws=100)

    one_scene = [r for r in left if r["scene_index"] == 0]
    with pytest.raises(ValueError):
        paired_video_metrics(one_scene, one_scene, draws=100)
