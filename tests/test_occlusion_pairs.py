import pytest

from tiny_omni_decision.occlusion_pairs import QUESTIONS, build_pairs


def test_answer_remap_and_unchanged_target_retained():
    questions = [
        dict(id=i, question=q, options=options, answer_id=target)
        for i, (q, options, target) in enumerate(
            [(QUESTIONS["initial"], ["c", "a", "b"], 1), (QUESTIONS["final"], ["b", "c", "a"], 2)]
        )
    ]
    data = {"video": dict(metadata={"split": "train"}, mc_question=questions)}
    result = build_pairs(data, "train")
    assert len(result["pairs"]) == 1
    assert [q["target"] for q in result["pairs"][0]["questions"]] == [0, 0]
    questions.pop()
    assert not build_pairs(data, "train")["pairs"]
    assert len(build_pairs(data, "train")["held"]) == 1
    with pytest.raises(ValueError):
        build_pairs(data, "test")
