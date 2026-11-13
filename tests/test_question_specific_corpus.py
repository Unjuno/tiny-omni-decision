import pytest

from scripts.question_specific_corpus import convert


def test_preserves_each_questions_options_and_video_split():
    source = {"v": {"mc_question": [
        dict(id=1, question="first", options=["a", "b", "c"], answer_id=1),
        dict(id=2, question="last", options=["d", "a", "b"], answer_id=0),
    ]}}
    rows = convert(source, ["v"], "development")
    assert rows[0]["options"] == ["a", "b", "c"]
    assert rows[1]["options"] == ["d", "a", "b"]
    assert [r["target"] for r in rows] == [1, 0]
    assert {r["split"] for r in rows} == {"development"}
    assert len({r["id"] for r in rows}) == 2


def test_duplicate_question_ids_rejected():
    q = dict(id=1, question="q", options=["a", "b"], answer_id=0)
    with pytest.raises(ValueError, match="duplicate"):
        convert({"v": {"mc_question": [q, q]}}, ["v"], "fit")
