from __future__ import annotations

from scripts.pretrained_reuse.evaluate_jaqket_dev2_once import _split_overlap


def _row(qid: str, question: str, answer: str, options: list[str]) -> dict[str, object]:
    return {
        "qid": qid,
        "question": question,
        "answer_entity": answer,
        "answer_candidates": options,
    }


def test_dev2_overlap_audit_checks_qid_question_target_and_candidate_set() -> None:
    options = [f"choice-{index}" for index in range(20)]
    options[0] = "Answer"
    dev1 = [_row("id-1", "A question?", "Answer", options)]
    dev2 = [_row("id-2", "a question?", "answer", list(reversed(options)))]

    overlap = _split_overlap(dev1, dev2)

    assert overlap == {
        "qid": 0,
        "normalized_question": 1,
        "answer_entity": 1,
        "unordered_candidate_set": 1,
    }


def test_dev2_overlap_audit_allows_candidate_option_titles_to_repeat() -> None:
    dev1_options = [f"choice-{index}" for index in range(20)]
    dev1_options[0] = "Answer 1"
    dev2_options = [f"other-{index}" for index in range(20)]
    dev2_options[0] = "Answer 2"
    dev2_options[1] = dev1_options[1]
    dev1 = [_row("id-1", "question 1", "Answer 1", dev1_options)]
    dev2 = [_row("id-2", "question 2", "Answer 2", dev2_options)]

    assert _split_overlap(dev1, dev2) == {
        "qid": 0,
        "normalized_question": 0,
        "answer_entity": 0,
        "unordered_candidate_set": 0,
    }
