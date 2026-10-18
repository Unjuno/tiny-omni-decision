import pytest

from scripts.clotho_aqa_split import split_questions


def row(source="1", uploader="Person", question="Is there speech?"):
    return dict(source_id=source, uploader=uploader, file_name=source + ".wav",
                question=question, votes=["yes", "no", "yes"], target=1)


def test_deterministic_and_uploader_grouped():
    rows = [row(str(i), "person" + str(i // 3)) for i in range(100)]
    first = split_questions(rows)
    assert first == split_questions(list(reversed(rows)))
    owners = {}
    for split, values in first.items():
        for value in values:
            assert owners.setdefault(value["uploader"], split) == split
    assert sum(map(len, first.values())) == 100
    assert all(first.values())


def test_all_questions_and_aliases_stay_together():
    result = split_questions([row(), row(question="Is there music?", uploader=" PERSON ")])
    assert sorted(map(len, result.values())) == [0, 0, 2]


@pytest.mark.parametrize("rows", [
    [row(), row(uploader="Other", question="Another?")],
    [row(), row(question=" IS THERE SPEECH? ")],
    [dict(row(), target=0)],
    [dict(row(), votes=["yes"])],
])
def test_reject_bad_identity_or_votes(rows):
    with pytest.raises(ValueError):
        split_questions(rows)
