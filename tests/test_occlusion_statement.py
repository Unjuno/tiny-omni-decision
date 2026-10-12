import pytest

from tiny_omni_decision.occlusion_statement import OPTIONS, QUESTIONS, occlusion_statement


def test_all_relations_and_conditions_preserved():
    outputs = set()
    for question, time in QUESTIONS.items():
        for candidate in OPTIONS:
            result = occlusion_statement(question, candidate)
            assert f"At the {time}," in result
            assert "person's viewpoint" in result
            assert result.endswith(candidate[0].lower() + candidate[1:])
            outputs.add(result)
    assert len(outputs) == len(OPTIONS) * len(QUESTIONS)


def test_unknown_input_fails_closed():
    with pytest.raises(ValueError):
        occlusion_statement("Where?", next(iter(OPTIONS)))
    with pytest.raises(ValueError):
        occlusion_statement(next(iter(QUESTIONS)), "Under object A.")
