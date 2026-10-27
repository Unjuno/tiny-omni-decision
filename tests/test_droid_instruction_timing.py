import pytest

from scripts.audit_droid_instruction_timing import availability


def test_target_and_future_instruction_not_available():
    seq = [b""] * 10
    seq[8] = b"future"
    result = availability([seq], 10)
    assert result["windows"] == 2
    assert result["at_previous_step"] == 1
    assert result["at_any_past_step"] == 1


def test_past_instruction_and_alternative_fields():
    first = [b""] * 10
    second = [b""] * 10
    second[2] = b"instruction"
    result = availability([first, second], 10)
    assert result["at_any_past_step"] == 2
    assert result["at_previous_step"] == 0
    assert result["episode_any"] == 1


def test_bad_length_fails_closed():
    with pytest.raises(ValueError, match="length"):
        availability([[b""] * 9], 10)
