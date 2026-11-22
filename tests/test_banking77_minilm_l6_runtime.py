import pytest

from tiny_omni_decision.banking77_minilm_l6 import (
    CLASS_COUNT,
    Banking77MiniLML6Decision,
    class_ids,
)


def test_banking77_class_ids_are_stable_and_complete():
    ids = class_ids()
    assert len(ids) == CLASS_COUNT
    assert ids[0] == "banking77:activate_my_card"
    assert ids[-1] == "banking77:wrong_exchange_rate_for_cash_withdrawal"
    assert len(set(ids)) == CLASS_COUNT


def test_decision_returns_option_ids_in_caller_order():
    runtime = Banking77MiniLML6Decision.__new__(Banking77MiniLML6Decision)
    runtime.probabilities = lambda text: [1 / CLASS_COUNT] * CLASS_COUNT
    options = [
        {"option_id": f"choice-{index}", "class_id": class_id}
        for index, class_id in enumerate(class_ids())
    ]
    result = runtime.decide("where is my card", options[::-1])
    assert result["schema_version"] == "1.0"
    assert result["status"] == "selected"
    assert result["selected_option_id"] == "choice-51"
    assert result["options"][0]["option_id"] == "choice-76"
    assert len(result["options"]) == CLASS_COUNT


def test_decision_rejects_incomplete_taxonomy():
    runtime = Banking77MiniLML6Decision.__new__(Banking77MiniLML6Decision)
    runtime.probabilities = lambda text: [1 / CLASS_COUNT] * CLASS_COUNT
    options = [
        {"option_id": f"choice-{index}", "class_id": class_id}
        for index, class_id in enumerate(class_ids()[:-1])
    ]
    with pytest.raises(ValueError, match="all classes"):
        runtime.decide("where is my card", options)
