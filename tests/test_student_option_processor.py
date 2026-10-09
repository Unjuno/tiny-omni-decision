from tiny_omni_decision.student import processor_inputs_for_option


def test_missing_single_option_uses_regular_processor_without_two_choice_guard():
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        return kwargs

    result = processor_inputs_for_option(processor, "candidate")
    assert result == {
        "text": ["task: sentence similarity | query: candidate"],
        "return_tensors": "pt",
    }
    assert calls == [result]
