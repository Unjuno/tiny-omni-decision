from scripts.evaluate_instruction_condition_probe import examples, prompt


def test_instruction_prompt_does_not_leak_target():
    for row in examples():
        assert row["state"] in prompt(row)
        assert row["question"] in prompt(row)
        assert prompt(row) == prompt(dict(row, target=(row["target"] + 1) % 3))
        assert prompt(row).endswith("Return only the option label A, B, or C.")
