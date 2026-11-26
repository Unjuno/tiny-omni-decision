import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_text_token_contract_preserves_query_option_format_and_order():
    from scripts.train_shared_text_video import text_token_contract

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == dict(max_length=128, truncation=True)
            self.texts = texts
            return {"input_ids": [[ord(c) for c in text][:128] for text in texts]}

    rows = [dict(question="What happens?", options=["wrong", "correct"])]
    tokenizer = Tokenizer()
    first = text_token_contract(rows, tokenizer)
    assert tokenizer.texts == [
        "Question: What happens?",
        "Candidate answer: wrong",
        "Candidate answer: correct",
    ]
    assert first["strings"] == 3
    assert first == text_token_contract(rows, tokenizer)
    changed = [dict(rows[0], options=["correct", "wrong"])]
    assert first["ids_sha256"] != text_token_contract(changed, tokenizer)["ids_sha256"]


def test_worker_environment_adds_repo_source_without_mutating_parent(monkeypatch, tmp_path):
    import os

    from scripts.train_shared_text_video import worker_environment

    monkeypatch.setenv("PYTHONPATH", "existing-path")
    result = worker_environment(tmp_path)
    assert result["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "src"),
        str(tmp_path),
        "existing-path",
    ]
    assert os.environ["PYTHONPATH"] == "existing-path"
