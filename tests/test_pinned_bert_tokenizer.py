import hashlib
import sys
from types import SimpleNamespace

import pytest

from tiny_omni_decision import pinned_bert_tokenizer as loader
from tiny_omni_decision.pinned_bert_tokenizer import validate_tokenizer


class Tokenizer:
    vocab_size = 30522
    pad_token_id, unk_token_id, cls_token_id, sep_token_id = 0, 100, 101, 102

    def __call__(self, text, **kwargs):
        return {"input_ids": [2034] if text == "first" else [2034, 2117, 2353]}


def test_pinned_ids_admitted():
    validate_tokenizer(Tokenizer())


def test_silently_ignored_vocabulary_rejected():
    with pytest.raises(ValueError, match="size"):
        validate_tokenizer(SimpleNamespace(vocab_size=5))


def test_unknown_collapse_rejected():
    class Broken(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [100, 100, 100]}

    with pytest.raises(ValueError, match="collapsed"):
        validate_tokenizer(Broken())


def test_special_ids_rejected():
    t = Tokenizer()
    t.cls_token_id = 2
    with pytest.raises(ValueError, match="special"):
        validate_tokenizer(t)


def test_wrong_vocab_hash_rejected(tmp_path):
    path = tmp_path / "vocab.txt"
    path.write_text("wrong", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        loader.load_pinned_bert_tokenizer(path)


@pytest.mark.parametrize("argument", ["vocab", "vocab_file"])
def test_supported_constructor_receives_pinned_path(tmp_path, monkeypatch, argument):
    path = tmp_path / "vocab.txt"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(loader, "VOCAB_SHA256", hashlib.sha256(b"fixture").hexdigest())

    class Modern(Tokenizer):
        def __init__(self, vocab, do_lower_case):
            self.path, self.lowercase = vocab, do_lower_case

    class Legacy(Tokenizer):
        def __init__(self, vocab_file, do_lower_case):
            self.path, self.lowercase = vocab_file, do_lower_case

    constructor = Modern if argument == "vocab" else Legacy
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BertTokenizer=constructor))
    tokenizer = loader.load_pinned_bert_tokenizer(path)
    assert tokenizer.path == str(path)
    assert tokenizer.lowercase is True
