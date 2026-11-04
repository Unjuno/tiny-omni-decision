"""Fail-closed loader for the pinned uncased BERT vocabulary used by All-in-one."""

import hashlib
import inspect
from pathlib import Path

VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"


def validate_tokenizer(tokenizer):
    if tokenizer.vocab_size != 30522:
        raise ValueError("BERT vocabulary size mismatch")
    if (
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    ) != (0, 100, 101, 102):
        raise ValueError("BERT special token mismatch")
    ids = tokenizer("first second third", add_special_tokens=False)["input_ids"]
    if len(ids) != 3 or len(set(ids)) != 3 or tokenizer.unk_token_id in ids:
        raise ValueError("BERT ordinal tokens collapsed")
    if tokenizer("first", add_special_tokens=False)["input_ids"] != [2034]:
        raise ValueError("BERT known token mismatch")


def load_pinned_bert_tokenizer(vocab_path):
    path = Path(vocab_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != VOCAB_SHA256:
        raise ValueError("BERT vocabulary hash mismatch")
    from transformers import BertTokenizer

    arguments = inspect.signature(BertTokenizer.__init__).parameters
    if "vocab" in arguments:
        kwargs = {"vocab": str(path)}
    elif "vocab_file" in arguments:
        kwargs = {"vocab_file": str(path)}
    else:
        raise ValueError("unsupported BERT tokenizer constructor")
    tokenizer = BertTokenizer(**kwargs, do_lower_case=True)
    validate_tokenizer(tokenizer)
    return tokenizer
