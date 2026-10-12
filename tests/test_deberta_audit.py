import json

from scripts.audit_deberta_xsmall import serializable_loading_info


def test_transformers_loading_sets_serialize_without_losing_names():
    x = serializable_loading_info(
        {"missing_keys": set(), "unexpected_keys": {"b", "a"}, "errors": []}
    )
    assert json.loads(json.dumps(x)) == {
        "missing_keys": [],
        "unexpected_keys": ["a", "b"],
        "errors": [],
    }
