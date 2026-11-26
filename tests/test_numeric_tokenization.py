from scripts.audit_numeric_tokenization import numeric_spans


def test_numeric_spans_exclude_labels_and_other_fields():
    text = 'State: [{"id": "123", "cost": 123, "risk": 42, "time": 1000, "other": 9}]'
    assert [text[a:b] for a, b in numeric_spans(text)] == ["123", "42", "1000"]
