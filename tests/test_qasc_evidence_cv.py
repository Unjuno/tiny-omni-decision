from scripts.prepare_qasc_evidence_cv import _source_group, enrich_train_rows


def _prepared(row_id, group, question, options, target):
    return {
        "id": row_id,
        "source_record_id": f"{group}:{row_id}",
        "state": "",
        "question": question,
        "options": options,
        "target": target,
        "split": "train",
    }


def _source(row_id, question, options, target, fact1, fact2):
    labels = list("ABCDEFGH")
    return {
        "id": row_id,
        "question": question,
        "choices": {"label": labels, "text": options},
        "answerKey": labels[options.index(target)],
        "fact1": fact1,
        "fact2": fact2,
    }


def test_evidence_adapter_joins_rows_and_preserves_fact_group_and_target():
    options = [f"answer {index}" for index in range(8)]
    prepared = [
        _prepared(
            "q1", _source_group("Fact one.", "Fact two."), "What is true?", options, options[3]
        )
    ]
    source = [_source("q1", "What is true?", options, options[3], "Fact one.", "Fact two.")]

    enriched, report = enrich_train_rows(prepared, source)

    assert enriched[0]["state"] == "Evidence:\nFact one.\nFact two."
    assert enriched[0]["source_record_id"] == prepared[0]["source_record_id"]
    assert enriched[0]["target"] == options[3]
    assert enriched[0]["options"] == options
    assert enriched[0]["split"] == "train"
    assert report["rows"] == 1
    assert report["source_id_order_preserved"] is True


def test_evidence_adapter_rejects_mismatched_source_rows():
    options = [f"answer {index}" for index in range(8)]
    prepared = [
        _prepared(
            "q1", _source_group("Fact one.", "Fact two."), "What is true?", options, options[3]
        )
    ]
    source = [_source("q2", "What is true?", options, options[3], "Fact one.", "Fact two.")]

    try:
        enrich_train_rows(prepared, source)
    except ValueError as error:
        assert "ID order" in str(error)
    else:
        raise AssertionError("mismatched source rows must be rejected")

