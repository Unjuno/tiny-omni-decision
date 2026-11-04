import json

from scripts.audit_qasc_evidence_cv_result import _probabilities
from scripts.prepare_qasc_evidence_cv import _source_group, enrich_train_rows
from scripts.run_qasc_evidence_train_cv import _fold_plan_matches, _paired_group_bootstrap


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


def test_paired_bootstrap_rejects_mismatched_target_or_fold():
    rows = [
        {
            "id": "q1",
            "source_record_id": "group-a:q1",
            "question": "Which answer is correct?",
            "options": [f"answer {index}" for index in range(8)],
            "target": "answer 0",
        }
    ]
    probabilities = [1 / 8] * 8
    candidate = [
        {
            "id": "q1",
            "target_index": 0,
            "prediction_index": 0,
            "fold": 1,
            "probabilities": probabilities,
        }
    ]
    reference = [
        {
            "id": "q1",
            "target_index": 1,
            "prediction_index": 1,
            "fold": 2,
            "probabilities": probabilities,
        }
    ]

    try:
        _paired_group_bootstrap(rows, candidate, reference, replicates=10)
    except ValueError as error:
        assert "target or fold" in str(error)
    else:
        raise AssertionError("paired bootstrap must reject mismatched targets or folds")


def test_paired_bootstrap_rejects_invalid_option_probabilities():
    rows = [
        {
            "id": "q1",
            "source_record_id": "group-a:q1",
            "question": "Which answer is correct?",
            "options": [f"answer {index}" for index in range(8)],
            "target": "answer 0",
        }
    ]
    candidate = [
        {
            "id": "q1",
            "target_index": 0,
            "prediction_index": 0,
            "fold": 0,
            "probabilities": [0.5, 0.5],
        }
    ]
    reference = [
        {
            "id": "q1",
            "target_index": 0,
            "prediction_index": 0,
            "fold": 0,
            "probabilities": [1 / 8] * 8,
        }
    ]

    try:
        _paired_group_bootstrap(rows, candidate, reference, replicates=10)
    except ValueError as error:
        assert "eight-choice distributions" in str(error)
    else:
        raise AssertionError("paired bootstrap must reject malformed option probabilities")


def test_fold_comparison_checks_exact_membership_independent_of_assignment_file_order(tmp_path):
    rows = [{"id": f"q{index}"} for index in range(5)]
    assignment = [0, 1, 2, 3, 4]
    plan = {
        "folds": 5,
        "seed": 17,
        "folds_report": [{"fold": index, "rows": 1} for index in range(5)],
    }
    reference_assignment = [{"id": f"q{index}", "fold": index} for index in reversed(range(5))]
    plan_path = tmp_path / "fold-plan.json"
    assignment_path = tmp_path / "fold-assignment.jsonl"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assignment_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reference_assignment), encoding="utf-8"
    )

    assert _fold_plan_matches(rows, assignment, plan_path, assignment_path)

    reference_assignment[0]["fold"] = 0
    assignment_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reference_assignment), encoding="utf-8"
    )
    try:
        _fold_plan_matches(rows, assignment, plan_path, assignment_path)
    except ValueError as error:
        assert "membership" in str(error)
    else:
        raise AssertionError("fold comparison must reject changed ID-to-fold assignment")


def test_saved_probability_audit_requires_eight_finite_normalized_options():
    assert _probabilities([{"probabilities": [1 / 8] * 8}]).shape == (1, 8)

    try:
        _probabilities([{"probabilities": [0.5, 0.5]}])
    except ValueError as error:
        assert "eight option probabilities" in str(error)
    else:
        raise AssertionError("saved probability audit must reject non-eight-way outputs")

    try:
        _probabilities([{"probabilities": [float("nan")] * 8}])
    except ValueError as error:
        assert "finite and nonnegative" in str(error)
    else:
        raise AssertionError("saved probability audit must reject non-finite values")
