import pytest

from tiny_omni_decision.qasc_cv import build_qasc_train_folds, qasc_train_component_ids


def _row(row_id, group, question, options, target):
    return {
        "id": row_id,
        "source_record_id": f"{group}:{row_id}",
        "question": question,
        "options": options,
        "target": target,
        "split": "train",
    }


def _rows():
    rows = []
    for index in range(30):
        options = [f"answer {j}" for j in range(8)]
        rows.append(
            _row(
                f"id-{index}",
                f"pair-{index // 2}",
                f"Which result applies to case {index // 2}?",
                options,
                options[index % 8],
            )
        )
    return rows


def test_qasc_folds_are_deterministic_complete_and_grouped():
    rows = _rows()
    first, report = build_qasc_train_folds(rows, folds=5, seed=17)
    second, _ = build_qasc_train_folds(rows, folds=5, seed=17)

    assert first == second
    assert len(first) == len(rows)
    assert all(0 <= fold < 5 for fold in first)
    assert set(first) == set(range(5))
    assert report["rows"] == len(rows)
    assert report["connected_components"] == 15
    assert report["group_integrity_verified"] is True
    for index in range(0, len(rows), 2):
        assert first[index] == first[index + 1]


def test_qasc_folds_join_normalized_question_and_content_duplicates():
    options = [f"choice {i}" for i in range(8)]
    rows = [
        _row("a", "pair-a", "Question repeated?", options, options[0]),
        _row("b", "pair-b", "  QUESTION   REPEATED? ", options, options[1]),
        _row("c", "pair-c", "Different question?", list(reversed(options)), options[-1]),
        _row("d", "pair-d", "Other question?", options, options[2]),
    ]
    assignment, report = build_qasc_train_folds(rows, folds=2, seed=17)

    assert assignment[0] == assignment[1]
    assert report["connected_components"] == 3
    assert assignment[2] != assignment[0] or assignment[3] != assignment[0]


def test_qasc_component_ids_match_question_overlap_clusters_and_are_stable():
    options = [f"choice {i}" for i in range(8)]
    rows = [
        _row("a", "pair-a", "Question repeated?", options, options[0]),
        _row("b", "pair-b", " QUESTION   REPEATED? ", options, options[1]),
        _row("c", "pair-c", "Independent question?", list(reversed(options)), options[-1]),
    ]

    first = qasc_train_component_ids(rows)
    second = qasc_train_component_ids(rows)

    assert first == second
    assert first[0] == first[1]
    assert first[2] != first[0]


def test_qasc_folds_reject_too_few_independent_components():
    rows = [
        _row("a", "same", "Question A?", [f"x{i}" for i in range(8)], "x0"),
        _row("b", "same", "Question B?", [f"y{i}" for i in range(8)], "y0"),
    ]
    with pytest.raises(ValueError, match="independent connected components"):
        build_qasc_train_folds(rows, folds=2, seed=17)


def test_qasc_fold_overlap_audit_rejects_split_fact_pair():
    from tiny_omni_decision.qasc_cv import audit_qasc_train_folds

    rows = _rows()[:4]
    with pytest.raises(ValueError, match="fold leakage"):
        audit_qasc_train_folds(rows, [0, 1, 0, 1], folds=2)


def test_qasc_fold_overlap_audit_accepts_disjoint_components():
    from tiny_omni_decision.qasc_cv import audit_qasc_train_folds

    rows = _rows()
    assignment, _ = build_qasc_train_folds(rows, folds=5, seed=17)
    audits = audit_qasc_train_folds(rows, assignment, folds=5)

    assert len(audits) == 5
    assert all(row["train_held_fact_pair_overlap"] == 0 for row in audits)
    assert all(row["train_held_question_overlap"] == 0 for row in audits)
    assert all(row["train_held_content_overlap"] == 0 for row in audits)
