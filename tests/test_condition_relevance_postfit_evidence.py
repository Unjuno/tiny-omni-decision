import copy

import pytest
from test_condition_relevance_postfit import pair


def fixture():
    from scripts.decision_semantic_training import learning_rate

    run, _ = pair()
    run["config"].update(learning_rate=2e-5, warmup_updates=31)
    schema = {f"task{i}": 2 for i in range(47)}
    metric = dict(count=2, accuracy=0.5, nll=1.0, brier=0.5, ece=0.1)
    history = [
        dict(
            step=step,
            purpose="validation",
            metrics={t: dict(metric) for t in schema},
            macro={k: metric[k] for k in ("accuracy", "nll", "brier", "ece")},
            minimum_accuracy=0.5,
            best_step=0,
            consumed=dict(consumed=18 * step, unique_questions=18 * step, repeated_questions=0),
        )
        for step in (0, 256, 512, 768, 1024)
    ]
    updates = [
        dict(
            step=step,
            learning_rate=learning_rate(step, 1024, 31, 2e-5),
            consumed=18 * step,
            ce=0.5,
            kl=0.1,
            relevance_loss=0.3,
            gradient_norm=1.0,
            auxiliary_gradient_norm_after_clip=0.0,
            seconds=1.0,
        )
        for step in range(1, 1025)
    ]
    order = [
        [[source, i] for source, n in enumerate((8, 8, 2)) for i in range(step * n, (step + 1) * n)]
        for step in range(1024)
    ]
    for part in ("validation", "fixed-train"):
        for suffix in ("scores.npy", "metrics.json", "predictions.jsonl"):
            run["artifacts"][f"reload-{part}-{suffix}"] = run["artifacts"][f"{part}-0-{suffix}"]
    return run, history, updates, order, schema


def validate(args):
    from scripts.condition_relevance_postfit_evidence import audit_fit_evidence

    return audit_fit_evidence(*args)


def test_complete_timeline_and_selector_evidence():
    result = validate(fixture())
    assert result["updates"] == 1024
    assert result["positive_lr_updates"] == 1023
    assert result["consumed_questions"] == 18432
    assert result["best_step"] == 0
    assert result["selected_consumed_questions"] == 0
    assert result["bucket_counts"] == [8192, 8192, 2048]


@pytest.mark.parametrize(
    "change",
    [
        "partial",
        "lr",
        "consumption",
        "nonfinite",
        "ce_aux",
        "duplicated_order",
        "bool_index",
        "missing_eval",
        "selector",
        "macro",
        "reload",
        "schema",
        "history_consumption",
    ],
)
def test_changed_evidence_rejected(change):
    args = fixture()
    run, history, updates, order, schema = args
    if change == "partial":
        updates.pop()
    if change == "lr":
        updates[511]["learning_rate"] = 0.0
    if change == "consumption":
        updates[400]["consumed"] -= 1
    if change == "nonfinite":
        updates[300]["gradient_norm"] = float("nan")
    if change == "ce_aux":
        updates[50]["auxiliary_gradient_norm_after_clip"] = 0.1
    if change == "duplicated_order":
        order[1][0] = order[0][0]
    if change == "bool_index":
        order[0][0][1] = False
    if change == "missing_eval":
        history.pop(1)
    if change == "selector":
        history[-1]["best_step"] = 1024
    if change == "macro":
        history[-1]["macro"]["nll"] = 0.5
    if change == "reload":
        run["artifacts"]["reload-validation-scores.npy"] = "9" * 64
    if change == "schema":
        schema.pop("task1")
    if change == "history_consumption":
        history[1]["consumed"]["unique_questions"] -= 1
    with pytest.raises(ValueError):
        validate(args)


def test_nonzero_selected_step_keeps_accounting_separate():
    args = fixture()
    run, history, _, _, _ = args
    for entry in history[2:]:
        entry["metrics"] = copy.deepcopy(entry["metrics"])
        for values in entry["metrics"].values():
            values["nll"] = 0.99
        entry["macro"]["nll"] = 0.99
        entry["best_step"] = 512
    run["best_step"] = 512
    run["selected_consumed"].update(consumed=9216, unique_questions=9216)
    for part in ("validation", "fixed-train"):
        for suffix in ("scores.npy", "metrics.json", "predictions.jsonl"):
            run["artifacts"][f"{part}-512-{suffix}"] = "e" * 64
            run["artifacts"][f"reload-{part}-{suffix}"] = "e" * 64
    result = validate(args)
    assert result["best_step"] == 512 and result["selected_consumed_questions"] == 9216
