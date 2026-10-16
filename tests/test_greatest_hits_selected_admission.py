import copy

import numpy as np
import pytest


def receipt():
    return dict(
        status="PASS_MATCHED_AV_TRAIN",
        arm="C0",
        step=32768,
        optimizer_updates=32768,
        encoder_updates=0,
        best_step=512,
        best_final_reload_verified=True,
        input_digest="inputs",
        config_sha256="config",
        config=dict(plan_sha256="plan"),
        source_commit="source",
        versions={"torch": "v"},
    )


def test_selected_checkpoint_requires_completed_training_and_exact_binding():
    from scripts.greatest_hits_selected_admission import require_selected

    run = receipt()
    best = dict(step=512, checkpoint="checkpoint-00512.pt", checkpoint_sha256="weights")
    saved = dict(step=512, sampler_sha256="plan", config_sha256="config")
    require_selected(run, best, saved, "weights")
    for key, value in (
        ("status", "PASS_MATCHED_AV_SMOKE"),
        ("step", 512),
        ("optimizer_updates", 512),
        ("encoder_updates", 1),
        ("best_final_reload_verified", False),
        ("best_step", 1024),
    ):
        changed = copy.deepcopy(run)
        changed[key] = value
        with pytest.raises(ValueError):
            require_selected(changed, best, saved, "weights")
    for key in ("step", "sampler_sha256", "config_sha256"):
        changed = dict(saved, **{key: "wrong"})
        with pytest.raises(ValueError):
            require_selected(run, best, changed, "weights")
    with pytest.raises(ValueError):
        require_selected(run, best, saved, "other")


def test_natural_arrays_align_by_id_preserving_full_option_order():
    from scripts.greatest_hits_selected_admission import natural_reference

    rows = [
        dict(id="b", split="validation", options=["x", "y", "z"]),
        dict(id="a", split="validation", options=["x", "y"]),
    ]
    saved = list(reversed(rows))
    scores = np.full((2, 40), -1e30, dtype=np.float32)
    scores[0, :2], scores[1, :3] = [1, 2], [3, 4, 5]
    result = natural_reference(rows, saved, scores)
    np.testing.assert_array_equal(result["a"], [1, 2])
    np.testing.assert_array_equal(result["b"], [3, 4, 5])
    for change in ("duplicate", "options", "missing", "split", "padding", "nan"):
        meta, matrix = copy.deepcopy(saved), scores.copy()
        if change == "duplicate":
            meta[1] = meta[0]
        if change == "options":
            meta[0]["options"].reverse()
        if change == "missing":
            meta.pop()
        if change == "split":
            meta[0]["split"] = "train"
        if change == "padding":
            matrix[0, 2] = 1
        if change == "nan":
            matrix[0, 0] = np.nan
        with pytest.raises(ValueError):
            natural_reference(rows, meta, matrix)


def test_all_five_selections_must_be_complete_and_share_input():
    from scripts.greatest_hits_selected_admission import require_selection_set

    runs = {arm: dict(receipt(), arm=arm) for arm in ("C0", "E0G0", "E0G1", "E1G0", "E1G1")}
    require_selection_set(runs)
    runs["E1G1"]["status"] = "RUNNING"
    with pytest.raises(ValueError):
        require_selection_set(runs)
    runs["E1G1"]["status"] = "PASS_MATCHED_AV_TRAIN"
    runs["E1G1"]["input_digest"] = "different"
    with pytest.raises(ValueError):
        require_selection_set(runs)
    runs.pop("E1G1")
    with pytest.raises(ValueError):
        require_selection_set(runs)


@pytest.mark.parametrize("field", ["source_commit", "config", "versions"])
def test_selection_set_rejects_nonmatched_training_conditions(field):
    from scripts.greatest_hits_selected_admission import require_selection_set

    runs = {arm: dict(receipt(), arm=arm, source_commit="source", versions={"torch": "v"})
            for arm in ("C0", "E0G0", "E0G1", "E1G0", "E1G1")}
    runs["E1G1"][field] = "different"
    with pytest.raises(ValueError):
        require_selection_set(runs)
